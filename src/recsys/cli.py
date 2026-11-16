"""Command-line entry point: python -m recsys.cli <command>."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from recsys.config import Config
from recsys.data import (
    download,
    load_item_features,
    load_phase,
    load_ratings,
    n_users_total,
    prepare,
)
from recsys.evaluate import EvalResult, evaluate
from recsys.models.als import fit_als
from recsys.models.popularity import PopularityScorer, popularity_counts
from recsys.train_two_tower import train_two_tower

RESULTS = Path("results")


def log(msg) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    RESULTS.mkdir(exist_ok=True)
    with open(RESULTS / "train_log.txt", "a") as f:
        f.write(line + "\n")


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2) + "\n")


def save_eval(name: str, phase: str, res: EvalResult) -> None:
    d = RESULTS / "per_user"
    d.mkdir(parents=True, exist_ok=True)
    np.savez(
        d / f"{phase}_{name}.npz",
        users=res.users,
        candidates=res.candidates,
        candidate_scores=res.candidate_scores,
        **{f"m_{k}": v for k, v in res.per_user.items()},
    )


def ks(cfg: Config) -> tuple[int, ...]:
    return tuple(cfg.eval["ks"])


def phase_stats(p, n_users: int) -> dict:
    nt = np.array([len(p.targets[int(u)]) for u in p.eval_users])
    return {
        "cutoff_utc": time.strftime("%Y-%m-%d", time.gmtime(p.cutoff)),
        "end_utc": None if p.end is None else time.strftime("%Y-%m-%d", time.gmtime(p.end)),
        "fit_positives": int(len(p.fit)),
        "fit_users": int(p.fit.user.nunique()),
        "fit_items": int(p.fit.item.nunique()),
        "candidate_items": int(p.eligible.sum()),
        "eval_users": int(len(p.eval_users)),
        "target_positives": int(nt.sum()),
        "targets_per_user_median": float(np.median(nt)),
        "target_positives_dropped_unseen_item": int(p.n_target_dropped),
        "target_users_dropped_no_history": int(p.n_target_users_no_history),
        "n_users_total": n_users,
    }


# ---------------------------------------------------------------- commands


def cmd_data(cfg: Config) -> None:
    download(cfg)
    prepare(cfg)


def cmd_tune_als(cfg: Config) -> None:
    ratings = load_ratings(cfg)
    nu = n_users_total(ratings)
    val = load_phase(cfg, "val", ratings)
    runs = []
    for override in cfg.als["grid"]:
        params = {k: cfg.als[k] for k in ("factors", "regularization", "alpha", "iterations")}
        params.update(override)
        sc, secs = fit_als(val, nu, seed=cfg.als["seed"], **params)
        res = evaluate(sc, val, ks(cfg))
        runs.append({"params": params, "fit_seconds": round(secs, 1), "val": res.summary()})
        log(f"als {params} fit {secs:.0f}s val {res.summary()}")
    best = max(runs, key=lambda r: r["val"]["recall@50"])
    write_json(
        RESULTS / "tuning_als.json", {"selection": "val recall@50", "runs": runs, "best": best}
    )


def cmd_tune_tt(cfg: Config) -> None:
    ratings = load_ratings(cfg)
    nu = n_users_total(ratings)
    val = load_phase(cfg, "val", ratings)
    feats = load_item_features(cfg, "val")
    base = dict(cfg.two_tower)
    screen = []
    for override in base["grid"]:
        c = dict(base, **override)
        c["max_epochs"] = base["screen_epochs"]
        c["samples_per_epoch"] = base["screen_samples_per_epoch"]
        _, info = train_two_tower(
            c,
            val,
            nu,
            feats,
            seed=0,
            early_stop=True,
            log=lambda m, o=override: log(f"screen {o} {m}"),
        )
        best_r50 = max(h["val_recall@50"] for h in info["history"])
        screen.append(
            {
                "override": override,
                "best_val_recall@50": best_r50,
                "train_seconds": info["train_seconds"],
            }
        )
    chosen = max(screen, key=lambda r: r["best_val_recall@50"])["override"]
    log(f"two-tower screening chose {chosen}")
    c = dict(base, **chosen)
    sc, info = train_two_tower(
        c, val, nu, feats, seed=0, early_stop=True, log=lambda m: log(f"tt-val {m}")
    )
    res = evaluate(sc, val, ks(cfg))
    save_eval("two_tower_seed0", "val", res)
    write_json(
        RESULTS / "tuning_two_tower.json",
        {
            "selection": "val recall@50 (screening), early stopping on val recall@50 (full run)",
            "screening": screen,
            "chosen": chosen,
            "full_run": info,
            "val": res.summary(),
        },
    )


def cmd_final(cfg: Config, only: str | None = None) -> None:
    """Refit on everything before test_start with tuned settings and score the test window once."""
    ratings = load_ratings(cfg)
    nu = n_users_total(ratings)
    test = load_phase(cfg, "test", ratings)
    out_path = RESULTS / "final_runs.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    out["test_phase"] = phase_stats(test, nu)

    if only in (None, "popularity"):
        for name, days in (
            ("popularity", None),
            ("popularity_recent", cfg.popularity["recent_days"]),
        ):
            t0 = time.time()
            sc = PopularityScorer(popularity_counts(test, days))
            fit_s = time.time() - t0
            res = evaluate(sc, test, ks(cfg), cfg.eval["candidates"])
            save_eval(name, "test", res)
            out[name] = {
                "fit_seconds": round(fit_s, 2),
                "test": res.summary(),
                "params": {"recent_days": days},
            }
            log(f"final {name} {res.summary()}")
            write_json(out_path, out)

    if only in (None, "als"):
        best = json.loads((RESULTS / "tuning_als.json").read_text())["best"]["params"]
        sc, secs = fit_als(test, nu, seed=cfg.als["seed"], **best)
        res = evaluate(sc, test, ks(cfg), cfg.eval["candidates"])
        save_eval("als", "test", res)
        out["als"] = {"fit_seconds": round(secs, 1), "params": best, "test": res.summary()}
        log(f"final als {res.summary()}")
        write_json(out_path, out)

    if only in (None, "two_tower"):
        tune = json.loads((RESULTS / "tuning_two_tower.json").read_text())
        c = dict(cfg.two_tower, **tune["chosen"])
        epochs = tune["full_run"]["best_epoch"]
        feats = load_item_features(cfg, "test")
        seeds = {}
        for seed in cfg.two_tower["seeds"]:
            sc, info = train_two_tower(
                c,
                test,
                nu,
                feats,
                seed=seed,
                early_stop=False,
                fixed_epochs=epochs,
                log=lambda m, s=seed: log(f"tt-test seed{s} {m}"),
            )
            res = evaluate(sc, test, ks(cfg), cfg.eval["candidates"])
            save_eval(f"two_tower_seed{seed}", "test", res)
            info["test"] = res.summary()
            seeds[str(seed)] = info
            log(f"final two_tower seed {seed} {res.summary()}")
            out["two_tower"] = {"params": c, "epochs": epochs, "seeds": seeds}
            write_json(out_path, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["data", "tune-als", "tune-tt", "final", "report"])
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    cfg = Config.load(args.config)
    log(f"command {args.command} torch {torch.__version__} python {platform.python_version()}")
    t0 = time.time()
    if args.command == "data":
        cmd_data(cfg)
    elif args.command == "tune-als":
        cmd_tune_als(cfg)
    elif args.command == "tune-tt":
        cmd_tune_tt(cfg)
    elif args.command == "final":
        cmd_final(cfg, args.only)
    elif args.command == "report":
        from recsys.report import build_report

        build_report(cfg)
    log(f"command {args.command} done in {time.time() - t0:.0f}s wall")


if __name__ == "__main__":
    main()
