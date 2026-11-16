"""Aggregate saved runs into results/metrics.json and results/report.md."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import torch

from recsys.config import Config
from recsys.data import load_phase, load_ratings
from recsys.stats import bootstrap_ci, paired_bootstrap, sign_test

RESULTS = Path("results")
METRICS = ["recall@10", "recall@50", "ndcg@10", "mrr"]
BUCKETS = [(1, 9), (10, 49), (50, 199), (200, 10**9)]


def load_npz(name: str) -> dict:
    f = np.load(RESULTS / "per_user" / f"test_{name}.npz")
    return {k: f[k] for k in f.files}


def hardware() -> dict:
    def sysctl(key):
        try:
            return subprocess.check_output(["sysctl", "-n", key], text=True).strip()
        except Exception:
            return None

    mem = sysctl("hw.memsize")
    return {
        "chip": sysctl("machdep.cpu.brand_string"),
        "memory_gb": round(int(mem) / 2**30) if mem else None,
        "os": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
    }


def ci_block(per_user: dict, n_boot: int, seed: int) -> dict:
    out = {}
    for m in METRICS:
        mean, lo, hi = bootstrap_ci(per_user[f"m_{m}"], n_boot, seed)
        out[m] = {"mean": mean, "ci95": [lo, hi]}
    return out


def fmt(x: float) -> str:
    return f"{x:.4f}"


def build_report(cfg: Config) -> None:
    nb = cfg.eval["bootstrap_samples"]
    seed = cfg.eval["seed"]
    final = json.loads((RESULTS / "final_runs.json").read_text())
    tune_als = json.loads((RESULTS / "tuning_als.json").read_text())
    tune_tt = json.loads((RESULTS / "tuning_two_tower.json").read_text())
    ranker = None
    if (RESULTS / "ranker.json").exists():
        ranker = json.loads((RESULTS / "ranker.json").read_text())
    seeds = [str(s) for s in cfg.two_tower["seeds"]]

    ratings = load_ratings(cfg)
    val = load_phase(cfg, "val", ratings)
    test = load_phase(cfg, "test", ratings)
    pos = ratings[ratings.rating >= cfg.data["positive_threshold"]]

    runs = {
        "popularity": load_npz("popularity"),
        "popularity_recent": load_npz("popularity_recent"),
        "als": load_npz("als"),
    }
    for s in seeds:
        runs[f"two_tower_seed{s}"] = load_npz(f"two_tower_seed{s}")
        if ranker:
            runs[f"ranker_seed{s}"] = load_npz(f"ranker_seed{s}")
    users = runs["als"]["users"]
    for name, r in runs.items():
        assert np.array_equal(r["users"], users), f"user order differs for {name}"

    # ------------------------------------------------------------ main table
    table = {}
    for name in ("popularity", "popularity_recent", "als"):
        table[name] = ci_block(runs[name], nb, seed)
        table[name]["coverage@10"] = final[name]["test"]["coverage@10"]
    for fam in ("two_tower", "ranker") if ranker else ("two_tower",):
        per_seed = {s: ci_block(runs[f"{fam}_seed{s}"], nb, seed) for s in seeds}
        cov = (
            [final["two_tower"]["seeds"][s]["test"]["coverage@10"] for s in seeds]
            if fam == "two_tower"
            else [ranker["test"][s]["coverage@10"] for s in seeds]
        )
        agg = {}
        for m in METRICS:
            vals = [per_seed[s][m]["mean"] for s in seeds]
            agg[m] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "per_seed": vals,
            }
        agg["coverage@10"] = {
            "mean": float(np.mean(cov)),
            "std": float(np.std(cov, ddof=1)),
            "per_seed": cov,
        }
        table[fam] = {"across_seeds": agg, "per_seed_ci": per_seed}

    # ------------------------------------------------------------ significance
    def compare(a: str, b: str) -> dict:
        res = {}
        for m in METRICS:
            res[m] = paired_bootstrap(runs[a][f"m_{m}"], runs[b][f"m_{m}"], nb, seed)
        res["sign_test_ndcg@10"] = sign_test(runs[a]["m_ndcg@10"], runs[b]["m_ndcg@10"])
        res["sign_test_recall@50"] = sign_test(runs[a]["m_recall@50"], runs[b]["m_recall@50"])
        return res

    sig = {}
    for s in seeds:
        sig[f"two_tower_seed{s}_vs_als"] = compare(f"two_tower_seed{s}", "als")
        sig[f"two_tower_seed{s}_vs_popularity_recent"] = compare(
            f"two_tower_seed{s}", "popularity_recent"
        )
        if ranker:
            sig[f"ranker_seed{s}_vs_two_tower_seed{s}"] = compare(
                f"ranker_seed{s}", f"two_tower_seed{s}"
            )
            sig[f"ranker_seed{s}_vs_als"] = compare(f"ranker_seed{s}", "als")
            sig[f"ranker_seed{s}_vs_popularity_recent"] = compare(
                f"ranker_seed{s}", "popularity_recent"
            )

    # ------------------------------------------------------------ analysis
    hist_len = np.bincount(test.fit.user.to_numpy(), minlength=int(ratings.user.max()) + 1)[users]
    buckets = {}
    show = ["popularity_recent", "als", "two_tower_seed0"] + (["ranker_seed0"] if ranker else [])
    for lo, hi in BUCKETS:
        mask = (hist_len >= lo) & (hist_len <= hi)
        key = f"{lo}-{hi}" if hi < 10**9 else f"{lo}+"
        b = {"n_users": int(mask.sum())}
        for name in show:
            b[name] = {}
            for m in ("recall@50", "ndcg@10"):
                if mask.sum():
                    mean, l_, h_ = bootstrap_ci(runs[name][f"m_{m}"][mask], nb, seed)
                    b[name][m] = {"mean": mean, "ci95": [l_, h_]}
        buckets[key] = b

    counts = np.bincount(test.fit.item.to_numpy(), minlength=test.n_items).astype(float)
    elig = np.flatnonzero(test.eligible)
    order = elig[np.argsort(-counts[elig], kind="stable")]
    pct = np.zeros(test.n_items)
    pct[order] = 1.0 - np.arange(len(order)) / len(order)  # 1.0 = most popular
    head_n = max(1, int(0.01 * len(order)))
    head = np.zeros(test.n_items, bool)
    head[order[:head_n]] = True
    tail_tg = [test.targets[int(u)][~head[test.targets[int(u)]]] for u in users]
    has_tail = np.array([len(t) > 0 for t in tail_tg])
    all_tg = np.concatenate([test.targets[int(u)] for u in users])
    pop_bias = {
        "definition": "head = top 1% of candidate items by positives before test_start "
        f"({head_n} items); popularity percentile 1.0 = most popular",
        "targets": {
            "share_of_test_positives_on_head_items": float(head[all_tg].mean()),
            "mean_popularity_percentile": float(pct[all_tg].mean()),
        },
        "users_with_tail_targets": int(has_tail.sum()),
    }
    for name in (
        ["popularity", "popularity_recent", "als"]
        + [f"two_tower_seed{s}" for s in seeds]
        + ([f"ranker_seed{s}" for s in seeds] if ranker else [])
    ):
        top10 = runs[name]["candidates"][:, :10]
        top50 = runs[name]["candidates"][:, :50]
        tail_rec = [
            np.isin(top50[i], tail_tg[i]).sum() / len(tail_tg[i]) for i in np.flatnonzero(has_tail)
        ]
        pop_bias[name] = {
            "top10_share_head_items": float(head[top10].mean()),
            "top10_mean_popularity_percentile": float(pct[top10].mean()),
            "tail_recall@50": float(np.mean(tail_rec)),
        }

    # ------------------------------------------------------------ dataset + timings
    dataset = {
        "name": "MovieLens 25M (GroupLens)",
        "url": cfg.data["url"],
        "md5": cfg.data["md5"],
        "ratings": int(len(ratings)),
        "users": int(ratings.user.nunique()),
        "items_rated": int(ratings.item.nunique()),
        "positives_rating_ge_4": int(len(pos)),
        "positive_definition": f"rating >= {cfg.data['positive_threshold']}",
        "split": {
            "type": "global temporal cutoff",
            "val_window": [cfg.data["val_start"], cfg.data["test_start"]],
            "test_window": [cfg.data["test_start"], "end of data (2019-11-21)"],
            "val_phase": final.get("val_phase") or None,
            "test_phase": final["test_phase"],
        },
    }
    from recsys.cli import phase_stats

    dataset["split"]["val_phase"] = phase_stats(val, int(ratings.user.max()) + 1)

    timings = {
        "als_tuning_fit_seconds": [r["fit_seconds"] for r in tune_als["runs"]],
        "two_tower_screening_train_seconds": [r["train_seconds"] for r in tune_tt["screening"]],
        "two_tower_val_full_run_train_seconds": tune_tt["full_run"]["train_seconds"],
        "final_fit_seconds": {
            "popularity": final["popularity"]["fit_seconds"],
            "popularity_recent": final["popularity_recent"]["fit_seconds"],
            "als": final["als"]["fit_seconds"],
            "two_tower_per_seed": [final["two_tower"]["seeds"][s]["train_seconds"] for s in seeds],
        },
    }
    if ranker:
        timings["ranker_fit_seconds"] = ranker["fit_seconds"]
    timings["total_training_seconds"] = round(
        sum(timings["als_tuning_fit_seconds"])
        + sum(timings["two_tower_screening_train_seconds"])
        + timings["two_tower_val_full_run_train_seconds"]
        + timings["final_fit_seconds"]["als"]
        + sum(timings["final_fit_seconds"]["two_tower_per_seed"])
        + timings.get("ranker_fit_seconds", 0.0),
        1,
    )

    metrics = {
        "dataset": dataset,
        "evaluation": {
            "ranking": "full catalogue: every candidate item (positive before cutoff) is scored; "
            "items the user rated before the cutoff are excluded",
            "test_users": int(len(users)),
            "bootstrap": f"{nb} resamples over users, percentile 95% CI",
            "metrics": "Recall@K = hits/|targets|; NDCG@K binary gains; MRR over full ranking",
        },
        "hardware": hardware(),
        "tuning": {
            "als": tune_als,
            "two_tower": {k: v for k, v in tune_tt.items() if k != "full_run"}
            | {
                "full_run_history": tune_tt["full_run"]["history"],
                "best_epoch": tune_tt["full_run"]["best_epoch"],
            },
        },
        "test": table,
        "significance": sig,
        "analysis": {"history_length_buckets": buckets, "popularity_bias": pop_bias},
        "ranker": ranker,
        "timings": timings,
        "final_runs": final,
    }
    (RESULTS / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    write_markdown(metrics, seeds)


def write_markdown(m: dict, seeds: list[str]) -> None:
    lines = [
        "| Model | Recall@10 | Recall@50 | NDCG@10 | MRR | Coverage@10 |",
        "|---|---|---|---|---|---|",
    ]
    names = {
        "popularity": "Popularity (all-time)",
        "popularity_recent": "Popularity (last 30 days)",
        "als": "iALS matrix factorisation",
    }
    for k, label in names.items():
        r = m["test"][k]
        cells = [
            f"{fmt(r[x]['mean'])} [{fmt(r[x]['ci95'][0])}, {fmt(r[x]['ci95'][1])}]" for x in METRICS
        ]
        lines.append(f"| {label} | " + " | ".join(cells) + f" | {fmt(r['coverage@10'])} |")
    for fam, label in (("two_tower", "Two-tower (3 seeds)"), ("ranker", "Two-tower + LambdaRank")):
        if fam not in m["test"]:
            continue
        r = m["test"][fam]["across_seeds"]
        s0 = m["test"][fam]["per_seed_ci"]["0"]
        cells = [
            f"{fmt(r[x]['mean'])} ± {fmt(r[x]['std'])} (seed 0 CI {fmt(s0[x]['ci95'][0])}, "
            f"{fmt(s0[x]['ci95'][1])})"
            for x in METRICS
        ]
        cov = r["coverage@10"]
        lines.append(
            f"| {label} | " + " | ".join(cells) + f" | {fmt(cov['mean'])} ± {fmt(cov['std'])} |"
        )
    lines += [
        "",
        "Significance (paired bootstrap on per-user differences, sign test on NDCG@10):",
        "",
    ]
    lines.append("| Comparison | Metric | Diff | 95% CI | p (bootstrap) | Sign test W/L, p |")
    lines.append("|---|---|---|---|---|---|")
    for comp, res in m["significance"].items():
        st = res["sign_test_ndcg@10"]
        for x in METRICS:
            r = res[x]
            lines.append(
                f"| {comp} | {x} | {r['diff']:+.4f} | [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
                f"| {r['p_bootstrap']:.4f} | {st['wins']}/{st['losses']}, {st['p']:.2e} |"
            )
    (RESULTS / "report.md").write_text("\n".join(lines) + "\n")
