"""Second stage: LightGBM LambdaRank over the two-tower top-100 candidates.

The ranker is trained on the val phase only (two-tower val run, seed 0, candidates for val users,
labels = val-window positives), using an 80/20 user split of val users for early stopping. It is
then applied unchanged to each test-phase two-tower seed's candidates, with features rebuilt from
data before test_start. Ranks beyond the candidate list are untouched, so a user whose first
relevant item is outside the top-100 keeps the stage-1 reciprocal rank.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

from recsys.config import Config
from recsys.data import load_item_features, load_phase, load_ratings
from recsys.metrics import batch_metrics, coverage
from recsys.split import PhaseData

RESULTS = Path("results")

FEATURES = [
    "stage1_score",
    "stage1_rank",
    "log_pop_all",
    "log_pop_30d",
    "log_pop_7d",
    "pop_trend",
    "item_age_days",
    "year_bucket",
    "n_genres",
    "user_genre_affinity",
    "log_user_hist",
    "user_days_since_last",
    "user_mean_rating",
]


def phase_context(phase: PhaseData, ratings, feats) -> dict:
    fit = phase.fit
    n = phase.n_items
    item = fit.item.to_numpy()
    ts = fit.ts.to_numpy()
    c = phase.cutoff

    def cnt(days):
        m = ts >= c - days * 86400 if days else np.ones(len(ts), bool)
        return np.bincount(item[m], minlength=n).astype(np.float64)

    first_seen = np.full(n, c, dtype=np.int64)
    np.minimum.at(first_seen, item, ts)
    g = feats["genres"]
    n_users = int(ratings.user.max()) + 1
    users = fit.user.to_numpy()
    prof = np.zeros((n_users, g.shape[1]))
    np.add.at(prof, users, g[item])
    norm = np.linalg.norm(prof, axis=1, keepdims=True)
    prof = prof / np.maximum(norm, 1e-9)
    hist = np.bincount(users, minlength=n_users)
    last = np.zeros(n_users, dtype=np.int64)
    np.maximum.at(last, users, ts)
    before = ratings[ratings.ts < c]
    sums = np.bincount(before.user, weights=before.rating, minlength=n_users)
    cnts = np.bincount(before.user, minlength=n_users)
    p30, p7 = cnt(30), cnt(7)
    return {
        "pop_all": cnt(None),
        "pop_30d": p30,
        "pop_7d": p7,
        "pop_trend": np.log1p(p7 * 30 / 7) - np.log1p(p30),
        "age": (c - first_seen) / 86400,
        "genres": g,
        "gnorm": g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9),
        "years": feats["years"],
        "profile": prof,
        "hist": hist,
        "days_since_last": (c - last) / 86400,
        "mean_rating": sums / np.maximum(cnts, 1),
    }


def build_features(ctx: dict, users: np.ndarray, cands: np.ndarray, scores: np.ndarray):
    B, K = cands.shape
    it = cands.reshape(-1)
    uu = np.repeat(users, K)
    aff = (ctx["profile"][uu] * ctx["gnorm"][it]).sum(1)
    X = np.column_stack(
        [
            scores.reshape(-1),
            np.tile(np.arange(K), B),
            np.log1p(ctx["pop_all"][it]),
            np.log1p(ctx["pop_30d"][it]),
            np.log1p(ctx["pop_7d"][it]),
            ctx["pop_trend"][it],
            ctx["age"][it],
            ctx["years"][it],
            ctx["genres"][it].sum(1),
            aff,
            np.log1p(ctx["hist"][uu]),
            ctx["days_since_last"][uu],
            ctx["mean_rating"][uu],
        ]
    ).astype(np.float32)
    return X


def labels_for(phase: PhaseData, users: np.ndarray, cands: np.ndarray) -> np.ndarray:
    return np.stack([np.isin(cands[b], phase.targets[int(u)]) for b, u in enumerate(users)]).astype(
        np.int32
    )


def rerank_metrics(phase, users, cands, pred, stage1_mrr, ks, n_eligible):
    B, K = cands.shape
    order = np.argsort(-pred.reshape(B, K), axis=1, kind="stable")
    rer = np.take_along_axis(cands, order, axis=1)
    lab = labels_for(phase, users, rer)
    has = lab.any(1)
    first = np.where(has, lab.argmax(1) + 1, 0)
    targets = [phase.targets[int(u)] for u in users]
    m = batch_metrics(rer, np.where(has, first, 1), targets, ks)
    m["mrr"] = np.where(has, 1.0 / np.maximum(first, 1), stage1_mrr)
    cov = {f"coverage@{k}": coverage(rer, k, n_eligible) for k in ks}
    return m, cov, rer


def run_ranker(cfg: Config, log) -> None:
    rc = cfg.ranker
    ks = tuple(cfg.eval["ks"])
    ratings = load_ratings(cfg)
    val = load_phase(cfg, "val", ratings)
    vctx = phase_context(val, ratings, load_item_features(cfg, "val"))
    v = np.load(RESULTS / "per_user" / "val_two_tower_seed0.npz")
    users, cands, scores = v["users"], v["candidates"], v["candidate_scores"]
    K = cands.shape[1]
    X = build_features(vctx, users, cands, scores)
    y = labels_for(val, users, cands).reshape(-1)
    rng = np.random.default_rng(rc["seed"])
    is_tr = rng.random(len(users)) < 0.8
    rows_tr = np.repeat(is_tr, K)
    t0 = time.time()
    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "eval_at": [10],
        "learning_rate": rc["learning_rate"],
        "num_leaves": rc["num_leaves"],
        "min_data_in_leaf": rc["min_child_samples"],
        "seed": rc["seed"],
        "deterministic": True,
        "verbose": -1,
    }
    dtr = lgb.Dataset(X[rows_tr], y[rows_tr], group=[K] * int(is_tr.sum()), feature_name=FEATURES)
    dho = lgb.Dataset(X[~rows_tr], y[~rows_tr], group=[K] * int((~is_tr).sum()), reference=dtr)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=rc["n_estimators"],
        valid_sets=[dho],
        callbacks=[lgb.early_stopping(rc["early_stopping_rounds"], verbose=False)],
    )
    fit_s = time.time() - t0
    log(f"ranker trained in {fit_s:.1f}s, best iteration {booster.best_iteration}")
    Path("models").mkdir(exist_ok=True)
    booster.save_model("models/ranker.lgb")

    # Val-holdout users: the only val numbers that are not in-sample for the ranker.
    ho = ~is_tr
    pred = booster.predict(X[~rows_tr])
    m_ho, _, _ = rerank_metrics(
        val, users[ho], cands[ho], pred, v["m_mrr"][ho], ks, int(val.eligible.sum())
    )
    stage1_ho = {k: float(v[f"m_{k}"][ho].mean()) for k in ("recall@10", "ndcg@10", "mrr")}

    test = load_phase(cfg, "test", ratings)
    tctx = phase_context(test, ratings, load_item_features(cfg, "test"))
    out = {
        "features": FEATURES,
        "train_users": int(is_tr.sum()),
        "holdout_users": int(ho.sum()),
        "best_iteration": int(booster.best_iteration),
        "fit_seconds": round(fit_s, 1),
        "val_holdout": {
            "stage1": stage1_ho,
            "reranked": {k: float(m_ho[k].mean()) for k in ("recall@10", "ndcg@10", "mrr")},
        },
        "feature_importance_gain": dict(
            zip(FEATURES, booster.feature_importance("gain").round(1).tolist(), strict=True)
        ),
        "test": {},
    }
    for seed in cfg.two_tower["seeds"]:
        f = np.load(RESULTS / "per_user" / f"test_two_tower_seed{seed}.npz")
        tu, tcands, tscores = f["users"], f["candidates"], f["candidate_scores"]
        Xt = build_features(tctx, tu, tcands, tscores)
        pred = booster.predict(Xt)
        m, cov, rer = rerank_metrics(
            test, tu, tcands, pred, f["m_mrr"], ks, int(test.eligible.sum())
        )
        recall_ceiling = labels_for(test, tu, tcands).sum(1) / np.array(
            [len(test.targets[int(u)]) for u in tu]
        )
        np.savez(
            RESULTS / "per_user" / f"test_ranker_seed{seed}.npz",
            users=tu,
            candidates=rer,
            **{f"m_{k}": val_ for k, val_ in m.items()},
        )
        summ = {k: float(val_.mean()) for k, val_ in m.items()}
        summ.update(cov)
        summ["candidate_recall@100"] = float(recall_ceiling.mean())
        out["test"][str(seed)] = summ
        log(f"ranker on test seed {seed}: {summ}")
    (RESULTS / "ranker.json").write_text(json.dumps(out, indent=2) + "\n")


def main() -> None:
    """Separate entry point: LightGBM and PyTorch each ship their own libomp on macOS, and
    loading both in one process aborts, so this module never imports torch."""
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    def log(msg) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(RESULTS / "train_log.txt", "a") as f:
            f.write(line + "\n")

    t0 = time.time()
    log("command ranker")
    run_ranker(Config.load(args.config), log)
    log(f"command ranker done in {time.time() - t0:.0f}s wall")


if __name__ == "__main__":
    main()
