"""Top-K ranking metrics.

Definitions (per user, then averaged uniformly over users):

* Recall@K  = |top-K intersect T| / |T|, with T the user's target positives.
* NDCG@K    = DCG@K / IDCG@K, binary gains, DCG = sum_{rank r in top-K, hit} 1 / log2(r + 1),
              IDCG over min(K, |T|) hits.
* MRR       = 1 / rank of the first relevant item in the full-catalogue ranking (not truncated).
* Coverage@K = distinct items appearing in any user's top-K / number of candidate items.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def recall_at_k(ranked: Sequence[int], relevant: Sequence[int], k: int) -> float:
    rel = set(relevant)
    if not rel:
        raise ValueError("user has no relevant items")
    return len(rel.intersection(list(ranked)[:k])) / len(rel)


def ndcg_at_k(ranked: Sequence[int], relevant: Sequence[int], k: int) -> float:
    rel = set(relevant)
    if not rel:
        raise ValueError("user has no relevant items")
    dcg = sum(1.0 / np.log2(r + 2) for r, item in enumerate(list(ranked)[:k]) if item in rel)
    idcg = sum(1.0 / np.log2(r + 2) for r in range(min(k, len(rel))))
    return float(dcg / idcg)


def reciprocal_rank(ranked: Sequence[int], relevant: Sequence[int]) -> float:
    """Reciprocal rank over a complete ranking (0 if nothing relevant is ranked)."""
    rel = set(relevant)
    for r, item in enumerate(ranked):
        if item in rel:
            return 1.0 / (r + 1)
    return 0.0


def batch_metrics(
    topk: np.ndarray,
    first_rank: np.ndarray,
    targets: Sequence[np.ndarray],
    ks: Sequence[int],
) -> dict[str, np.ndarray]:
    """Vectorised per-user metrics.

    topk: int [B, >= max(ks)] ranked item ids; first_rank: int [B], 1-based full-catalogue rank of
    the best-ranked target; targets: B arrays of target item ids.
    """
    B, width = topk.shape
    if width < max(ks):
        raise ValueError("topk is narrower than the largest K")
    hits = np.zeros((B, width), dtype=bool)
    n_rel = np.empty(B, dtype=np.float64)
    for b, t in enumerate(targets):
        hits[b] = np.isin(topk[b], t)
        n_rel[b] = len(t)
    disc = 1.0 / np.log2(np.arange(2, width + 2))
    out: dict[str, np.ndarray] = {}
    for k in ks:
        h = hits[:, :k]
        out[f"recall@{k}"] = h.sum(1) / n_rel
        dcg = (h * disc[:k]).sum(1)
        ideal = np.cumsum(disc[:k])[np.minimum(n_rel, k).astype(int) - 1]
        out[f"ndcg@{k}"] = dcg / ideal
    out["mrr"] = 1.0 / first_rank.astype(np.float64)
    return out


def coverage(topk: np.ndarray, k: int, n_candidates: int) -> float:
    return float(np.unique(topk[:, :k]).size / n_candidates)
