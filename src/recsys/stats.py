"""Bootstrap confidence intervals and paired significance tests over users."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def bootstrap_ci(
    x: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05, chunk: int = 250
) -> tuple[float, float, float]:
    """Mean and percentile bootstrap CI of a per-user metric (users resampled with replacement)."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = []
    for s in range(0, n_boot, chunk):
        idx = rng.integers(0, len(x), size=(min(chunk, n_boot - s), len(x)))
        means.append(x[idx].mean(1))
    m = np.concatenate(means)
    return float(x.mean()), float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))


def paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> dict:
    """Paired bootstrap on the per-user difference a - b (same users, same order).

    p is the two-sided bootstrap p-value: twice the smaller tail mass of the resampled mean
    difference on either side of zero (floored at 1 / n_boot).
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("paired arrays must align")
    d = a - b
    rng = np.random.default_rng(seed)
    boots = []
    for s in range(0, n_boot, 250):
        idx = rng.integers(0, len(d), size=(min(250, n_boot - s), len(d)))
        boots.append(d[idx].mean(1))
    m = np.concatenate(boots)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return {
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "diff": float(d.mean()),
        "rel_diff": float(d.mean() / b.mean()) if b.mean() != 0 else None,
        "ci_low": float(np.quantile(m, alpha / 2)),
        "ci_high": float(np.quantile(m, 1 - alpha / 2)),
        "p_bootstrap": float(max(p, 1.0 / n_boot)),
    }


def sign_test(a: np.ndarray, b: np.ndarray) -> dict:
    """Exact two-sided sign test on per-user wins/losses (ties dropped)."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    wins = int((d > 0).sum())
    losses = int((d < 0).sum())
    ties = int((d == 0).sum())
    p = binomtest(wins, wins + losses, 0.5).pvalue if wins + losses else 1.0
    return {"wins": wins, "losses": losses, "ties": ties, "p": float(p)}
