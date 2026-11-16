"""Popularity baselines: all-time and recent (last N days before the cutoff)."""

from __future__ import annotations

import numpy as np
import torch

from recsys.split import PhaseData


def popularity_counts(phase: PhaseData, recent_days: int | None = None) -> np.ndarray:
    fit = phase.fit
    if recent_days is not None:
        fit = fit[fit.ts >= phase.cutoff - recent_days * 86400]
    return np.bincount(fit.item.to_numpy(), minlength=phase.n_items).astype(np.float64)


class PopularityScorer:
    """Same ranking for every user. Ties are broken by item index so ranks are unique."""

    def __init__(self, counts: np.ndarray):
        order = np.lexsort((np.arange(len(counts)), -counts))
        score = np.empty(len(counts), dtype=np.float32)
        score[order] = np.arange(len(counts), 0, -1, dtype=np.float32)
        self.item_score = torch.from_numpy(score)

    def score(self, users: np.ndarray) -> torch.Tensor:
        return self.item_score[None, :].expand(len(users), -1)
