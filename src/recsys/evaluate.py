"""Full-catalogue evaluation shared by every model."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch

from recsys.metrics import batch_metrics, coverage
from recsys.split import PhaseData


class Scorer(Protocol):
    def score(self, users: np.ndarray) -> torch.Tensor:
        """Return float scores [len(users), n_items] for the given user indices."""


@dataclass
class EvalResult:
    users: np.ndarray
    per_user: dict[str, np.ndarray]
    candidates: np.ndarray  # int32 [n_users, n_cand] ranked item ids
    candidate_scores: np.ndarray
    coverage: dict[str, float]
    seconds: float

    def summary(self) -> dict[str, float]:
        s = {k: float(v.mean()) for k, v in self.per_user.items()}
        s.update(self.coverage)
        s["n_users"] = int(len(self.users))
        return s


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def evaluate(
    scorer: Scorer,
    phase: PhaseData,
    ks: tuple[int, ...] = (10, 50),
    n_cand: int = 100,
    chunk: int = 2048,
    users: np.ndarray | None = None,
    dev: torch.device | None = None,
) -> EvalResult:
    dev = dev or device()
    t0 = time.time()
    users = phase.eval_users if users is None else users
    ineligible = torch.from_numpy(~phase.eligible).to(dev)
    n_cand = max(n_cand, max(ks))
    per_user: dict[str, list[np.ndarray]] = {}
    cands, cand_scores = [], []
    for s in range(0, len(users), chunk):
        ub = users[s : s + chunk]
        scores = scorer.score(ub).to(dev, torch.float32).clone()
        B = len(ub)
        scores[:, ineligible] = -torch.inf
        rows, cols = [], []
        trows, tcols = [], []
        for b, u in enumerate(ub):
            ex = phase.exclude[int(u)]
            rows.append(np.full(len(ex), b))
            cols.append(ex)
            t = phase.targets[int(u)]
            trows.append(np.full(len(t), b))
            tcols.append(t)
        r = torch.from_numpy(np.concatenate(rows).astype(np.int64)).to(dev)
        c = torch.from_numpy(np.concatenate(cols).astype(np.int64)).to(dev)
        scores[r, c] = -torch.inf
        tr = torch.from_numpy(np.concatenate(trows).astype(np.int64)).to(dev)
        tc = torch.from_numpy(np.concatenate(tcols).astype(np.int64)).to(dev)
        tmask = torch.zeros_like(scores, dtype=torch.bool)
        tmask[tr, tc] = True
        best = torch.where(tmask, scores, torch.full_like(scores, -torch.inf)).max(1).values
        first_rank = (scores > best[:, None]).sum(1) + 1
        top_s, top_i = torch.topk(scores, n_cand, dim=1)
        top_i = top_i.cpu().numpy().astype(np.int32)
        m = batch_metrics(
            top_i,
            first_rank.cpu().numpy(),
            [phase.targets[int(u)] for u in ub],
            ks,
        )
        for k, v in m.items():
            per_user.setdefault(k, []).append(v)
        cands.append(top_i)
        cand_scores.append(top_s.cpu().numpy())
        del scores, tmask
        assert top_i.shape[0] == B
    candidates = np.concatenate(cands)
    n_eligible = int(phase.eligible.sum())
    cov = {f"coverage@{k}": coverage(candidates, k, n_eligible) for k in ks}
    return EvalResult(
        users=users,
        per_user={k: np.concatenate(v) for k, v in per_user.items()},
        candidates=candidates,
        candidate_scores=np.concatenate(cand_scores),
        coverage=cov,
        seconds=time.time() - t0,
    )
