"""Implicit-feedback matrix factorisation (iALS, Hu, Koren and Volinsky 2008) via `implicit`."""

from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import torch

from recsys.split import PhaseData


def user_item_matrix(phase: PhaseData, n_users: int) -> sp.csr_matrix:
    u = phase.fit.user.to_numpy()
    i = phase.fit.item.to_numpy()
    m = sp.csr_matrix((np.ones(len(u), dtype=np.float32), (u, i)), shape=(n_users, phase.n_items))
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


class ALSScorer:
    def __init__(self, user_factors: np.ndarray, item_factors: np.ndarray):
        self.U = torch.from_numpy(np.ascontiguousarray(user_factors, dtype=np.float32))
        self.V = torch.from_numpy(np.ascontiguousarray(item_factors, dtype=np.float32))

    def score(self, users: np.ndarray) -> torch.Tensor:
        return self.U[torch.from_numpy(np.asarray(users, dtype=np.int64))] @ self.V.T


def fit_als(
    phase: PhaseData,
    n_users: int,
    factors: int,
    regularization: float,
    alpha: float,
    iterations: int,
    seed: int,
) -> tuple[ALSScorer, float]:
    from implicit.cpu.als import AlternatingLeastSquares

    X = user_item_matrix(phase, n_users)
    model = AlternatingLeastSquares(
        factors=factors,
        regularization=regularization,
        alpha=alpha,
        iterations=iterations,
        random_state=seed,
        calculate_training_loss=False,
    )
    t0 = time.time()
    model.fit(X, show_progress=False)
    secs = time.time() - t0
    return ALSScorer(model.user_factors, model.item_factors), secs
