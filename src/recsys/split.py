"""Global temporal split.

Every interaction is assigned to exactly one window by its timestamp:

    fit    : t <  cutoff
    target : cutoff <= t < end

Two phases share the same functions:

* ``val``  phase: cutoff = val_start, end = test_start. Used for all tuning and early stopping.
* ``test`` phase: cutoff = test_start, end = +inf. Models are refit on everything before
  test_start with the hyperparameters chosen in the val phase, then scored once.

Users are evaluated in a phase only if they have at least one positive in the fit window (so
every model has some history for them) and at least one positive target on an eligible item.
Eligible items are those with at least one positive in the fit window: an ID-based model cannot
rank an item it has never seen, so target positives on other items are dropped and counted.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PhaseData:
    """Everything a model and the evaluator need for one phase."""

    name: str
    cutoff: int
    end: int | None
    n_items: int  # item index space size, index 0 is padding
    fit: pd.DataFrame  # positives with t < cutoff, columns user, item, ts
    eligible: np.ndarray  # bool [n_items], candidate items
    eval_users: np.ndarray  # int user indices evaluated in this phase
    targets: dict[int, np.ndarray]  # user -> eligible target items
    exclude: dict[int, np.ndarray]  # user -> items rated (any rating) before cutoff
    n_target_dropped: int  # target positives on ineligible items
    n_target_users_no_history: int  # users with targets but no fit positives


def temporal_phase(
    ratings: pd.DataFrame,
    n_items: int,
    cutoff: int,
    end: int | None,
    positive_threshold: float,
    name: str,
) -> PhaseData:
    """Build one phase from a ratings frame with columns user, item, rating, ts."""
    before = ratings[ratings.ts < cutoff]
    if end is None:
        window = ratings[ratings.ts >= cutoff]
    else:
        window = ratings[(ratings.ts >= cutoff) & (ratings.ts < end)]

    fit = before[before.rating >= positive_threshold][["user", "item", "ts"]]
    fit = fit.sort_values(["user", "ts", "item"], kind="stable").reset_index(drop=True)

    eligible = np.zeros(n_items, dtype=bool)
    eligible[np.unique(fit.item.to_numpy())] = True
    eligible[0] = False

    tgt = window[window.rating >= positive_threshold]
    fit_users = np.unique(fit.user.to_numpy())
    has_hist = np.isin(tgt.user.to_numpy(), fit_users)
    n_no_hist = int(np.unique(tgt.user.to_numpy()[~has_hist]).size)
    tgt = tgt[has_hist]
    on_eligible = eligible[tgt.item.to_numpy()]
    n_dropped = int((~on_eligible).sum())
    tgt = tgt[on_eligible]

    targets = {int(u): np.unique(g.to_numpy()) for u, g in tgt.groupby("user").item}
    eval_users = np.array(sorted(targets), dtype=np.int64)

    ex = before[before.user.isin(eval_users)]
    exclude = {int(u): np.unique(g.to_numpy()) for u, g in ex.groupby("user").item}
    for u in eval_users:
        exclude.setdefault(int(u), np.array([], dtype=np.int64))

    return PhaseData(
        name=name,
        cutoff=cutoff,
        end=end,
        n_items=n_items,
        fit=fit,
        eligible=eligible,
        eval_users=eval_users,
        targets=targets,
        exclude=exclude,
        n_target_dropped=n_dropped,
        n_target_users_no_history=n_no_hist,
    )


def user_offsets(fit: pd.DataFrame, n_users: int) -> np.ndarray:
    """CSR-style offsets into ``fit`` (which must be sorted by user, ts)."""
    counts = np.bincount(fit.user.to_numpy(), minlength=n_users)
    off = np.zeros(n_users + 1, dtype=np.int64)
    np.cumsum(counts, out=off[1:])
    return off
