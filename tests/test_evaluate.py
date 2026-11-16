"""End-to-end evaluator check with a fixed scorer and hand-checked ranks."""

import numpy as np
import torch

from recsys.evaluate import evaluate
from recsys.split import PhaseData


class FixedScorer:
    def __init__(self, s):
        self.s = torch.tensor(s, dtype=torch.float32)

    def score(self, users):
        return self.s[torch.as_tensor(users)]


def test_masks_and_full_rank():
    n_items = 6
    eligible = np.array([False, True, True, True, True, False])  # item 5 ineligible
    scores = [[0, 1, 2, 3, 4, 9], [0, 5, 4, 3, 2, 9]]
    phase = PhaseData(
        name="t",
        cutoff=0,
        end=None,
        n_items=n_items,
        fit=None,
        eligible=eligible,
        eval_users=np.array([0, 1]),
        targets={0: np.array([2]), 1: np.array([3, 4])},
        exclude={0: np.array([4]), 1: np.array([], dtype=np.int64)},
        n_target_dropped=0,
        n_target_users_no_history=0,
    )
    res = evaluate(FixedScorer(scores), phase, ks=(1, 2), n_cand=3, dev=torch.device("cpu"))
    # user 0: item 5 ineligible, item 4 excluded -> ranking 3, 2, 1; target 2 at rank 2
    assert res.candidates[0].tolist() == [3, 2, 1]
    assert res.per_user["mrr"][0] == 0.5
    assert res.per_user["recall@1"][0] == 0.0 and res.per_user["recall@2"][0] == 1.0
    # user 1: ranking 1, 2, 3, 4; targets 3, 4 -> first at rank 3
    assert res.per_user["mrr"][1] == 1 / 3
    assert res.per_user["recall@2"][1] == 0.0
    assert res.coverage["coverage@1"] == 2 / 4
