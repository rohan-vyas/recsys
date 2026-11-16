"""Metric correctness against hand-computed values."""

import math

import numpy as np
import pytest

from recsys.metrics import batch_metrics, coverage, ndcg_at_k, recall_at_k, reciprocal_rank


def test_recall_hand_computed():
    ranked = [5, 3, 9, 1, 7]
    relevant = [3, 7, 42]
    # top-2 = {5, 3}: one of three relevant items
    assert recall_at_k(ranked, relevant, 2) == pytest.approx(1 / 3)
    # top-5 contains 3 and 7: two of three
    assert recall_at_k(ranked, relevant, 5) == pytest.approx(2 / 3)


def test_ndcg_hand_computed():
    ranked = [5, 3, 9, 1, 7]
    relevant = [3, 7, 42]
    # hits at ranks 2 and 5: DCG = 1/log2(3) + 1/log2(6)
    dcg = 1 / math.log2(3) + 1 / math.log2(6)
    # three relevant, K=5: IDCG = 1 + 1/log2(3) + 1/log2(4)
    idcg = 1 + 1 / math.log2(3) + 0.5
    assert ndcg_at_k(ranked, relevant, 5) == pytest.approx(dcg / idcg)
    # K=2, single hit at rank 2, IDCG over min(2, 3) = 2 positions
    assert ndcg_at_k(ranked, relevant, 2) == pytest.approx(
        (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    )


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k([1, 2, 3], [1, 2], 3) == pytest.approx(1.0)
    assert ndcg_at_k([4, 5, 6], [1, 2], 3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank([5, 3, 9], [9]) == pytest.approx(1 / 3)
    assert reciprocal_rank([5, 3, 9], [5, 9]) == 1.0
    assert reciprocal_rank([5, 3, 9], [8]) == 0.0


def test_batch_matches_scalar():
    rng = np.random.default_rng(0)
    n_items = 50
    topk, targets, first = [], [], []
    for _ in range(40):
        ranking = rng.permutation(np.arange(1, n_items))
        t = rng.choice(np.arange(1, n_items), size=rng.integers(1, 8), replace=False)
        topk.append(ranking[:20])
        targets.append(t)
        first.append(next(r + 1 for r, it in enumerate(ranking) if it in set(t)))
        full = ranking
        assert reciprocal_rank(full, t) == pytest.approx(1 / first[-1])
    m = batch_metrics(np.array(topk), np.array(first), targets, (5, 10))
    for b in range(40):
        for k in (5, 10):
            assert m[f"recall@{k}"][b] == pytest.approx(recall_at_k(topk[b], targets[b], k))
            assert m[f"ndcg@{k}"][b] == pytest.approx(ndcg_at_k(topk[b], targets[b], k))
        assert m["mrr"][b] == pytest.approx(1 / first[b])


def test_coverage():
    topk = np.array([[1, 2, 3], [1, 4, 5]])
    assert coverage(topk, 2, 10) == pytest.approx(3 / 10)  # {1, 2, 4}
    assert coverage(topk, 3, 10) == pytest.approx(5 / 10)
