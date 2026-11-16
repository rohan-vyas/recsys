"""Forward pass, history gathering and a few training steps on tiny data."""

import numpy as np
import pandas as pd
import torch

from recsys.models.two_tower import HistoryIndex, TwoTower, TwoTowerScorer
from recsys.split import user_offsets


def tiny_model(n_items=11):
    rng = np.random.default_rng(0)
    genres = (rng.random((n_items, 20)) > 0.7).astype(np.float32)
    years = rng.integers(0, 17, n_items)
    tags = rng.integers(0, 6, (n_items, 4))
    torch.manual_seed(0)
    return TwoTower(
        n_items,
        genres,
        years,
        tags,
        n_years=17,
        n_tags=6,
        dim=8,
        hidden=16,
        tag_dim=4,
        year_dim=3,
        short_len=2,
    )


def test_history_is_strictly_before_and_right_aligned():
    fit = pd.DataFrame({"user": [0, 0, 0, 1, 1], "item": [3, 5, 7, 2, 9], "ts": [1, 2, 3, 1, 2]})
    off = user_offsets(fit, 2)
    h = HistoryIndex(fit.item.to_numpy(), fit.user.to_numpy(), off, hist_len=3)
    got = h.before(np.array([0, 1, 2, 3, 4]))
    assert got.tolist() == [[0, 0, 0], [0, 0, 3], [0, 3, 5], [0, 0, 0], [0, 0, 2]]
    assert h.latest(np.array([0, 1])).tolist() == [[3, 5, 7], [0, 2, 9]]


def test_forward_shapes_and_normalised():
    m = tiny_model()
    hist = torch.tensor([[0, 1, 2], [3, 4, 5]])
    u = m.user_tower(hist)
    v = m.all_items()
    assert u.shape == (2, 8) and v.shape == (11, 8)
    assert torch.allclose(u.norm(dim=1), torch.ones(2), atol=1e-5)
    assert torch.allclose(v.norm(dim=1), torch.ones(11), atol=1e-5)


def test_loss_finite_and_decreases():
    m = tiny_model()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    hist = torch.tensor([[0, 1, 2], [0, 3, 4], [0, 5, 6], [7, 8, 9]])
    pos = torch.tensor([3, 5, 7, 10])
    log_q = torch.log(torch.full((11,), 1 / 11))
    first = m.loss(hist, pos, log_q).item()
    for _ in range(50):
        loss = m.loss(hist, pos, log_q)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert np.isfinite(first) and loss.item() < first


def test_duplicate_items_in_batch_are_masked():
    m = tiny_model()
    hist = torch.tensor([[0, 1, 2], [0, 3, 4]])
    pos = torch.tensor([5, 5])  # same item twice: neither may be the other's negative
    loss = m.loss(hist, pos, None)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-6)


def test_scorer_scores_full_catalogue():
    m = tiny_model()
    fit = pd.DataFrame({"user": [0, 0, 1], "item": [3, 5, 2], "ts": [1, 2, 1]})
    off = user_offsets(fit, 2)
    h = HistoryIndex(fit.item.to_numpy(), fit.user.to_numpy(), off, hist_len=3)
    s = TwoTowerScorer(m, h, torch.device("cpu")).score(np.array([0, 1]))
    assert s.shape == (2, 11)
