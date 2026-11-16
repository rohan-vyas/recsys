"""The temporal split must never let information from at or after the cutoff into the fit set."""

import numpy as np

from recsys.split import temporal_phase, user_offsets


def test_fit_strictly_before_cutoff(toy_ratings):
    p = temporal_phase(toy_ratings, 13, cutoff=50, end=80, positive_threshold=4.0, name="val")
    assert (p.fit.ts < 50).all()
    assert len(p.fit) > 0


def test_targets_inside_window_and_positive(toy_ratings):
    p = temporal_phase(toy_ratings, 13, cutoff=50, end=80, positive_threshold=4.0, name="val")
    window = toy_ratings[(toy_ratings.ts >= 50) & (toy_ratings.ts < 80) & (toy_ratings.rating >= 4)]
    for u, items in p.targets.items():
        allowed = set(window[window.user == u].item)
        assert set(items) <= allowed
        assert len(items) > 0


def test_no_target_overlaps_fit_history(toy_ratings):
    p = temporal_phase(toy_ratings, 13, cutoff=50, end=80, positive_threshold=4.0, name="val")
    for u, items in p.targets.items():
        seen = set(p.fit[p.fit.user == u].item)
        assert not seen.intersection(items)
        assert not set(p.exclude[u]).intersection(items)


def test_cold_users_and_unseen_items_excluded(toy_ratings):
    p = temporal_phase(toy_ratings, 13, cutoff=50, end=None, positive_threshold=4.0, name="test")
    assert 5 not in p.targets  # no history before the cutoff
    assert not p.eligible[12]  # item 12 first appears after the cutoff
    assert not p.eligible[0]  # padding index
    assert all(12 not in t for t in p.targets.values())
    assert p.n_target_dropped >= 1


def test_exclude_covers_all_pre_cutoff_ratings(toy_ratings):
    p = temporal_phase(toy_ratings, 13, cutoff=50, end=None, positive_threshold=4.0, name="test")
    for u in p.eval_users:
        before = toy_ratings[(toy_ratings.user == u) & (toy_ratings.ts < 50)]
        assert set(before.item) == set(p.exclude[int(u)])


def test_fit_sorted_by_user_then_time(toy_ratings):
    p = temporal_phase(toy_ratings, 13, cutoff=50, end=None, positive_threshold=4.0, name="test")
    f = p.fit
    key = list(zip(f.user, f.ts, strict=True))
    assert key == sorted(key)
    off = user_offsets(f, 6)
    assert off[-1] == len(f)
    for u in range(6):
        assert (f.user.to_numpy()[off[u] : off[u + 1]] == u).all()


def test_real_config_windows_are_ordered():
    from recsys.config import Config
    from recsys.data import to_ts

    cfg = Config.load()
    assert to_ts(cfg.data["val_start"]) < to_ts(cfg.data["test_start"])
    assert np.isclose(cfg.data["positive_threshold"], 4.0)
