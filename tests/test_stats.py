import numpy as np

from recsys.stats import bootstrap_ci, paired_bootstrap, sign_test


def test_bootstrap_ci_brackets_mean():
    x = np.random.default_rng(0).random(500)
    mean, lo, hi = bootstrap_ci(x, n_boot=500)
    assert lo < mean < hi
    assert mean == x.mean()


def test_paired_bootstrap_detects_shift_and_null():
    rng = np.random.default_rng(1)
    b = rng.random(1000)
    shifted = paired_bootstrap(b + 0.05, b, n_boot=500)
    assert shifted["ci_low"] > 0 and shifted["p_bootstrap"] <= 0.002
    same = paired_bootstrap(b, b.copy(), n_boot=500)
    assert same["diff"] == 0.0


def test_sign_test_counts():
    r = sign_test(np.array([1, 2, 3, 0]), np.array([0, 2, 1, 1]))
    assert (r["wins"], r["losses"], r["ties"]) == (2, 1, 1)
    assert 0 < r["p"] <= 1
