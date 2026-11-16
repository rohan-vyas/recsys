import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def toy_ratings() -> pd.DataFrame:
    """6 users, items 1..12, timestamps 0..99 with a cutoff at 50 and an end at 80."""
    rng = np.random.default_rng(1)
    rows = []
    for u in range(6):
        for item in rng.choice(np.arange(1, 13), size=9, replace=False):
            rows.append(
                (
                    u,
                    int(item),
                    float(rng.choice([2.0, 3.0, 4.0, 4.5, 5.0])),
                    int(rng.integers(0, 100)),
                )
            )
    # user 5 has only post-cutoff activity: a cold user that must be excluded
    rows = [r for r in rows if not (r[0] == 5 and r[3] < 50)]
    # item 12 appears only after the cutoff: an ineligible item
    rows = [r for r in rows if not (r[1] == 12 and r[3] < 50)]
    rows.append((0, 12, 5.0, 60))
    return pd.DataFrame(rows, columns=["user", "item", "rating", "ts"]).drop_duplicates(
        ["user", "item"]
    )
