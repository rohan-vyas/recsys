"""Download and preprocess MovieLens 25M.

Raw files are untrusted input: they are parsed as CSV with fixed dtypes and never executed.
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from recsys.config import Config
from recsys.split import PhaseData, temporal_phase

GENRES = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "IMAX",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
    "(no genres listed)",
]
YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def to_ts(date: str) -> int:
    return int(pd.Timestamp(date, tz="UTC").timestamp())


def download(cfg: Config) -> Path:
    raw = Path(cfg.data["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    zpath = raw / "ml-25m.zip"
    if not zpath.exists():
        print(f"downloading {cfg.data['url']}")
        urllib.request.urlretrieve(cfg.data["url"], zpath)
    md5 = hashlib.md5(zpath.read_bytes()).hexdigest()
    if md5 != cfg.data["md5"]:
        raise RuntimeError(f"md5 mismatch for {zpath}: {md5}")
    wanted = {f"ml-25m/{n}" for n in ("ratings.csv", "movies.csv", "tags.csv", "README.txt")}
    with zipfile.ZipFile(zpath) as zf:
        for member in zf.namelist():
            if member in wanted:
                # Fixed member names only, so no path traversal from archive entries.
                zf.extract(member, raw)
    return raw / "ml-25m"


def year_bucket(year: float) -> int:
    """0 unknown, 1 before 1950, then 5-year buckets from 1950 to 2020+."""
    if np.isnan(year):
        return 0
    if year < 1950:
        return 1
    return 2 + min(int((year - 1950) // 5), 14)


N_YEAR_BUCKETS = 17


def tag_features(
    tags: pd.DataFrame, n_items: int, cutoff: int, top_tags: int, per_item: int
) -> np.ndarray:
    """Per-item tag ids using only tags applied before ``cutoff`` (no future leakage)."""
    t = tags[tags.ts < cutoff]
    vocab = t.tag.value_counts().head(top_tags).index
    tag_id = {tag: i + 1 for i, tag in enumerate(vocab)}
    t = t[t.tag.isin(tag_id)]
    counts = t.groupby(["item", "tag"]).size().rename("n").reset_index()
    counts = counts.sort_values(["item", "n", "tag"], ascending=[True, False, True])
    counts["rank"] = counts.groupby("item").cumcount()
    counts = counts[counts["rank"] < per_item]
    out = np.zeros((n_items, per_item), dtype=np.int32)
    out[counts.item.to_numpy(), counts["rank"].to_numpy()] = counts.tag.map(tag_id).to_numpy()
    return out


def prepare(cfg: Config) -> dict:
    src = Path(cfg.data["raw_dir"]) / "ml-25m"
    out = Path(cfg.data["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)

    r = pd.read_csv(
        src / "ratings.csv",
        engine="pyarrow",
        dtype={"userId": "int64", "movieId": "int64", "rating": "float32", "timestamp": "int64"},
    )
    movies = pd.read_csv(
        src / "movies.csv", dtype={"movieId": "int64", "title": str, "genres": str}
    )
    tags = pd.read_csv(
        src / "tags.csv",
        dtype={"userId": "int64", "movieId": "int64", "tag": str, "timestamp": "int64"},
    )

    # Item index 0 is padding; every movie in movies.csv gets an index.
    movie_ids = np.sort(movies.movieId.unique())
    item_of = pd.Series(np.arange(1, len(movie_ids) + 1, dtype=np.int32), index=movie_ids)
    n_items = len(movie_ids) + 1
    user_ids = np.sort(r.userId.unique())
    user_of = pd.Series(np.arange(len(user_ids), dtype=np.int32), index=user_ids)

    ratings = pd.DataFrame(
        {
            "user": user_of.loc[r.userId.to_numpy()].to_numpy(),
            "item": item_of.loc[r.movieId.to_numpy()].to_numpy(),
            "rating": r.rating.to_numpy(np.float32),
            "ts": r.timestamp.to_numpy(np.int64),
        }
    )
    del r
    ratings.to_parquet(out / "ratings.parquet", index=False)

    genres = np.zeros((n_items, len(GENRES)), dtype=np.float32)
    years = np.zeros(n_items, dtype=np.int64)
    for mid, title, g in movies[["movieId", "title", "genres"]].itertuples(index=False):
        i = item_of[mid]
        for name in str(g).split("|"):
            if name in GENRES:
                genres[i, GENRES.index(name)] = 1.0
        m = YEAR_RE.search(str(title).strip())
        years[i] = year_bucket(float(m.group(1)) if m else float("nan"))

    tags = tags[tags.movieId.isin(item_of.index)]
    tags = pd.DataFrame(
        {
            "item": item_of.loc[tags.movieId.to_numpy()].to_numpy(),
            "tag": tags.tag.astype(str).str.strip().str.lower(),
            "ts": tags.timestamp.to_numpy(np.int64),
        }
    )
    feats = {"genres": genres, "years": years}
    for phase, date in (("val", cfg.data["val_start"]), ("test", cfg.data["test_start"])):
        feats[f"tags_{phase}"] = tag_features(
            tags, n_items, to_ts(date), cfg.data["top_tags"], cfg.data["tags_per_item"]
        )
    np.savez(out / "item_features.npz", **feats)
    np.save(out / "movie_ids.npy", movie_ids)

    titles = pd.Series(movies.title.to_numpy(), index=item_of.loc[movies.movieId].to_numpy())
    titles.sort_index().to_frame("title").to_parquet(out / "titles.parquet")
    summary = {
        "n_ratings": int(len(ratings)),
        "n_users": int(len(user_ids)),
        "n_items_in_movies_csv": int(len(movie_ids)),
        "n_items_rated": int(ratings.item.nunique()),
    }
    print(summary)
    return summary


def load_ratings(cfg: Config) -> pd.DataFrame:
    return pd.read_parquet(Path(cfg.data["processed_dir"]) / "ratings.parquet")


def load_item_features(cfg: Config, phase: str) -> dict[str, np.ndarray]:
    f = np.load(Path(cfg.data["processed_dir"]) / "item_features.npz")
    return {"genres": f["genres"], "years": f["years"], "tags": f[f"tags_{phase}"]}


def load_phase(cfg: Config, phase: str, ratings: pd.DataFrame | None = None) -> PhaseData:
    if ratings is None:
        ratings = load_ratings(cfg)
    n_items = len(np.load(Path(cfg.data["processed_dir"]) / "movie_ids.npy")) + 1
    thr = cfg.data["positive_threshold"]
    if phase == "val":
        return temporal_phase(
            ratings,
            n_items,
            to_ts(cfg.data["val_start"]),
            to_ts(cfg.data["test_start"]),
            thr,
            "val",
        )
    if phase == "test":
        return temporal_phase(ratings, n_items, to_ts(cfg.data["test_start"]), None, thr, "test")
    raise ValueError(phase)


def n_users_total(ratings: pd.DataFrame) -> int:
    return int(ratings.user.max()) + 1
