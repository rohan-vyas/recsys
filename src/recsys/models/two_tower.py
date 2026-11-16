"""Two-tower retrieval model.

User tower: the user is represented only by their positive history (no user-ID embedding), so the
same network serves users whose history grew after training. Inputs are the mean embedding of the
last ``hist_len`` positives, the mean of the last ``short_len`` positives and log history length.

Item tower: item-ID embedding, genre multi-hot, release-year bucket and a bag of user tags (tags
counted only before the phase cutoff).

Training predicts each positive from the positives that precede it in time, with in-batch
softmax over the other targets in the batch, logQ sampling-bias correction (Yi et al. 2019) and
masking of in-batch duplicates of the true item.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class ItemTower(nn.Module):
    def __init__(
        self,
        n_items: int,
        n_genres: int,
        n_years: int,
        n_tags: int,
        dim: int,
        hidden: int,
        tag_dim: int,
        year_dim: int,
    ):
        super().__init__()
        self.id_emb = nn.Embedding(n_items, dim, padding_idx=0)
        self.year_emb = nn.Embedding(n_years, year_dim)
        self.tag_emb = nn.EmbeddingBag(n_tags, tag_dim, mode="sum", padding_idx=0)
        self.mlp = nn.Sequential(
            nn.Linear(dim + n_genres + year_dim + tag_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, dim),
        )
        nn.init.normal_(self.id_emb.weight, std=0.05)

    def forward(
        self, items: torch.Tensor, genres: torch.Tensor, years: torch.Tensor, tags: torch.Tensor
    ) -> torch.Tensor:
        n_tags = (tags > 0).sum(1, keepdim=True).clamp(min=1)
        tag_vec = self.tag_emb(tags) / n_tags
        x = torch.cat([self.id_emb(items), genres, self.year_emb(years), tag_vec], dim=1)
        return F.normalize(self.mlp(x), dim=-1)


class UserTower(nn.Module):
    def __init__(self, n_items: int, dim: int, hidden: int, short_len: int):
        super().__init__()
        self.short_len = short_len
        self.hist_emb = nn.Embedding(n_items, dim, padding_idx=0)
        self.mlp = nn.Sequential(nn.Linear(2 * dim + 1, hidden), nn.ReLU(), nn.Linear(hidden, dim))
        nn.init.normal_(self.hist_emb.weight, std=0.05)

    def forward(self, hist: torch.Tensor) -> torch.Tensor:
        """hist: int [B, L], right-aligned (most recent last), 0 = padding."""
        mask = (hist > 0).float()
        e = self.hist_emb(hist) * mask[..., None]
        n = mask.sum(1, keepdim=True)
        long = e.sum(1) / n.clamp(min=1)
        s_mask = mask[:, -self.short_len :]
        short = e[:, -self.short_len :].sum(1) / s_mask.sum(1, keepdim=True).clamp(min=1)
        x = torch.cat([long, short, torch.log1p(n)], dim=1)
        return F.normalize(self.mlp(x), dim=-1)


class TwoTower(nn.Module):
    def __init__(
        self,
        n_items: int,
        genres: np.ndarray,
        years: np.ndarray,
        tags: np.ndarray,
        n_years: int,
        n_tags: int,
        dim: int = 64,
        hidden: int = 256,
        tag_dim: int = 32,
        year_dim: int = 16,
        short_len: int = 5,
        temperature: float = 0.05,
    ):
        super().__init__()
        self.register_buffer("genres", torch.as_tensor(genres, dtype=torch.float32))
        self.register_buffer("years", torch.as_tensor(years, dtype=torch.long))
        self.register_buffer("tags", torch.as_tensor(tags, dtype=torch.long))
        self.item_tower = ItemTower(
            n_items, genres.shape[1], n_years, n_tags, dim, hidden, tag_dim, year_dim
        )
        self.user_tower = UserTower(n_items, dim, hidden, short_len)
        self.temperature = temperature

    def items(self, idx: torch.Tensor) -> torch.Tensor:
        return self.item_tower(idx, self.genres[idx], self.years[idx], self.tags[idx])

    def all_items(self, chunk: int = 16384) -> torch.Tensor:
        n = self.genres.shape[0]
        dev = self.genres.device
        return torch.cat(
            [self.items(torch.arange(s, min(s + chunk, n), device=dev)) for s in range(0, n, chunk)]
        )

    def loss(
        self, hist: torch.Tensor, pos: torch.Tensor, log_q: torch.Tensor | None
    ) -> torch.Tensor:
        u = self.user_tower(hist)
        v = self.items(pos)
        logits = u @ v.T / self.temperature
        if log_q is not None:
            logits = logits - log_q[pos][None, :]
        same = pos[:, None] == pos[None, :]
        eye = torch.eye(len(pos), dtype=torch.bool, device=pos.device)
        logits = logits.masked_fill(same & ~eye, float("-inf"))
        labels = torch.arange(len(pos), device=pos.device)
        return F.cross_entropy(logits, labels)


class HistoryIndex:
    """Gathers right-aligned history windows from a user-sorted positive item array."""

    def __init__(self, items: np.ndarray, users: np.ndarray, offsets: np.ndarray, hist_len: int):
        self.items = np.concatenate([[0], items]).astype(np.int64)  # shift by 1, slot 0 = pad
        self.users = users
        self.offsets = offsets
        self.hist_len = hist_len
        self.back = np.arange(hist_len, 0, -1)

    def before(self, positions: np.ndarray) -> np.ndarray:
        """History strictly before each global position (in the sorted fit array)."""
        start = self.offsets[self.users[positions]] if len(positions) else positions
        idx = positions[:, None] - self.back[None, :]
        valid = idx >= start[:, None]
        return np.where(valid, self.items[np.where(valid, idx + 1, 0)], 0)

    def latest(self, users: np.ndarray) -> np.ndarray:
        """Most recent ``hist_len`` positives of each user (inference)."""
        end = self.offsets[users + 1]
        idx = end[:, None] - self.back[None, :]
        valid = idx >= self.offsets[users][:, None]
        return np.where(valid, self.items[np.where(valid, idx + 1, 0)], 0)


class TwoTowerScorer:
    def __init__(self, model: TwoTower, hist: HistoryIndex, dev: torch.device):
        self.model = model.eval()
        self.hist = hist
        self.dev = dev
        with torch.no_grad():
            self.item_vecs = model.all_items()

    @torch.no_grad()
    def user_vectors(self, users: np.ndarray) -> torch.Tensor:
        h = torch.from_numpy(self.hist.latest(np.asarray(users, dtype=np.int64))).to(self.dev)
        return self.model.user_tower(h)

    @torch.no_grad()
    def score(self, users: np.ndarray) -> torch.Tensor:
        return self.user_vectors(users) @ self.item_vecs.T
