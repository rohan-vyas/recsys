"""Training loop for the two-tower model."""

from __future__ import annotations

import copy
import math
import random
import time

import numpy as np
import torch

from recsys.data import N_YEAR_BUCKETS
from recsys.evaluate import device, evaluate
from recsys.models.two_tower import HistoryIndex, TwoTower, TwoTowerScorer
from recsys.split import PhaseData, user_offsets


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(cfg: dict, n_items: int, feats: dict[str, np.ndarray]) -> TwoTower:
    return TwoTower(
        n_items=n_items,
        genres=feats["genres"],
        years=feats["years"],
        tags=feats["tags"],
        n_years=N_YEAR_BUCKETS,
        n_tags=int(feats["tags"].max()) + 1,
        dim=cfg["dim"],
        hidden=cfg["hidden"],
        tag_dim=cfg["tag_dim"],
        year_dim=cfg["year_dim"],
        short_len=cfg["short_len"],
        temperature=cfg["temperature"],
    )


def train_two_tower(
    cfg: dict,
    phase: PhaseData,
    n_users: int,
    feats: dict[str, np.ndarray],
    seed: int,
    early_stop: bool,
    fixed_epochs: int | None = None,
    log=print,
) -> tuple[TwoTowerScorer, dict]:
    """Train on ``phase.fit``.

    With ``early_stop`` the model is scored on ``phase``'s own eval users after every epoch (only
    ever used with the val phase) and the best epoch by Recall@50 is kept. Otherwise exactly
    ``fixed_epochs`` epochs are run. The LR schedule always spans ``max_epochs`` so a fixed-epoch
    refit follows the same schedule as the tuning run up to its stopping point.
    """
    set_seed(seed)
    dev = device()
    fit = phase.fit
    off = user_offsets(fit, n_users)
    users = fit.user.to_numpy().astype(np.int64)
    items = fit.item.to_numpy().astype(np.int64)
    hist = HistoryIndex(items, users, off, cfg["hist_len"])

    positions = np.arange(len(fit), dtype=np.int64)
    keep = positions > off[users]  # at least one earlier positive
    if cfg.get("train_since_days"):
        # Recency window on the *target* only; histories still use all earlier positives.
        keep &= fit.ts.to_numpy() >= phase.cutoff - cfg["train_since_days"] * 86400
    positions = positions[keep]
    counts = np.bincount(items[positions], minlength=phase.n_items).astype(np.float64)
    log_q = None
    if cfg["logq_correction"]:
        log_q = torch.from_numpy(np.log(np.maximum(counts, 1.0) / counts.sum())).float().to(dev)

    model = build_model(cfg, phase.n_items, feats).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    bs = cfg["batch_size"]
    per_epoch = min(cfg["samples_per_epoch"], len(positions))
    steps_per_epoch = per_epoch // bs
    total = steps_per_epoch * cfg["max_epochs"]
    warm = cfg["warmup_steps"]

    def lr_at(step: int) -> float:
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = np.random.default_rng(seed)
    n_epochs = cfg["max_epochs"] if fixed_epochs is None else fixed_epochs
    history = []
    best = (-1.0, 0, None)
    bad = 0
    t_train = 0.0
    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        sample = rng.choice(positions, size=per_epoch, replace=False)
        losses = []
        for s in range(steps_per_epoch):
            pb = sample[s * bs : (s + 1) * bs]
            h = torch.from_numpy(hist.before(pb)).to(dev)
            p = torch.from_numpy(items[pb]).to(dev)
            loss = model.loss(h, p, log_q)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            if s % 50 == 0:
                losses.append(loss.item())
        if dev.type == "mps":
            torch.mps.synchronize()
        dt = time.time() - t0
        t_train += dt
        rec = {"epoch": epoch, "loss": float(np.mean(losses)), "seconds": round(dt, 1)}
        if early_stop:
            sc = TwoTowerScorer(model, hist, dev)
            res = evaluate(sc, phase, dev=dev)
            r50 = float(res.per_user["recall@50"].mean())
            rec["val_recall@50"] = r50
            rec["val_recall@10"] = float(res.per_user["recall@10"].mean())
            if r50 > best[0]:
                best = (r50, epoch, copy.deepcopy(model.state_dict()))
                bad = 0
            else:
                bad += 1
        history.append(rec)
        log(rec)
        if early_stop and bad >= cfg["patience"]:
            break
    if early_stop:
        model.load_state_dict(best[2])
        best_epoch = best[1]
    else:
        best_epoch = n_epochs
    info = {
        "seed": seed,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "train_seconds": round(t_train, 1),
        "train_examples_available": int(len(positions)),
        "examples_per_epoch": int(steps_per_epoch * bs),
        "history": history,
        "device": dev.type,
    }
    return TwoTowerScorer(model, hist, dev), info
