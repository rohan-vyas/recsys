from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class Config:
    data: dict
    eval: dict
    popularity: dict
    als: dict
    two_tower: dict
    ranker: dict

    @classmethod
    def load(cls, path: str | Path = "configs/default.yaml") -> Config:
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls(**raw)
