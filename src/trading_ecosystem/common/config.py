from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT / "configs"
DEFAULT_DATA_DIR = ROOT / "data"


def project_root() -> Path:
    return ROOT


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        candidate = CONFIG_DIR / p
        p = candidate if candidate.exists() else Path(path)
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {p}")
    return data


def data_dir() -> Path:
    import os

    raw = os.getenv("DATA_DIR", str(DEFAULT_DATA_DIR))
    path = Path(raw)
    if not path.is_absolute():
        path = ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    (path / "raw").mkdir(exist_ok=True)
    (path / "processed").mkdir(exist_ok=True)
    return path
