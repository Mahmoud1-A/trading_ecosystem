"""Artifact persistence for trials (trades, equity, fills, orders)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dir(path: Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any] | list[Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


def write_frame(path: Path, frame: pd.DataFrame) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    else:
        frame.to_csv(path, index=False)
    return path


def persist_run_artifacts(
    run_dir: Path,
    *,
    equity: pd.Series | None = None,
    trades: pd.DataFrame | None = None,
    fills: list[dict[str, Any]] | None = None,
    orders: list[dict[str, Any]] | None = None,
    risk_events: list[dict[str, Any]] | None = None,
    metrics: dict[str, Any] | None = None,
    config_snapshot: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Persist reproducibility artifacts; return relative path map."""
    run_dir = ensure_dir(run_dir)
    paths: dict[str, str] = {}
    if config_snapshot is not None:
        paths["config"] = str(write_json(run_dir / "config_snapshot.json", config_snapshot))
    if metrics is not None:
        paths["metrics"] = str(write_json(run_dir / "metrics.json", metrics))
    if equity is not None and len(equity):
        eq_df = equity.reset_index()
        eq_df.columns = ["timestamp", "equity"] if eq_df.shape[1] == 2 else list(eq_df.columns)
        paths["equity_curve"] = str(write_frame(run_dir / "equity_curve.csv", eq_df))
    if trades is not None and not trades.empty:
        paths["trades"] = str(write_frame(run_dir / "trades.csv", trades))
    if fills is not None:
        paths["fills"] = str(write_json(run_dir / "fills.json", fills))
    if orders is not None:
        paths["orders"] = str(write_json(run_dir / "orders.json", orders))
    if risk_events is not None:
        paths["risk_events"] = str(write_json(run_dir / "risk_events.json", risk_events))
    return paths
