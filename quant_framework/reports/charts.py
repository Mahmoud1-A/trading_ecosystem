"""Equity curve chart persistence."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def save_equity_curve_chart(
    equity: pd.Series,
    path: Path,
    *,
    title: str = "Equity Curve",
) -> Path:
    """Save equity curve PNG. Requires matplotlib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to save equity charts") from exc

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(equity.index if isinstance(equity.index, pd.DatetimeIndex) else range(len(equity)), equity.values)
    ax.set_title(title)
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info("Saved equity chart to %s", path)
    return path
