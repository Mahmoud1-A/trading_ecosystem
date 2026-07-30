from __future__ import annotations

from pathlib import Path
from typing import Any

from trading_ecosystem.common.config import CONFIG_DIR, load_yaml
from trading_ecosystem.strategies.base import Strategy
from trading_ecosystem.strategies.breakout_atr import BreakoutATR
from trading_ecosystem.strategies.donchian_atr import DonchianATR
from trading_ecosystem.strategies.gap_fade import GapFade
from trading_ecosystem.strategies.seasonality import SeasonalityWindow
from trading_ecosystem.strategies.zscore_revert import ZScoreRevert

_REGISTRY: dict[str, type[Strategy]] = {
    "DonchianATR": DonchianATR,
    "BreakoutATR": BreakoutATR,
    "GapFade": GapFade,
    "ZScoreRevert": ZScoreRevert,
    "SeasonalityWindow": SeasonalityWindow,
}


def register_strategy(name: str, cls: type[Strategy]) -> None:
    _REGISTRY[name] = cls


def available_strategies() -> list[str]:
    return sorted(_REGISTRY)


def build_strategy(
    class_name: str,
    strategy_id: str,
    symbols: list[str],
    params: dict[str, Any] | None = None,
) -> Strategy:
    cls = _REGISTRY.get(class_name)
    if cls is None:
        raise ValueError(f"Unknown strategy class: {class_name}")
    return cls(strategy_id=strategy_id, symbols=list(symbols), params=dict(params or {}))


def build_strategies(path: str | Path | None = None) -> list[Strategy]:
    cfg_path = Path(path) if path else CONFIG_DIR / "strategies.yaml"
    raw = load_yaml(cfg_path)
    strategies: list[Strategy] = []
    for item in raw.get("strategies", []):
        if not item.get("enabled", True):
            continue
        strategies.append(
            build_strategy(
                class_name=str(item["class"]),
                strategy_id=str(item["id"]),
                symbols=list(item.get("symbols") or []),
                params=dict(item.get("params") or {}),
            )
        )
    return strategies
