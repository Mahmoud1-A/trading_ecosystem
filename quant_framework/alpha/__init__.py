"""Alpha package — strategy protocol and implementations."""

from alpha.mean_reversion import MeanReversionStrategy
from alpha.protocol import StrategyProtocol, StrategyState
from alpha.runner import StrategyRunResult, build_signal_fn, run_strategy_backtest

__all__ = [
    "MeanReversionStrategy",
    "StrategyProtocol",
    "StrategyRunResult",
    "StrategyState",
    "build_signal_fn",
    "run_strategy_backtest",
]
