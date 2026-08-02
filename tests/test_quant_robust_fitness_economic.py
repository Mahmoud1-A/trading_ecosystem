from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "quant_framework"))

from discovery.fitness import (
    ECONOMIC_RETURN_BELOW_MINIMUM,
    FoldOOSMetrics,
    RobustFitness,
)


def _fold(
    *,
    calmar: float,
    max_drawdown: float,
    expectancy: float = 0.02,
    sharpe: float = 1.0,
    profit_factor: float = 1.30,
    n_trades: int = 10,
) -> FoldOOSMetrics:
    return FoldOOSMetrics(
        fold_id=0,
        expectancy=expectancy,
        sharpe=sharpe,
        profit_factor=profit_factor,
        calmar=calmar,
        max_drawdown=max_drawdown,
        n_trades=n_trades,
    )


def test_high_calmar_from_microscopic_drawdown_fails_economic_gate() -> None:
    result = RobustFitness(
        min_total_oos_trades=8,
        min_risk_normalized_annual_return=0.12,
    ).score(
        [_fold(calmar=20.0, max_drawdown=-0.0005)]
    )

    assert result.rejected is True
    assert result.rejection_reason == ECONOMIC_RETURN_BELOW_MINIMUM
    assert result.components["median_implied_annual_return"] == 0.01
    assert result.components["median_permitted_risk_scale"] == 4.0
    assert result.components["median_risk_normalized_annual_return"] == 0.04


def test_economically_useful_return_passes_at_common_risk_budget() -> None:
    result = RobustFitness(
        min_total_oos_trades=8,
        min_risk_normalized_annual_return=0.12,
    ).score(
        [_fold(calmar=3.0, max_drawdown=-0.04)]
    )

    assert result.rejected is False
    assert result.components["median_implied_annual_return"] == 0.12
    assert result.components["median_risk_normalized_annual_return"] == 0.15


def test_raw_calmar_cannot_dominate_economic_ranking() -> None:
    fitness = RobustFitness(
        min_total_oos_trades=8,
        min_risk_normalized_annual_return=0.0,
    )
    metric_gamer = fitness.score(
        [_fold(calmar=30.0, max_drawdown=-0.0003)]
    )
    economically_stronger = fitness.score(
        [_fold(calmar=2.5, max_drawdown=-0.05)]
    )

    assert metric_gamer.rejected is False
    assert economically_stronger.rejected is False
    assert economically_stronger.fitness > metric_gamer.fitness
    assert metric_gamer.components["pos_oos_calmar_capped"] == 3.0
