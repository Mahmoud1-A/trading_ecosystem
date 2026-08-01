"""Robust OOS fitness — training metrics never promote candidates."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


class FitnessError(ValueError):
    pass


@dataclass(frozen=True)
class FoldOOSMetrics:
    """Out-of-sample metrics for a single validation fold."""

    fold_id: int
    expectancy: float
    sharpe: float
    profit_factor: float
    calmar: float
    max_drawdown: float
    drawdown_duration: float = 0.0
    worst_day: float = 0.0
    turnover: float = 0.0
    prop_breach_prob: float = 0.0
    regime_entropy: float = 1.0
    n_trades: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "fold_id": self.fold_id,
            "expectancy": self.expectancy,
            "sharpe": self.sharpe,
            "profit_factor": self.profit_factor,
            "calmar": self.calmar,
            "max_drawdown": self.max_drawdown,
            "drawdown_duration": self.drawdown_duration,
            "worst_day": self.worst_day,
            "turnover": self.turnover,
            "prop_breach_prob": self.prop_breach_prob,
            "regime_entropy": self.regime_entropy,
            "n_trades": self.n_trades,
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> FoldOOSMetrics:
        return FoldOOSMetrics(
            fold_id=int(raw.get("fold_id", 0)),
            expectancy=float(raw.get("expectancy", 0.0)),
            sharpe=float(raw.get("sharpe", 0.0)),
            profit_factor=float(raw.get("profit_factor", 0.0)),
            calmar=float(raw.get("calmar", 0.0)),
            max_drawdown=float(raw.get("max_drawdown", 0.0)),
            drawdown_duration=float(raw.get("drawdown_duration", 0.0)),
            worst_day=float(raw.get("worst_day", 0.0)),
            turnover=float(raw.get("turnover", 0.0)),
            prop_breach_prob=float(raw.get("prop_breach_prob", 0.0)),
            regime_entropy=float(raw.get("regime_entropy", 1.0)),
            n_trades=int(raw.get("n_trades", 0)),
        )


@dataclass(frozen=True)
class FitnessResult:
    fitness: float
    ranking_source: str
    components: dict[str, float]
    fold_scores: tuple[float, ...]
    rejected: bool = False
    rejection_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "fitness": self.fitness,
            "ranking_source": self.ranking_source,
            "components": dict(self.components),
            "fold_scores": list(self.fold_scores),
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
        }

    @staticmethod
    def _coerce_components(raw: Mapping[str, Any] | None) -> dict[str, float]:
        """Accept finite scalars only; expand fold_trade_counts lists; never float(list)."""
        components: dict[str, float] = {}
        for key, value in dict(raw or {}).items():
            name = str(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                out = float(value)
                if math.isfinite(out):
                    components[name] = out
                continue
            if isinstance(value, str):
                try:
                    out = float(value.strip())
                except ValueError:
                    continue
                if math.isfinite(out):
                    components[name] = out
                continue
            if name == "fold_trade_counts" and isinstance(value, (list, tuple)):
                for i, item in enumerate(value):
                    if isinstance(item, bool):
                        continue
                    try:
                        components[f"fold_{i}_oos_trades"] = float(int(item))
                    except (TypeError, ValueError):
                        continue
                continue
            # list / tuple / dict metadata — skip; never call float() directly.
        return components

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> FitnessResult:
        return FitnessResult(
            fitness=float(raw.get("fitness", 0.0)),
            ranking_source=str(raw.get("ranking_source") or "validation_oos"),
            components=FitnessResult._coerce_components(raw.get("components")),
            fold_scores=tuple(float(x) for x in (raw.get("fold_scores") or ())),
            rejected=bool(raw.get("rejected", False)),
            rejection_reason=raw.get("rejection_reason"),
        )


RANKING_SOURCE_OOS = "validation_oos"
RANKING_SOURCE_IS = "training_is"

NO_OOS_TRADES = "NO_OOS_TRADES"
INSUFFICIENT_OOS_TRADES = "INSUFFICIENT_OOS_TRADES"
NEGATIVE_EXPECTANCY = "NEGATIVE_EXPECTANCY"
PF_BELOW_ONE = "PF_BELOW_ONE"
MAX_DRAWDOWN_EXCEEDED = "MAX_DRAWDOWN_EXCEEDED"

# Exact SCORE_QUALIFIED hard rejects (fitness gate — never advance to Stress).
SCORE_QUALIFIED_REJECT_REASONS = frozenset(
    {
        NO_OOS_TRADES,
        INSUFFICIENT_OOS_TRADES,
        NEGATIVE_EXPECTANCY,
        PF_BELOW_ONE,
        MAX_DRAWDOWN_EXCEEDED,
    }
)


def _median(xs: Sequence[float]) -> float:
    if not xs:
        return 0.0
    return float(np.median(np.asarray(xs, dtype=float)))


def _trimmed_mean(xs: Sequence[float], trim: float = 0.1) -> float:
    arr = np.sort(np.asarray(xs, dtype=float))
    if len(arr) == 0:
        return 0.0
    k = int(len(arr) * trim)
    if len(arr) - 2 * k <= 0:
        return float(np.mean(arr))
    return float(np.mean(arr[k : len(arr) - k]))


def _lower_confidence_bound(xs: Sequence[float], z: float = 1.0) -> float:
    arr = np.asarray(xs, dtype=float)
    if len(arr) == 0:
        return 0.0
    if len(arr) == 1:
        return float(arr[0])
    return float(np.mean(arr) - z * np.std(arr, ddof=1) / np.sqrt(len(arr)))


def _percentile(xs: Sequence[float], q: float = 25.0) -> float:
    if not xs:
        return 0.0
    return float(np.percentile(np.asarray(xs, dtype=float), q))


@dataclass
class RobustFitness:
    """
    Aggregate fold-level OOS evidence into a single ranking score.

    One exceptional fold cannot dominate: medians, trimmed means, and lower
    confidence bounds are preferred over raw means.

    SCORE_QUALIFIED hard gates (all required):
    - median OOS expectancy > 0
    - median profit factor > 1
    - total OOS trades >= ``min_total_oos_trades`` (and per-fold floors)
    - worst |MaxDD| <= ``max_oos_drawdown``
    """

    complexity_penalty: float = 0.05
    drawdown_penalty: float = 1.0
    turnover_penalty: float = 0.1
    similarity_penalty: float = 0.5
    multiple_testing_penalty: float = 0.0
    min_folds: int = 1
    # Closed OOS trade floors (synced from SearchBudget in the controller).
    min_total_oos_trades: int = 8
    min_oos_trades_per_fold: int = 1
    # Absolute drawdown ceiling as a positive fraction (0.20 == 20%).
    max_oos_drawdown: float = 0.20

    def score(
        self,
        folds: Sequence[FoldOOSMetrics],
        *,
        ranking_source: str = RANKING_SOURCE_OOS,
        complexity: float = 0.0,
        behavioral_similarity: float = 0.0,
        cost_sensitivity: float = 0.0,
        parameter_instability: float = 0.0,
        train_metrics: Mapping[str, float] | None = None,
    ) -> FitnessResult:
        if ranking_source != RANKING_SOURCE_OOS:
            raise FitnessError(
                f"ranking_source must be {RANKING_SOURCE_OOS!r}; "
                f"got {ranking_source!r}. Training metrics cannot promote."
            )
        if train_metrics:
            # Explicitly ignore IS metrics for ranking — presence is diagnostic only
            pass
        if len(folds) < self.min_folds:
            return FitnessResult(
                fitness=float("-inf"),
                ranking_source=ranking_source,
                components={},
                fold_scores=(),
                rejected=True,
                rejection_reason="insufficient_oos_folds",
            )

        # Trade counts are mandatory evidence. Expectancy/PF are trade-based and
        # become 0.0 with empty trade lists, while MaxDD/Calmar come from the
        # equity curve and can still be non-zero without closed trades — never
        # score-qualify on equity-only movement.
        fold_trade_counts = tuple(max(0, int(f.n_trades)) for f in folds)
        total_oos_trades = int(sum(fold_trade_counts))
        expectancies = [float(f.expectancy) for f in folds]
        pfs = [float(f.profit_factor) for f in folds]
        dds = [float(f.max_drawdown) for f in folds]
        median_expectancy = _median(expectancies)
        median_pf = _median(pfs)
        # Worst drawdown magnitude (fold max_drawdown is typically <= 0).
        worst_dd = float(min(dds)) if dds else 0.0
        abs_worst_dd = abs(worst_dd)
        trade_components = {
            "total_oos_trades": float(total_oos_trades),
            "minimum_required_oos_trades": float(self.min_total_oos_trades),
            "minimum_required_oos_trades_per_fold": float(self.min_oos_trades_per_fold),
            "median_oos_expectancy": float(median_expectancy),
            "median_profit_factor": float(median_pf),
            "worst_oos_max_drawdown": float(worst_dd),
            "max_oos_drawdown_limit": float(self.max_oos_drawdown),
            **{f"fold_{i}_oos_trades": float(n) for i, n in enumerate(fold_trade_counts)},
        }
        if total_oos_trades == 0:
            return FitnessResult(
                fitness=float("-inf"),
                ranking_source=ranking_source,
                components=trade_components,
                fold_scores=(),
                rejected=True,
                rejection_reason=NO_OOS_TRADES,
            )
        if total_oos_trades < self.min_total_oos_trades or any(
            n < self.min_oos_trades_per_fold for n in fold_trade_counts
        ):
            return FitnessResult(
                fitness=float("-inf"),
                ranking_source=ranking_source,
                components=trade_components,
                fold_scores=(),
                rejected=True,
                rejection_reason=INSUFFICIENT_OOS_TRADES,
            )
        if not (median_expectancy > 0.0):
            return FitnessResult(
                fitness=float("-inf"),
                ranking_source=ranking_source,
                components=trade_components,
                fold_scores=(),
                rejected=True,
                rejection_reason=NEGATIVE_EXPECTANCY,
            )
        if not (median_pf > 1.0):
            return FitnessResult(
                fitness=float("-inf"),
                ranking_source=ranking_source,
                components=trade_components,
                fold_scores=(),
                rejected=True,
                rejection_reason=PF_BELOW_ONE,
            )
        if abs_worst_dd > float(self.max_oos_drawdown) + 1e-15:
            return FitnessResult(
                fitness=float("-inf"),
                ranking_source=ranking_source,
                components=trade_components,
                fold_scores=(),
                rejected=True,
                rejection_reason=MAX_DRAWDOWN_EXCEEDED,
            )

        sharpes = [f.sharpe for f in folds]
        calmars = [f.calmar for f in folds]
        turnovers = [f.turnover for f in folds]
        breaches = [f.prop_breach_prob for f in folds]
        regimes = [f.regime_entropy for f in folds]

        fold_scores = tuple(
            0.35 * f.expectancy
            + 0.25 * f.sharpe
            + 0.15 * min(f.profit_factor, 3.0)
            + 0.15 * f.calmar
            - self.drawdown_penalty * abs(f.max_drawdown)
            - 0.5 * f.prop_breach_prob
            for f in folds
        )

        positive = {
            "median_oos_expectancy": median_expectancy,
            "oos_dsr_proxy": _lower_confidence_bound(sharpes),
            "oos_profit_factor": median_pf,
            "oos_calmar": _median(calmars),
            "fold_stability": 1.0 / (1.0 + float(np.std(fold_scores)) if fold_scores else 1.0),
            "regime_breadth": _median(regimes),
            "low_prop_breach": 1.0 - _median(breaches),
            "worst_fold_score": float(min(fold_scores)) if fold_scores else 0.0,
            "pct25_fold_score": _percentile(fold_scores, 25),
            "trimmed_mean_fold": _trimmed_mean(fold_scores),
        }
        negative = {
            "max_drawdown": _median([abs(x) for x in dds]),
            "turnover": _median(turnovers),
            "complexity": complexity,
            "behavioral_similarity": behavioral_similarity,
            "cost_sensitivity": cost_sensitivity,
            "parameter_instability": parameter_instability,
            "multiple_testing_penalty": self.multiple_testing_penalty,
        }

        fitness = (
            0.40 * positive["trimmed_mean_fold"]
            + 0.20 * positive["pct25_fold_score"]
            + 0.15 * positive["worst_fold_score"]
            + 0.10 * positive["fold_stability"]
            + 0.10 * positive["low_prop_breach"]
            + 0.05 * positive["regime_breadth"]
            - self.complexity_penalty * negative["complexity"]
            - self.turnover_penalty * negative["turnover"]
            - self.similarity_penalty * negative["behavioral_similarity"]
            - 0.25 * negative["cost_sensitivity"]
            - 0.25 * negative["parameter_instability"]
            - negative["multiple_testing_penalty"]
        )

        components = {
            **trade_components,
            **{f"pos_{k}": v for k, v in positive.items()},
            **{f"neg_{k}": v for k, v in negative.items()},
        }
        return FitnessResult(
            fitness=float(fitness),
            ranking_source=ranking_source,
            components=components,
            fold_scores=fold_scores,
        )


def assert_oos_only_promotion(
    *,
    train_score: float | None,
    oos_fitness: float,
    promote_threshold: float,
) -> bool:
    """Return True only when OOS fitness clears the bar; train_score is ignored."""
    _ = train_score  # never used for promotion
    return oos_fitness >= promote_threshold
