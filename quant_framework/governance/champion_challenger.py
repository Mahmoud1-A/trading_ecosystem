"""Champion / challenger governance — backtest alone cannot replace a champion."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from governance.degradation import DecayState
from governance.drift import MetricSnapshot


class ChallengerMode(str, Enum):
    PAPER = "PAPER"
    SHADOW = "SHADOW"


class ReplacementError(PermissionError):
    pass


@dataclass
class StrategyRole:
    strategy_id: str
    lineage_id: str
    decay_state: DecayState
    backtest_sharpe: float = 0.0
    paper_sharpe: float = 0.0
    shadow_sharpe: float = 0.0
    contribution: float = 0.0
    risk_score: float = 0.0  # higher = worse
    execution_score: float = 0.0  # higher = better
    vault_passed: bool = False
    paper_gates_passed: bool = False
    metrics: MetricSnapshot = field(default_factory=MetricSnapshot)


@dataclass
class ReplacementDecision:
    accepted: bool
    reasons: list[str]
    champion_id: str
    challenger_id: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "accepted": self.accepted,
            "reasons": list(self.reasons),
            "champion_id": self.champion_id,
            "challenger_id": self.challenger_id,
        }


@dataclass
class ChampionChallenger:
    champion: StrategyRole | None = None
    challengers: dict[str, StrategyRole] = field(default_factory=dict)

    def set_champion(self, role: StrategyRole) -> None:
        self.champion = role

    def register_challenger(self, role: StrategyRole, *, mode: ChallengerMode) -> None:
        role_meta = StrategyRole(
            strategy_id=role.strategy_id,
            lineage_id=role.lineage_id,
            decay_state=role.decay_state,
            backtest_sharpe=role.backtest_sharpe,
            paper_sharpe=role.paper_sharpe,
            shadow_sharpe=role.shadow_sharpe,
            contribution=role.contribution,
            risk_score=role.risk_score,
            execution_score=role.execution_score,
            vault_passed=role.vault_passed,
            paper_gates_passed=role.paper_gates_passed,
            metrics=role.metrics,
        )
        # Tag mode in contribution note via id namespace
        self.challengers[f"{mode.value}:{role.strategy_id}"] = role_meta

    def evaluate_replacement(
        self,
        challenger_key: str,
        *,
        require_vault: bool = True,
        require_paper_gates: bool = True,
        min_observation_ok: bool = True,
        risk_gate_ok: bool = True,
        execution_gate_ok: bool = True,
        contribution_gate_ok: bool = True,
    ) -> ReplacementDecision:
        if self.champion is None:
            raise ReplacementError("no champion set")
        challenger = self.challengers.get(challenger_key)
        if challenger is None:
            raise ReplacementError(f"unknown challenger {challenger_key}")

        reasons: list[str] = []

        # Backtest alone is insufficient
        if challenger.backtest_sharpe > self.champion.backtest_sharpe:
            if challenger.paper_sharpe <= self.champion.paper_sharpe and challenger.shadow_sharpe <= 0:
                reasons.append("cannot_replace_on_backtest_alone")

        if require_vault and not challenger.vault_passed:
            reasons.append("vault_gate_failed")
        if require_paper_gates and not challenger.paper_gates_passed:
            reasons.append("paper_gates_failed")
        if not min_observation_ok:
            reasons.append("observation_gate_failed")
        if not risk_gate_ok or challenger.risk_score > self.champion.risk_score:
            if not risk_gate_ok:
                reasons.append("risk_gate_failed")
            elif challenger.risk_score > self.champion.risk_score + 1e-12:
                reasons.append("risk_worse_than_champion")
        if not execution_gate_ok or challenger.execution_score < self.champion.execution_score:
            if not execution_gate_ok:
                reasons.append("execution_gate_failed")
            elif challenger.execution_score + 1e-12 < self.champion.execution_score:
                reasons.append("execution_worse_than_champion")
        if not contribution_gate_ok or challenger.contribution <= self.champion.contribution:
            if not contribution_gate_ok:
                reasons.append("contribution_gate_failed")
            elif challenger.contribution <= self.champion.contribution:
                reasons.append("contribution_not_improved")

        if challenger.decay_state in {DecayState.SUSPENDED, DecayState.RETIRED, DecayState.REDUCE_ONLY}:
            reasons.append("challenger_not_healthy")

        # Need out-of-sample evidence (paper or shadow), not backtest only
        if challenger.paper_sharpe <= 0 and challenger.shadow_sharpe <= 0:
            reasons.append("cannot_replace_on_backtest_alone")

        accepted = len(reasons) == 0
        return ReplacementDecision(
            accepted=accepted,
            reasons=reasons,
            champion_id=self.champion.strategy_id,
            challenger_id=challenger.strategy_id,
        )

    def replace_if_allowed(self, challenger_key: str, **gates: Any) -> StrategyRole:
        decision = self.evaluate_replacement(challenger_key, **gates)
        if not decision.accepted:
            raise ReplacementError(",".join(decision.reasons))
        new_champ = self.challengers[challenger_key]
        self.champion = new_champ
        return new_champ
