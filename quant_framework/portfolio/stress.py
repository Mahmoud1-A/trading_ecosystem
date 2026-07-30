"""Portfolio-level stress tests."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

from portfolio.constraints import MemberMeta
from portfolio.marginal_contribution import portfolio_quality


PORTFOLIO_STRESS_SCENARIOS = (
    "base",
    "corr_shock",
    "vol_shock",
    "prop_shock",
    "turnover_shock",
    "remove_best_member",
)


@dataclass(frozen=True)
class PortfolioStressResult:
    scenario: str
    quality: float
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "quality": self.quality,
            "passed": self.passed,
            "details": dict(self.details),
        }


@dataclass
class PortfolioStressTester:
    min_quality_ratio: float = 0.4

    def run(
        self,
        weights: Mapping[str, float],
        members: Sequence[MemberMeta],
        pairwise: Mapping[tuple[str, str], float],
        *,
        scenarios: tuple[str, ...] | None = None,
    ) -> list[PortfolioStressResult]:
        base_q = portfolio_quality(weights, members, pairwise)
        results: list[PortfolioStressResult] = []
        for scenario in scenarios or PORTFOLIO_STRESS_SCENARIOS:
            q, details = self._apply(scenario, weights, members, pairwise, base_q)
            passed = q >= base_q * self.min_quality_ratio or (base_q <= 0 and q >= base_q)
            results.append(
                PortfolioStressResult(scenario=scenario, quality=q, passed=passed, details=details)
            )
        return results

    def _apply(
        self,
        scenario: str,
        weights: Mapping[str, float],
        members: Sequence[MemberMeta],
        pairwise: Mapping[tuple[str, str], float],
        base_q: float,
    ) -> tuple[float, dict[str, Any]]:
        if scenario == "base":
            return base_q, {}
        if scenario == "corr_shock":
            shocked = {k: min(1.0, v + 0.2) for k, v in pairwise.items()}
            return portfolio_quality(weights, members, shocked), {"delta_corr": 0.2}
        if scenario == "vol_shock":
            shocked_m = [
                replace(m, turnover=m.turnover * 1.5, expected_oos=m.expected_oos * 0.8)
                for m in members
            ]
            return portfolio_quality(weights, shocked_m, pairwise), {}
        if scenario == "prop_shock":
            shocked_m = [
                replace(m, prop_breach_prob=min(1.0, m.prop_breach_prob + 0.15)) for m in members
            ]
            return portfolio_quality(weights, shocked_m, pairwise), {}
        if scenario == "turnover_shock":
            shocked_m = [replace(m, turnover=m.turnover * 2.0) for m in members]
            return portfolio_quality(weights, shocked_m, pairwise), {}
        if scenario == "remove_best_member":
            if len(weights) <= 1:
                return base_q * 0.5, {"removed": None}
            best = max(
                weights.keys(),
                key=lambda cid: next(m.expected_oos for m in members if m.candidate_id == cid),
            )
            remaining = {cid: w for cid, w in weights.items() if cid != best}
            s = sum(remaining.values()) or 1.0
            remaining = {cid: w / s for cid, w in remaining.items()}
            rem_m = [m for m in members if m.candidate_id in remaining]
            return portfolio_quality(remaining, rem_m, pairwise), {"removed": best}
        return base_q, {}
