"""Portfolio optimizers — no unconstrained mean-variance as default."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

import numpy as np

from portfolio.constraints import (
    ConstraintViolation,
    MemberMeta,
    PortfolioConstraints,
    validate_weights,
)
from portfolio.correlation import correlation_penalty
from portfolio.marginal_contribution import (
    marginal_contribution,
    portfolio_quality,
    risk_parity_weights,
)
from portfolio.risk_budget import RiskBudget


class ConstructionMethod(str, Enum):
    EQUAL_RISK_CONTRIBUTION = "equal_risk_contribution"
    VOLATILITY_SCALING = "volatility_scaling"
    CONSTRAINED_RISK_PARITY = "constrained_risk_parity"
    GREEDY_MARGINAL = "greedy_marginal_contribution"
    ROBUST_COVARIANCE = "robust_covariance_allocation"
    HIERARCHICAL_CLUSTER = "hierarchical_cluster_allocation"


class UnstableAllocationError(ValueError):
    pass


@dataclass
class OptimizerResult:
    weights: dict[str, float]
    method: ConstructionMethod
    quality: float
    correlation_penalty: float
    accepted: bool
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "weights": dict(self.weights),
            "method": self.method.value,
            "quality": self.quality,
            "correlation_penalty": self.correlation_penalty,
            "accepted": self.accepted,
            "reason": self.reason,
        }


@dataclass
class PortfolioOptimizer:
    constraints: PortfolioConstraints
    risk_budget: RiskBudget | None = None
    min_quality: float = -1e9
    max_mean_corr: float = 0.9

    def optimize(
        self,
        members: Sequence[MemberMeta],
        *,
        pairwise: Mapping[tuple[str, str], float],
        vols: Mapping[str, float] | None = None,
        method: ConstructionMethod = ConstructionMethod.GREEDY_MARGINAL,
        pnl_by_id: Mapping[str, Sequence[float]] | None = None,
    ) -> OptimizerResult:
        if not members:
            raise UnstableAllocationError("empty member set")
        vols = vols or {m.candidate_id: 0.1 for m in members}

        if method is ConstructionMethod.EQUAL_RISK_CONTRIBUTION:
            weights = risk_parity_weights(vols)
        elif method is ConstructionMethod.VOLATILITY_SCALING:
            weights = risk_parity_weights(vols)
        elif method is ConstructionMethod.CONSTRAINED_RISK_PARITY:
            weights = self._constrained_risk_parity(members, vols)
        elif method is ConstructionMethod.GREEDY_MARGINAL:
            weights = self._greedy_marginal(members, pairwise)
        elif method is ConstructionMethod.ROBUST_COVARIANCE:
            weights = self._robust_covariance(members, pairwise, vols)
        elif method is ConstructionMethod.HIERARCHICAL_CLUSTER:
            weights = self._hierarchical(members, pairwise)
        else:
            raise UnstableAllocationError(f"unsupported method {method}")

        # Never default to unconstrained mean-variance
        try:
            validate_weights(weights, members, self.constraints, correlation_matrix=dict(pairwise))
        except ConstraintViolation as exc:
            return OptimizerResult(
                weights=weights,
                method=method,
                quality=float("-inf"),
                correlation_penalty=correlation_penalty(weights, pairwise),
                accepted=False,
                reason=str(exc),
            )

        quality = portfolio_quality(weights, members, pairwise)
        corr_pen = correlation_penalty(weights, pairwise)
        mean_corr = _mean_corr(pairwise)
        if mean_corr > self.max_mean_corr and len(weights) > 1:
            return OptimizerResult(
                weights=weights,
                method=method,
                quality=quality,
                correlation_penalty=corr_pen,
                accepted=False,
                reason="unstable_high_mean_correlation",
            )
        if quality < self.min_quality:
            return OptimizerResult(
                weights=weights,
                method=method,
                quality=quality,
                correlation_penalty=corr_pen,
                accepted=False,
                reason="quality_below_minimum",
            )
        rb = self.risk_budget or RiskBudget()
        if not rb.within_limits(weights, vols):
            return OptimizerResult(
                weights=weights,
                method=method,
                quality=quality,
                correlation_penalty=corr_pen,
                accepted=False,
                reason="risk_budget_exceeded",
            )
        return OptimizerResult(
            weights=weights,
            method=method,
            quality=quality,
            correlation_penalty=corr_pen,
            accepted=True,
        )

    def _constrained_risk_parity(
        self, members: Sequence[MemberMeta], vols: Mapping[str, float]
    ) -> dict[str, float]:
        weights = risk_parity_weights({m.candidate_id: vols[m.candidate_id] for m in members})
        # Cap individual weights
        capped = {
            cid: min(w, self.constraints.max_strategy_weight) for cid, w in weights.items()
        }
        s = sum(capped.values()) or 1.0
        return {cid: w / s for cid, w in capped.items()}

    def _greedy_marginal(
        self,
        members: Sequence[MemberMeta],
        pairwise: Mapping[tuple[str, str], float],
    ) -> dict[str, float]:
        ranked = sorted(members, key=lambda m: m.expected_oos, reverse=True)
        selected: list[MemberMeta] = []
        weights: dict[str, float] = {}
        for m in ranked:
            if len(selected) >= self.constraints.max_members:
                break
            if not selected:
                selected.append(m)
                weights = {m.candidate_id: 1.0}
                continue
            delta = marginal_contribution(
                m.candidate_id,
                current_weights=weights,
                members=list(selected) + [m],
                pairwise=pairwise,
                new_weight=1.0 / (len(selected) + 1),
            )
            if delta <= 0:
                continue  # must improve marginal portfolio quality
            selected.append(m)
            # Equal weights among selected after each add
            w = 1.0 / len(selected)
            weights = {s.candidate_id: w for s in selected}
            # Cap
            if any(x > self.constraints.max_strategy_weight for x in weights.values()):
                weights = {
                    cid: min(x, self.constraints.max_strategy_weight) for cid, x in weights.items()
                }
                s = sum(weights.values())
                weights = {cid: x / s for cid, x in weights.items()}
        if not weights:
            top = ranked[0]
            weights = {top.candidate_id: 1.0}
        return weights

    def _robust_covariance(
        self,
        members: Sequence[MemberMeta],
        pairwise: Mapping[tuple[str, str], float],
        vols: Mapping[str, float],
    ) -> dict[str, float]:
        # Shrinkage toward diagonal — inverse-vol with correlation discount
        scores: dict[str, float] = {}
        for m in members:
            disc = 1.0
            for other in members:
                if other.candidate_id == m.candidate_id:
                    continue
                corr = pairwise.get(
                    (m.candidate_id, other.candidate_id),
                    pairwise.get((other.candidate_id, m.candidate_id), 0.0),
                )
                disc *= 1.0 - 0.5 * max(0.0, corr)
            scores[m.candidate_id] = max(m.expected_oos, 0.0) * disc / max(vols[m.candidate_id], 1e-8)
        s = sum(scores.values()) or 1.0
        weights = {cid: v / s for cid, v in scores.items()}
        return _renorm_cap(weights, self.constraints.max_strategy_weight)

    def _hierarchical(
        self,
        members: Sequence[MemberMeta],
        pairwise: Mapping[tuple[str, str], float],
    ) -> dict[str, float]:
        # Cluster by family then equal-weight clusters
        families: dict[str, list[MemberMeta]] = {}
        for m in members:
            families.setdefault(m.family, []).append(m)
        cluster_w = 1.0 / len(families)
        weights: dict[str, float] = {}
        for _fam, ms in families.items():
            w = cluster_w / len(ms)
            for m in ms:
                weights[m.candidate_id] = w
        _ = pairwise
        return weights


def _mean_corr(pairwise: Mapping[tuple[str, str], float]) -> float:
    if not pairwise:
        return 0.0
    return float(np.mean(list(pairwise.values())))


def _renorm_cap(weights: dict[str, float], max_w: float) -> dict[str, float]:
    capped = {cid: min(w, max_w) for cid, w in weights.items()}
    s = sum(capped.values()) or 1.0
    return {cid: w / s for cid, w in capped.items()}
