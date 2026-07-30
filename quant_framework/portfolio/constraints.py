"""Portfolio construction constraints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


class ConstraintViolation(ValueError):
    pass


@dataclass(frozen=True)
class PortfolioConstraints:
    max_members: int = 10
    min_members: int = 1
    max_strategy_weight: float = 0.40
    max_family_weight: float = 0.60
    max_symbol_exposure: float = 0.50
    max_asset_class_exposure: float = 0.80
    max_regime_concentration: float = 0.70
    max_session_concentration: float = 0.70
    max_pairwise_correlation: float = 0.85
    max_turnover: float = 5.0
    max_margin_utilization: float = 0.80
    max_prop_breach_probability: float = 0.25

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemberMeta:
    candidate_id: str
    lineage_id: str
    family: str
    symbols: tuple[str, ...]
    asset_class: str
    regime_bucket: str
    session: str
    turnover: float
    margin_usage: float
    prop_breach_prob: float
    expected_oos: float


def validate_weights(
    weights: Mapping[str, float],
    members: Sequence[MemberMeta],
    constraints: PortfolioConstraints,
    *,
    correlation_matrix: Mapping[tuple[str, str], float] | None = None,
) -> None:
    """Raise ConstraintViolation if any portfolio constraint is breached."""
    by_id = {m.candidate_id: m for m in members}
    ids = list(weights.keys())
    if len(ids) > constraints.max_members:
        raise ConstraintViolation(f"max_members {constraints.max_members} exceeded")
    if len(ids) < constraints.min_members:
        raise ConstraintViolation(f"min_members {constraints.min_members} not met")

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ConstraintViolation(f"weights must sum to 1, got {total}")

    for cid, w in weights.items():
        if w < 0:
            raise ConstraintViolation(f"negative weight for {cid}")
        # Single-member portfolios necessarily have weight 1.0
        if len(ids) > 1 and w > constraints.max_strategy_weight + 1e-12:
            raise ConstraintViolation(
                f"max_strategy_weight {constraints.max_strategy_weight} exceeded by {cid}"
            )
        if cid not in by_id:
            raise ConstraintViolation(f"unknown member {cid}")

    # Family concentration (vacuous for a singleton)
    if len(ids) > 1:
        fam: dict[str, float] = {}
        for cid, w in weights.items():
            f = by_id[cid].family
            fam[f] = fam.get(f, 0.0) + w
        for f, w in fam.items():
            if w > constraints.max_family_weight + 1e-12:
                raise ConstraintViolation(f"max_family_weight exceeded for {f}")

        # Symbol exposure
        sym: dict[str, float] = {}
        for cid, w in weights.items():
            for s in by_id[cid].symbols:
                sym[s] = sym.get(s, 0.0) + w
        for s, w in sym.items():
            if w > constraints.max_symbol_exposure + 1e-12:
                raise ConstraintViolation(f"max_symbol_exposure exceeded for {s}")

        # Asset class
        ac: dict[str, float] = {}
        for cid, w in weights.items():
            a = by_id[cid].asset_class
            ac[a] = ac.get(a, 0.0) + w
        for a, w in ac.items():
            if w > constraints.max_asset_class_exposure + 1e-12:
                raise ConstraintViolation(f"max_asset_class_exposure exceeded for {a}")

        # Regime / session
        reg: dict[str, float] = {}
        sess: dict[str, float] = {}
        for cid, w in weights.items():
            m = by_id[cid]
            reg[m.regime_bucket] = reg.get(m.regime_bucket, 0.0) + w
            sess[m.session] = sess.get(m.session, 0.0) + w
        for k, w in reg.items():
            if w > constraints.max_regime_concentration + 1e-12:
                raise ConstraintViolation(f"max_regime_concentration exceeded for {k}")
        for k, w in sess.items():
            if w > constraints.max_session_concentration + 1e-12:
                raise ConstraintViolation(f"max_session_concentration exceeded for {k}")

    # Turnover / margin / prop
    turnover = sum(weights[cid] * by_id[cid].turnover for cid in ids)
    margin = sum(weights[cid] * by_id[cid].margin_usage for cid in ids)
    prop = sum(weights[cid] * by_id[cid].prop_breach_prob for cid in ids)
    if turnover > constraints.max_turnover + 1e-12:
        raise ConstraintViolation("max_turnover exceeded")
    if margin > constraints.max_margin_utilization + 1e-12:
        raise ConstraintViolation("max_margin_utilization exceeded")
    if prop > constraints.max_prop_breach_probability + 1e-12:
        raise ConstraintViolation("max_prop_breach_probability exceeded")

    if correlation_matrix:
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                corr = correlation_matrix.get((a, b), correlation_matrix.get((b, a), 0.0))
                if corr > constraints.max_pairwise_correlation + 1e-12:
                    # High correlation is a soft constraint enforced via penalty in
                    # the optimizer; hard block when both weights are material
                    if weights[a] > 0.05 and weights[b] > 0.05:
                        raise ConstraintViolation(
                            f"max_pairwise_correlation exceeded for {a},{b}"
                        )
