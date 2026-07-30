"""Marginal risk-adjusted contribution helpers."""

from __future__ import annotations

from typing import Mapping, Sequence

from portfolio.correlation import correlation_penalty
from portfolio.constraints import MemberMeta


def portfolio_quality(
    weights: Mapping[str, float],
    members: Sequence[MemberMeta],
    pairwise: Mapping[tuple[str, str], float],
) -> float:
    """Scalar portfolio quality — OOS contribution minus correlation / risk penalties."""
    by_id = {m.candidate_id: m for m in members}
    expected = sum(weights[cid] * by_id[cid].expected_oos for cid in weights)
    corr_pen = correlation_penalty(weights, pairwise)
    prop_pen = sum(weights[cid] * by_id[cid].prop_breach_prob for cid in weights)
    turn_pen = 0.05 * sum(weights[cid] * by_id[cid].turnover for cid in weights)
    # Diversification bonus: reward lower average pairwise correlation among members
    n = len(weights)
    div_bonus = 0.0
    if n > 1:
        corrs = []
        ids = list(weights.keys())
        for i, a in enumerate(ids):
            for b in ids[i + 1 :]:
                corrs.append(pairwise.get((a, b), pairwise.get((b, a), 0.0)))
        mean_c = float(sum(corrs) / len(corrs)) if corrs else 0.0
        div_bonus = 0.35 * (1.0 - mean_c)
    return float(expected - 2.0 * corr_pen - 0.75 * prop_pen - turn_pen + div_bonus)


def marginal_contribution(
    candidate_id: str,
    *,
    current_weights: Mapping[str, float],
    members: Sequence[MemberMeta],
    pairwise: Mapping[tuple[str, str], float],
    new_weight: float = 0.1,
) -> float:
    """
    Quality delta from adding ``candidate_id`` at ``new_weight`` (renormalized).

    Positive => improves portfolio; used to reject non-improving additions.
    """
    if not current_weights:
        by_id = {m.candidate_id: m for m in members}
        return float(by_id[candidate_id].expected_oos)

    base = portfolio_quality(current_weights, members, pairwise)
    scale = 1.0 - new_weight
    proposed = {cid: w * scale for cid, w in current_weights.items()}
    proposed[candidate_id] = proposed.get(candidate_id, 0.0) + new_weight
    # renormalize
    total = sum(proposed.values())
    proposed = {cid: w / total for cid, w in proposed.items()}
    return portfolio_quality(proposed, members, pairwise) - base


def risk_parity_weights(vols: Mapping[str, float]) -> dict[str, float]:
    inv = {cid: 1.0 / max(v, 1e-8) for cid, v in vols.items()}
    s = sum(inv.values())
    return {cid: v / s for cid, v in inv.items()}
