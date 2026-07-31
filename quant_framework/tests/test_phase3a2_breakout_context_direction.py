"""Phase 3A.2: breakout context features must not cast direction votes."""

from __future__ import annotations

from discovery.candidate import build_candidate
from discovery.expression_tree import constant_node, feature_node, op_node
from discovery.family_generator import materialize_family_spec
from discovery.multi_family_campaign import (
    FAMILY_DIRECTION_INCOHERENT,
    validate_family_direction_coherence,
)
from discovery.operators import OperatorId
from discovery.types import CreationMethod, ValueType


def _breakout_cand(
    *,
    entry_op: OperatorId,
    conditions: list[tuple[str, ValueType, OperatorId, float]],
    seed: int = 1,
):
    """Build a breakout candidate; multiple conditions are AND-combined."""
    cmp_nodes = []
    for feature_id, feature_type, cmp_op, threshold in conditions:
        feat = feature_node(feature_id, feature_type)
        cmp_nodes.append(op_node(cmp_op, feat, constant_node(threshold)))
    cond = cmp_nodes[0]
    for extra in cmp_nodes[1:]:
        cond = op_node(OperatorId.AND, cond, extra)
    entry = op_node(entry_op, cond)
    return build_candidate(
        entry_tree=entry,
        strategy_family="breakout",
        creation_method=CreationMethod.MUTATION,
        generation=1,
        parent_ids=("parent_a",),
        grammar_version="strategy_dsl_v1",
        feature_set_version="feature_set_v1_phase6b",
        random_seed=seed,
        family_provenance={"family_id": "breakout"},
    )


class TestBreakoutContextNotDirectional:
    def test_upside_breakout_plus_compression_with_entry_long_accepted(self) -> None:
        spec = materialize_family_spec("breakout", seed=1)
        cand = _breakout_cand(
            entry_op=OperatorId.ENTRY_LONG,
            conditions=[
                (
                    "price.breakout_distance_20",
                    ValueType.RATIO,
                    OperatorId.GREATER_THAN,
                    0.02,
                ),
                (
                    "vol.range_compression_20",
                    ValueType.RATIO,
                    OperatorId.GREATER_THAN,
                    0.3,
                ),
            ],
        )
        result = validate_family_direction_coherence(cand, spec)
        assert result.coherent
        assert result.details["expected_direction"] == "ENTRY_LONG"
        assert result.details["entry_direction"] == "ENTRY_LONG"
        assert result.details["coherence_reason"] == "direction_coherent"
        assert any(
            e["condition_feature"] == "price.breakout_distance_20"
            for e in result.details["directional_evidence"]
        )
        assert any(
            e["condition_feature"] == "vol.range_compression_20"
            for e in result.details["context_evidence"]
        )
        assert all(
            e["condition_feature"] != "vol.range_compression_20"
            for e in result.details["directional_evidence"]
        )

    def test_downside_breakout_plus_compression_with_entry_short_accepted(self) -> None:
        spec = materialize_family_spec("breakout", seed=1)
        cand = _breakout_cand(
            entry_op=OperatorId.ENTRY_SHORT,
            conditions=[
                (
                    "price.breakdown_distance_20",
                    ValueType.RATIO,
                    OperatorId.LESS_THAN,
                    -0.02,
                ),
                (
                    "vol.prior_range_compression_20",
                    ValueType.RATIO,
                    OperatorId.LESS_THAN,
                    0.5,
                ),
            ],
        )
        result = validate_family_direction_coherence(cand, spec)
        assert result.coherent
        assert result.details["expected_direction"] == "ENTRY_SHORT"
        assert result.details["entry_direction"] == "ENTRY_SHORT"
        assert any(
            e["condition_feature"] == "vol.prior_range_compression_20"
            for e in result.details["context_evidence"]
        )

    def test_upside_breakout_with_entry_short_rejected(self) -> None:
        spec = materialize_family_spec("breakout", seed=1)
        cand = _breakout_cand(
            entry_op=OperatorId.ENTRY_SHORT,
            conditions=[
                (
                    "price.breakout_distance_20",
                    ValueType.RATIO,
                    OperatorId.GREATER_THAN,
                    0.02,
                ),
            ],
        )
        result = validate_family_direction_coherence(cand, spec)
        assert not result.coherent
        assert result.rejection_reason == FAMILY_DIRECTION_INCOHERENT
        assert result.details["expected_direction"] == "ENTRY_LONG"
        assert result.details["entry_direction"] == "ENTRY_SHORT"

    def test_downside_breakout_with_entry_long_rejected(self) -> None:
        spec = materialize_family_spec("breakout", seed=1)
        cand = _breakout_cand(
            entry_op=OperatorId.ENTRY_LONG,
            conditions=[
                (
                    "price.breakdown_distance_20",
                    ValueType.RATIO,
                    OperatorId.LESS_THAN,
                    -0.02,
                ),
            ],
        )
        result = validate_family_direction_coherence(cand, spec)
        assert not result.coherent
        assert result.rejection_reason == FAMILY_DIRECTION_INCOHERENT
        assert result.details["expected_direction"] == "ENTRY_SHORT"
        assert result.details["entry_direction"] == "ENTRY_LONG"

    def test_compression_only_long_rejected_as_unproven(self) -> None:
        spec = materialize_family_spec("breakout", seed=1)
        cand = _breakout_cand(
            entry_op=OperatorId.ENTRY_LONG,
            conditions=[
                (
                    "vol.prior_range_compression_20",
                    ValueType.RATIO,
                    OperatorId.LESS_THAN,
                    0.5,
                ),
            ],
        )
        result = validate_family_direction_coherence(cand, spec)
        assert not result.coherent
        assert result.rejection_reason == FAMILY_DIRECTION_INCOHERENT
        assert result.details["coherence_reason"] == "no_provable_directional_condition"
        assert result.details["directional_evidence"] == []
        assert len(result.details["context_evidence"]) == 1

    def test_compression_only_short_rejected_as_unproven(self) -> None:
        spec = materialize_family_spec("breakout", seed=1)
        cand = _breakout_cand(
            entry_op=OperatorId.ENTRY_SHORT,
            conditions=[
                (
                    "vol.prior_range_compression_20",
                    ValueType.RATIO,
                    OperatorId.LESS_THAN,
                    0.5,
                ),
            ],
        )
        result = validate_family_direction_coherence(cand, spec)
        assert not result.coherent
        assert result.rejection_reason == FAMILY_DIRECTION_INCOHERENT
        assert result.details["coherence_reason"] == "no_provable_directional_condition"

    def test_volume_only_entry_rejected_as_unproven(self) -> None:
        spec = materialize_family_spec("breakout", seed=1)
        cand = _breakout_cand(
            entry_op=OperatorId.ENTRY_LONG,
            conditions=[
                (
                    "liq.volume_pct_20",
                    ValueType.RANK,
                    OperatorId.GREATER_THAN,
                    0.7,
                ),
            ],
        )
        result = validate_family_direction_coherence(cand, spec)
        assert not result.coherent
        assert result.rejection_reason == FAMILY_DIRECTION_INCOHERENT
        assert result.details["coherence_reason"] == "no_provable_directional_condition"
        assert result.details["directional_evidence"] == []
        assert any(
            e["condition_feature"] == "liq.volume_pct_20"
            for e in result.details["context_evidence"]
        )

    def test_downside_breakout_not_rejected_merely_for_compression_context(self) -> None:
        """Prior compression context must not cast a LONG vote against SHORT breakdown."""
        spec = materialize_family_spec("breakout", seed=1)
        cand = _breakout_cand(
            entry_op=OperatorId.ENTRY_SHORT,
            conditions=[
                (
                    "price.breakdown_distance_20",
                    ValueType.RATIO,
                    OperatorId.LESS_THAN,
                    -0.02,
                ),
                (
                    "vol.prior_range_compression_20",
                    ValueType.RATIO,
                    OperatorId.LESS_THAN,
                    0.5,
                ),
            ],
        )
        result = validate_family_direction_coherence(cand, spec)
        assert result.coherent
        assert result.details["expected_direction"] == "ENTRY_SHORT"
        assert result.details["coherence_reason"] != "conflicting_directional_conditions"
        assert all(
            e.get("expected_direction") is None or "expected_direction" not in e
            or e["condition_feature"] != "vol.prior_range_compression_20"
            for e in result.details["directional_evidence"]
        )

    def test_negative_breakout_distance_does_not_prove_short(self) -> None:
        """Negative distance from prior maximum is not a downside-breakout proof."""
        spec = materialize_family_spec("breakout", seed=1)
        cand = _breakout_cand(
            entry_op=OperatorId.ENTRY_SHORT,
            conditions=[
                (
                    "price.breakout_distance_20",
                    ValueType.RATIO,
                    OperatorId.LESS_THAN,
                    -0.02,
                ),
            ],
        )
        result = validate_family_direction_coherence(cand, spec)
        assert not result.coherent
        assert result.rejection_reason == FAMILY_DIRECTION_INCOHERENT
        assert result.details["coherence_reason"] == "breakout_distance_does_not_prove_short"

    def test_conflicting_genuine_directional_conditions_rejected(self) -> None:
        spec = materialize_family_spec("breakout", seed=1)
        cand = _breakout_cand(
            entry_op=OperatorId.ENTRY_LONG,
            conditions=[
                (
                    "price.breakout_distance_20",
                    ValueType.RATIO,
                    OperatorId.GREATER_THAN,
                    0.02,
                ),
                (
                    "price.return_5",
                    ValueType.RETURN,
                    OperatorId.LESS_THAN,
                    -0.02,
                ),
            ],
        )
        result = validate_family_direction_coherence(cand, spec)
        assert not result.coherent
        assert result.rejection_reason == FAMILY_DIRECTION_INCOHERENT
        assert result.details["coherence_reason"] == "conflicting_directional_conditions"
        assert len(result.details["directional_evidence"]) == 2
        assert result.details["context_evidence"] == []
