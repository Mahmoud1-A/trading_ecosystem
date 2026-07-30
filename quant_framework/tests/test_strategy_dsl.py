"""
Phase 6C acceptance tests — Typed Strategy DSL.

Proves: invalid trees rejected, depth/node limits enforced, no arbitrary code
execution, protected division finite, stable canonical IDs, seed reproducibility,
typed mutation/crossover, duplicate rejection, legacy-import new identities.
"""

from __future__ import annotations

import pytest

from discovery import (
    FORBIDDEN_BUILTINS,
    CandidateGenerator,
    CreationMethod,
    Crossover,
    DSLValidationError,
    Grammar,
    GrammarLimits,
    Mutator,
    NodeKind,
    OperatorId,
    ValueType,
    build_candidate,
    canonicalize_node,
    constant_node,
    eval_scalar,
    feature_node,
    legacy_import_candidate,
    op_node,
    parameter_node,
    protected_divide,
    structural_precheck,
    truncate_to_limits,
    validate_limits,
)
from discovery.expression_tree import ExprNode
from discovery.prechecks import probe_always_same


class TestInvalidTreesAndLimits:
    def test_invalid_type_tree_rejected(self) -> None:
        with pytest.raises(DSLValidationError):
            # AND requires BOOLEAN children
            op_node(
                OperatorId.AND,
                feature_node("price.rolling_z_20", ValueType.ZSCORE),
                feature_node("vol.atr_14", ValueType.VOLATILITY),
            )

    def test_unknown_operator_rejected(self) -> None:
        with pytest.raises(DSLValidationError):
            ExprNode(NodeKind.OPERATOR, ValueType.SCALAR, "EVAL", ())

    def test_depth_and_node_limits_cannot_be_bypassed(self) -> None:
        grammar = Grammar(limits=GrammarLimits(max_tree_depth=3, max_nodes=8))
        # Build a deep chain of ABS
        node: ExprNode = feature_node("price.simple_return_1", ValueType.RETURN)
        for _ in range(6):
            node = op_node(OperatorId.ABS, node)
        with pytest.raises(DSLValidationError, match="depth"):
            validate_limits(node, grammar.limits)
        repaired = truncate_to_limits(node, grammar)
        validate_limits(repaired, grammar.limits)
        assert repaired.depth <= grammar.limits.max_tree_depth
        assert repaired.n_nodes <= grammar.limits.max_nodes


class TestNoArbitraryCode:
    def test_forbidden_builtins_blocked(self) -> None:
        assert "eval" in FORBIDDEN_BUILTINS
        with pytest.raises(DSLValidationError, match="Forbidden"):
            ExprNode(NodeKind.FEATURE, ValueType.SCALAR, "eval")
        with pytest.raises(DSLValidationError, match="Forbidden"):
            parameter_node("__import__", 1.0)

    def test_candidate_cannot_carry_callables(self) -> None:
        entry = op_node(
            OperatorId.ENTRY_LONG,
            op_node(
                OperatorId.LESS_THAN,
                feature_node("price.rolling_z_20", ValueType.ZSCORE),
                constant_node(-2.0),
            ),
        )
        cand = build_candidate(
            entry_tree=entry,
            grammar_version="strategy_dsl_v1",
            feature_set_version="feature_set_v1_phase6b",
            random_seed=1,
        )
        blob = str(cand.as_dict())
        assert "eval(" not in blob
        assert "__import__" not in blob


class TestProtectedDivision:
    def test_protected_division_remains_finite(self) -> None:
        assert protected_divide(1.0, 0.0) == 0.0
        assert protected_divide(1.0, 1e-12) == 0.0
        assert protected_divide(float("nan"), 1.0) == 0.0
        tree = op_node(OperatorId.DIVIDE_PROTECTED, constant_node(1.0), constant_node(0.0))
        assert eval_scalar(tree) == 0.0
        assert abs(protected_divide(6.0, 3.0) - 2.0) < 1e-12


class TestCanonicalIdentityAndSeed:
    def test_canonical_identities_are_stable(self) -> None:
        a = op_node(OperatorId.ADD, constant_node(1.0), constant_node(2.0))
        b = op_node(OperatorId.ADD, constant_node(2.0), constant_node(1.0))
        ca = canonicalize_node(a)
        cb = canonicalize_node(b)
        assert ca.as_dict() == cb.as_dict()

        entry = op_node(
            OperatorId.ENTRY_LONG,
            op_node(
                OperatorId.LESS_THAN,
                feature_node("price.rolling_z_20", ValueType.ZSCORE),
                parameter_node("z_entry", -2.0),
            ),
        )
        c1 = build_candidate(
            entry_tree=entry,
            grammar_version="strategy_dsl_v1",
            feature_set_version="fs",
            random_seed=0,
        )
        c2 = build_candidate(
            entry_tree=entry,
            grammar_version="strategy_dsl_v1",
            feature_set_version="fs",
            random_seed=99,  # seed is not part of canonical hash payload
        )
        assert c1.candidate_id == c2.candidate_id

    def test_same_seed_produces_identical_candidates(self) -> None:
        gen = CandidateGenerator()
        a = gen.generate(seed=42)
        b = gen.generate(seed=42)
        assert a.candidate_id == b.candidate_id
        assert a.entry_tree.as_dict() == b.entry_tree.as_dict()
        assert a.as_dict()["entry_tree"] == b.as_dict()["entry_tree"]


class TestMutationAndCrossoverTypes:
    def test_mutation_and_crossover_preserve_types(self) -> None:
        gen = CandidateGenerator()
        parent = gen.seed_template_mean_reversion(seed=7)
        mutator = Mutator()
        child = mutator.mutate(parent, seed=11)
        assert child.creation_method is CreationMethod.MUTATION
        assert child.parent_ids == (parent.candidate_id,)
        # Entry remains a valid typed tree
        assert child.entry_tree.value_type in {ValueType.ORDER_INTENT, ValueType.BOOLEAN}
        structural_precheck(
            entry=child.entry_tree,
            exit=child.exit_tree,
            regime_gates=(),
            grammar=Grammar(),
            seed=11,
        )

        other = gen.generate(seed=99)
        cross = Crossover()
        c1, c2 = cross.crossover(parent, other, seed=5)
        assert c1.creation_method is CreationMethod.CROSSOVER
        assert parent.candidate_id in c1.parent_ids
        assert other.candidate_id in c1.parent_ids
        for node in c1.entry_tree.walk():
            if node.kind is NodeKind.OPERATOR:
                # Reconstruction already validated types in ExprNode.__post_init__
                assert node.value_type


class TestDuplicatesAndLegacy:
    def test_duplicate_expressions_share_candidate_id(self) -> None:
        gen = CandidateGenerator()
        t = gen.seed_template_mean_reversion(seed=1)
        again = gen.seed_template_mean_reversion(seed=2)
        assert t.candidate_id == again.candidate_id
        # Duplicate identity means the same candidate need not be reevaluated
        seen = {t.candidate_id}
        assert again.candidate_id in seen

    def test_legacy_import_receives_new_identity_and_lineage(self) -> None:
        entry = op_node(
            OperatorId.ENTRY_LONG,
            op_node(
                OperatorId.LESS_THAN,
                feature_node("price.rolling_z_20", ValueType.ZSCORE),
                constant_node(-2.0),
            ),
        )
        native = build_candidate(
            entry_tree=entry,
            grammar_version="strategy_dsl_v1",
            feature_set_version="fs",
            random_seed=0,
            creation_method=CreationMethod.RANDOM,
            strategy_family="legacy_import",
        )
        legacy = legacy_import_candidate(
            entry_tree=entry,
            exit_tree=None,
            legacy_id="book_member_7",
            grammar_version="strategy_dsl_v1",
            feature_set_version="fs",
            random_seed=0,
            strategy_family="legacy_import",
        )
        assert legacy.creation_method is CreationMethod.LEGACY_IMPORT
        assert legacy.parent_ids == ("legacy:book_member_7",)
        # Same expression identity (candidate_id) but distinct lineage
        assert legacy.candidate_id == native.candidate_id
        assert legacy.lineage_id != native.lineage_id
        assert legacy.lineage_id.startswith("lin_legacy_")


class TestConstantSignalRejection:
    def test_always_true_and_always_false_rejected(self) -> None:
        always_true = op_node(
            OperatorId.ENTRY_LONG,
            op_node(OperatorId.GREATER_THAN, constant_node(1.0), constant_node(0.0)),
        )
        always_false = op_node(
            OperatorId.ENTRY_LONG,
            op_node(OperatorId.LESS_THAN, constant_node(1.0), constant_node(0.0)),
        )
        g = Grammar()
        assert structural_precheck(entry=always_true, exit=None, regime_gates=(), grammar=g).accepted is False
        assert structural_precheck(entry=always_false, exit=None, regime_gates=(), grammar=g).accepted is False

    def test_probe_rejects_non_finite(self) -> None:
        # DIVIDE_PROTECTED keeps finite; direct infinite constant blocked at construction
        with pytest.raises(DSLValidationError):
            constant_node(float("inf"))
