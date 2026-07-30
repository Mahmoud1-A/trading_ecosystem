"""Deterministic random generation of typed strategy candidates."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from discovery.candidate import StrategyCandidate, build_candidate
from discovery.expression_tree import (
    DSLValidationError,
    ExprNode,
    constant_node,
    feature_node,
    op_node,
    parameter_node,
)
from discovery.feature_domains import (
    INVALID_FEATURE_THRESHOLD_DOMAIN,
    sample_threshold_for_leaf,
    validate_tree_threshold_domains,
)
from discovery.grammar import GRAMMAR_VERSION, Grammar
from discovery.operators import OPERATOR_REGISTRY, OperatorId
from discovery.prechecks import structural_precheck
from discovery.regime_gates import compile_regime_constraints
from discovery.repair import clamp_lookbacks, repair_entry
from discovery.typecheck import check_ast_types, check_strategy_trees, parse_dsl_validation_error
from discovery.types import CreationMethod, NUMERIC_TYPES, ValueType

if TYPE_CHECKING:
    from discovery.family_spec import FamilySpec


class CandidateGenerator:
    def __init__(
        self,
        grammar: Grammar | None = None,
        *,
        feature_set_version: str = "feature_set_v1_phase6b",
        cost_model_version: str = "cost_v1",
        strategy_family: str = "dsl_generated",
        family_spec: FamilySpec | None = None,
    ) -> None:
        self.family_spec = family_spec
        if family_spec is not None:
            self.grammar = family_spec.to_grammar()
            self.strategy_family = family_spec.family_id
            if self.strategy_family in {"", "dsl_generated"}:
                raise ValueError("FamilySpec must not resolve to generic dsl_generated")
        else:
            self.grammar = grammar or Grammar()
            self.strategy_family = strategy_family
        self.feature_set_version = feature_set_version
        self.cost_model_version = cost_model_version

    @classmethod
    def from_family_spec(
        cls,
        family_spec: FamilySpec,
        *,
        feature_set_version: str = "feature_set_v1_phase6b",
        cost_model_version: str = "cost_v1",
    ) -> CandidateGenerator:
        return cls(
            family_spec=family_spec,
            feature_set_version=feature_set_version,
            cost_model_version=cost_model_version,
        )

    def _family_provenance(self) -> dict[str, Any]:
        if self.family_spec is None:
            return {}
        return {
            "family_id": self.family_spec.family_id,
            "family_hash": self.family_spec.canonical_hash(),
            "grammar_fingerprint": self.family_spec.effective_grammar_fingerprint(),
            "hypothesis": self.family_spec.hypothesis,
            "entry_patterns": list(self.family_spec.entry_patterns),
            "exit_patterns": list(self.family_spec.exit_patterns),
            "allowed_features": list(self.family_spec.allowed_features),
            "allowed_operators": list(self.family_spec.allowed_operators),
            "regime_constraints": list(self.family_spec.regime_constraints),
            "direction_support": ["ENTRY_LONG", "ENTRY_SHORT"],
            **dict(self.family_spec.provenance),
        }

    def _rng(self, seed: int) -> np.random.Generator:
        return np.random.default_rng(seed)

    def _features_of_type(self, value_type: ValueType) -> list:
        return [l for l in self.grammar.feature_leaves if l.value_type is value_type]

    def _pick_feature(self, rng: np.random.Generator, *preferred_ids: str) -> ExprNode:
        by_id = {l.feature_id: l for l in self.grammar.feature_leaves}
        for fid in preferred_ids:
            if fid in by_id:
                leaf = by_id[fid]
                return feature_node(leaf.feature_id, leaf.value_type)
        leaves = list(self.grammar.feature_leaves)
        leaf = leaves[int(rng.integers(0, len(leaves)))]
        return feature_node(leaf.feature_id, leaf.value_type)

    def _sample_param(
        self,
        rng: np.random.Generator,
        name: str,
        default_lo: float,
        default_hi: float,
    ) -> ExprNode:
        ranges = dict(self.family_spec.parameter_ranges) if self.family_spec else {}
        lo, hi = ranges.get(name, (default_lo, default_hi))
        val = float(rng.uniform(float(lo), float(hi)))
        return parameter_node(name, val)

    def _random_feature(self, rng: np.random.Generator, *, value_type: ValueType | None = None) -> ExprNode:
        leaves = list(self.grammar.feature_leaves)
        if value_type is not None:
            typed = self._features_of_type(value_type)
            if not typed:
                raise ValueError(f"no feature leaves of type {value_type}")
            leaves = typed
        leaf = leaves[int(rng.integers(0, len(leaves)))]
        return feature_node(leaf.feature_id, leaf.value_type)

    def _random_const(self, rng: np.random.Generator, leaf=None) -> ExprNode:
        if leaf is not None:
            return constant_node(sample_threshold_for_leaf(leaf, rng))
        # Untyped fallback: mild ratio-scale noise (not N(0,1) for arbitrary features).
        return constant_node(float(rng.uniform(-0.05, 0.05)))

    def _random_param(self, rng: np.random.Generator, idx: int, leaf=None) -> ExprNode:
        if leaf is not None:
            return parameter_node(f"p{idx}", sample_threshold_for_leaf(leaf, rng))
        return parameter_node(f"p{idx}", float(rng.uniform(-2, 2)))

    def _allowed_ops(self, target_type: ValueType) -> list[OperatorId]:
        return list(self.grammar.operators_returning(target_type))

    def _boolean_leaf(self, rng: np.random.Generator, param_counter: list[int]) -> ExprNode:
        numeric_leaves = [l for l in self.grammar.feature_leaves if l.value_type in NUMERIC_TYPES]
        if not numeric_leaves:
            numeric_leaves = list(self.grammar.feature_leaves)
        leaf = numeric_leaves[int(rng.integers(0, len(numeric_leaves)))]
        left = feature_node(leaf.feature_id, leaf.value_type)
        right = (
            self._random_const(rng, leaf)
            if rng.random() < 0.7
            else self._random_param(rng, param_counter[0], leaf)
        )
        if right.kind.value == "PARAMETER":
            param_counter[0] += 1
        cmp_ops = [
            oid
            for oid in (
                OperatorId.LESS_THAN,
                OperatorId.GREATER_THAN,
                OperatorId.LESS_EQUAL,
                OperatorId.GREATER_EQUAL,
            )
            if self.grammar.allows_operator(oid)
        ]
        if not cmp_ops:
            cmp_ops = [OperatorId.LESS_THAN, OperatorId.GREATER_THAN]
        return op_node(cmp_ops[int(rng.integers(0, len(cmp_ops)))], left, right)

    def _leaf(
        self,
        rng: np.random.Generator,
        *,
        target_type: ValueType,
        param_counter: list[int],
    ) -> ExprNode:
        """Return a node whose value_type exactly matches target_type."""
        if target_type is ValueType.BOOLEAN:
            return self._boolean_leaf(rng, param_counter)
        if target_type is ValueType.ORDER_INTENT:
            cond = self._leaf(rng, target_type=ValueType.BOOLEAN, param_counter=param_counter)
            return op_node(OperatorId.ENTRY_LONG, cond)
        if target_type is ValueType.REGIME:
            typed = self._features_of_type(ValueType.REGIME)
            if typed:
                return self._random_feature(rng, value_type=ValueType.REGIME)
            return constant_node(float(int(rng.integers(0, 3))), ValueType.REGIME)
        if target_type is ValueType.SCALAR:
            if self._features_of_type(ValueType.SCALAR) and rng.random() < 0.4:
                return self._random_feature(rng, value_type=ValueType.SCALAR)
            if rng.random() < 0.5:
                return self._random_const(rng)
            node = self._random_param(rng, param_counter[0])
            param_counter[0] += 1
            return node
        typed = self._features_of_type(target_type)
        if typed:
            return self._random_feature(rng, value_type=target_type)
        numeric_leaves = [l for l in self.grammar.feature_leaves if l.value_type in NUMERIC_TYPES]

        def _num_feat() -> ExprNode:
            leaf = numeric_leaves[int(rng.integers(0, len(numeric_leaves)))]
            return feature_node(leaf.feature_id, leaf.value_type)

        if target_type is ValueType.RATIO:
            return op_node(OperatorId.DIVIDE_PROTECTED, _num_feat(), self._random_const(rng))
        if target_type is ValueType.VOLATILITY:
            return op_node(OperatorId.ROLLING_STD, _num_feat(), constant_node(20.0))
        if target_type is ValueType.RETURN:
            return op_node(OperatorId.PERCENT_CHANGE, _num_feat())
        if target_type is ValueType.RANK:
            return op_node(OperatorId.ROLLING_RANK, _num_feat(), constant_node(20.0))
        if target_type is ValueType.ZSCORE:
            return op_node(OperatorId.ROLLING_ZSCORE, _num_feat(), constant_node(20.0))
        raise DSLValidationError(f"cannot synthesize leaf for type {target_type.value}")

    def _child_for_slot(
        self,
        rng: np.random.Generator,
        *,
        allowed: frozenset[ValueType],
        max_depth: int,
        param_counter: list[int],
        lookback: bool,
    ) -> ExprNode:
        if lookback:
            lb = int(rng.integers(2, min(self.grammar.limits.max_rolling_lookback, 40) + 1))
            return constant_node(float(lb))
        ordered = sorted(
            allowed,
            key=lambda t: (
                0 if t in {ValueType.BOOLEAN, ValueType.REGIME, ValueType.SCALAR} else 1,
                t.value,
            ),
        )
        last_err: Exception | None = None
        for _ in range(max(3, len(ordered) * 2)):
            child_type = ordered[int(rng.integers(0, len(ordered)))]
            try:
                if max_depth <= 1 or rng.random() < 0.35:
                    child = self._leaf(rng, target_type=child_type, param_counter=param_counter)
                else:
                    child = self._grow(
                        rng,
                        target_type=child_type,
                        max_depth=max_depth - 1,
                        param_counter=param_counter,
                    )
                if child.value_type in allowed:
                    return child
                for alt in ordered:
                    fixed = self._leaf(rng, target_type=alt, param_counter=param_counter)
                    if fixed.value_type in allowed:
                        return fixed
            except DSLValidationError as exc:
                last_err = exc
                continue
        if last_err is not None:
            raise last_err
        raise DSLValidationError(
            f"could not build child for allowed types {[t.value for t in allowed]}"
        )

    def _grow(
        self,
        rng: np.random.Generator,
        *,
        target_type: ValueType,
        max_depth: int,
        param_counter: list[int],
    ) -> ExprNode:
        if max_depth <= 1 or rng.random() < 0.35:
            return self._leaf(rng, target_type=target_type, param_counter=param_counter)

        candidates = self._allowed_ops(target_type)
        if not candidates:
            return self._leaf(rng, target_type=target_type, param_counter=param_counter)

        order = list(candidates)
        rng.shuffle(order)
        for op in order[: min(8, len(order))]:
            spec = OPERATOR_REGISTRY[op]
            try:
                children = [
                    self._child_for_slot(
                        rng,
                        allowed=allowed,
                        max_depth=max_depth,
                        param_counter=param_counter,
                        lookback=spec.lookback_child == i,
                    )
                    for i, allowed in enumerate(spec.input_types)
                ]
                node = op_node(op, *children)
                check_ast_types(node, path="grow", operation="generate")
                return node
            except DSLValidationError:
                continue
        return self._leaf(rng, target_type=target_type, param_counter=param_counter)

    def _pattern_entry_cond(self, pattern: str, rng: np.random.Generator) -> ExprNode:
        """Build a BOOLEAN condition shaped by the family entry pattern."""
        if pattern in {"zscore_extreme", "vwap_zscore"}:
            feat = self._pick_feature(rng, "price.rolling_z_20", "liq.dist_session_vwap")
            thr = self._sample_param(rng, "entry_threshold", -2.5, -1.0)
            return op_node(OperatorId.LESS_THAN, feat, thr)
        if pattern == "pullback_to_ema":
            feat = self._pick_feature(rng, "price.dist_rolling_mean_20", "liq.dist_session_vwap")
            thr = self._sample_param(rng, "entry_threshold", -0.04, -0.002)
            return op_node(OperatorId.LESS_THAN, feat, thr)
        if pattern in {"dist_mean_threshold", "vwap_deviation"}:
            feat = self._pick_feature(
                rng, "price.dist_rolling_mean_20", "liq.dist_session_vwap", "price.rolling_z_20"
            )
            if feat.name == "price.rolling_z_20":
                thr = self._sample_param(rng, "entry_threshold", -2.5, -1.0)
            else:
                thr = self._sample_param(rng, "entry_threshold", -0.04, -0.002)
            return op_node(OperatorId.LESS_THAN, feat, thr)
        if pattern in {"return_persistence", "cross_momentum", "trend_resume_cross"}:
            feat = self._pick_feature(rng, "price.return_5", "price.simple_return_1")
            thr = self._sample_param(rng, "entry_threshold", 0.0, 0.02)
            if self.grammar.allows_operator(OperatorId.CROSS_ABOVE) and rng.random() < 0.5:
                base = self._pick_feature(rng, "price.simple_return_1", "price.log_return_1")
                return op_node(OperatorId.CROSS_ABOVE, feat, base)
            return op_node(OperatorId.GREATER_THAN, feat, thr)
        if pattern in {"breakout_distance", "compression_release"}:
            feat = self._pick_feature(
                rng, "price.breakout_distance_20", "vol.range_compression_20", "liq.volume_pct_20"
            )
            thr = self._sample_param(rng, "entry_threshold", 0.0, 0.05)
            return op_node(OperatorId.GREATER_THAN, feat, thr)
        if pattern in {"vol_expansion_break", "compression_then_move"}:
            feat = self._pick_feature(
                rng, "vol.range_compression_20", "vol.realized_20", "vol.norm_atr_14"
            )
            vol_thr = self._sample_param(rng, "vol_threshold", 0.15, 0.85)
            move = self._pick_feature(rng, "price.simple_return_1", "price.return_5")
            ret_thr = self._sample_param(rng, "directional_return_threshold", 0.0002, 0.008)
            left = op_node(OperatorId.GREATER_THAN, feat, vol_thr)
            # Directional confirmation uses return-scale threshold (not exit_vol).
            right = op_node(OperatorId.GREATER_THAN, op_node(OperatorId.ABS, move), ret_thr)
            if self.grammar.allows_operator(OperatorId.AND) and rng.random() < 0.7:
                return op_node(OperatorId.AND, left, right)
            return left
        if pattern in {"gap_fade_entry", "vwap_gap_reversion"}:
            # Normalized overnight gap (instrument-independent), not one-bar return.
            gap = self._pick_feature(rng, "price.close_to_open")
            thr = self._sample_param(rng, "entry_threshold", 0.0005, 0.012)
            core = op_node(OperatorId.GREATER_THAN, op_node(OperatorId.ABS, gap), thr)
            if pattern == "vwap_gap_reversion" and self.grammar.allows_operator(OperatorId.OR):
                vwap = self._pick_feature(rng, "liq.dist_session_vwap")
                vwap_cond = op_node(
                    OperatorId.GREATER_THAN,
                    op_node(OperatorId.ABS, vwap),
                    self._sample_param(rng, "exit_threshold", 0.001, 0.01),
                )
                # OR keeps the gate economically meaningful without always-false AND.
                core = op_node(OperatorId.OR, core, vwap_cond)
            elif self.grammar.allows_operator(OperatorId.AND) and rng.random() < 0.5:
                minutes = self._pick_feature(rng, "temp.minutes_since_open")
                tmax = self._sample_param(rng, "minutes_open_max", 90.0, 390.0)
                window = op_node(OperatorId.LESS_THAN, minutes, tmax)
                core = op_node(OperatorId.AND, core, window)
            return core
        if pattern in {"session_extreme", "open_fade"}:
            feat = self._pick_feature(
                rng, "price.rolling_z_20", "liq.dist_session_vwap", "price.simple_return_1"
            )
            thr = self._sample_param(rng, "entry_threshold", -2.5, -1.0)
            core = op_node(OperatorId.LESS_THAN, feat, thr)
            if self.grammar.allows_operator(OperatorId.AND) and rng.random() < 0.4:
                minutes = self._pick_feature(rng, "temp.minutes_since_open")
                tmax = self._sample_param(rng, "minutes_open_max", 60.0, 390.0)
                window = op_node(OperatorId.LESS_THAN, minutes, tmax)
                return op_node(OperatorId.AND, core, window)
            return core
        # Fallback: family-constrained boolean leaf
        return self._boolean_leaf(rng, [0])

    def _pattern_exit_tree(self, pattern: str, rng: np.random.Generator) -> ExprNode:
        if pattern in {"session_flatten"} and self.grammar.allows_operator(OperatorId.SESSION_EXIT):
            minutes = self._pick_feature(rng, "temp.minutes_since_open")
            cond = op_node(
                OperatorId.GREATER_THAN,
                minutes,
                self._sample_param(rng, "minutes_open_max", 60.0, 240.0),
            )
            return op_node(OperatorId.SESSION_EXIT, cond)
        if pattern in {"time_decay_exit"} and self.grammar.allows_operator(OperatorId.TIME_EXIT):
            return op_node(
                OperatorId.TIME_EXIT,
                self._sample_param(rng, "lookback", 5.0, 40.0),
            )
        feat = self._pick_feature(
            rng,
            "price.rolling_z_20",
            "liq.dist_session_vwap",
            "price.dist_rolling_mean_20",
            "price.return_5",
            "vol.realized_20",
        )
        thr = self._sample_param(rng, "exit_threshold", -0.5, 0.5)
        if pattern in {"vol_mean_revert", "realized_collapse"}:
            feat = self._pick_feature(rng, "vol.range_compression_20", "vol.realized_20")
            thr = self._sample_param(rng, "exit_vol_threshold", 0.2, 0.9)
            cond = op_node(OperatorId.LESS_THAN, feat, thr)
            return op_node(OperatorId.EXIT_SIGNAL, cond)
        if pattern == "gap_filled":
            feat = self._pick_feature(rng, "price.close_to_open", "liq.dist_session_vwap")
            thr = self._sample_param(rng, "exit_threshold", 0.0, 0.005)
            cond = op_node(OperatorId.LESS_THAN, op_node(OperatorId.ABS, feat), thr)
            return op_node(OperatorId.EXIT_SIGNAL, cond)
        if pattern in {"zscore_normalize", "mean_reentry", "vwap_reentry", "vwap_touch"}:
            cond = op_node(OperatorId.GREATER_THAN, feat, thr)
        elif pattern in {"momentum_fade", "trend_exhaustion", "ema_loss", "breakout_failure"}:
            cond = op_node(OperatorId.LESS_THAN, feat, thr)
        else:
            cond = op_node(OperatorId.GREATER_THAN, feat, thr)
        return op_node(OperatorId.EXIT_SIGNAL, cond)

    def generate_entry(self, seed: int) -> ExprNode:
        rng = self._rng(seed)
        prefer_short = bool(rng.random() < 0.45) and self.grammar.allows_operator(OperatorId.ENTRY_SHORT)
        entry_op = OperatorId.ENTRY_SHORT if prefer_short else OperatorId.ENTRY_LONG
        if self.family_spec is not None and self.family_spec.entry_patterns:
            pattern = self.family_spec.entry_patterns[
                int(rng.integers(0, len(self.family_spec.entry_patterns)))
            ]
            cond = self._pattern_entry_cond(pattern, rng)
            # Gap fade: fade direction opposite the gap sign → short after up-gap.
            if pattern in {"gap_fade_entry", "vwap_gap_reversion"} and self.grammar.allows_operator(
                OperatorId.ENTRY_SHORT
            ):
                entry_op = OperatorId.ENTRY_SHORT if rng.random() < 0.5 else OperatorId.ENTRY_LONG
            entry = op_node(entry_op, cond)
            entry = repair_entry(clamp_lookbacks(entry, self.grammar), self.grammar)
            check_ast_types(entry, path="entry", operation="generate")
            return entry
        counter = [0]
        raw = self._grow(
            rng,
            target_type=ValueType.BOOLEAN,
            max_depth=min(4, self.grammar.limits.max_tree_depth),
            param_counter=counter,
        )
        entry = repair_entry(clamp_lookbacks(raw, self.grammar), self.grammar)
        if entry.name == OperatorId.ENTRY_LONG.value and prefer_short:
            entry = op_node(OperatorId.ENTRY_SHORT, entry.children[0]) if entry.children else entry
        elif entry.value_type is ValueType.BOOLEAN:
            entry = op_node(entry_op, entry)
        check_ast_types(entry, path="entry", operation="generate")
        return entry

    def generate_exit(self, seed: int) -> ExprNode:
        rng = self._rng(seed + 17)
        if self.family_spec is not None and self.family_spec.exit_patterns:
            pattern = self.family_spec.exit_patterns[
                int(rng.integers(0, len(self.family_spec.exit_patterns)))
            ]
            exit_tree = self._pattern_exit_tree(pattern, rng)
            check_ast_types(exit_tree, path="exit", operation="generate")
            return exit_tree
        numeric_leaves = [l for l in self.grammar.feature_leaves if l.value_type in NUMERIC_TYPES]
        leaf = numeric_leaves[int(rng.integers(0, len(numeric_leaves)))]
        feature = feature_node(leaf.feature_id, leaf.value_type)
        thr = self._random_const(rng)
        cond = op_node(OperatorId.GREATER_THAN, feature, thr)
        exit_tree = op_node(OperatorId.EXIT_SIGNAL, cond)
        check_ast_types(exit_tree, path="exit", operation="generate")
        return exit_tree

    def generate(
        self,
        seed: int,
        *,
        with_exit: bool = True,
        with_stop: bool = True,
        max_attempts: int = 32,
    ) -> StrategyCandidate:
        if self.strategy_family == "dsl_generated" and self.family_spec is not None:
            raise RuntimeError("family-bound generator must not emit dsl_generated")
        last_reason = "unknown"
        last_exc: Exception | None = None
        for attempt in range(max_attempts):
            s = seed + attempt * 997
            try:
                entry = self.generate_entry(s)
                exit_tree = self.generate_exit(s) if with_exit else None
                stop = None
                target = None
                if with_stop:
                    atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
                    if any(l.feature_id == "vol.atr_14" for l in self.grammar.feature_leaves):
                        stop_mult = self._sample_param(
                            self._rng(s + 3), "atr_stop_mult", 1.0, 2.5
                        )
                        target_mult = self._sample_param(
                            self._rng(s + 5), "atr_target_mult", 1.5, 4.0
                        )
                        stop = op_node(OperatorId.ATR_STOP, atr, stop_mult)
                        target = op_node(OperatorId.ATR_TARGET, atr, target_mult)
                regime_gates: tuple = ()
                if self.family_spec is not None and self.family_spec.regime_constraints:
                    regime_gates = compile_regime_constraints(self.family_spec.regime_constraints)
                domain_reason = validate_tree_threshold_domains(entry, self.grammar)
                if domain_reason is None and exit_tree is not None:
                    domain_reason = validate_tree_threshold_domains(exit_tree, self.grammar)
                if domain_reason is not None:
                    last_reason = domain_reason
                    continue
                check_strategy_trees(
                    entry, exit_tree, stop, target, *regime_gates, operation="generate"
                )
                check = structural_precheck(
                    entry=entry,
                    exit=exit_tree,
                    regime_gates=regime_gates,
                    grammar=self.grammar,
                    seed=s,
                )
                if not check.accepted:
                    last_reason = check.reason
                    continue
                direction = (
                    "ENTRY_SHORT"
                    if entry.name == OperatorId.ENTRY_SHORT.value
                    else "ENTRY_LONG"
                )
                prov = {
                    **self._family_provenance(),
                    "direction": direction,
                    "regime_gates_compiled": [c for c in (self.family_spec.regime_constraints if self.family_spec else ())],
                }
                cand = build_candidate(
                    entry_tree=entry,
                    exit_tree=exit_tree,
                    stop=stop,
                    target=target,
                    regime_gates=regime_gates,
                    strategy_family=self.strategy_family,
                    creation_method=CreationMethod.RANDOM,
                    grammar_version=self.grammar.version,
                    feature_set_version=self.feature_set_version,
                    cost_model_version=self.cost_model_version,
                    random_seed=s,
                    family_provenance=prov,
                )
                check_strategy_trees(
                    cand.entry_tree,
                    cand.exit_tree,
                    cand.stop,
                    cand.target,
                    *cand.regime_gates,
                    operation="generate",
                    candidate_id=cand.candidate_id,
                )
                return cand
            except DSLValidationError as exc:
                last_exc = parse_dsl_validation_error(exc, operation="generate")
                last_reason = str(last_exc)
                continue
        if last_exc is not None:
            raise parse_dsl_validation_error(last_exc, operation="generate")
        raise RuntimeError(f"Failed to generate valid candidate: {last_reason}")

    def seed_template_mean_reversion(self, seed: int = 0) -> StrategyCandidate:
        z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
        entry_cond = op_node(OperatorId.LESS_THAN, z, parameter_node("z_entry", -2.0))
        entry = op_node(OperatorId.ENTRY_LONG, entry_cond)
        exit_cond = op_node(OperatorId.GREATER_THAN, z, parameter_node("z_exit", -0.25))
        exit_tree = op_node(OperatorId.EXIT_SIGNAL, exit_cond)
        atr = feature_node("vol.atr_14", ValueType.VOLATILITY)
        stop = op_node(OperatorId.ATR_STOP, atr, constant_node(1.5))
        cand = build_candidate(
            entry_tree=entry,
            exit_tree=exit_tree,
            stop=stop,
            strategy_family="mean_reversion_template",
            creation_method=CreationMethod.SEED_TEMPLATE,
            grammar_version=GRAMMAR_VERSION,
            feature_set_version=self.feature_set_version,
            cost_model_version=self.cost_model_version,
            random_seed=seed,
        )
        check_strategy_trees(
            cand.entry_tree,
            cand.exit_tree,
            cand.stop,
            operation="seed_template",
            candidate_id=cand.candidate_id,
        )
        return cand
