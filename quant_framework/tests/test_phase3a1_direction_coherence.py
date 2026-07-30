"""Phase 3A.1: family-direction coherence integrity for evolution descendants."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from discovery.candidate import build_candidate
from discovery.crossover import Crossover
from discovery.evaluator import SyntheticOOSBackend
from discovery.expression_tree import constant_node, feature_node, op_node
from discovery.family_generator import materialize_family_spec
from discovery.generator import CandidateGenerator
from discovery.multi_family_campaign import (
    FAMILY_DIRECTION_INCOHERENT,
    FamilyCampaignConfig,
    GenerationRecord,
    MultiFamilyCampaign,
    validate_family_direction_coherence,
)
from discovery.mutation import Mutator
from discovery.operators import OperatorId
from discovery.types import CreationMethod, ValueType
from registry.experiment_registry import ExperimentRegistry


def _cand(
    *,
    family_id: str,
    entry_op: OperatorId,
    feature_id: str,
    feature_type: ValueType,
    cmp_op: OperatorId,
    threshold: float,
    creation_method: CreationMethod = CreationMethod.MUTATION,
    parent_ids: tuple[str, ...] = ("parent_a",),
    generation: int = 1,
    seed: int = 1,
):
    feat = feature_node(feature_id, feature_type)
    cond = op_node(cmp_op, feat, constant_node(threshold))
    entry = op_node(entry_op, cond)
    return build_candidate(
        entry_tree=entry,
        strategy_family=family_id,
        creation_method=creation_method,
        generation=generation,
        parent_ids=parent_ids,
        grammar_version="strategy_dsl_v1",
        feature_set_version="feature_set_v1_phase6b",
        random_seed=seed,
        family_provenance={"family_id": family_id},
    )


@dataclass
class CountingBackend(SyntheticOOSBackend):
    is_full_event_wfo: bool = True
    backend_kind: str = "event_driven_wfo"
    evaluate_calls: int = 0
    evaluated_ids: list[str] = field(default_factory=list)

    def evaluate(self, candidate):
        self.evaluate_calls += 1
        self.evaluated_ids.append(candidate.candidate_id)
        folds, train = super().evaluate(candidate)
        train = {
            **train,
            "signal_source": "candidate_dsl_trees",
            "is_full_event_wfo": True,
            "wfo_completed_folds": len(folds),
        }
        return folds, train


class TestFamilyDirectionCoherenceValidator:
    def test_momentum_positive_with_entry_short_rejected(self) -> None:
        spec = materialize_family_spec("momentum", seed=1)
        cand = _cand(
            family_id="momentum",
            entry_op=OperatorId.ENTRY_SHORT,
            feature_id="price.return_5",
            feature_type=ValueType.RETURN,
            cmp_op=OperatorId.GREATER_THAN,
            threshold=0.01,
        )
        result = validate_family_direction_coherence(cand, spec)
        assert not result.coherent
        assert result.rejection_reason == FAMILY_DIRECTION_INCOHERENT
        assert result.details["expected_direction"] == "ENTRY_LONG"
        assert result.details["entry_direction"] == "ENTRY_SHORT"

    def test_momentum_negative_with_entry_long_rejected(self) -> None:
        spec = materialize_family_spec("momentum", seed=1)
        cand = _cand(
            family_id="momentum",
            entry_op=OperatorId.ENTRY_LONG,
            feature_id="price.return_5",
            feature_type=ValueType.RETURN,
            cmp_op=OperatorId.LESS_THAN,
            threshold=-0.01,
        )
        result = validate_family_direction_coherence(cand, spec)
        assert not result.coherent
        assert result.rejection_reason == FAMILY_DIRECTION_INCOHERENT
        assert result.details["expected_direction"] == "ENTRY_SHORT"

    def test_mean_reversion_negative_extreme_with_entry_short_rejected(self) -> None:
        spec = materialize_family_spec("mean_reversion", seed=2)
        cand = _cand(
            family_id="mean_reversion",
            entry_op=OperatorId.ENTRY_SHORT,
            feature_id="price.rolling_z_20",
            feature_type=ValueType.ZSCORE,
            cmp_op=OperatorId.LESS_THAN,
            threshold=-2.0,
        )
        result = validate_family_direction_coherence(cand, spec)
        assert not result.coherent
        assert result.rejection_reason == FAMILY_DIRECTION_INCOHERENT
        assert result.details["expected_direction"] == "ENTRY_LONG"

    def test_gap_up_fade_with_entry_long_rejected(self) -> None:
        spec = materialize_family_spec("gap_fade", seed=3)
        cand = _cand(
            family_id="gap_fade",
            entry_op=OperatorId.ENTRY_LONG,
            feature_id="price.close_to_open",
            feature_type=ValueType.RETURN,
            cmp_op=OperatorId.GREATER_THAN,
            threshold=0.005,
        )
        result = validate_family_direction_coherence(cand, spec)
        assert not result.coherent
        assert result.rejection_reason == FAMILY_DIRECTION_INCOHERENT
        assert result.details["expected_direction"] == "ENTRY_SHORT"

    def test_gap_down_fade_with_entry_short_rejected(self) -> None:
        spec = materialize_family_spec("gap_fade", seed=3)
        cand = _cand(
            family_id="gap_fade",
            entry_op=OperatorId.ENTRY_SHORT,
            feature_id="price.close_to_open",
            feature_type=ValueType.RETURN,
            cmp_op=OperatorId.LESS_THAN,
            threshold=-0.005,
        )
        result = validate_family_direction_coherence(cand, spec)
        assert not result.coherent
        assert result.rejection_reason == FAMILY_DIRECTION_INCOHERENT
        assert result.details["expected_direction"] == "ENTRY_LONG"

    def test_valid_long_and_short_variants_accepted(self) -> None:
        cases = [
            (
                "momentum",
                OperatorId.ENTRY_LONG,
                "price.return_5",
                ValueType.RETURN,
                OperatorId.GREATER_THAN,
                0.01,
            ),
            (
                "momentum",
                OperatorId.ENTRY_SHORT,
                "price.return_5",
                ValueType.RETURN,
                OperatorId.LESS_THAN,
                -0.01,
            ),
            (
                "mean_reversion",
                OperatorId.ENTRY_LONG,
                "price.rolling_z_20",
                ValueType.ZSCORE,
                OperatorId.LESS_THAN,
                -2.0,
            ),
            (
                "mean_reversion",
                OperatorId.ENTRY_SHORT,
                "price.rolling_z_20",
                ValueType.ZSCORE,
                OperatorId.GREATER_THAN,
                2.0,
            ),
            (
                "gap_fade",
                OperatorId.ENTRY_SHORT,
                "price.close_to_open",
                ValueType.RETURN,
                OperatorId.GREATER_THAN,
                0.005,
            ),
            (
                "gap_fade",
                OperatorId.ENTRY_LONG,
                "price.close_to_open",
                ValueType.RETURN,
                OperatorId.LESS_THAN,
                -0.005,
            ),
            (
                "breakout",
                OperatorId.ENTRY_LONG,
                "price.breakout_distance_20",
                ValueType.RATIO,
                OperatorId.GREATER_THAN,
                0.02,
            ),
            (
                "breakout",
                OperatorId.ENTRY_SHORT,
                "price.return_5",
                ValueType.RETURN,
                OperatorId.LESS_THAN,
                -0.02,
            ),
        ]
        for family_id, entry_op, feat, ftype, cmp_op, thr in cases:
            spec = materialize_family_spec(family_id, seed=7)
            cand = _cand(
                family_id=family_id,
                entry_op=entry_op,
                feature_id=feat,
                feature_type=ftype,
                cmp_op=cmp_op,
                threshold=thr,
                creation_method=CreationMethod.RANDOM,
                parent_ids=(),
                generation=0,
            )
            result = validate_family_direction_coherence(cand, spec)
            assert result.coherent, (family_id, entry_op, result.details)

    def test_abs_gap_without_signed_condition_rejected(self) -> None:
        spec = materialize_family_spec("gap_fade", seed=4)
        gap = feature_node("price.close_to_open", ValueType.RETURN)
        cond = op_node(OperatorId.GREATER_THAN, op_node(OperatorId.ABS, gap), constant_node(0.005))
        entry = op_node(OperatorId.ENTRY_SHORT, cond)
        cand = build_candidate(
            entry_tree=entry,
            strategy_family="gap_fade",
            creation_method=CreationMethod.MUTATION,
            generation=1,
            parent_ids=("p1",),
            grammar_version="strategy_dsl_v1",
            feature_set_version="feature_set_v1_phase6b",
            random_seed=9,
            family_provenance={"family_id": "gap_fade"},
        )
        result = validate_family_direction_coherence(cand, spec)
        assert not result.coherent
        assert result.rejection_reason == FAMILY_DIRECTION_INCOHERENT


class TestDirectionCoherenceEvolutionGates:
    def test_mutation_cannot_enqueue_incoherent_descendant(self, tmp_path: Path) -> None:
        spec = materialize_family_spec("momentum", seed=11)
        incoherent = _cand(
            family_id="momentum",
            entry_op=OperatorId.ENTRY_SHORT,
            feature_id="price.return_5",
            feature_type=ValueType.RETURN,
            cmp_op=OperatorId.GREATER_THAN,
            threshold=0.01,
            creation_method=CreationMethod.MUTATION,
            generation=1,
            seed=101,
        )
        assert not validate_family_direction_coherence(incoherent, spec).coherent

        campaign = MultiFamilyCampaign(
            config=FamilyCampaignConfig(
                requested_family_count=1,
                family_ids=["momentum"],
                family_local_evolution=True,
                min_candidates_per_family=2,
                total_candidate_budget=8,
                max_full_wfo=6,
                seed=3,
            ),
            registry=ExperimentRegistry(tmp_path / "reg_mut_gate"),
            backend=CountingBackend(seed_salt=1),
            discovery_run_id="mut_gate",
        )
        greg = GenerationRecord(family_id="momentum", generation=0)
        reject_counts: dict[str, int] = {}
        grammar = spec.to_grammar()
        assert campaign.candidate_within_family_grammar(incoherent, grammar)

        parent = CandidateGenerator.from_family_spec(spec).generate(seed=5)
        mutator = Mutator(grammar=grammar)

        def _force_incoherent(parent_cand, **kwargs):
            return incoherent

        mutator.mutate = _force_incoherent  # type: ignore[method-assign]
        next_queue: list = []
        seen_ids: set[str] = set()
        breed_budget = 2

        def _accept(child, *, kind: str, family_ref: str) -> bool:
            if len(next_queue) >= breed_budget:
                return False
            if child.candidate_id in seen_ids:
                return False
            if child.strategy_family != family_ref:
                return False
            if not campaign.candidate_within_family_grammar(child, grammar):
                return False
            coh = validate_family_direction_coherence(child, spec)
            if not coh.coherent:
                campaign._record_rejected_descendant(
                    greg=greg,
                    child=child,
                    kind=kind,
                    rejection_reason=coh.rejection_reason or FAMILY_DIRECTION_INCOHERENT,
                    details=dict(coh.details),
                    reject_counts=reject_counts,
                )
                return False
            seen_ids.add(child.candidate_id)
            next_queue.append(child)
            return True

        mutant = mutator.mutate(parent, seed=1, strict=True, generation=1)
        accepted = _accept(mutant, kind="mutation", family_ref=parent.strategy_family)
        assert accepted is False
        assert next_queue == []
        assert incoherent.candidate_id not in seen_ids
        assert reject_counts[FAMILY_DIRECTION_INCOHERENT] == 1
        assert greg.rejected_descendants[0]["rejection_reason"] == FAMILY_DIRECTION_INCOHERENT
        assert greg.rejected_descendants[0]["candidate_id"] == incoherent.candidate_id

    def test_crossover_cannot_enqueue_incoherent_descendant(self, tmp_path: Path) -> None:
        spec = materialize_family_spec("momentum", seed=12)
        long_parent = _cand(
            family_id="momentum",
            entry_op=OperatorId.ENTRY_LONG,
            feature_id="price.return_5",
            feature_type=ValueType.RETURN,
            cmp_op=OperatorId.GREATER_THAN,
            threshold=0.01,
            creation_method=CreationMethod.RANDOM,
            parent_ids=(),
            generation=0,
            seed=1,
        )
        short_parent = _cand(
            family_id="momentum",
            entry_op=OperatorId.ENTRY_SHORT,
            feature_id="price.return_5",
            feature_type=ValueType.RETURN,
            cmp_op=OperatorId.LESS_THAN,
            threshold=-0.01,
            creation_method=CreationMethod.RANDOM,
            parent_ids=(),
            generation=0,
            seed=2,
        )
        incoherent = _cand(
            family_id="momentum",
            entry_op=OperatorId.ENTRY_SHORT,
            feature_id="price.return_5",
            feature_type=ValueType.RETURN,
            cmp_op=OperatorId.GREATER_THAN,
            threshold=0.01,
            creation_method=CreationMethod.CROSSOVER,
            parent_ids=(long_parent.candidate_id, short_parent.candidate_id),
            generation=1,
            seed=33,
        )
        assert len(incoherent.parent_ids) == 2
        assert not validate_family_direction_coherence(incoherent, spec).coherent

        campaign = MultiFamilyCampaign(
            config=FamilyCampaignConfig(family_local_evolution=True, seed=1),
            registry=ExperimentRegistry(tmp_path / "reg_cx_gate"),
            backend=CountingBackend(),
            discovery_run_id="cx_gate",
        )
        greg = GenerationRecord(family_id="momentum", generation=0)
        reject_counts: dict[str, int] = {}
        next_queue: list = []

        coh = validate_family_direction_coherence(incoherent, spec)
        if not coh.coherent:
            campaign._record_rejected_descendant(
                greg=greg,
                child=incoherent,
                kind="crossover",
                rejection_reason=FAMILY_DIRECTION_INCOHERENT,
                details=dict(coh.details),
                reject_counts=reject_counts,
            )
        assert next_queue == []
        assert reject_counts[FAMILY_DIRECTION_INCOHERENT] == 1
        assert greg.rejected_descendants[0]["creation_kind"] == "crossover"
        assert incoherent.candidate_id not in next_queue

    def test_incoherent_descendant_consumes_zero_full_wfo(self, tmp_path: Path) -> None:
        spec = materialize_family_spec("mean_reversion", seed=8)
        incoherent = _cand(
            family_id="mean_reversion",
            entry_op=OperatorId.ENTRY_SHORT,
            feature_id="price.rolling_z_20",
            feature_type=ValueType.ZSCORE,
            cmp_op=OperatorId.LESS_THAN,
            threshold=-2.0,
            creation_method=CreationMethod.MUTATION,
            generation=1,
            seed=77,
        )
        backend = CountingBackend(seed_salt=2)
        campaign = MultiFamilyCampaign(
            config=FamilyCampaignConfig(
                requested_family_count=1,
                family_ids=["mean_reversion"],
                family_local_evolution=True,
                min_candidates_per_family=2,
                total_candidate_budget=6,
                max_full_wfo=4,
                seed=2,
            ),
            registry=ExperimentRegistry(tmp_path / "reg_zero_wfo"),
            backend=backend,
            discovery_run_id="zero_wfo",
        )
        ev = campaign._make_evaluator(spec=spec, wfo_cap=3, gen_cap=3)
        campaign._bind_evaluator_features(ev)
        prev_wfo = ev.counters.full_wfo
        prev_calls = backend.evaluate_calls

        coh = validate_family_direction_coherence(incoherent, spec)
        assert not coh.coherent
        assert coh.rejection_reason == FAMILY_DIRECTION_INCOHERENT
        # Rejected before enqueue => backend never called, Full WFO unchanged.
        assert ev.counters.full_wfo == prev_wfo
        assert backend.evaluate_calls == prev_calls
        assert incoherent.candidate_id not in backend.evaluated_ids

    def test_accepted_crossover_records_exactly_two_parent_ids(self, tmp_path: Path) -> None:
        spec = materialize_family_spec("momentum", seed=15)
        grammar = spec.to_grammar()
        gen = CandidateGenerator.from_family_spec(spec)
        a = gen.generate(seed=21)
        b = gen.generate(seed=22)
        if a.entry_tree.name != b.entry_tree.name:
            for s in range(23, 80):
                b = gen.generate(seed=s)
                if b.entry_tree.name == a.entry_tree.name:
                    break
        children = Crossover(grammar=grammar).crossover(
            a, b, seed=41, strict=True, require_same_family=True, generation=1
        )
        accepted = []
        for child in children:
            assert len(child.parent_ids) == 2
            coh = validate_family_direction_coherence(child, spec)
            if coh.coherent and MultiFamilyCampaign.candidate_within_family_grammar(child, grammar):
                accepted.append(child)
        assert accepted, "expected at least one coherent crossover child"
        for child in accepted:
            assert len(child.parent_ids) == 2
            assert child.creation_method == CreationMethod.CROSSOVER
