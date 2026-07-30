"""Tests for Strategy Family Generator + MultiFamilyCampaign."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest

from discovery.evaluator import SyntheticOOSBackend
from discovery.family_generator import StrategyFamilyGenerator, materialize_family_spec
from discovery.family_spec import (
    DuplicateFamilyError,
    HomogeneousFamilyGrammarError,
    assert_diverse_family_grammars,
    dedupe_family_specs,
)
from discovery.generator import CandidateGenerator
from discovery.multi_family_campaign import (
    FamilyCampaignConfig,
    MultiFamilyCampaign,
    adaptive_reallocate_wfo,
    equal_initial_allocation,
)
from discovery.operators import OperatorId
from registry.experiment_registry import ExperimentRegistry


def _op_hist(cand) -> Counter:
    return Counter(
        n.name for n in cand.entry_tree.walk() if n.kind.value == "OPERATOR"
    )


def _feat_hist(cand) -> Counter:
    return Counter(cand.feature_ids)


class TestFamilySpecDiversity:
    def test_eight_blueprints_have_distinct_grammars(self) -> None:
        gen = StrategyFamilyGenerator(seed=7)
        families = gen.generate(count=8)
        fps = {f.effective_grammar_fingerprint() for f in families}
        assert len(fps) == 8
        assert_diverse_family_grammars(families)

    def test_duplicate_family_specs_rejected(self) -> None:
        a = materialize_family_spec("mean_reversion", seed=1)
        b = materialize_family_spec("mean_reversion", seed=1)
        assert a.canonical_hash() == b.canonical_hash()
        with pytest.raises(DuplicateFamilyError):
            dedupe_family_specs([a, b])

    def test_homogeneous_grammar_fails_fast(self) -> None:
        a = materialize_family_spec("momentum", seed=3)
        # Force a clone with same grammar payload but different id via object replace
        clone = materialize_family_spec("momentum", seed=3)
        # Bypass dedupe and assert diversity helper on identical fingerprints
        with pytest.raises(HomogeneousFamilyGrammarError):
            assert_diverse_family_grammars([a, clone])


class TestFamilyAstDistributions:
    def test_different_families_generate_different_ast_distributions(self) -> None:
        families = StrategyFamilyGenerator(seed=11).generate(count=4)
        feat_sets = []
        op_sets = []
        for spec in families:
            gen = CandidateGenerator.from_family_spec(spec)
            assert gen.strategy_family == spec.family_id
            assert gen.strategy_family != "dsl_generated"
            feats = Counter()
            ops = Counter()
            for i in range(12):
                cand = gen.generate(seed=1000 + i * 17)
                assert cand.strategy_family == spec.family_id
                assert cand.family_provenance.get("family_id") == spec.family_id
                assert "family_hash" in cand.family_provenance
                # Features must stay inside family allow-list (+ ATR utility).
                allowed = set(spec.allowed_features)
                assert set(cand.feature_ids).issubset(allowed)
                feats.update(_feat_hist(cand))
                ops.update(_op_hist(cand))
            feat_sets.append(frozenset(feats.keys()))
            op_sets.append(frozenset(ops.keys()))
        # At least two families must differ in feature OR operator support used.
        assert len({frozenset(s) for s in feat_sets}) >= 2 or len({frozenset(s) for s in op_sets}) >= 2
        # Stronger: pairwise fingerprints of (features∪ops) differ across families.
        signatures = [frozenset(f) | frozenset(o) for f, o in zip(feat_sets, op_sets)]
        assert len(set(signatures)) >= 3

    def test_no_family_silently_becomes_dsl_generated(self) -> None:
        for spec in StrategyFamilyGenerator(seed=5).generate(count=8):
            gen = CandidateGenerator.from_family_spec(spec)
            cand = gen.generate(seed=42)
            assert cand.strategy_family == spec.family_id
            assert cand.strategy_family != "dsl_generated"
            assert cand.family_provenance["family_id"] == spec.family_id


class TestCampaignBudget:
    def test_equal_allocation_respects_minimum_quota(self) -> None:
        ids = ["a", "b", "c", "d", "e", "f"]
        alloc = equal_initial_allocation(
            family_ids=ids, total_budget=60, min_per_family=10
        )
        assert sum(alloc.values()) == 60
        assert all(v >= 10 for v in alloc.values())
        assert all(v == 10 for v in alloc.values())

    def test_allocation_never_starves_minimum(self) -> None:
        ids = ["a", "b", "c"]
        alloc = equal_initial_allocation(
            family_ids=ids, total_budget=32, min_per_family=10
        )
        assert sum(alloc.values()) == 32
        assert all(v >= 10 for v in alloc.values())

    def test_insufficient_budget_raises(self) -> None:
        with pytest.raises(ValueError):
            equal_initial_allocation(
                family_ids=["a", "b", "c"], total_budget=20, min_per_family=10
            )

    def test_adaptive_reallocation_keeps_min_quota(self) -> None:
        ids = ["a", "b", "c", "d", "e", "f"]
        scores = {"a": 1.0, "b": 0.5, "c": -1.0, "d": 2.0, "e": 0.0, "f": -2.0}
        alloc = adaptive_reallocate_wfo(
            family_ids=ids, max_full_wfo=18, early_scores=scores, min_quota=1
        )
        assert sum(alloc.values()) == 18
        assert all(v >= 1 for v in alloc.values())
        assert alloc["d"] >= alloc["f"]

    def test_campaign_budget_distributed_and_provenance_retained(self, tmp_path: Path) -> None:
        registry = ExperimentRegistry(tmp_path / "reg")
        cfg = FamilyCampaignConfig(
            requested_family_count=3,
            min_candidates_per_family=4,
            total_candidate_budget=12,
            max_full_wfo=6,
            adaptive_reallocation=True,
            seed=21,
            population_size=2,
            stagnation_generations=99,
            min_oos_trades=1,
            min_oos_trades_per_fold=1,
            max_runtime_seconds=30,
        )
        campaign = MultiFamilyCampaign(
            config=cfg,
            registry=registry,
            backend=SyntheticOOSBackend(),
            discovery_run_id="test_mfc",
        )
        result = campaign.run()
        assert len(result.families) == 3
        fps = {f.effective_grammar_fingerprint() for f in result.families}
        assert len(fps) == 3
        gen_alloc = result.budget_allocation["generated"]
        assert sum(gen_alloc.values()) == 12
        assert all(v >= 4 for v in gen_alloc.values())
        for st in result.family_stats:
            assert st.family_id != "dsl_generated"
            assert st.generated >= 1
            assert st.allocation_generated >= 4
        # Candidates retain family provenance in registry snapshots.
        families_seen = set()
        for trial in registry.all_trials():
            if trial.strategy_family == "invalid_dsl_placeholder":
                continue
            families_seen.add(trial.strategy_family)
            assert trial.strategy_family != "dsl_generated"
            prov = (trial.config_snapshot or {}).get("family_provenance") or {}
            if trial.rejection_reason and "precheck" in str(trial.rejection_reason):
                continue
            if prov:
                assert prov.get("family_id") == trial.strategy_family
        assert len(families_seen) >= 2
