"""Generation-budget semantics: exact initial population + evolutionary reserve."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from discovery.crossover import Crossover, CrossoverError
from discovery.evaluator import EvalOutcome, SyntheticOOSBackend
from discovery.family_generator import StrategyFamilyGenerator
from discovery.generator import CandidateGenerator
from discovery.multi_family_campaign import (
    CROSS_FAMILY_CROSSOVER_UNSUPPORTED,
    NO_STRUCTURAL_PARENTS,
    NOT_PARENT_ELIGIBLE,
    STRUCTURAL_PARENT_ELIGIBLE,
    FamilyCampaignConfig,
    MultiFamilyCampaign,
    evolution_budget_split,
)
from discovery.types import CreationMethod
from registry.experiment_registry import ExperimentRegistry


@dataclass
class HonestFullWfoBackend(SyntheticOOSBackend):
    is_full_event_wfo: bool = True
    backend_kind: str = "event_driven_wfo"

    def evaluate(self, candidate):
        folds, train = super().evaluate(candidate)
        train = {
            **train,
            "signal_source": "candidate_dsl_trees",
            "is_full_event_wfo": True,
            "wfo_completed_folds": len(folds),
        }
        return folds, train


def _cfg(**overrides) -> FamilyCampaignConfig:
    base: dict[str, Any] = dict(
        requested_family_count=8,
        min_candidates_per_family=6,
        initial_candidates_per_family=6,
        total_candidate_budget=96,
        max_full_wfo=32,
        max_evaluated_candidates=96,
        adaptive_reallocation=True,
        seed=11,
        min_oos_trades=1,
        min_oos_trades_per_fold=1,
        max_runtime_seconds=120,
        family_local_evolution=True,
        evolution_generations=3,
        population_size=3,
        stagnation_generations=99,
        allow_cross_family_crossover=False,
        minimum_improvement=1e-4,
    )
    base.update(overrides)
    return FamilyCampaignConfig(**base)


def _run(tmp_path: Path, **cfg_overrides):
    registry = ExperimentRegistry(tmp_path / "reg_budget_drain")
    campaign = MultiFamilyCampaign(
        config=_cfg(**cfg_overrides),
        registry=registry,
        backend=HonestFullWfoBackend(seed_salt=3),
        discovery_run_id="test_budget_drain",
    )
    return campaign.run(), campaign


class TestEvolutionBudgetSplit:
    def test_campaign_01_split_reserves_evolutionary_slots(self) -> None:
        ids = [f"f{i}" for i in range(8)]
        split = evolution_budget_split(
            family_ids=ids, total_budget=96, initial_per_family=6
        )
        assert split["initial_population_budget"] == 48
        assert split["evolutionary_candidate_budget"] == 48
        assert all(v == 6 for v in split["initial_alloc"].values())
        assert sum(split["evolutionary_alloc"].values()) == 48
        assert sum(split["generated_cap_alloc"].values()) == 96
        # Generation-0 capacity is exact — not 96/8=12.
        assert all(v == 6 for v in split["initial_alloc"].values())
        assert all(v == 12 for v in split["generated_cap_alloc"].values())


class TestAlphaBudgetDrainSemantics:
    def test_eight_by_six_creates_exactly_48_generation_zero(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path)
        alloc = result.budget_allocation
        assert alloc["generation_0_generated"] == 48
        assert alloc["initial_candidates_generated"] == 48
        assert alloc["initial_population_budget"] == 48
        assert sum(s.generation_0_generated for s in result.family_stats) == 48
        assert all(s.generation_0_generated == 6 for s in result.family_stats)

    def test_total_96_leaves_48_evolutionary_slots(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path)
        alloc = result.budget_allocation
        assert alloc["evolutionary_candidate_budget"] == 48
        assert alloc["generated_cap"] == 96
        assert (
            alloc["initial_population_budget"] + alloc["evolutionary_candidate_budget"]
            == 96
        )

    def test_generation_0_never_expands_to_exhaust_budget(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path)
        alloc = result.budget_allocation
        assert alloc["generation_0_generated"] == 48
        assert alloc["generation_0_generated"] != 96
        # Old bug: 96/8=12 gen-0 per family.
        assert all(s.generation_0_generated == 6 for s in result.family_stats)
        assert all(s.generation_0_generated != 12 for s in result.family_stats)

    def test_structural_parent_produces_generation_1_mutations(self, tmp_path: Path) -> None:
        result, campaign = _run(tmp_path)
        mut_ids = {
            cid
            for g in result.generation_records
            for cid in g.mutation_candidate_ids
        }
        assert mut_ids, "expected mutation children from structural parents"
        assert result.budget_allocation["descendants_generated"] > 0
        assert result.budget_allocation["highest_generation_reached"] >= 1
        found = False
        for trial in campaign.registry.all_trials():
            snap = trial.config_snapshot or {}
            if trial.candidate_id not in mut_ids:
                continue
            assert int(snap.get("generation", -1)) >= 1
            parents = list(snap.get("parent_ids") or [])
            assert len(parents) == 1
            found = True
            break
        assert found, "mutated descendants must be evaluated and registered with parent_ids"

    def test_mutation_children_have_one_parent_id(self, tmp_path: Path) -> None:
        result, campaign = _run(tmp_path)
        mut_ids = {
            cid
            for g in result.generation_records
            for cid in g.mutation_candidate_ids
        }
        assert mut_ids
        found = 0
        for trial in campaign.registry.all_trials():
            if trial.candidate_id not in mut_ids:
                continue
            snap = trial.config_snapshot or {}
            parents = list(snap.get("parent_ids") or [])
            assert len(parents) == 1
            assert snap.get("creation_method") == CreationMethod.MUTATION.value
            found += 1
        assert found >= 1

    def test_crossover_children_have_two_parent_ids(self, tmp_path: Path) -> None:
        result, campaign = _run(tmp_path)
        cx_ids = {
            cid
            for g in result.generation_records
            for cid in g.crossover_candidate_ids
        }
        assert cx_ids, "expected family-local crossover children"
        found = False
        for trial in campaign.registry.all_trials():
            if trial.candidate_id not in cx_ids:
                continue
            snap = trial.config_snapshot or {}
            parents = list(snap.get("parent_ids") or [])
            assert len(parents) == 2
            assert snap.get("creation_method") == CreationMethod.CROSSOVER.value
            found = True
            break
        assert found, "crossover descendants must be evaluated and registered with two parent_ids"

    def test_descendants_consume_generated_and_evaluated_budgets(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path)
        alloc = result.budget_allocation
        assert alloc["generated_total"] == (
            alloc["initial_candidates_generated"]
            + alloc["evolutionary_candidates_generated"]
        )
        assert alloc["campaign_evaluated"] <= 96
        assert alloc["evolutionary_candidates_generated"] == alloc["descendants_generated"]
        assert alloc["descendants_generated"] > 0

    def test_no_parent_families_report_no_structural_parents(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _never_parent(cls, rec):  # noqa: ANN001
            return NOT_PARENT_ELIGIBLE, False, "forced_no_parent"

        monkeypatch.setattr(
            MultiFamilyCampaign,
            "classify_parent_eligibility",
            classmethod(_never_parent),
        )
        result, _ = _run(
            tmp_path,
            requested_family_count=2,
            initial_candidates_per_family=3,
            min_candidates_per_family=3,
            total_candidate_budget=12,
            max_full_wfo=8,
            max_evaluated_candidates=12,
            adaptive_reallocation=True,
            evolution_generations=3,
        )
        alloc = result.budget_allocation
        assert alloc["generation_0_generated"] == 6
        assert alloc["descendants_generated"] == 0
        assert all(s.family_stop_reason == NO_STRUCTURAL_PARENTS for s in result.family_stats)
        assert all(s.descendants_generated == 0 for s in result.family_stats)
        # Must not refill unused evo with extra random gen-0.
        assert all(s.generation_0_generated == 3 for s in result.family_stats)

    def test_adaptive_reallocation_moves_evo_to_active_families(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        families = StrategyFamilyGenerator(seed=19).generate(count=2)
        blocked = families[0].family_id
        live = families[1].family_id
        registry = ExperimentRegistry(tmp_path / "reg_realloc")
        real_classify = MultiFamilyCampaign.classify_parent_eligibility

        def _classify(cls, rec):  # noqa: ANN001
            status, is_sq, reason = real_classify(rec)
            fam = None
            for trial in registry.all_trials():
                if trial.candidate_id == rec.candidate_id:
                    fam = trial.strategy_family
                    break
            if fam == blocked:
                return NOT_PARENT_ELIGIBLE, False, "blocked_family"
            return status, is_sq, reason

        monkeypatch.setattr(
            MultiFamilyCampaign,
            "classify_parent_eligibility",
            classmethod(_classify),
        )
        result = MultiFamilyCampaign(
            config=_cfg(
                requested_family_count=2,
                family_ids=[blocked, live],
                initial_candidates_per_family=2,
                min_candidates_per_family=2,
                total_candidate_budget=10,
                max_full_wfo=8,
                max_evaluated_candidates=10,
                adaptive_reallocation=True,
                evolution_generations=3,
                population_size=2,
                seed=19,
            ),
            registry=registry,
            backend=HonestFullWfoBackend(seed_salt=5),
            discovery_run_id="realloc",
        ).run()
        by_id = {s.family_id: s for s in result.family_stats}
        assert by_id[blocked].family_stop_reason == NO_STRUCTURAL_PARENTS
        assert by_id[blocked].descendants_generated == 0
        assert by_id[blocked].generation_0_generated == 2
        assert by_id[live].generation_0_generated == 2
        assert by_id[live].descendants_generated >= 1
        assert result.budget_allocation["generated_total"] <= 10

    def test_cross_family_crossover_remains_forbidden(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match=CROSS_FAMILY_CROSSOVER_UNSUPPORTED):
            MultiFamilyCampaign(
                config=_cfg(allow_cross_family_crossover=True, requested_family_count=2),
                registry=ExperimentRegistry(tmp_path / "xfam"),
                backend=HonestFullWfoBackend(),
                discovery_run_id="xfam",
            ).run()
        families = StrategyFamilyGenerator(seed=3).generate(count=2)
        a = CandidateGenerator.from_family_spec(families[0]).generate(seed=1)
        b = CandidateGenerator.from_family_spec(families[1]).generate(seed=2)
        with pytest.raises(CrossoverError):
            Crossover(grammar=families[0].to_grammar()).crossover(
                a, b, seed=1, strict=True, require_same_family=True, generation=1
            )

    def test_generated_total_never_exceeds_96(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path)
        assert result.budget_allocation["generated_total"] <= 96
        assert result.budget_allocation["campaign_generated"] <= 96

    def test_full_wfo_never_exceeds_cap(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path, max_full_wfo=32)
        assert result.budget_allocation["campaign_full_wfo"] <= 32
        assert sum(s.full_wfo for s in result.family_stats) <= 32

    def test_end_to_end_controlled_artifact_with_lineage(self, tmp_path: Path) -> None:
        result, campaign = _run(tmp_path)
        alloc = result.budget_allocation
        assert alloc["generation_0_generated"] == 48
        assert alloc["descendants_generated"] > 0
        assert alloc["highest_generation_reached"] >= 1
        assert alloc["generated_total"] <= 96
        assert alloc["campaign_full_wfo"] <= 32
        # Funnel exposes required fields.
        for st in result.family_stats:
            payload = st.as_dict()
            for key in (
                "generation_0_generated",
                "descendants_generated",
                "mutation_children",
                "crossover_children",
                "highest_generation_reached",
                "structural_parents_found",
                "family_stop_reason",
            ):
                assert key in payload
        # At least one candidate with parent_ids.
        lined = False
        sample_mutation = None
        sample_crossover = None
        for trial in campaign.registry.all_trials():
            snap = trial.config_snapshot or {}
            parents = list(snap.get("parent_ids") or [])
            if not parents:
                continue
            lined = True
            if len(parents) == 1 and sample_mutation is None:
                sample_mutation = {
                    "candidate_id": trial.candidate_id,
                    "parent_ids": parents,
                    "generation": snap.get("generation"),
                    "creation_method": snap.get("creation_method"),
                }
            if len(parents) == 2 and sample_crossover is None:
                sample_crossover = {
                    "candidate_id": trial.candidate_id,
                    "parent_ids": parents,
                    "generation": snap.get("generation"),
                    "creation_method": snap.get("creation_method"),
                }
        assert lined
        assert sample_mutation is not None
        assert sample_crossover is not None
        # Compatibility: min_candidates maps when initial field absent.
        cfg = FamilyCampaignConfig(
            min_candidates_per_family=6,
            initial_candidates_per_family=None,
            family_local_evolution=True,
        )
        assert cfg.effective_initial_candidates_per_family() == 6


class TestMinCandidatesCompatibility:
    def test_min_maps_to_initial_when_absent(self) -> None:
        cfg = FamilyCampaignConfig(min_candidates_per_family=6)
        assert cfg.effective_initial_candidates_per_family() == 6
        assert cfg.as_dict()["initial_candidates_per_family"] == 6
