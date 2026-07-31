"""Phase 3A: budget-safe family-local evolution focused tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from discovery.crossover import Crossover, CrossoverError
from discovery.evaluator import SyntheticOOSBackend
from discovery.family_generator import StrategyFamilyGenerator
from discovery.generator import CandidateGenerator
from discovery.multi_family_campaign import (
    CROSS_FAMILY_CROSSOVER_UNSUPPORTED,
    FAMILY_STAGNATION,
    NOT_PARENT_ELIGIBLE,
    PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_RESEARCH_SHORTLIST,
    SCORE_QUALIFIED,
    STRUCTURAL_PARENT_ELIGIBLE,
    FamilyCampaignConfig,
    MultiFamilyCampaign,
)
from discovery.mutation import Mutator
from discovery.types import CreationMethod
from registry.experiment_registry import ExperimentRegistry


@dataclass
class HonestFullWfoBackend(SyntheticOOSBackend):
    """Controlled backend that honestly proves Full WFO completion artifacts."""

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


def _evo_cfg(**overrides) -> FamilyCampaignConfig:
    base = dict(
        requested_family_count=1,
        min_candidates_per_family=4,
        total_candidate_budget=16,
        max_full_wfo=12,
        adaptive_reallocation=False,
        seed=11,
        min_oos_trades=1,
        min_oos_trades_per_fold=1,
        max_runtime_seconds=45,
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
    registry = ExperimentRegistry(tmp_path / "reg_phase3a")
    campaign = MultiFamilyCampaign(
        config=_evo_cfg(**cfg_overrides),
        registry=registry,
        backend=HonestFullWfoBackend(seed_salt=3),
        discovery_run_id="test_phase3a",
    )
    return campaign.run(), campaign


class TestPhase3AFamilyLocalEvolution:
    def test_mutated_descendant_receives_full_wfo(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path)
        mut_ids = {
            cid
            for g in result.generation_records
            for cid in g.mutation_candidate_ids
        }
        wfo_mut = {
            cid
            for g in result.generation_records
            for cid in g.completed_full_wfo_candidate_ids
            if cid in mut_ids
        }
        assert mut_ids, "expected at least one mutated descendant"
        assert wfo_mut, "mutated descendant must receive completed Full WFO"
        assert sum(s.full_wfo for s in result.family_stats) >= 1

    def test_crossover_descendant_records_two_parent_ids(self, tmp_path: Path) -> None:
        result, campaign = _run(tmp_path)
        cx_ids = {
            cid
            for g in result.generation_records
            for cid in g.crossover_candidate_ids
        }
        assert cx_ids, "expected at least one crossover descendant"
        found_parents: list[str] | None = None
        for trial in campaign.registry.all_trials():
            snap = trial.config_snapshot or {}
            if trial.candidate_id not in cx_ids:
                continue
            parents = list(snap.get("parent_ids") or [])
            if len(parents) == 2:
                found_parents = parents
                assert snap.get("creation_method") == CreationMethod.CROSSOVER.value
                break
        assert found_parents is not None and len(found_parents) == 2

    def test_descendants_evaluated_in_declared_generation(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path)
        for g in result.generation_records:
            for cid in g.mutation_candidate_ids + g.crossover_candidate_ids:
                # Descendants created during generation g are evaluated in g+1.
                eval_gens = [
                    gg.generation
                    for gg in result.generation_records
                    if cid in gg.evaluated_candidate_ids
                ]
                assert eval_gens, f"{cid} never evaluated"
                assert eval_gens == [g.generation + 1], (
                    f"{cid} created in gen {g.generation} but evaluated in {eval_gens}"
                )

    def test_generation_count_matches_completed_generations(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path)
        completed = len(result.generation_records)
        assert result.discovery_results[0].generations == completed
        assert completed >= 2

    def test_mutation_stays_inside_family_grammar(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path)
        spec = result.families[0]
        grammar = spec.to_grammar()
        gen = CandidateGenerator.from_family_spec(spec)
        parent = gen.generate(seed=33)
        mutator = Mutator(grammar=grammar)
        child = mutator.mutate(parent, seed=44, strict=True, generation=1)
        assert child.strategy_family == parent.strategy_family
        assert MultiFamilyCampaign.candidate_within_family_grammar(child, grammar)
        assert (child.family_provenance or {}).get("family_id") == (
            parent.family_provenance or {}
        ).get("family_id") or child.strategy_family == spec.family_id

    def test_crossover_stays_inside_same_family(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path)
        spec = result.families[0]
        grammar = spec.to_grammar()
        gen = CandidateGenerator.from_family_spec(spec)
        a = gen.generate(seed=5)
        b = gen.generate(seed=6)
        c1, c2 = Crossover(grammar=grammar).crossover(
            a, b, seed=9, strict=True, require_same_family=True, generation=1
        )
        assert c1.strategy_family == spec.family_id
        assert c2.strategy_family == spec.family_id
        assert MultiFamilyCampaign.candidate_within_family_grammar(c1, grammar)
        assert MultiFamilyCampaign.candidate_within_family_grammar(c2, grammar)
        assert len(c1.parent_ids) == 2

    def test_cross_family_crossover_fails_explicitly(self, tmp_path: Path) -> None:
        registry = ExperimentRegistry(tmp_path / "reg_xfam")
        with pytest.raises(RuntimeError, match=CROSS_FAMILY_CROSSOVER_UNSUPPORTED):
            MultiFamilyCampaign(
                config=_evo_cfg(allow_cross_family_crossover=True),
                registry=registry,
                backend=HonestFullWfoBackend(),
                discovery_run_id="xfam",
            ).run()

        families = StrategyFamilyGenerator(seed=3).generate(count=2)
        assert len(families) == 2
        a = CandidateGenerator.from_family_spec(families[0]).generate(seed=1)
        b = CandidateGenerator.from_family_spec(families[1]).generate(seed=2)
        with pytest.raises(CrossoverError):
            Crossover(grammar=families[0].to_grammar()).crossover(
                a, b, seed=1, strict=True, require_same_family=True, generation=1
            )

    def test_budgets_never_exceeded(self, tmp_path: Path) -> None:
        cfg = _evo_cfg(
            total_candidate_budget=10,
            max_full_wfo=6,
            max_evaluated_candidates=9,
            min_candidates_per_family=4,
            population_size=2,
            evolution_generations=4,
        )
        registry = ExperimentRegistry(tmp_path / "reg_budget")
        result = MultiFamilyCampaign(
            config=cfg,
            registry=registry,
            backend=HonestFullWfoBackend(seed_salt=1),
            discovery_run_id="budget",
        ).run()
        alloc = result.budget_allocation
        assert alloc["campaign_generated"] <= cfg.total_candidate_budget
        assert alloc["campaign_evaluated"] <= int(cfg.max_evaluated_candidates)
        assert alloc["campaign_full_wfo"] <= cfg.max_full_wfo
        for st in result.family_stats:
            assert st.generated <= st.allocation_generated
            assert st.full_wfo <= st.allocation_wfo

    def test_invalid_descendants_consume_zero_full_wfo(self, tmp_path: Path) -> None:
        result, campaign = _run(tmp_path, evolution_generations=2, population_size=2)
        before = sum(s.full_wfo for s in result.family_stats)
        spec = result.families[0]
        ev = campaign._make_evaluator(spec=spec, wfo_cap=3, gen_cap=3)
        campaign._bind_evaluator_features(ev)
        prev = ev.counters.full_wfo
        ev.register_invalid_dsl(ValueError("phase3a_invalid"), seed=1, operation="mutation")
        assert ev.counters.full_wfo == prev
        assert sum(s.full_wfo for s in result.family_stats) == before

    def test_stagnation_stops_correctly(self, tmp_path: Path) -> None:
        result, _ = _run(
            tmp_path,
            evolution_generations=6,
            population_size=2,
            stagnation_generations=1,
            minimum_improvement=1e9,
            total_candidate_budget=20,
            max_full_wfo=16,
        )
        assert result.aggregated_stop_reason == FAMILY_STAGNATION
        stagnated = [g for g in result.generation_records if g.stop_reason == FAMILY_STAGNATION]
        assert stagnated
        g = stagnated[-1]
        assert g.stagnation_count >= 1
        assert g.minimum_required_improvement == 1e9

    def test_structural_parent_does_not_imply_score_qualified(self, tmp_path: Path) -> None:
        result, _ = _run(tmp_path)
        assert result.pipeline_level == PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_RESEARCH_SHORTLIST
        assert result.post_wfo_pipeline_complete is False
        assert result.stress_pipeline_complete is True
        assert result.robustness_pipeline_complete is True
        assert result.statistics_pipeline_complete is True
        assert result.clustering_pipeline_complete is True
        assert result.research_shortlist_pipeline_complete is True
        assert result.vault_pipeline_complete is False
        assert result.paper_pipeline_complete is False
        assert result.live_pipeline_complete is False
        structural = 0
        score_q = 0
        for g in result.generation_records:
            for cid, status in g.parent_eligibility.items():
                if status == STRUCTURAL_PARENT_ELIGIBLE:
                    structural += 1
                    if cid in g.score_qualified_candidate_ids:
                        score_q += 1
                assert status in {
                    STRUCTURAL_PARENT_ELIGIBLE,
                    NOT_PARENT_ELIGIBLE,
                }
                # Score Qualified is tracked separately — never encoded as parent status.
                assert status != SCORE_QUALIFIED
        assert structural >= 1
        # Structural eligibility must be allowed without Score Qualified.
        assert structural >= score_q
        for st in result.family_stats:
            # parent_status_counts may track SCORE_QUALIFIED as a separate tally
            assert STRUCTURAL_PARENT_ELIGIBLE in st.parent_status_counts
        assert result.as_dict()["finalists"] == []
        assert result.as_dict()["promoted"] == []
        assert result.as_dict()["vault_candidates"] == []
        assert result.as_dict()["paper_candidates"] == []
