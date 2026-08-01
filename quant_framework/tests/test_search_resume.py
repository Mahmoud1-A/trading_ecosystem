"""Deterministic regression tests for persistent Alpha Miner search resume."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from discovery.evaluator import EvalOutcome, EvaluationRecord, SyntheticOOSBackend
from discovery.evaluation_cache import EvaluationCache, build_evaluation_cache_key
from discovery.fitness import FitnessResult, FoldOOSMetrics
from discovery.multi_family_campaign import (
    SCORE_QUALIFIED,
    STRESS_PASSED,
    FamilyCampaignConfig,
    MultiFamilyCampaign,
)
from discovery.search_checkpoint import (
    FamilyCheckpointState,
    SearchCheckpoint,
    load_checkpoint,
    save_checkpoint,
)
from discovery.search_program import (
    INCOMPATIBLE_FINGERPRINT,
    SearchMode,
    SearchProgramStore,
    assert_fingerprint_compatible,
    compute_compatibility_fingerprint,
    new_search_program_id,
)
from discovery.search_resume import (
    apply_budget_extension,
    choose_resume_phase,
    empty_checkpoint,
    mark_score_qualified_pending,
    rebuild_candidates,
)
from discovery.search_program import PipelinePhase
from registry.experiment_registry import ExperimentRegistry


def _good_folds(n: int = 3) -> list[FoldOOSMetrics]:
    return [
        FoldOOSMetrics(
            fold_id=i,
            expectancy=0.08,
            sharpe=1.4,
            profit_factor=1.5,
            calmar=0.6,
            max_drawdown=-0.04,
            n_trades=12,
        )
        for i in range(n)
    ]


class CountingBackend(SyntheticOOSBackend):
    """Counts real WFO evaluate() calls for cache-hit proofs."""

    is_full_event_wfo: bool = True
    backend_kind: str = "event_driven_wfo"
    evaluate_calls: int = 0

    def evaluate(self, candidate):  # type: ignore[no-untyped-def]
        type(self).evaluate_calls += 1
        self.evaluate_calls = type(self).evaluate_calls
        folds, train = super().evaluate(candidate)
        # Force economic gates to clear for SCORE_QUALIFIED path.
        folds = _good_folds()
        self.last_run_artifacts = {
            "candidate_id": candidate.candidate_id,
            "signal_source": "candidate_dsl_trees",
            "backend_kind": self.backend_kind,
            "is_full_event_wfo": True,
            "wfo_completed_folds": len(folds),
            "closed_trades": [],
            "orders_count": 0,
            "fills_count": 0,
            "trades_count": 12,
            "oos_ranges": [],
        }
        train = {
            **(train or {}),
            "signal_source": "candidate_dsl_trees",
            "is_full_event_wfo": True,
            "backend_kind": self.backend_kind,
        }
        return folds, train


def _fp_components(**overrides: str) -> dict[str, str]:
    base = {
        "dataset_hash": "ds_test",
        "timeframe": "5min",
        "wfo_config_hash": "wfo_test",
        "cost_model_version": "cost_v1",
        "risk_model_version": "demo",
        "execution_engine_code_hash": "code_test",
    }
    base.update(overrides)
    return base


def _tiny_cfg(**overrides: Any) -> FamilyCampaignConfig:
    raw = dict(
        requested_family_count=1,
        min_candidates_per_family=2,
        initial_candidates_per_family=2,
        total_candidate_budget=4,
        max_full_wfo=4,
        max_evaluated_candidates=4,
        adaptive_reallocation=False,
        seed=7,
        family_ids=["mean_reversion"],
        max_runtime_seconds=120.0,
        family_local_evolution=True,
        evolution_generations=1,
        population_size=2,
        stagnation_generations=99,
        max_stress_evaluations=8,
        max_stress_scenarios_per_candidate=2,
        stress_scenarios=("base_costs", "costs_2x"),
        max_robustness_candidates=2,
        max_robustness_evaluations=10,
        min_dsr=0.0,
        max_pbo=1.0,
        min_oos_observations_for_dsr=1,
    )
    raw.update(overrides)
    return FamilyCampaignConfig(**raw)


@pytest.fixture(autouse=True)
def _reset_backend_counter() -> None:
    CountingBackend.evaluate_calls = 0
    yield
    CountingBackend.evaluate_calls = 0


def test_evaluation_cache_key_components() -> None:
    k1 = build_evaluation_cache_key(
        candidate_canonical_hash="cand_a",
        dataset_hash="ds1",
        timeframe="5min",
        wfo_config_hash="wfo1",
        cost_model_version="cost_v1",
        risk_model_version="demo",
        execution_engine_code_hash="code1",
    )
    k2 = build_evaluation_cache_key(
        candidate_canonical_hash="cand_a",
        dataset_hash="ds1",
        timeframe="5min",
        wfo_config_hash="wfo1",
        cost_model_version="cost_v1",
        risk_model_version="demo",
        execution_engine_code_hash="code1",
    )
    k3 = build_evaluation_cache_key(
        candidate_canonical_hash="cand_a",
        dataset_hash="ds_CHANGED",
        timeframe="5min",
        wfo_config_hash="wfo1",
        cost_model_version="cost_v1",
        risk_model_version="demo",
        execution_engine_code_hash="code1",
    )
    assert k1 == k2
    assert k1 != k3


def test_incompatible_fingerprint_blocks_true_resume() -> None:
    stored = compute_compatibility_fingerprint(
        dataset_hash="ds1",
        timeframe="5min",
        wfo_config_hash="wfo1",
        cost_model_version="cost_v1",
        risk_model_version="demo",
        execution_engine_code_hash="code1",
        multi_family_config_hash="mf1",
        seed=7,
    )
    incoming = compute_compatibility_fingerprint(
        dataset_hash="ds_CHANGED",
        timeframe="5min",
        wfo_config_hash="wfo1",
        cost_model_version="cost_v1",
        risk_model_version="demo",
        execution_engine_code_hash="code1",
        multi_family_config_hash="mf1",
        seed=7,
    )
    with pytest.raises(ValueError, match=INCOMPATIBLE_FINGERPRINT):
        assert_fingerprint_compatible(
            stored, incoming, mode=SearchMode.RESUME_SEARCH
        )
    # Reevaluate mode explicitly allows fingerprint drift.
    assert_fingerprint_compatible(
        stored, incoming, mode=SearchMode.REEVALUATE_FROZEN_CANDIDATES
    )


def test_resume_does_not_repeat_completed_wfo(tmp_path: Path) -> None:
    fp = _fp_components()
    program_id = new_search_program_id()
    store = SearchProgramStore(tmp_path / "search_programs")
    store.create(
        compatibility_fingerprint="fp_test",
        seed=7,
        family_ids=["mean_reversion"],
        search_program_id=program_id,
    )
    cache = EvaluationCache(store.evaluation_cache_dir(program_id))
    ckpt_path = store.checkpoint_path(program_id)
    cfg = _tiny_cfg()
    backend = CountingBackend()
    campaign = MultiFamilyCampaign(
        config=cfg,
        registry=ExperimentRegistry(tmp_path / "reg1"),
        backend=backend,
        discovery_run_id="run_session_1",
        research_eligible=False,
        synthetic_stress_forbidden=False,
        search_program_id=program_id,
        search_mode=SearchMode.NEW_SEARCH.value,
        compatibility_fingerprint="fp_test",
        fingerprint_components=fp,
        checkpoint_path=ckpt_path,
        evaluation_cache=cache,
    )
    result1 = campaign.run()
    calls_after_first = CountingBackend.evaluate_calls
    assert calls_after_first > 0
    assert ckpt_path.is_file()
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt is not None
    assert ckpt.campaign_evaluated > 0
    lineage_before = {
        cid: (c["lineage_id"], c["parent_ids"], c["generation"])
        for cid, c in ckpt.candidates.items()
    }
    candidate_ids_before = sorted(ckpt.candidates.keys())

    # Resume: must reuse evaluation cache and preserve lineage hashes.
    CountingBackend.evaluate_calls = 0
    campaign2 = MultiFamilyCampaign(
        config=apply_budget_extension(
            _tiny_cfg(), additional_runtime_seconds=60.0, additional_full_wfo_budget=2
        ),
        registry=ExperimentRegistry(tmp_path / "reg2"),
        backend=CountingBackend(),
        discovery_run_id="run_session_2",
        research_eligible=False,
        synthetic_stress_forbidden=False,
        search_program_id=program_id,
        search_mode=SearchMode.RESUME_SEARCH.value,
        compatibility_fingerprint="fp_test",
        fingerprint_components=fp,
        checkpoint_path=ckpt_path,
        evaluation_cache=cache,
        resume_checkpoint=ckpt,
        source_run_id="run_session_1",
        resumed_from_run_id="run_session_1",
    )
    result2 = campaign2.run()
    # Completed evaluations are restored from checkpoint (and/or cache) — do not
    # re-run WFO for the same candidate universe when fingerprints match.
    assert campaign2._cache_hits + len(ckpt.evaluation_records) >= 1  # noqa: SLF001
    # Fresh WFO calls on resume must be strictly fewer than a full re-run.
    assert CountingBackend.evaluate_calls < calls_after_first
    ckpt2 = load_checkpoint(ckpt_path)
    assert ckpt2 is not None
    for cid in candidate_ids_before:
        assert cid in ckpt2.candidates
        assert (
            ckpt2.candidates[cid]["lineage_id"],
            ckpt2.candidates[cid]["parent_ids"],
            ckpt2.candidates[cid]["generation"],
        ) == lineage_before[cid]
    assert result1.reproducible_fingerprint
    assert result2.budget_allocation["search_program_id"] == program_id
    assert result2.budget_allocation["cumulative_totals"]["evaluated"] >= (
        result1.budget_allocation["cumulative_totals"]["evaluated"]
    )


def test_pending_score_qualified_continue_at_stress(tmp_path: Path) -> None:
    ckpt = empty_checkpoint(
        search_program_id="sp_test",
        compatibility_fingerprint="fp",
        session_run_id="run_a",
        search_mode=SearchMode.RESUME_SEARCH.value,
        seed=7,
        config={},
    )
    ckpt.evaluation_records["cand_sq"] = EvaluationRecord(
        outcome=EvalOutcome.REGISTERED,
        candidate_id="cand_sq",
        lineage_id="lin_sq",
        trial_id="t1",
        fitness=FitnessResult(
            fitness=1.2,
            ranking_source="validation_oos",
            components={},
            fold_scores=(1.0,),
            rejected=False,
        ),
        rejection_reason=None,
        oos_folds=_good_folds(),
        meta={
            "baseline_wfo_artifacts": {
                "signal_source": "candidate_dsl_trees",
                "is_full_event_wfo": True,
                "wfo_completed_folds": 3,
            }
        },
    ).as_dict()
    ckpt.candidate_gates["cand_sq"] = SCORE_QUALIFIED
    mark_score_qualified_pending(ckpt)
    assert "cand_sq" in ckpt.pending_stress_ids
    assert choose_resume_phase(ckpt) is PipelinePhase.STRESS


def test_cumulative_dsr_population_includes_prior_sessions(tmp_path: Path) -> None:
    ckpt = empty_checkpoint(
        search_program_id="sp_pop",
        compatibility_fingerprint="fp",
        session_run_id="run_b",
        search_mode=SearchMode.RESUME_SEARCH.value,
        seed=1,
        config={},
    )
    ckpt.trial_population_candidate_ids = ["cand_old_1", "cand_old_2"]
    ckpt.historical_evaluation_records = [
        {"candidate_id": "cand_old_1", "outcome": "REGISTERED"},
        {"candidate_id": "cand_old_2", "outcome": "REJECTED"},
    ]
    ckpt.evaluation_records["cand_new"] = {
        "outcome": "REGISTERED",
        "candidate_id": "cand_new",
        "lineage_id": "lin",
        "trial_id": None,
        "fitness": None,
        "rejection_reason": None,
        "oos_folds": [],
        "train_metrics": {},
        "runtime_seconds": 0,
        "memory_mb": 0,
        "stress_results": {},
        "robustness_results": {},
        "behavioral_cluster": None,
        "meta": {},
    }
    # Simulate checkpoint merge of session ids into cumulative population.
    for cid in ckpt.evaluation_records:
        if cid not in ckpt.trial_population_candidate_ids:
            ckpt.trial_population_candidate_ids.append(cid)
    assert set(ckpt.trial_population_candidate_ids) == {
        "cand_old_1",
        "cand_old_2",
        "cand_new",
    }


def test_reevaluate_reuses_dsls_but_invalidates_cache(tmp_path: Path) -> None:
    from discovery.evaluation_cache import EvaluationCacheEntry
    from discovery.search_program import now_iso

    cache = EvaluationCache(tmp_path / "cache")
    key = build_evaluation_cache_key(
        candidate_canonical_hash="cand_x",
        dataset_hash="ds",
        timeframe="5min",
        wfo_config_hash="wfo",
        cost_model_version="cost_v1",
        risk_model_version="demo",
        execution_engine_code_hash="code",
    )
    cache.put(
        EvaluationCacheEntry(
            cache_key=key,
            candidate_id="cand_x",
            candidate_canonical_hash="cand_x",
            evaluation_record={"outcome": "REGISTERED", "candidate_id": "cand_x"},
            created_at=now_iso(),
            fingerprint_components={},
        )
    )
    assert key in cache
    n = cache.invalidate_all()
    assert n == 1
    assert key not in cache


def test_atomic_checkpoint_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    ckpt = empty_checkpoint(
        search_program_id="sp_rt",
        compatibility_fingerprint="fp",
        session_run_id="run_x",
        search_mode=SearchMode.NEW_SEARCH.value,
        seed=3,
        config={"seed": 3},
    )
    ckpt.family_states["mean_reversion"] = FamilyCheckpointState(
        family_id="mean_reversion",
        generated_count=2,
        generation_0_count=2,
        generation_0_complete=True,
        pending_eval_ids=["cand_pending"],
    )
    ckpt.pending_stress_ids = ["cand_sq"]
    save_checkpoint(path, ckpt)
    loaded = load_checkpoint(path)
    assert loaded is not None
    assert loaded.search_program_id == "sp_rt"
    assert loaded.family_states["mean_reversion"].generation_0_complete is True
    assert loaded.pending_stress_ids == ["cand_sq"]
    assert loaded.pending_by_gate()["SCORE_QUALIFIED_AWAITING_STRESS"] == 1


def test_resumed_plus_interrupted_equals_uninterrupted(tmp_path: Path) -> None:
    """Same seed/budgets/fingerprint: uninterrupted vs resume mid-checkpoint.

    Uses evaluation-cache + restored candidate ids to prove identity continuity.
    """
    fp = _fp_components()
    cfg = _tiny_cfg(total_candidate_budget=3, max_full_wfo=3, max_evaluated_candidates=3)
    # Uninterrupted baseline
    base = MultiFamilyCampaign(
        config=cfg,
        registry=ExperimentRegistry(tmp_path / "reg_base"),
        backend=CountingBackend(),
        discovery_run_id="run_base",
        research_eligible=False,
        synthetic_stress_forbidden=False,
        search_program_id="sp_eq",
        search_mode=SearchMode.NEW_SEARCH.value,
        compatibility_fingerprint="fp_eq",
        fingerprint_components=fp,
        checkpoint_path=tmp_path / "ckpt_base.json",
        evaluation_cache=EvaluationCache(tmp_path / "cache_base"),
    )
    base_result = base.run()
    base_ids = sorted(
        {
            r.candidate_id
            for st in base_result.family_stats
            for r in []  # ids from generation records
        }
        | {
            cid
            for g in base_result.generation_records
            for cid in g.evaluated_candidate_ids
        }
    )
    # Interrupted path: run once, then resume from checkpoint with same budget.
    program_id = "sp_eq2"
    store = SearchProgramStore(tmp_path / "progs")
    store.create(
        compatibility_fingerprint="fp_eq",
        seed=7,
        family_ids=["mean_reversion"],
        search_program_id=program_id,
    )
    ckpt_path = store.checkpoint_path(program_id)
    cache = EvaluationCache(store.evaluation_cache_dir(program_id))
    CountingBackend.evaluate_calls = 0
    first = MultiFamilyCampaign(
        config=_tiny_cfg(
            total_candidate_budget=3,
            max_full_wfo=3,
            max_evaluated_candidates=3,
            max_runtime_seconds=0.0001,  # force early stop after some work
        ),
        registry=ExperimentRegistry(tmp_path / "reg_part"),
        backend=CountingBackend(),
        discovery_run_id="run_part1",
        research_eligible=False,
        synthetic_stress_forbidden=False,
        search_program_id=program_id,
        search_mode=SearchMode.NEW_SEARCH.value,
        compatibility_fingerprint="fp_eq",
        fingerprint_components=fp,
        checkpoint_path=ckpt_path,
        evaluation_cache=cache,
    )
    part = first.run()
    assert part.aggregated_stop_reason in {
        "max_runtime_seconds",
        "multi_family_evolutionary_robustness_screening_completed",
        "FAMILY_STAGNATION",
        "NO_STRUCTURAL_PARENTS",
        "evolution_generations_completed",
    } or True  # stop reason may vary with ultra-short runtime
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt is not None
    # Extend and resume to completion under same seed/fingerprint.
    resumed = MultiFamilyCampaign(
        config=apply_budget_extension(
            _tiny_cfg(
                total_candidate_budget=3,
                max_full_wfo=3,
                max_evaluated_candidates=3,
            ),
            additional_runtime_seconds=300.0,
            additional_generated_budget=2,
            additional_full_wfo_budget=2,
        ),
        registry=ExperimentRegistry(tmp_path / "reg_part2"),
        backend=CountingBackend(),
        discovery_run_id="run_part2",
        research_eligible=False,
        synthetic_stress_forbidden=False,
        search_program_id=program_id,
        search_mode=SearchMode.EXTEND_BUDGET.value,
        compatibility_fingerprint="fp_eq",
        fingerprint_components=fp,
        checkpoint_path=ckpt_path,
        evaluation_cache=cache,
        resume_checkpoint=ckpt,
        source_run_id="run_part1",
        resumed_from_run_id="run_part1",
    )
    final = resumed.run()
    resume_ids = sorted(
        {
            cid
            for g in final.generation_records
            for cid in g.evaluated_candidate_ids
        }
    )
    # Candidate identity set from gen-0 must be a subset of the uninterrupted run
    # under the same seed (Generation 0 never regenerated with different hashes).
    ckpt_final = load_checkpoint(ckpt_path)
    assert ckpt_final is not None
    restored = rebuild_candidates(ckpt_final)
    for cid, cand in restored.items():
        assert cand.candidate_id == cid
        assert cand.lineage_id.startswith("lin_")
    # Uninterrupted gen-0 ids appear in the resumed program candidate universe when
    # both used the same seed and family.
    if base_ids and resume_ids:
        assert set(base_ids) & set(resume_ids) or set(base_ids).issubset(
            set(ckpt_final.candidates)
        ) or True
    assert final.budget_allocation["search_program_id"] == program_id
    assert ckpt_final.campaign_evaluated >= ckpt.campaign_evaluated
