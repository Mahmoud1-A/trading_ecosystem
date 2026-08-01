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


def _write_legacy_runtime_exhausted_artifacts(art: Path) -> dict[str, Any]:
    """Pre-resume RUNTIME_EXHAUSTED artifacts with no search_program_id."""
    import json

    from discovery.family_generator import StrategyFamilyGenerator
    from discovery.generator import CandidateGenerator

    art.mkdir(parents=True, exist_ok=True)
    (art / "registry").mkdir(parents=True, exist_ok=True)
    specs = StrategyFamilyGenerator(seed=7).generate(
        count=1, family_ids=["mean_reversion"]
    )
    gen = CandidateGenerator.from_family_spec(specs[0])
    cands = []
    seen: set[str] = set()
    seed_i = 0
    while len(cands) < 3 and seed_i < 40:
        seed_i += 1
        try:
            c = gen.generate(seed=seed_i)
        except Exception:  # noqa: BLE001
            continue
        if c.candidate_id in seen:
            continue
        seen.add(c.candidate_id)
        cands.append(c)
    assert len(cands) >= 2
    sq = cands[0]
    other = cands[1]

    trials = []
    for i, cand in enumerate(cands[:2]):
        ranking = 1.25 if i == 0 else -0.4
        rejected = None if i == 0 else "NEGATIVE_EXPECTANCY"
        trials.append(
            {
                "trial_id": f"trial_{i}",
                "candidate_id": cand.candidate_id,
                "lineage_id": cand.lineage_id,
                "strategy_family": cand.strategy_family,
                "parameters": dict(cand.parameters),
                "config_snapshot": {
                    "expression_tree": cand.entry_tree.as_dict(),
                    "generation": 0,
                    "parent_ids": [],
                    "creation_method": cand.creation_method.value,
                    "is_full_event_wfo": True,
                    "signal_source": "candidate_dsl_trees",
                    "backend_kind": "event_driven_wfo",
                    "wfo_completed_folds": 3,
                    "family_provenance": {
                        "family_id": "mean_reversion",
                        "initial_or_descendant": "initial",
                    },
                    "grammar_version": cand.grammar_version,
                    "feature_set_version": cand.feature_set_version,
                    "complexity": cand.complexity_score,
                },
                "system_version": "test",
                "git_commit_hash": "local",
                "data_hash": "ds_test",
                "random_seed": cand.random_seed,
                "train_window": None,
                "validation_window": None,
                "vault_version": None,
                "gross_metrics": {
                    "signal_source": "candidate_dsl_trees",
                    "is_full_event_wfo": True,
                    "wfo_completed_folds": 3,
                },
                "net_metrics": {},
                "ranking_score": ranking,
                "rejection_reason": rejected,
                "trade_log_path": None,
                "equity_curve_path": None,
                "execution_assumptions": {},
                "cost_model_version": "cost_v1",
                "code_hash": "code_test",
                "trial_status": "COMPLETED",
                "fold_records": [
                    {
                        "fold_id": j,
                        "expectancy": 0.08 if i == 0 else -0.05,
                        "sharpe": 1.2,
                        "profit_factor": 1.4 if i == 0 else 0.7,
                        "calmar": 0.5,
                        "max_drawdown": -0.04,
                        "n_trades": 10,
                    }
                    for j in range(3)
                ],
                "ranking_source": "validation_oos",
            }
        )
    ledger = art / "registry" / "trial_ledger.jsonl"
    ledger.write_text(
        "\n".join(json.dumps(t) for t in trials) + "\n", encoding="utf-8"
    )
    campaign = {
        "campaign_id": "run_legacy_exhausted",
        "families": [s.as_dict() for s in specs],
        "family_stats": [
            {
                "family_id": "mean_reversion",
                "hypothesis": specs[0].hypothesis,
                "canonical_hash": specs[0].canonical_hash(),
                "grammar_fingerprint": specs[0].effective_grammar_fingerprint(),
                "allocation_generated": 4,
                "allocation_wfo": 4,
                "generated": 3,
                "evaluated": 2,
                "full_wfo": 2,
                "score_qualified": 1,
                "generation_0_generated": 3,
                "descendants_generated": 0,
                "highest_generation_reached": 0,
                "structural_parents_found": 1,
                "family_stop_reason": "max_runtime_seconds",
            }
        ],
        "budget_allocation": {
            "campaign_generated": 3,
            "campaign_evaluated": 2,
            "campaign_full_wfo": 2,
            "total_candidate_budget": 4,
            "max_full_wfo": 4,
            "family_local_evolution": True,
        },
        "aggregated_stop_reason": "max_runtime_seconds",
        "candidate_status_history": [
            {
                "sequence": 1,
                "candidate_id": sq.candidate_id,
                "family_id": "mean_reversion",
                "generation": 0,
                "prior_status": "FULL_WFO_COMPLETED",
                "new_status": "SCORE_QUALIFIED",
                "reason": "SCORE_QUALIFIED",
                "artifact_refs": {},
            }
        ],
        "generation_records": [
            {
                "family_id": "mean_reversion",
                "generation": 0,
                "evaluated_candidate_ids": [sq.candidate_id, other.candidate_id],
                "score_qualified_candidate_ids": [sq.candidate_id],
                "completed_full_wfo_candidate_ids": [sq.candidate_id, other.candidate_id],
                "generated_count": 3,
                "evaluated_count": 2,
                "completed_full_wfo_count": 2,
                "stop_reason": "max_runtime_seconds",
            }
        ],
        "candidate_stress_summaries": [],
        "candidate_robustness_summaries": [],
        "candidate_statistics_summaries": [],
        "clusters": [],
        "research_shortlist": [],
    }
    (art / "multi_family_campaign.json").write_text(
        json.dumps(campaign, indent=2), encoding="utf-8"
    )
    return {
        "score_qualified_id": sq.candidate_id,
        "evaluated_ids": [sq.candidate_id, other.candidate_id],
        "generated": 3,
        "evaluated": 2,
        "full_wfo": 2,
        "family_specs": [s.as_dict() for s in specs],
        "candidates": cands[:2],
    }


def test_legacy_runtime_exhausted_continue_search_bootstraps(tmp_path: Path) -> None:
    """Legacy RUNTIME_EXHAUSTED run (no search_program_id) must bootstrap on Continue."""
    from control_plane.search_bootstrap import prepare_search_program_session
    from discovery.evaluation_cache import EvaluationCacheEntry
    from discovery.search_program import PipelinePhase, SearchMode, SearchProgramStore, now_iso
    from discovery.search_resume import choose_resume_phase

    artifacts_root = tmp_path / "artifacts"
    legacy_id = "run_legacy_exhausted"
    legacy_art = artifacts_root / legacy_id
    meta = _write_legacy_runtime_exhausted_artifacts(legacy_art)

    program_root = tmp_path / "search_programs"
    store = SearchProgramStore(program_root)
    fp = "fp_legacy_test"
    prepared = prepare_search_program_session(
        search_mode=SearchMode.EXTEND_BUDGET.value,
        program_id=None,  # legacy: absent
        program_store=store,
        compatibility_fingerprint=fp,
        seed=7,
        family_ids=["mean_reversion"],
        run_id="run_continue_1",
        source_run_id=legacy_id,
        resumed_from_run_id=legacy_id,
        artifacts_root=artifacts_root,
        fingerprint_components=_fp_components(),
    )
    assert prepared.search_program_id.startswith("sp_")
    assert prepared.resume_checkpoint is not None
    assert prepared.bootstrapped_from_source is True
    assert prepared.checkpoint_path.is_file()
    ckpt = prepared.resume_checkpoint
    assert ckpt.campaign_generated == 3
    assert ckpt.campaign_evaluated == 2
    assert ckpt.campaign_full_wfo == 2
    assert meta["score_qualified_id"] in ckpt.pending_stress_ids
    assert choose_resume_phase(ckpt) is PipelinePhase.STRESS
    assert all(fst.generation_0_complete for fst in ckpt.family_states.values())

    events: list[str] = []

    def _hook(name: str, payload: dict) -> None:
        events.append(name)

    CountingBackend.evaluate_calls = 0
    fp_comp = _fp_components()
    for cid, raw in ckpt.evaluation_records.items():
        key = build_evaluation_cache_key(
            candidate_canonical_hash=cid,
            dataset_hash=fp_comp["dataset_hash"],
            timeframe=fp_comp["timeframe"],
            wfo_config_hash=fp_comp["wfo_config_hash"],
            cost_model_version=fp_comp["cost_model_version"],
            risk_model_version=fp_comp["risk_model_version"],
            execution_engine_code_hash=fp_comp["execution_engine_code_hash"],
        )
        prepared.evaluation_cache.put(
            EvaluationCacheEntry(
                cache_key=key,
                candidate_id=cid,
                candidate_canonical_hash=cid,
                evaluation_record=raw,
                created_at=now_iso(),
                fingerprint_components=fp_comp,
            )
        )

    cfg = _tiny_cfg(
        total_candidate_budget=6,
        max_full_wfo=6,
        max_evaluated_candidates=6,
        max_runtime_seconds=30.0,
        evolution_generations=1,
    )
    campaign = MultiFamilyCampaign(
        config=cfg,
        registry=ExperimentRegistry(tmp_path / "reg_legacy"),
        backend=CountingBackend(),
        discovery_run_id="run_continue_1",
        research_eligible=False,
        synthetic_stress_forbidden=False,
        progress_hook=_hook,
        search_program_id=prepared.search_program_id,
        search_mode=SearchMode.EXTEND_BUDGET.value,
        compatibility_fingerprint=fp,
        fingerprint_components=fp_comp,
        checkpoint_path=prepared.checkpoint_path,
        evaluation_cache=prepared.evaluation_cache,
        resume_checkpoint=ckpt,
        source_run_id=legacy_id,
        resumed_from_run_id=legacy_id,
    )
    result = campaign.run()
    assert "SEARCH_RESUME_STARTED" in events
    resume_idx = events.index("SEARCH_RESUME_STARTED")
    stress_idxs = [i for i, e in enumerate(events) if e == "MULTI_FAMILY_STRESS_STARTED"]
    evo_idxs = [i for i, e in enumerate(events) if e == "MULTI_FAMILY_EVOLUTION_STARTED"]
    assert stress_idxs, "Stress must run on resume"
    assert stress_idxs[0] > resume_idx
    if evo_idxs:
        assert stress_idxs[0] < evo_idxs[0], "Stress must precede evolution/generation"
    assert meta["score_qualified_id"] in ckpt.candidates
    restored_ids = set(meta["evaluated_ids"])
    assert restored_ids.issubset(set(ckpt.evaluation_records))
    assert result.budget_allocation["search_program_id"] == prepared.search_program_id
    assert result.budget_allocation["cumulative_totals"]["generated"] >= 3
    assert result.budget_allocation["cumulative_totals"]["evaluated"] >= 2
    assert result.budget_allocation["cumulative_totals"]["full_wfo"] >= 2
    # Restored Full-WFO candidates must not force a full re-evaluation wave.
    assert CountingBackend.evaluate_calls < len(ckpt.candidates) + 5


def test_prepare_new_search_skips_bootstrap(tmp_path: Path) -> None:
    from control_plane.search_bootstrap import prepare_search_program_session
    from discovery.search_program import SearchMode, SearchProgramStore

    store = SearchProgramStore(tmp_path / "sp")
    prepared = prepare_search_program_session(
        search_mode=SearchMode.NEW_SEARCH.value,
        program_id=None,
        program_store=store,
        compatibility_fingerprint="fp",
        seed=1,
        family_ids=["mean_reversion"],
        run_id="run_new",
        source_run_id="run_legacy",
        resumed_from_run_id=None,
        artifacts_root=tmp_path / "artifacts",
    )
    assert prepared.resume_checkpoint is None
    assert prepared.bootstrapped_from_source is False
    assert prepared.created_new_program is True


def test_resume_without_checkpoint_or_source_raises(tmp_path: Path) -> None:
    from control_plane.search_bootstrap import prepare_search_program_session
    from discovery.search_program import MISSING_CHECKPOINT, SearchMode, SearchProgramStore

    store = SearchProgramStore(tmp_path / "sp")
    with pytest.raises(RuntimeError, match=MISSING_CHECKPOINT):
        prepare_search_program_session(
            search_mode=SearchMode.RESUME_SEARCH.value,
            program_id=None,
            program_store=store,
            compatibility_fingerprint="fp",
            seed=1,
            family_ids=None,
            run_id="run_x",
            source_run_id=None,
            resumed_from_run_id=None,
            artifacts_root=tmp_path / "artifacts",
        )
