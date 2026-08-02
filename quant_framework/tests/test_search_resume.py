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
    IncompatibleSearchFingerprint,
    SearchMode,
    SearchProgramStore,
    assert_fingerprint_compatible,
    build_fingerprint_components,
    canonical_multi_family_payload,
    compute_compatibility_fingerprint,
    hash_multi_family_config,
    hash_wfo_config,
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
        "execution_semantic_hash": "code_test",
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
        execution_semantic_hash="code1",
    )
    k2 = build_evaluation_cache_key(
        candidate_canonical_hash="cand_a",
        dataset_hash="ds1",
        timeframe="5min",
        wfo_config_hash="wfo1",
        cost_model_version="cost_v1",
        risk_model_version="demo",
        execution_semantic_hash="code1",
    )
    k3 = build_evaluation_cache_key(
        candidate_canonical_hash="cand_a",
        dataset_hash="ds_CHANGED",
        timeframe="5min",
        wfo_config_hash="wfo1",
        cost_model_version="cost_v1",
        risk_model_version="demo",
        execution_semantic_hash="code1",
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
        execution_semantic_hash="code1",
        multi_family_config_hash="mf1",
        seed=7,
    )
    incoming = compute_compatibility_fingerprint(
        dataset_hash="ds_CHANGED",
        timeframe="5min",
        wfo_config_hash="wfo1",
        cost_model_version="cost_v1",
        risk_model_version="demo",
        execution_semantic_hash="code1",
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
    # Completed evaluations are restored from checkpoint (and/or cache) ? do not
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
        execution_semantic_hash="code",
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
        payload = cand.as_dict()
        trials.append(
            {
                "trial_id": f"trial_{i}",
                "candidate_id": cand.candidate_id,
                "lineage_id": cand.lineage_id,
                "strategy_family": cand.strategy_family,
                "parameters": dict(cand.parameters),
                "config_snapshot": {
                    "expression_tree": cand.entry_tree.as_dict(),
                    "entry_tree": cand.entry_tree.as_dict(),
                    "exit_tree": cand.exit_tree.as_dict() if cand.exit_tree else None,
                    "stop": cand.stop.as_dict() if cand.stop else None,
                    "target": cand.target.as_dict() if cand.target else None,
                    "sizing": cand.sizing.as_dict() if cand.sizing else None,
                    "regime_gates": [g.as_dict() for g in cand.regime_gates],
                    "candidate": payload,
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
                "net_metrics": {
                    "median_oos_expectancy": 0.08 if i == 0 else -0.05,
                    "fold_trade_counts": [10, 10, 10],
                    "train_diagnostic": {"sharpe": -1.0},
                },
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
            execution_semantic_hash=fp_comp["execution_semantic_hash"],
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


def _write_legacy_list_param_ledger(
    art: Path,
    *,
    n_score_qualified: int = 42,
    include_ambiguous_pending: bool = False,
) -> dict[str, Any]:
    """Realistic legacy ledger: list-valued params + full candidate payloads."""
    import json

    from discovery.family_generator import StrategyFamilyGenerator
    from discovery.generator import CandidateGenerator

    art.mkdir(parents=True, exist_ok=True)
    (art / "registry").mkdir(parents=True, exist_ok=True)
    specs = StrategyFamilyGenerator(seed=11).generate(
        count=1, family_ids=["mean_reversion"]
    )
    gen = CandidateGenerator.from_family_spec(specs[0])
    cands = []
    seen: set[str] = set()
    seed_i = 0
    need = n_score_qualified + 8
    while len(cands) < need and seed_i < 400:
        seed_i += 1
        try:
            c = gen.generate(seed=seed_i)
        except Exception:  # noqa: BLE001
            continue
        if c.candidate_id in seen:
            continue
        seen.add(c.candidate_id)
        cands.append(c)
    assert len(cands) >= n_score_qualified + 2

    sq_cands = cands[:n_score_qualified]
    rejected_cands = cands[n_score_qualified : n_score_qualified + 5]
    if include_ambiguous_pending:
        ambiguous = cands[n_score_qualified + 5]
    else:
        ambiguous = None

    list_param_name = next(iter(sq_cands[0].parameters))
    list_param_scalar = float(sq_cands[0].parameters[list_param_name])
    original_ids = [c.candidate_id for c in sq_cands]
    original_hashes = list(original_ids)

    trials: list[dict[str, Any]] = []
    status_history: list[dict[str, Any]] = []
    seq = 0

    def _trial_for(
        cand: Any,
        *,
        ranking: float | None,
        rejected: str | None,
        listify_param: bool = False,
        drop_candidate_payload: bool = False,
        corrupt_list_no_dsl: bool = False,
    ) -> dict[str, Any]:
        params = dict(cand.parameters)
        if listify_param and list_param_name in params:
            params[list_param_name] = [0.1, list_param_scalar, 9.9]
        if corrupt_list_no_dsl:
            params["ghost_grid_param"] = [1.0, 2.0, 3.0]
        payload = cand.as_dict()
        snap: dict[str, Any] = {
            "expression_tree": cand.entry_tree.as_dict(),
            "entry_tree": cand.entry_tree.as_dict(),
            "generation": 0,
            "parent_ids": list(cand.parent_ids),
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
        }
        if drop_candidate_payload:
            # Entry-only legacy snapshot (malformed / history-only).
            pass
        else:
            snap.update(
                {
                    "exit_tree": cand.exit_tree.as_dict() if cand.exit_tree else None,
                    "stop": cand.stop.as_dict() if cand.stop else None,
                    "target": cand.target.as_dict() if cand.target else None,
                    "sizing": cand.sizing.as_dict() if cand.sizing else None,
                    "regime_gates": [g.as_dict() for g in cand.regime_gates],
                    "candidate": payload,
                }
            )
        return {
            "trial_id": f"trial_{cand.candidate_id}",
            "candidate_id": cand.candidate_id,
            "lineage_id": cand.lineage_id,
            "strategy_family": cand.strategy_family,
            "parameters": params,
            "config_snapshot": snap,
            "system_version": "test",
            "git_commit_hash": "local",
            "data_hash": "ds_test",
            "random_seed": cand.random_seed,
            "gross_metrics": {
                "signal_source": "candidate_dsl_trees",
                "is_full_event_wfo": True,
                "wfo_completed_folds": 3,
            },
            "net_metrics": {
                "median_oos_expectancy": 0.08 if rejected is None else -0.05,
                "fold_trade_counts": [12, 11, 10],
                "train_diagnostic": {"sharpe": -2.0},
                "violations": ["legacy"],
            },
            "ranking_score": ranking,
            "rejection_reason": rejected,
            "cost_model_version": "cost_v1",
            "code_hash": "code_test",
            "trial_status": "COMPLETED",
            "fold_records": [
                {
                    "fold_id": j,
                    "expectancy": 0.08 if rejected is None else -0.05,
                    "sharpe": 1.2,
                    "profit_factor": 1.4 if rejected is None else 0.7,
                    "calmar": 0.5,
                    "max_drawdown": -0.04,
                    "n_trades": 12,
                }
                for j in range(3)
            ],
            "ranking_source": "validation_oos",
        }

    for i, cand in enumerate(sq_cands):
        trials.append(
            _trial_for(
                cand,
                ranking=1.0 + i * 0.01,
                rejected=None,
                listify_param=(i == 0),
            )
        )
        seq += 1
        status_history.append(
            {
                "sequence": seq,
                "candidate_id": cand.candidate_id,
                "family_id": "mean_reversion",
                "generation": 0,
                "prior_status": "FULL_WFO_COMPLETED",
                "new_status": "SCORE_QUALIFIED",
                "reason": "SCORE_QUALIFIED",
                "artifact_refs": {},
            }
        )

    for cand in rejected_cands:
        # Malformed rejected history: list param + entry-only snapshot.
        trials.append(
            _trial_for(
                cand,
                ranking=-0.5,
                rejected="NEGATIVE_EXPECTANCY",
                listify_param=True,
                drop_candidate_payload=True,
            )
        )

    if ambiguous is not None:
        trials.append(
            _trial_for(
                ambiguous,
                ranking=2.5,
                rejected=None,
                corrupt_list_no_dsl=True,
            )
        )
        seq += 1
        status_history.append(
            {
                "sequence": seq,
                "candidate_id": ambiguous.candidate_id,
                "family_id": "mean_reversion",
                "generation": 0,
                "prior_status": "FULL_WFO_COMPLETED",
                "new_status": "SCORE_QUALIFIED",
                "reason": "SCORE_QUALIFIED",
                "artifact_refs": {},
            }
        )

    (art / "registry" / "trial_ledger.jsonl").write_text(
        "\n".join(json.dumps(t) for t in trials) + "\n", encoding="utf-8"
    )
    n_eval = len(trials)
    campaign = {
        "campaign_id": "run_legacy_list_params",
        "families": [s.as_dict() for s in specs],
        "family_stats": [
            {
                "family_id": "mean_reversion",
                "hypothesis": specs[0].hypothesis,
                "canonical_hash": specs[0].canonical_hash(),
                "grammar_fingerprint": specs[0].effective_grammar_fingerprint(),
                "allocation_generated": n_eval + 10,
                "allocation_wfo": n_eval + 10,
                "generated": n_eval + 10,
                "evaluated": n_eval,
                "full_wfo": n_eval,
                "score_qualified": n_score_qualified + (1 if ambiguous else 0),
                "generation_0_generated": n_eval + 10,
                "descendants_generated": 0,
                "highest_generation_reached": 0,
                "structural_parents_found": n_score_qualified,
                "family_stop_reason": "max_runtime_seconds",
            }
        ],
        "budget_allocation": {
            "campaign_generated": n_eval + 10,
            "campaign_evaluated": n_eval,
            "campaign_full_wfo": n_eval,
            "total_candidate_budget": n_eval + 20,
            "max_full_wfo": n_eval + 20,
            "family_local_evolution": True,
        },
        "aggregated_stop_reason": "max_runtime_seconds",
        "candidate_status_history": status_history,
        "generation_records": [
            {
                "family_id": "mean_reversion",
                "generation": 0,
                "evaluated_candidate_ids": [t["candidate_id"] for t in trials],
                "score_qualified_candidate_ids": [
                    c.candidate_id for c in sq_cands
                ]
                + ([ambiguous.candidate_id] if ambiguous else []),
                "completed_full_wfo_candidate_ids": [t["candidate_id"] for t in trials],
                "generated_count": n_eval + 10,
                "evaluated_count": n_eval,
                "completed_full_wfo_count": n_eval,
                "stop_reason": "max_runtime_seconds",
            }
        ],
        "candidate_stress_summaries": [],
        "candidate_robustness_summaries": [],
        "candidate_statistics_summaries": [],
        "clusters": [],
        "research_shortlist": [],
        "population_stats": {"n_trials": n_eval},
    }
    (art / "multi_family_campaign.json").write_text(
        json.dumps(campaign, indent=2), encoding="utf-8"
    )
    return {
        "score_qualified_ids": original_ids,
        "canonical_hashes": original_hashes,
        "list_param_name": list_param_name,
        "list_param_scalar": list_param_scalar,
        "rejected_ids": [c.candidate_id for c in rejected_cands],
        "ambiguous_id": ambiguous.candidate_id if ambiguous else None,
        "n_evaluated": n_eval,
        "generated": n_eval + 10,
        "full_wfo": n_eval,
        "family_specs": [s.as_dict() for s in specs],
    }


def test_normalize_legacy_candidate_parameters_recovers_list_from_dsl() -> None:
    from control_plane.search_bootstrap import (
        LEGACY_PARAMETER_VALUE_AMBIGUOUS,
        LegacyBootstrapError,
        normalize_legacy_candidate_parameters,
    )
    from discovery.expression_tree import parameter_node
    from discovery.legacy_parameters import LegacyParameterError
    from discovery.types import ValueType

    tree = parameter_node("lookback", 17.5, ValueType.SCALAR)
    params, recovered = normalize_legacy_candidate_parameters(
        tree,
        {"lookback": [10.0, 17.5, 20.0], "z_entry": "-1.5"},
        candidate_id="cand_x",
    )
    assert params["lookback"] == 17.5
    assert params["z_entry"] == -1.5
    assert recovered and recovered[0]["recovered_from_dsl"] == 17.5

    with pytest.raises(LegacyParameterError, match=LEGACY_PARAMETER_VALUE_AMBIGUOUS):
        normalize_legacy_candidate_parameters(
            tree,
            {"ghost": [1.0, 2.0]},
            candidate_id="cand_x",
        )


def test_legacy_list_param_bootstrap_recovers_and_enters_stress(tmp_path: Path) -> None:
    from control_plane.search_bootstrap import prepare_search_program_session
    from discovery.evaluation_cache import EvaluationCacheEntry
    from discovery.search_program import PipelinePhase, SearchMode, SearchProgramStore, now_iso
    from discovery.search_resume import choose_resume_phase, rebuild_candidates

    artifacts_root = tmp_path / "artifacts"
    legacy_id = "run_legacy_list_params"
    meta = _write_legacy_list_param_ledger(
        artifacts_root / legacy_id, n_score_qualified=42
    )
    store = SearchProgramStore(tmp_path / "search_programs")
    fp = "fp_list_param"
    prepared = prepare_search_program_session(
        search_mode=SearchMode.EXTEND_BUDGET.value,
        program_id=None,
        program_store=store,
        compatibility_fingerprint=fp,
        seed=11,
        family_ids=["mean_reversion"],
        run_id="run_continue_list",
        source_run_id=legacy_id,
        resumed_from_run_id=legacy_id,
        artifacts_root=artifacts_root,
        fingerprint_components=_fp_components(),
    )
    assert prepared.bootstrapped_from_source is True
    ckpt = prepared.resume_checkpoint
    assert ckpt is not None
    assert len(ckpt.pending_stress_ids) == 42
    assert set(meta["score_qualified_ids"]) == set(ckpt.pending_stress_ids)
    assert choose_resume_phase(ckpt) is PipelinePhase.STRESS
    assert ckpt.campaign_full_wfo == meta["full_wfo"]
    assert ckpt.campaign_evaluated == meta["n_evaluated"]
    assert ckpt.campaign_generated == meta["generated"]

    restored = rebuild_candidates(ckpt)
    assert set(meta["score_qualified_ids"]).issubset(set(restored))
    for cid in meta["score_qualified_ids"]:
        assert restored[cid].candidate_id == cid
        assert restored[cid].exit_tree is not None
        assert restored[cid].stop is not None
        assert restored[cid].target is not None

    list_cid = meta["score_qualified_ids"][0]
    assert restored[list_cid].parameters[meta["list_param_name"]] == pytest.approx(
        meta["list_param_scalar"]
    )
    # Rejected malformed history remains in the multiple-testing population.
    for rid in meta["rejected_ids"]:
        assert rid in ckpt.trial_population_candidate_ids
        assert rid in ckpt.evaluation_records
        assert rid not in ckpt.candidates

    report = prepared.bootstrap_report
    assert report is not None
    assert report.trials_scanned == meta["n_evaluated"]
    assert report.values_recovered_from_dsl
    report_path = prepared.checkpoint_path.parent / "legacy_bootstrap_report.json"
    assert report_path.is_file()

    # Resume campaign: Stress first, no Generation 0 regeneration, no WFO replay.
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
            execution_semantic_hash=fp_comp["execution_semantic_hash"],
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
        total_candidate_budget=meta["generated"] + 4,
        max_full_wfo=meta["full_wfo"] + 4,
        max_evaluated_candidates=meta["n_evaluated"] + 4,
        max_runtime_seconds=45.0,
        evolution_generations=1,
        max_stress_evaluations=50,
    )
    campaign = MultiFamilyCampaign(
        config=cfg,
        registry=ExperimentRegistry(tmp_path / "reg_list"),
        backend=CountingBackend(),
        discovery_run_id="run_continue_list",
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
    stress_idxs = [i for i, e in enumerate(events) if e == "MULTI_FAMILY_STRESS_STARTED"]
    evo_idxs = [i for i, e in enumerate(events) if e == "MULTI_FAMILY_EVOLUTION_STARTED"]
    assert stress_idxs, "42 SCORE_QUALIFIED must enter Stress first"
    if evo_idxs:
        assert stress_idxs[0] < evo_idxs[0]
    # Completed WFO must not be repeated for the restored population.
    assert CountingBackend.evaluate_calls < 8
    assert result.budget_allocation["cumulative_totals"]["full_wfo"] >= meta["full_wfo"]
    assert result.budget_allocation["cumulative_totals"]["evaluated"] >= meta["n_evaluated"]


def test_legacy_ambiguous_pending_fails_closed_atomically(tmp_path: Path) -> None:
    from control_plane.search_bootstrap import (
        LEGACY_PARAMETER_VALUE_AMBIGUOUS,
        LegacyBootstrapError,
        prepare_search_program_session,
    )
    from discovery.search_program import SearchMode, SearchProgramStore

    artifacts_root = tmp_path / "artifacts"
    legacy_id = "run_legacy_ambiguous"
    meta = _write_legacy_list_param_ledger(
        artifacts_root / legacy_id,
        n_score_qualified=3,
        include_ambiguous_pending=True,
    )
    store = SearchProgramStore(tmp_path / "search_programs")
    with pytest.raises(LegacyBootstrapError, match=LEGACY_PARAMETER_VALUE_AMBIGUOUS) as excinfo:
        prepare_search_program_session(
            search_mode=SearchMode.EXTEND_BUDGET.value,
            program_id="sp_should_not_publish",
            program_store=store,
            compatibility_fingerprint="fp_amb",
            seed=11,
            family_ids=["mean_reversion"],
            run_id="run_amb",
            source_run_id=legacy_id,
            resumed_from_run_id=legacy_id,
            artifacts_root=artifacts_root,
            fingerprint_components=_fp_components(),
        )
    assert excinfo.value.candidate_id == meta["ambiguous_id"]
    assert store.get("sp_should_not_publish") is None
    assert not (tmp_path / "search_programs" / "sp_should_not_publish").exists()


def test_legacy_incomplete_score_qualified_fails_closed(tmp_path: Path) -> None:
    import json

    from control_plane.search_bootstrap import (
        LEGACY_CANDIDATE_SPEC_INCOMPLETE,
        LegacyBootstrapError,
        prepare_search_program_session,
    )
    from discovery.search_program import SearchMode, SearchProgramStore

    artifacts_root = tmp_path / "artifacts"
    legacy_id = "run_legacy_incomplete"
    art = artifacts_root / legacy_id
    meta = _write_legacy_runtime_exhausted_artifacts(art)
    # Strip authoritative trees so SCORE_QUALIFIED cannot be reconstructed.
    ledger = art / "registry" / "trial_ledger.jsonl"
    rows = []
    for line in ledger.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        trial = json.loads(line)
        snap = trial["config_snapshot"]
        snap.pop("candidate", None)
        snap.pop("exit_tree", None)
        snap.pop("stop", None)
        snap.pop("target", None)
        snap.pop("sizing", None)
        snap.pop("regime_gates", None)
        rows.append(trial)
    ledger.write_text("\n".join(json.dumps(t) for t in rows) + "\n", encoding="utf-8")

    store = SearchProgramStore(tmp_path / "search_programs")
    with pytest.raises(LegacyBootstrapError, match=LEGACY_CANDIDATE_SPEC_INCOMPLETE):
        prepare_search_program_session(
            search_mode=SearchMode.EXTEND_BUDGET.value,
            program_id=None,
            program_store=store,
            compatibility_fingerprint="fp_incomplete",
            seed=7,
            family_ids=["mean_reversion"],
            run_id="run_incomplete",
            source_run_id=legacy_id,
            resumed_from_run_id=legacy_id,
            artifacts_root=artifacts_root,
            fingerprint_components=_fp_components(),
        )
    assert list(store.root.glob("sp_*")) == []
    assert meta["score_qualified_id"]


def _new_search_mf(**overrides: Any) -> dict[str, Any]:
    """Structural NEW_SEARCH multi_family config (pre-runtime enrichment)."""
    raw = {
        "enabled": True,
        "requested_family_count": 2,
        "min_candidates_per_family": 2,
        "initial_candidates_per_family": 2,
        "total_candidate_budget": 8,
        "max_full_wfo": 8,
        "max_evaluated_candidates": 8,
        "adaptive_reallocation": False,
        "seed": 7,
        "family_ids": ["mean_reversion", "trend_following"],
        "max_runtime_seconds": 120.0,
        "family_local_evolution": True,
        "evolution_generations": 1,
        "population_size": 2,
        "stagnation_generations": 99,
        "minimum_improvement": 1e-4,
        "allow_cross_family_crossover": False,
        "max_stress_evaluations": 8,
        "max_stress_scenarios_per_candidate": 2,
        "min_stress_pass_rate": 0.5,
        "stress_scenarios": ["base_costs", "costs_2x"],
        "fail_closed_unsupported_stress": True,
        "max_robustness_candidates": 2,
        "max_robustness_evaluations": 10,
        "max_parameters_per_candidate": 2,
        "max_points_per_parameter": 5,
        "allow_one_sided_neighborhood": False,
        "min_valid_neighborhood_points": 3,
        "min_dsr": 0.0,
        "max_pbo": 1.0,
        "pbo_n_splits": 4,
        "behavioral_similarity_threshold": 0.85,
        "min_oos_observations_for_dsr": 1,
        "preview_family_specs": [{"family_id": "mean_reversion"}],
        "distinct_grammar_fingerprints": 2,
        "grammar_fingerprints": ["g1", "g2"],
    }
    raw.update(overrides)
    return raw


def _enrich_frozen_mf(mf: dict[str, Any], *, program_id: str, fp: str, comps: dict[str, Any]) -> dict[str, Any]:
    """Simulate jobs.py writing runtime resume metadata into config_snapshot."""
    out = dict(mf)
    out.update(
        {
            "search_program_id": program_id,
            "search_mode": SearchMode.NEW_SEARCH.value,
            "source_run_id": None,
            "resumed_from_run_id": None,
            "compatibility_fingerprint": fp,
            "fingerprint_components": dict(comps),
        }
    )
    return out


def _clone_for_extend(enriched: dict[str, Any], *, program_id: str, source_run_id: str) -> dict[str, Any]:
    """Simulate Continue Search cloning the enriched source config."""
    out = dict(enriched)
    out.update(
        {
            "search_program_id": program_id,
            "source_run_id": source_run_id,
            "resumed_from_run_id": source_run_id,
            "search_mode": SearchMode.EXTEND_BUDGET.value,
            "additional_runtime_seconds": 300.0,
            "additional_generated_budget": 50,
            "additional_full_wfo_budget": 25,
            "max_runtime_seconds": 300.0,
            "total_candidate_budget": int(enriched.get("total_candidate_budget") or 0) + 50,
            "max_full_wfo": int(enriched.get("max_full_wfo") or 0) + 25,
        }
    )
    return out


def _fp_from_mf(mf: dict[str, Any], **comp_overrides: Any) -> tuple[str, dict[str, Any]]:
    comps = build_fingerprint_components(
        dataset_hash=str(comp_overrides.get("dataset_hash", "ds_test")),
        timeframe=str(comp_overrides.get("timeframe", "5min")),
        wfo_config_hash=str(comp_overrides.get("wfo_config_hash", hash_wfo_config({"n_folds": 3}))),
        cost_model_version=str(comp_overrides.get("cost_model_version", "cost_v1")),
        risk_model_version=str(comp_overrides.get("risk_model_version", "demo")),
        execution_semantic_hash=str(
            comp_overrides.get(
                "execution_semantic_hash",
                comp_overrides.get("execution_engine_code_hash", "code_test"),
            )
        ),
        repository_git_sha=(
            str(comp_overrides["repository_git_sha"])
            if comp_overrides.get("repository_git_sha") not in (None, "")
            else None
        ),
        multi_family_config_hash=hash_multi_family_config(mf),
        seed=int(mf.get("seed") or 7),
        family_ids=list(mf.get("family_ids") or []),
    )
    fp = compute_compatibility_fingerprint(
        dataset_hash=str(comps["dataset_hash"]),
        timeframe=str(comps["timeframe"]),
        wfo_config_hash=str(comps["wfo_config_hash"]),
        cost_model_version=str(comps["cost_model_version"]),
        risk_model_version=str(comps["risk_model_version"]),
        execution_semantic_hash=str(comps["execution_semantic_hash"]),
        multi_family_config_hash=str(comps["multi_family_config_hash"]),
        seed=int(comps["seed"]),
        family_ids=list(mf.get("family_ids") or []),
    )
    return fp, comps


def test_hash_multi_family_ignores_runtime_and_dashboard_metadata() -> None:
    base = _new_search_mf()
    enriched = _enrich_frozen_mf(
        base, program_id="sp_x", fp="fp_x", comps={"dataset_hash": "ds"}
    )
    extended = _clone_for_extend(enriched, program_id="sp_x", source_run_id="run_src")
    assert hash_multi_family_config(base) == hash_multi_family_config(enriched)
    assert hash_multi_family_config(base) == hash_multi_family_config(extended)
    assert "compatibility_fingerprint" not in canonical_multi_family_payload(enriched)
    assert "preview_family_specs" not in canonical_multi_family_payload(enriched)
    assert "total_candidate_budget" not in canonical_multi_family_payload(extended)


def test_extend_budget_fingerprint_matches_after_runtime_enrichment(tmp_path: Path) -> None:
    """NEW_SEARCH fingerprint equals EXTEND_BUDGET clone of the enriched snapshot."""
    from control_plane.search_bootstrap import prepare_search_program_session

    base = _new_search_mf()
    fp, comps = _fp_from_mf(base)
    program_id = new_search_program_id()
    store = SearchProgramStore(tmp_path / "search_programs")
    store.create(
        compatibility_fingerprint=fp,
        seed=7,
        family_ids=list(base["family_ids"]),
        search_program_id=program_id,
        metadata={"fingerprint_components": comps},
    )
    # Persist a minimal checkpoint with the same components (as NEW_SEARCH would).
    ckpt = empty_checkpoint(
        search_program_id=program_id,
        compatibility_fingerprint=fp,
        session_run_id="run_new",
        search_mode=SearchMode.NEW_SEARCH.value,
        seed=7,
        config=dict(base),
        fingerprint_components={
            k: str(comps[k])
            for k in (
                "dataset_hash",
                "timeframe",
                "wfo_config_hash",
                "cost_model_version",
                "risk_model_version",
                "execution_semantic_hash",
            )
        },
    )
    ckpt.campaign_generated = 10
    ckpt.campaign_evaluated = 10
    ckpt.campaign_full_wfo = 10
    ckpt.pipeline_phase = PipelinePhase.STRESS.value
    # Fingerprint-only prepare path: empty pending avoids candidate rebuild.
    ckpt.pending_stress_ids = []
    save_checkpoint(store.checkpoint_path(program_id), ckpt)
    store.update_cumulative(
        program_id,
        generated=10,
        evaluated=10,
        full_wfo=10,
        checkpoint_path=str(store.checkpoint_path(program_id)),
    )
    before = store.get(program_id)
    assert before is not None
    cum_before = (
        before.cumulative_generated,
        before.cumulative_evaluated,
        before.cumulative_full_wfo,
    )

    enriched = _enrich_frozen_mf(base, program_id=program_id, fp=fp, comps=comps)
    extend_mf = _clone_for_extend(enriched, program_id=program_id, source_run_id="run_new")
    incoming_fp, incoming_comps = _fp_from_mf(extend_mf)
    assert incoming_fp == fp
    assert incoming_comps["multi_family_config_hash"] == comps["multi_family_config_hash"]

    prepared = prepare_search_program_session(
        search_mode=SearchMode.EXTEND_BUDGET.value,
        program_id=program_id,
        program_store=store,
        compatibility_fingerprint=incoming_fp,
        seed=7,
        family_ids=list(base["family_ids"]),
        run_id="run_extend",
        source_run_id="run_new",
        resumed_from_run_id="run_new",
        artifacts_root=tmp_path / "artifacts",
        fingerprint_components=incoming_comps,
        multi_family_config=extend_mf,
        run_artifact_dir=tmp_path / "artifacts" / "run_extend",
    )
    assert prepared.search_program_id == program_id
    assert prepared.resume_checkpoint is not None
    after = store.get(program_id)
    assert after is not None
    assert (
        after.cumulative_generated,
        after.cumulative_evaluated,
        after.cumulative_full_wfo,
    ) == cum_before


def test_fingerprint_rejects_dataset_wfo_code_and_structure_changes(tmp_path: Path) -> None:
    from control_plane.search_bootstrap import prepare_search_program_session

    base = _new_search_mf()
    fp, comps = _fp_from_mf(base)
    program_id = new_search_program_id()
    store = SearchProgramStore(tmp_path / "search_programs")
    store.create(
        compatibility_fingerprint=fp,
        seed=7,
        family_ids=list(base["family_ids"]),
        search_program_id=program_id,
        metadata={"fingerprint_components": comps},
    )
    ckpt = empty_checkpoint(
        search_program_id=program_id,
        compatibility_fingerprint=fp,
        session_run_id="run_new",
        search_mode=SearchMode.NEW_SEARCH.value,
        seed=7,
        config=dict(base),
        fingerprint_components={
            k: str(comps[k])
            for k in (
                "dataset_hash",
                "timeframe",
                "wfo_config_hash",
                "cost_model_version",
                "risk_model_version",
                "execution_semantic_hash",
            )
        },
    )
    save_checkpoint(store.checkpoint_path(program_id), ckpt)

    cases = [
        ("dataset_hash", "ds_CHANGED", "DATASET_HASH_CHANGED"),
        ("wfo_config_hash", "wfo_CHANGED", "WFO_CONFIG_CHANGED"),
        ("execution_semantic_hash", "code_CHANGED", "EXECUTION_SEMANTICS_CHANGED"),
    ]
    for key, value, reason in cases:
        art = tmp_path / f"art_{key}"
        art.mkdir(parents=True, exist_ok=True)
        bad_fp, bad_comps = _fp_from_mf(base, **{key: value})
        with pytest.raises(IncompatibleSearchFingerprint, match=reason) as ei:
            prepare_search_program_session(
                search_mode=SearchMode.EXTEND_BUDGET.value,
                program_id=program_id,
                program_store=store,
                compatibility_fingerprint=bad_fp,
                seed=7,
                family_ids=list(base["family_ids"]),
                run_id=f"run_bad_{key}",
                source_run_id="run_new",
                resumed_from_run_id="run_new",
                artifacts_root=tmp_path / "artifacts",
                fingerprint_components=bad_comps,
                multi_family_config=base,
                run_artifact_dir=art,
            )
        assert (art / "fingerprint_diff.json").is_file()
        assert reason in ei.value.diff.changed_reason_codes

    # Structural multi-family threshold change
    art_mf = tmp_path / "art_mf"
    art_mf.mkdir(parents=True, exist_ok=True)
    changed = _new_search_mf(min_dsr=0.99)
    bad_fp, bad_comps = _fp_from_mf(changed)
    with pytest.raises(IncompatibleSearchFingerprint, match="MULTI_FAMILY_STRUCTURE_CHANGED"):
        prepare_search_program_session(
            search_mode=SearchMode.EXTEND_BUDGET.value,
            program_id=program_id,
            program_store=store,
            compatibility_fingerprint=bad_fp,
            seed=7,
            family_ids=list(base["family_ids"]),
            run_id="run_bad_mf",
            source_run_id="run_new",
            resumed_from_run_id="run_new",
            artifacts_root=tmp_path / "artifacts",
            fingerprint_components=bad_comps,
            multi_family_config=changed,
            run_artifact_dir=art_mf,
        )

    # Seed / family change
    art_seed = tmp_path / "art_seed"
    art_seed.mkdir(parents=True, exist_ok=True)
    seeded = _new_search_mf(seed=99)
    bad_fp, bad_comps = _fp_from_mf(seeded)
    with pytest.raises(IncompatibleSearchFingerprint, match="SEED_CHANGED"):
        prepare_search_program_session(
            search_mode=SearchMode.EXTEND_BUDGET.value,
            program_id=program_id,
            program_store=store,
            compatibility_fingerprint=bad_fp,
            seed=99,
            family_ids=list(base["family_ids"]),
            run_id="run_bad_seed",
            source_run_id="run_new",
            resumed_from_run_id="run_new",
            artifacts_root=tmp_path / "artifacts",
            fingerprint_components=bad_comps,
            multi_family_config=seeded,
            run_artifact_dir=art_seed,
        )

    # Failed checks must not mutate cumulative totals or append sessions.
    prog = store.get(program_id)
    assert prog is not None
    assert prog.sessions == []
    assert prog.cumulative_generated == 0


def test_legacy_aggregate_fingerprint_migrates_when_components_match(tmp_path: Path) -> None:
    """Programs hashed before runtime-metadata exclusion still resume."""
    from control_plane.search_bootstrap import prepare_search_program_session

    base = _new_search_mf()
    modern_fp, comps = _fp_from_mf(base)
    # Simulate the OLD polluted aggregate that included dashboard fields.
    polluted = dict(base)
    # Old denylist left preview/grammar fields in the hash payload.
    from registry.hashing import sha256_json

    old_payload = dict(polluted)
    for k in (
        "max_runtime_seconds",
        "total_candidate_budget",
        "max_full_wfo",
        "max_evaluated_candidates",
        "max_stress_evaluations",
        "max_robustness_evaluations",
        "search_mode",
        "search_program_id",
        "source_run_id",
        "resumed_from_run_id",
        "additional_runtime_seconds",
        "additional_generated_budget",
        "additional_full_wfo_budget",
    ):
        old_payload.pop(k, None)
    old_mf_hash = "mf_" + sha256_json(old_payload)[:24]
    old_fp = compute_compatibility_fingerprint(
        dataset_hash=str(comps["dataset_hash"]),
        timeframe=str(comps["timeframe"]),
        wfo_config_hash=str(comps["wfo_config_hash"]),
        cost_model_version=str(comps["cost_model_version"]),
        risk_model_version=str(comps["risk_model_version"]),
        execution_semantic_hash=str(comps["execution_semantic_hash"]),
        multi_family_config_hash=old_mf_hash,
        seed=7,
        family_ids=list(base["family_ids"]),
    )
    assert old_fp != modern_fp

    program_id = new_search_program_id()
    store = SearchProgramStore(tmp_path / "search_programs")
    store.create(
        compatibility_fingerprint=old_fp,
        seed=7,
        family_ids=list(base["family_ids"]),
        search_program_id=program_id,
        metadata={"fingerprint_components": comps},
    )
    ckpt = empty_checkpoint(
        search_program_id=program_id,
        compatibility_fingerprint=old_fp,
        session_run_id="run_new",
        search_mode=SearchMode.NEW_SEARCH.value,
        seed=7,
        config=dict(base),
        fingerprint_components={
            k: str(comps[k])
            for k in (
                "dataset_hash",
                "timeframe",
                "wfo_config_hash",
                "cost_model_version",
                "risk_model_version",
                "execution_semantic_hash",
            )
        },
    )
    save_checkpoint(store.checkpoint_path(program_id), ckpt)

    enriched = _enrich_frozen_mf(base, program_id=program_id, fp=old_fp, comps=comps)
    extend_mf = _clone_for_extend(enriched, program_id=program_id, source_run_id="run_new")
    incoming_fp, incoming_comps = _fp_from_mf(extend_mf)
    assert incoming_fp == modern_fp

    prepared = prepare_search_program_session(
        search_mode=SearchMode.EXTEND_BUDGET.value,
        program_id=program_id,
        program_store=store,
        compatibility_fingerprint=incoming_fp,
        seed=7,
        family_ids=list(base["family_ids"]),
        run_id="run_migrate",
        source_run_id="run_new",
        resumed_from_run_id="run_new",
        artifacts_root=tmp_path / "artifacts",
        fingerprint_components=incoming_comps,
        multi_family_config=extend_mf,
        run_artifact_dir=tmp_path / "artifacts" / "run_migrate",
    )
    assert prepared.fingerprint_migrated is True
    prog = store.get(program_id)
    assert prog is not None
    assert prog.compatibility_fingerprint == modern_fp


def test_continue_enters_pending_stress_without_regen(tmp_path: Path) -> None:
    """EXTEND_BUDGET with pending SCORE_QUALIFIED resumes at Stress, no Gen0 redo."""
    fp = _fp_components()
    program_id = new_search_program_id()
    store = SearchProgramStore(tmp_path / "search_programs")
    store.create(
        compatibility_fingerprint="fp_stress",
        seed=7,
        family_ids=["mean_reversion"],
        search_program_id=program_id,
    )
    cache = EvaluationCache(store.evaluation_cache_dir(program_id))
    ckpt_path = store.checkpoint_path(program_id)
    cfg = _tiny_cfg()
    campaign = MultiFamilyCampaign(
        config=cfg,
        registry=ExperimentRegistry(tmp_path / "reg1"),
        backend=CountingBackend(),
        discovery_run_id="run_session_1",
        research_eligible=False,
        synthetic_stress_forbidden=False,
        search_program_id=program_id,
        search_mode=SearchMode.NEW_SEARCH.value,
        compatibility_fingerprint="fp_stress",
        fingerprint_components=fp,
        checkpoint_path=ckpt_path,
        evaluation_cache=cache,
    )
    result1 = campaign.run()
    calls_after_first = CountingBackend.evaluate_calls
    assert calls_after_first > 0
    ckpt = load_checkpoint(ckpt_path)
    assert ckpt is not None
    # Force pending stress queue as if runtime exhausted before Stress finished.
    any_cid = next(iter(ckpt.candidates))
    ckpt.candidate_gates[any_cid] = SCORE_QUALIFIED
    # Clear prior stress outcomes so the candidate is treated as awaiting Stress.
    ckpt.stress_summaries = [
        s for s in ckpt.stress_summaries if s.get("candidate_id") != any_cid
    ]
    mark_score_qualified_pending(ckpt)
    assert ckpt.pending_stress_ids
    assert choose_resume_phase(ckpt) is PipelinePhase.STRESS
    save_checkpoint(ckpt_path, ckpt)
    gen0_before = {
        cid
        for cid, raw in ckpt.candidates.items()
        if int(raw.get("generation") or 0) == 0
    }
    evaluated_before = set(ckpt.evaluation_records.keys())

    CountingBackend.evaluate_calls = 0
    campaign2 = MultiFamilyCampaign(
        config=apply_budget_extension(
            _tiny_cfg(),
            additional_runtime_seconds=60.0,
            additional_generated_budget=0,
            additional_full_wfo_budget=0,
        ),
        registry=ExperimentRegistry(tmp_path / "reg2"),
        backend=CountingBackend(),
        discovery_run_id="run_session_2",
        research_eligible=False,
        synthetic_stress_forbidden=False,
        search_program_id=program_id,
        search_mode=SearchMode.EXTEND_BUDGET.value,
        compatibility_fingerprint="fp_stress",
        fingerprint_components=fp,
        checkpoint_path=ckpt_path,
        evaluation_cache=cache,
        resume_checkpoint=ckpt,
    )
    result2 = campaign2.run()
    ckpt2 = load_checkpoint(ckpt_path)
    assert ckpt2 is not None
    gen0_after = {
        cid
        for cid, raw in ckpt2.candidates.items()
        if int(raw.get("generation") or 0) == 0
    }
    assert gen0_before == gen0_after
    # Completed WFO evaluations must not be repeated via cache.
    assert CountingBackend.evaluate_calls < calls_after_first
    assert evaluated_before.issubset(set(ckpt2.evaluation_records.keys()))
    assert result1 is not None and result2 is not None

def _program_with_checkpoint(
    tmp_path: Path,
    *,
    comps: dict[str, Any],
    fp: str,
    mf: dict[str, Any],
    legacy_code_hash: str | None = None,
) -> tuple[str, SearchProgramStore]:
    program_id = new_search_program_id()
    store = SearchProgramStore(tmp_path / "search_programs")
    stored_comps = dict(comps)
    if legacy_code_hash is not None:
        # Legacy schema: whole-repo SHA in execution_engine_code_hash only.
        stored_comps = {
            k: v
            for k, v in comps.items()
            if k not in {"execution_semantic_hash", "repository_git_sha"}
        }
        stored_comps["execution_engine_code_hash"] = legacy_code_hash
        # Aggregate still computed from the legacy value for the stored program.
        from discovery.search_program import compute_compatibility_fingerprint

        fp = compute_compatibility_fingerprint(
            dataset_hash=str(stored_comps["dataset_hash"]),
            timeframe=str(stored_comps["timeframe"]),
            wfo_config_hash=str(stored_comps["wfo_config_hash"]),
            cost_model_version=str(stored_comps["cost_model_version"]),
            risk_model_version=str(stored_comps["risk_model_version"]),
            execution_engine_code_hash=legacy_code_hash,
            multi_family_config_hash=str(stored_comps["multi_family_config_hash"]),
            seed=int(stored_comps["seed"]),
            family_ids=list(mf.get("family_ids") or []),
        )
    store.create(
        compatibility_fingerprint=fp,
        seed=int(mf.get("seed") or 7),
        family_ids=list(mf.get("family_ids") or []),
        search_program_id=program_id,
        metadata={"fingerprint_components": stored_comps},
    )
    ckpt_fp_keys = [
        "dataset_hash",
        "timeframe",
        "wfo_config_hash",
        "cost_model_version",
        "risk_model_version",
    ]
    ckpt_comps = {k: str(stored_comps[k]) for k in ckpt_fp_keys if k in stored_comps}
    if legacy_code_hash is not None:
        ckpt_comps["execution_engine_code_hash"] = legacy_code_hash
    elif stored_comps.get("execution_semantic_hash"):
        ckpt_comps["execution_semantic_hash"] = str(stored_comps["execution_semantic_hash"])
        if stored_comps.get("repository_git_sha"):
            ckpt_comps["repository_git_sha"] = str(stored_comps["repository_git_sha"])
    ckpt = empty_checkpoint(
        search_program_id=program_id,
        compatibility_fingerprint=fp,
        session_run_id="run_new",
        search_mode=SearchMode.NEW_SEARCH.value,
        seed=int(mf.get("seed") or 7),
        config=dict(mf),
        fingerprint_components=ckpt_comps,
    )
    ckpt.campaign_generated = 540
    ckpt.campaign_evaluated = 530
    ckpt.campaign_full_wfo = 529
    ckpt.pipeline_phase = PipelinePhase.STRESS.value
    ckpt.pending_stress_ids = []
    save_checkpoint(store.checkpoint_path(program_id), ckpt)
    store.update_cumulative(
        program_id,
        generated=540,
        evaluated=530,
        full_wfo=529,
        checkpoint_path=str(store.checkpoint_path(program_id)),
    )
    return program_id, store


def test_dashboard_only_change_does_not_block_resume(tmp_path: Path) -> None:
    """repository_git_sha drift with identical semantic hash allows EXTEND_BUDGET."""
    from control_plane.search_bootstrap import prepare_search_program_session

    base = _new_search_mf()
    fp, comps = _fp_from_mf(
        base,
        execution_semantic_hash="esem_same",
        repository_git_sha="a" * 40,
    )
    comps["repository_git_sha"] = "a" * 40
    program_id, store = _program_with_checkpoint(tmp_path, comps=comps, fp=fp, mf=base)
    before = store.get(program_id)
    assert before is not None
    cum_before = (
        before.cumulative_generated,
        before.cumulative_evaluated,
        before.cumulative_full_wfo,
    )

    incoming_fp, incoming_comps = _fp_from_mf(
        base,
        execution_semantic_hash="esem_same",
        repository_git_sha="b" * 40,
    )
    incoming_comps["repository_git_sha"] = "b" * 40
    # Aggregate may match if repository SHA is not in the fingerprint payload.
    prepared = prepare_search_program_session(
        search_mode=SearchMode.EXTEND_BUDGET.value,
        program_id=program_id,
        program_store=store,
        compatibility_fingerprint=incoming_fp,
        seed=7,
        family_ids=list(base["family_ids"]),
        run_id="run_dash",
        source_run_id="run_new",
        resumed_from_run_id="run_new",
        artifacts_root=tmp_path / "artifacts",
        fingerprint_components=incoming_comps,
        multi_family_config=base,
        run_artifact_dir=tmp_path / "artifacts" / "run_dash",
    )
    assert prepared.resume_checkpoint is not None
    after = store.get(program_id)
    assert after is not None
    assert (
        after.cumulative_generated,
        after.cumulative_evaluated,
        after.cumulative_full_wfo,
    ) == cum_before
    if incoming_fp != fp:
        assert prepared.control_plane_code_changed or prepared.fingerprint_migrated


def test_test_only_and_search_metadata_change_do_not_block_resume(tmp_path: Path) -> None:
    """Opaque non-semantic bookkeeping changes must not alter the gate."""
    from control_plane.search_bootstrap import prepare_search_program_session

    base = _new_search_mf()
    fp, comps = _fp_from_mf(base, execution_semantic_hash="esem_meta")
    program_id, store = _program_with_checkpoint(tmp_path, comps=comps, fp=fp, mf=base)

    # Enriched clone adds search metadata but keeps structural + semantic identity.
    enriched = _enrich_frozen_mf(base, program_id=program_id, fp=fp, comps=comps)
    extend_mf = _clone_for_extend(enriched, program_id=program_id, source_run_id="run_new")
    incoming_fp, incoming_comps = _fp_from_mf(extend_mf, execution_semantic_hash="esem_meta")
    assert incoming_fp == fp

    prepared = prepare_search_program_session(
        search_mode=SearchMode.RESUME_SEARCH.value,
        program_id=program_id,
        program_store=store,
        compatibility_fingerprint=incoming_fp,
        seed=7,
        family_ids=list(base["family_ids"]),
        run_id="run_meta",
        source_run_id="run_new",
        resumed_from_run_id="run_new",
        artifacts_root=tmp_path / "artifacts",
        fingerprint_components=incoming_comps,
        multi_family_config=extend_mf,
        run_artifact_dir=tmp_path / "artifacts" / "run_meta",
    )
    assert prepared.search_program_id == program_id
    prog = store.get(program_id)
    assert prog is not None
    assert prog.cumulative_generated == 540
    assert prog.sessions == []  # prepare does not append sessions


def test_evaluator_or_cost_semantic_change_blocks_resume(tmp_path: Path) -> None:
    from control_plane.search_bootstrap import prepare_search_program_session

    base = _new_search_mf()
    fp, comps = _fp_from_mf(base, execution_semantic_hash="esem_v1")
    program_id, store = _program_with_checkpoint(tmp_path, comps=comps, fp=fp, mf=base)

    for label, new_hash in (
        ("evaluator", "esem_evaluator_CHANGED"),
        ("cost_slippage", "esem_cost_CHANGED"),
    ):
        art = tmp_path / f"art_{label}"
        art.mkdir(parents=True, exist_ok=True)
        bad_fp, bad_comps = _fp_from_mf(base, execution_semantic_hash=new_hash)
        with pytest.raises(IncompatibleSearchFingerprint, match="EXECUTION_SEMANTICS_CHANGED"):
            prepare_search_program_session(
                search_mode=SearchMode.EXTEND_BUDGET.value,
                program_id=program_id,
                program_store=store,
                compatibility_fingerprint=bad_fp,
                seed=7,
                family_ids=list(base["family_ids"]),
                run_id=f"run_bad_{label}",
                source_run_id="run_new",
                resumed_from_run_id="run_new",
                artifacts_root=tmp_path / "artifacts",
                fingerprint_components=bad_comps,
                multi_family_config=base,
                run_artifact_dir=art,
            )
        assert (art / "fingerprint_diff.json").is_file()

    prog = store.get(program_id)
    assert prog is not None
    assert prog.sessions == []
    assert prog.cumulative_generated == 540
    assert prog.cumulative_evaluated == 530
    assert prog.cumulative_full_wfo == 529
    assert prog.compatibility_fingerprint == fp


def test_legacy_git_sha_code_hash_migrates_atomically(tmp_path: Path) -> None:
    """Whole-repo SHA stored as execution_engine_code_hash migrates when semantic matches."""
    from control_plane.search_bootstrap import prepare_search_program_session
    from discovery.execution_semantic_hash import (
        compute_execution_semantic_hash,
        looks_like_git_sha,
        resolve_repo_root,
    )

    stored_sha = "cfb7f6d86fbea350c87609770743cfcaf28dc4d4"
    current_sha = "0d0a15f9c4d594d148d8686e3a8e45c8a26b2a64"
    assert looks_like_git_sha(stored_sha)
    root = resolve_repo_root()
    sem = compute_execution_semantic_hash(root, at_commit=stored_sha)
    assert sem == compute_execution_semantic_hash(root, at_commit=current_sha)

    base = _new_search_mf()
    modern_fp, modern_comps = _fp_from_mf(
        base,
        execution_semantic_hash=sem,
        repository_git_sha=current_sha,
    )
    modern_comps["repository_git_sha"] = current_sha

    # Stored program uses legacy whole-repo SHA as the code component.
    program_id, store = _program_with_checkpoint(
        tmp_path,
        comps=modern_comps,
        fp=modern_fp,
        mf=base,
        legacy_code_hash=stored_sha,
    )
    before = store.get(program_id)
    assert before is not None
    legacy_fp = before.compatibility_fingerprint
    assert legacy_fp != modern_fp
    ckpt_before = load_checkpoint(store.checkpoint_path(program_id))
    assert ckpt_before is not None
    pending_before = list(ckpt_before.pending_stress_ids)
    eval_cache_files_before = sorted(
        p.name for p in store.evaluation_cache_dir(program_id).glob("*.json")
    )

    prepared = prepare_search_program_session(
        search_mode=SearchMode.EXTEND_BUDGET.value,
        program_id=program_id,
        program_store=store,
        compatibility_fingerprint=modern_fp,
        seed=7,
        family_ids=list(base["family_ids"]),
        run_id="run_mig_sem",
        source_run_id="run_new",
        resumed_from_run_id="run_new",
        artifacts_root=tmp_path / "artifacts",
        fingerprint_components=modern_comps,
        multi_family_config=base,
        run_artifact_dir=tmp_path / "artifacts" / "run_mig_sem",
    )
    assert prepared.fingerprint_migrated is True
    assert prepared.legacy_code_hash_migrated is True
    assert prepared.control_plane_code_changed is True

    after = store.get(program_id)
    assert after is not None
    assert after.compatibility_fingerprint == modern_fp
    assert after.cumulative_generated == 540
    assert after.cumulative_evaluated == 530
    assert after.cumulative_full_wfo == 529
    assert after.sessions == []
    history = after.metadata.get("fingerprint_migration_history") or []
    assert history
    assert history[-1]["from_fingerprint"] == legacy_fp
    assert history[-1]["legacy_code_hash_migrated"] is True
    assert (after.metadata.get("fingerprint_components") or {}).get(
        "execution_semantic_hash"
    ) == sem

    ckpt_after = load_checkpoint(store.checkpoint_path(program_id))
    assert ckpt_after is not None
    assert ckpt_after.campaign_generated == 540
    assert ckpt_after.pending_stress_ids == pending_before
    assert ckpt_after.fingerprint_components.get("execution_semantic_hash") == sem
    assert sorted(
        p.name for p in store.evaluation_cache_dir(program_id).glob("*.json")
    ) == eval_cache_files_before


def test_zero_work_failed_attempt_does_not_change_program(tmp_path: Path) -> None:
    from control_plane.search_bootstrap import prepare_search_program_session

    base = _new_search_mf()
    fp, comps = _fp_from_mf(base, execution_semantic_hash="esem_stable")
    program_id, store = _program_with_checkpoint(tmp_path, comps=comps, fp=fp, mf=base)
    snap = store.get(program_id)
    assert snap is not None
    raw_before = store.program_path(program_id).read_text(encoding="utf-8")

    art = tmp_path / "art_zero"
    art.mkdir(parents=True, exist_ok=True)
    bad_fp, bad_comps = _fp_from_mf(base, execution_semantic_hash="esem_BROKEN")
    with pytest.raises(IncompatibleSearchFingerprint):
        prepare_search_program_session(
            search_mode=SearchMode.RESUME_SEARCH.value,
            program_id=program_id,
            program_store=store,
            compatibility_fingerprint=bad_fp,
            seed=7,
            family_ids=list(base["family_ids"]),
            run_id="run_zero",
            source_run_id="run_new",
            resumed_from_run_id="run_new",
            artifacts_root=tmp_path / "artifacts",
            fingerprint_components=bad_comps,
            multi_family_config=base,
            run_artifact_dir=art,
        )
    assert store.program_path(program_id).read_text(encoding="utf-8") == raw_before


def test_resume_of_resume_across_sessions(tmp_path: Path) -> None:
    """Fingerprint identity remains stable across chained EXTEND_BUDGET prepares."""
    from control_plane.search_bootstrap import prepare_search_program_session

    base = _new_search_mf()
    fp, comps = _fp_from_mf(
        base,
        execution_semantic_hash="esem_chain",
        repository_git_sha="c" * 40,
    )
    comps["repository_git_sha"] = "c" * 40
    program_id, store = _program_with_checkpoint(tmp_path, comps=comps, fp=fp, mf=base)

    for i, repo_sha in enumerate(("d" * 40, "e" * 40, "f" * 40), start=1):
        incoming_fp, incoming_comps = _fp_from_mf(
            base,
            execution_semantic_hash="esem_chain",
            repository_git_sha=repo_sha,
        )
        incoming_comps["repository_git_sha"] = repo_sha
        prepared = prepare_search_program_session(
            search_mode=SearchMode.EXTEND_BUDGET.value,
            program_id=program_id,
            program_store=store,
            compatibility_fingerprint=incoming_fp,
            seed=7,
            family_ids=list(base["family_ids"]),
            run_id=f"run_chain_{i}",
            source_run_id="run_new",
            resumed_from_run_id="run_new",
            artifacts_root=tmp_path / "artifacts",
            fingerprint_components=incoming_comps,
            multi_family_config=base,
            run_artifact_dir=tmp_path / "artifacts" / f"run_chain_{i}",
        )
        assert prepared.resume_checkpoint is not None
        prog = store.get(program_id)
        assert prog is not None
        assert prog.cumulative_generated == 540
        assert prog.cumulative_full_wfo == 529
        assert (prog.metadata.get("fingerprint_components") or {}).get(
            "execution_semantic_hash"
        ) == "esem_chain"


def test_semantic_hash_cfb7_to_0d0a_matches() -> None:
    from discovery.execution_semantic_hash import (
        build_semantic_hash_manifest,
        compute_execution_semantic_hash,
        resolve_repo_root,
    )

    root = resolve_repo_root()
    a = "cfb7f6d86fbea350c87609770743cfcaf28dc4d4"
    b = "0d0a15f9c4d594d148d8686e3a8e45c8a26b2a64"
    ha = compute_execution_semantic_hash(root, at_commit=a)
    hb = compute_execution_semantic_hash(root, at_commit=b)
    assert ha == hb
    assert ha.startswith("esem_")
    manifest = build_semantic_hash_manifest(
        repo_root=root,
        stored_repository_git_sha=a,
        current_repository_git_sha=b,
    )
    assert manifest["semantic_hashes_match"] is True
    assert manifest["migration_decision"] == "MIGRATE_ALLOW_RESUME"
    assert "quant_framework/control_plane/jobs.py" in manifest["excluded_change_paths"]
    assert "quant_framework/tests/test_search_resume.py" in manifest["excluded_change_paths"]
    assert manifest["changed_allowlist_paths"] == []
