"""Regression: REGIME_GATE typing + INVALID_DSL_TYPE campaign resilience."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from control_plane.alpha_results import build_alpha_miner_report
from data.acquisition.multiyear_orchestrator import (
    PROTECTED_MARCH_2024_ID,
    PROTECTED_YEARLY_2024_HASH,
    PROTECTED_YEARLY_2024_ID,
)
from data.catalog.catalog import DatasetCatalog
from discovery.candidate import build_candidate
from discovery.crossover import Crossover
from discovery.evaluator import CandidateEvaluator, EvalOutcome
from discovery.expression_tree import (
    DSLValidationError,
    constant_node,
    feature_node,
    op_node,
)
from discovery.generator import CandidateGenerator
from discovery.grammar import Grammar
from discovery.mutation import Mutator
from discovery.operators import OperatorId
from discovery.search_budget import SearchBudget
from discovery.search_controller import SearchController
from discovery.typecheck import INVALID_DSL_TYPE, check_ast_types
from discovery.types import CreationMethod, ValueType
from registry.experiment_registry import ExperimentRegistry


ARCHIVE_CATALOG = (
    Path(__file__).resolve().parents[1]
    / "data_import"
    / "dukascopy_archive"
    / "dataset_catalog"
)
MULTIYEAR_PARENT = (
    Path(__file__).resolve().parents[1]
    / "data_import"
    / "dukascopy_archive"
    / "multiyear"
    / "parent_job.json"
)


def _bool_child():
    return op_node(
        OperatorId.LESS_THAN,
        feature_node("price.rolling_z_20", ValueType.ZSCORE),
        constant_node(0.0),
    )


class TestRegimeGateContract:
    def test_rejects_scalar_child(self) -> None:
        with pytest.raises(DSLValidationError) as ei:
            op_node(OperatorId.REGIME_GATE, _bool_child(), constant_node(1.0))
        assert "SCALAR" in str(ei.value)
        assert "REGIME" in str(ei.value)

    def test_accepts_regime_child(self) -> None:
        node = op_node(
            OperatorId.REGIME_GATE,
            _bool_child(),
            feature_node("regime.trend_state", ValueType.REGIME),
        )
        check_ast_types(node)
        assert node.children[1].value_type is ValueType.REGIME

    def test_accepts_boolean_child(self) -> None:
        node = op_node(OperatorId.REGIME_GATE, _bool_child(), _bool_child())
        check_ast_types(node)
        assert node.children[1].value_type is ValueType.BOOLEAN


class TestGenerationMutationCrossover:
    def test_initial_generation_type_safe(self) -> None:
        gen = CandidateGenerator()
        failures = 0
        for seed in range(500):
            try:
                entry, _pattern = gen.generate_entry(seed)
                check_ast_types(entry)
            except Exception:  # noqa: BLE001
                failures += 1
        assert failures == 0

    def test_mutation_preserves_types(self) -> None:
        parent = CandidateGenerator().seed_template_mean_reversion(seed=1)
        mut = Mutator()
        for seed in range(40):
            child = mut.mutate(parent, seed=seed)
            check_ast_types(child.entry_tree)
            if child.exit_tree is not None:
                check_ast_types(child.exit_tree)

    def test_crossover_preserves_types(self) -> None:
        gen = CandidateGenerator()
        a = gen.seed_template_mean_reversion(seed=2)
        b = gen.generate(seed=3)
        cx = Crossover()
        for seed in range(20):
            c1, c2 = cx.crossover(a, b, seed=seed)
            check_ast_types(c1.entry_tree)
            check_ast_types(c2.entry_tree)


class TestInvalidDslCampaign:
    def test_invalid_becomes_invalid_dsl_type_and_recorded(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(tmp_path / "reg")
        budget = SearchBudget(
            max_generated_candidates=8,
            max_evaluated_candidates=8,
            max_full_wfo_evaluations=2,
            population_size=2,
            elite_count=1,
            stagnation_limit=8,
            minimum_generations_before_stagnation=2,
            max_runtime_seconds=30,
            max_candidates_per_family=20,
            max_candidates_per_complexity_tier=20,
            max_candidates_per_feature_family=20,
        )
        ctrl = SearchController(registry=reg, budget=budget, seed=1, discovery_run_id="dsl_inv")

        rec = ctrl.evaluator.register_invalid_dsl(
            DSLValidationError(
                "REGIME_GATE child[1] type SCALAR not in ['REGIME', 'BOOLEAN']"
            ),
            seed=99,
            operation="generate",
        )
        assert rec.outcome is EvalOutcome.INVALID_DSL_TYPE
        assert rec.rejection_reason == INVALID_DSL_TYPE
        assert ctrl.counters.invalid == 1
        reasons = [t.rejection_reason for t in reg.all_trials()]
        assert INVALID_DSL_TYPE in reasons
        assert rec.meta.get("operator") == "REGIME_GATE" or "REGIME_GATE" in str(
            rec.meta.get("message")
        )

        # Campaign continues after recording invalid — replacement generate works
        cand = ctrl._safe_generate(7)
        assert cand is not None
        assert ctrl.counters.invalid == 1

    def test_invalid_does_not_terminate_campaign(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(tmp_path / "reg2")
        budget = SearchBudget(
            max_generated_candidates=6,
            max_evaluated_candidates=6,
            max_full_wfo_evaluations=2,
            population_size=2,
            elite_count=1,
            stagnation_generations=8,
            minimum_generations_before_stagnation=2,
            max_runtime_seconds=20,
            max_candidates_per_family=20,
            max_candidates_per_complexity_tier=20,
            max_candidates_per_feature_family=20,
        )
        ctrl = SearchController(registry=reg, budget=budget, seed=5)
        real = ctrl._safe_generate
        n = {"i": 0}

        def flaky(seed, prefer_seed_template=False):
            n["i"] += 1
            if n["i"] == 2 and not prefer_seed_template:
                ctrl.evaluator.register_invalid_dsl(
                    DSLValidationError(
                        "REGIME_GATE child[1] type SCALAR not in ['REGIME', 'BOOLEAN']"
                    ),
                    seed=seed,
                    operation="generate",
                )
                return None
            return real(seed, prefer_seed_template=prefer_seed_template)

        with patch.object(ctrl, "_safe_generate", side_effect=flaky):
            result = ctrl.run()
        assert result.stop_reason is not None
        assert "REGIME_GATE child" not in result.stop_reason
        assert ctrl.counters.invalid >= 1

    def test_replacement_while_budget_remains(self, tmp_path: Path) -> None:
        reg = ExperimentRegistry(tmp_path / "reg3")
        budget = SearchBudget(
            max_generated_candidates=10,
            max_evaluated_candidates=10,
            population_size=3,
            elite_count=1,
            max_runtime_seconds=25,
            max_candidates_per_family=30,
            max_candidates_per_complexity_tier=30,
            max_candidates_per_feature_family=30,
            minimum_generations_before_stagnation=2,
            stagnation_generations=8,
        )
        ctrl = SearchController(registry=reg, budget=budget, seed=9)
        real = ctrl._safe_generate
        n = {"i": 0}

        def flaky(seed, prefer_seed_template=False):
            n["i"] += 1
            if n["i"] in {2, 3} and not prefer_seed_template:
                ctrl.evaluator.register_invalid_dsl(
                    DSLValidationError(
                        "REGIME_GATE child[1] type SCALAR not in ['REGIME', 'BOOLEAN']"
                    ),
                    seed=seed,
                    operation="generate",
                )
                return None
            return real(seed, prefer_seed_template=prefer_seed_template)

        with patch.object(ctrl, "_safe_generate", side_effect=flaky):
            ctrl.run()
        assert ctrl.counters.invalid >= 2
        assert ctrl.counters.generated >= 3  # replacements filled population
    def test_uncaught_dsl_maps_to_software_failure_not_nqc(self, tmp_path: Path) -> None:
        from control_plane.jobs import execute_run
        from control_plane.models import RunRecord, RunState, RunType, build_run_record
        from control_plane.events import NullEventSink

        run = build_run_record(
            run_type=RunType.ALPHA_MINER,
            system_version="test",
            config_snapshot={
                "dataset": "synthetic_demo",
                "smoke_test": True,
                "dataset_research_eligible": False,
                "dataset_smoke_test_only": True,
                "search_budget": {"max_generated_candidates": 4, "population_size": 2},
            },
            artifact_root=str(tmp_path / "artifacts"),
            random_seed=1,
        )
        Path(run.artifact_dir).mkdir(parents=True, exist_ok=True)
        sink = NullEventSink()

        def boom(*_a, **_k):
            raise DSLValidationError(
                "REGIME_GATE child[1] type SCALAR not in ['REGIME', 'BOOLEAN']"
            )

        with patch("control_plane.jobs.run_alpha_miner_job", side_effect=boom):
            out = execute_run(run, sink, cancel_check=lambda: False)
        assert out.state is RunState.FAILED
        assert out.terminal_reason == "DSL_TYPE_VALIDATION_FAILURE"
        summary = json.loads(
            (Path(run.artifact_dir) / "run_summary.json").read_text(encoding="utf-8")
        )
        assert summary["discovery_result"] == "SOFTWARE_FAILURE"
        assert summary["discovery_result"] != "NO_QUALIFIED_CANDIDATE"

    def test_completed_campaign_may_return_nqc(self, tmp_path: Path) -> None:
        from discovery.search_controller import DiscoveryRunResult

        reg = ExperimentRegistry(tmp_path / "reg4")
        # Register only precheck rejects so report is ALL_CANDIDATES or NQC path
        budget = SearchBudget(max_generated_candidates=2, population_size=2)
        result = DiscoveryRunResult(
            discovery_run_id="d",
            budget_id="b",
            stop_reason="max_generated_candidates",
            generations=1,
            evaluated=0,
            registered_trials=0,
            rankings=[],
            finalists=[],
            promoted=[],
            portfolio_pool={},
            clusters=[],
            reproducible_fingerprint="x",
        )
        report = build_alpha_miner_report(
            registry=reg,
            counters=ctrl_counters(),
            budget=budget,
            result=result,
            elapsed_seconds=1.0,
            evaluation_backend="event_driven_wfo",
        )
        assert report["discovery_result"] in {
            "NO_QUALIFIED_CANDIDATE",
            "ALL_CANDIDATES_PRECHECK_REJECTED",
        }
        assert report["discovery_result"] != "SOFTWARE_FAILURE"


def ctrl_counters():
    from discovery.search_budget import BudgetCounters

    return BudgetCounters()


class TestImmutabilityAcquisition:
    def test_yearly_hash_unchanged(self) -> None:
        cat = DatasetCatalog(ARCHIVE_CATALOG)
        e = cat.get(PROTECTED_YEARLY_2024_ID)
        assert e is not None
        assert (e.normalized_hash or e.raw_hash) == PROTECTED_YEARLY_2024_HASH

    def test_march_unchanged(self) -> None:
        cat = DatasetCatalog(ARCHIVE_CATALOG)
        m = cat.get(PROTECTED_MARCH_2024_ID)
        assert m is not None
        assert m.row_count == 644633

    def test_acquisition_running(self) -> None:
        import os
        import subprocess

        if not MULTIYEAR_PARENT.is_file():
            pytest.skip("no multiyear parent")
        parent = json.loads(MULTIYEAR_PARENT.read_text(encoding="utf-8"))
        state = str(parent.get("state") or "")
        assert state in {"RUNNING", "PAUSED", "COMPLETED"}
        assert 2024 not in (parent.get("years") or [])
        pid = parent.get("pid")
        # A completed immutable acquisition is healthy and has no live worker.
        # Only active lifecycle states are required to prove the PID is alive.
        if state in {"RUNNING", "PAUSED"} and pid:
            out = subprocess.run(
                ["tasklist", "/FI", f"PID eq {int(pid)}"],
                capture_output=True,
                text=True,
                check=False,
            )
            assert str(pid) in (out.stdout or "")


class TestBackendStillReal:
    def test_no_synthetic_fallback_gate(self) -> None:
        from control_plane.backend_resolution import (
            BackendResolutionError,
            resolve_alpha_miner_backend,
        )
        from data.catalog.silver_bar_resolution import REAL_DATA_BACKEND_UNAVAILABLE

        res = resolve_alpha_miner_backend(
            dataset_id=PROTECTED_YEARLY_2024_ID,
            smoke_test=False,
            research_eligible=True,
            smoke_test_only=False,
            timeframe="1m",
        )
        assert res.evaluation_backend == "event_driven_wfo"
        with pytest.raises(BackendResolutionError) as ei:
            resolve_alpha_miner_backend(
                dataset_id=PROTECTED_YEARLY_2024_ID,
                smoke_test=False,
                research_eligible=True,
                smoke_test_only=False,
                budget_cfg={"evaluation_backend": "synthetic_oos_probe"},
                timeframe="1m",
            )
        assert REAL_DATA_BACKEND_UNAVAILABLE in str(ei.value)
