"""Regression: genuine RESEARCH_ELIGIBLE Alpha Miner must use event-driven WFO."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from control_plane.backend_resolution import (
    BackendResolutionError,
    resolve_alpha_miner_backend,
)
from control_plane.models import RunType
from control_plane.security import ConfigValidationError, CreateRunRequest
from data.acquisition.multiyear_orchestrator import (
    PROTECTED_MARCH_2024_ID,
    PROTECTED_YEARLY_2024_HASH,
    PROTECTED_YEARLY_2024_ID,
)
from data.catalog.catalog import DatasetCatalog, SMOKE_DATASET_ID
from data.catalog.silver_bar_resolution import (
    REAL_DATA_BACKEND_UNAVAILABLE,
    compatible_timeframes_for_entry,
    resolve_bid_ask_bars,
    strategy_allows_timeframe,
)
from discovery.evaluator import SyntheticOOSBackend
from discovery.event_wfo_backend import EventDrivenDiscoveryBackend
from discovery.search_budget import BudgetCounters, SearchBudget


YEARLY_ID = PROTECTED_YEARLY_2024_ID
YEARLY_HASH = PROTECTED_YEARLY_2024_HASH
MARCH_ID = PROTECTED_MARCH_2024_ID
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


@pytest.fixture(scope="module")
def yearly_entry():
    if not (ARCHIVE_CATALOG / "dataset_catalog.json").is_file():
        pytest.skip("2024 Dukascopy catalog not present")
    cat = DatasetCatalog(ARCHIVE_CATALOG)
    entry = cat.get(YEARLY_ID)
    if entry is None:
        pytest.skip("yearly 2024 dataset missing")
    return entry


class TestBackendGate:
    def test_synthetic_demo_smoke_may_use_probe(self) -> None:
        res = resolve_alpha_miner_backend(
            dataset_id=SMOKE_DATASET_ID,
            smoke_test=True,
            research_eligible=False,
            smoke_test_only=True,
        )
        assert res.evaluation_backend == "synthetic_oos_probe"
        assert res.is_full_event_wfo is False

    def test_genuine_2024_never_uses_synthetic(self, yearly_entry) -> None:
        res = resolve_alpha_miner_backend(
            dataset_id=yearly_entry.dataset_id,
            smoke_test=False,
            research_eligible=True,
            smoke_test_only=False,
            strategy_family="mean_reversion_vwap_bb",
            timeframe="1m",
        )
        assert res.evaluation_backend == "event_driven_wfo"
        assert res.requires_silver_bars is True
        with pytest.raises(BackendResolutionError) as ei:
            resolve_alpha_miner_backend(
                dataset_id=yearly_entry.dataset_id,
                smoke_test=False,
                research_eligible=True,
                smoke_test_only=False,
                budget_cfg={"evaluation_backend": "synthetic_oos_probe"},
                timeframe="1m",
            )
        assert REAL_DATA_BACKEND_UNAVAILABLE in str(ei.value)

    def test_no_silent_fallback_helper(self, yearly_entry) -> None:
        """Requesting synthetic on genuine data raises — never returns probe."""
        with pytest.raises(BackendResolutionError):
            resolve_alpha_miner_backend(
                dataset_id=yearly_entry.dataset_id,
                smoke_test=False,
                research_eligible=True,
                smoke_test_only=False,
                budget_cfg={"evaluation_backend": "synthetic"},
            )


class TestSilverResolution:
    def test_resolve_1m_bid_ask(self, yearly_entry) -> None:
        resolved = resolve_bid_ask_bars(yearly_entry, timeframe="1m", max_rows=1000)
        assert resolved.timeframe == "1min"
        assert "bid" in resolved.frame.columns and "ask" in resolved.frame.columns
        assert resolved.bid_hash
        assert resolved.ask_hash
        assert len(resolved.frame) == 1000

    def test_resolve_5m_bid_ask(self, yearly_entry) -> None:
        resolved = resolve_bid_ask_bars(yearly_entry, timeframe="5m", max_rows=500)
        assert resolved.timeframe == "5min"
        assert len(resolved.frame) == 500
        assert (resolved.frame["ask"] >= resolved.frame["bid"]).all()

    def test_compatible_timeframes_expose_1m_5m(self, yearly_entry) -> None:
        tfs = compatible_timeframes_for_entry(yearly_entry)
        assert "1m" in tfs and "5m" in tfs

    def test_mean_reversion_rejects_tick(self) -> None:
        assert strategy_allows_timeframe("mean_reversion_vwap_bb", "1m")
        assert strategy_allows_timeframe("mean_reversion_vwap_bb", "5m")
        assert not strategy_allows_timeframe("mean_reversion_vwap_bb", "tick")

    def test_create_run_rejects_tick_for_mr(self, yearly_entry, tmp_path: Path) -> None:
        cat = DatasetCatalog(ARCHIVE_CATALOG)
        req = CreateRunRequest(
            run_type=RunType.ALPHA_MINER,
            dataset=YEARLY_ID,
            symbols=[yearly_entry.symbols[-1] if yearly_entry.symbols else "USA500IDXUSD"],
            timeframe="tick",
            strategy_family="mean_reversion_vwap_bb",
            smoke_test=False,
            confirm_alpha_miner=True,
            cost_model_version=yearly_entry.compatible_cost_models[0],
            risk_profile=yearly_entry.compatible_risk_profiles[0],
            feature_set_version=yearly_entry.compatible_feature_sets[0],
        )
        with pytest.raises(ConfigValidationError):
            req.validate_against_catalog(cat)


class TestRealWFOPath:
    def test_event_driven_invoked_for_real_backend(self, yearly_entry, tmp_path: Path) -> None:
        resolved = resolve_bid_ask_bars(yearly_entry, timeframe="1m", max_rows=8000)
        from config.walk_forward_config import WalkForwardConfig
        from discovery.generator import CandidateGenerator

        backend = EventDrivenDiscoveryBackend(
            bars=resolved.frame,
            require_real_bars=True,
            asset_class="cfd",
            intraday_only=True,
            artifact_dir=tmp_path,
            silver_resolution=resolved.as_dict(),
            wfo_config=WalkForwardConfig(
                train_window_days=2,
                validation_window_days=1,
                step_forward_days=1,
                purge_gap_bars=1,
                embargo_gap_bars=1,
                bars_per_day=1440,
                max_folds=2,
                param_grid={"lookback": [10], "z_entry": [2.0]},
            ),
        )
        with patch(
            "discovery.event_wfo_backend.run_event_driven_wfo",
            wraps=__import__(
                "validation.event_driven_wfo", fromlist=["run_event_driven_wfo"]
            ).run_event_driven_wfo,
        ) as spy:
            cand = CandidateGenerator().seed_template_mean_reversion(seed=7)
            folds, train = backend.evaluate(cand)
        assert spy.called
        assert backend.is_full_event_wfo is True
        assert train.get("is_full_event_wfo") is True
        assert "synthetic_oos_probe" not in str(train.get("evaluation_path"))
        assert train.get("training_ranges")
        assert train.get("oos_ranges")
        assert "orders_count" in train and "fills_count" in train and "trades_count" in train
        arts = list((tmp_path / "wfo_artifacts").glob("*_wfo.json"))
        assert arts
        payload = json.loads(arts[0].read_text(encoding="utf-8"))
        assert payload["training_ranges"]
        assert payload["oos_ranges"]
        assert folds is not None
        assert train.get("signal_source") == "candidate_dsl_trees"
        assert train.get("evaluation_path") == "dsl_event_driven_wfo"

    def test_generated_dsl_candidate_trades_on_bounded_2024_slice(
        self, yearly_entry, tmp_path: Path
    ) -> None:
        """Real Silver BID/ASK (bounded): at least one generated DSL candidate must close OOS trades."""
        resolved = resolve_bid_ask_bars(yearly_entry, timeframe="1m", max_rows=12_000)
        from config.walk_forward_config import WalkForwardConfig
        from discovery.generator import CandidateGenerator

        assert "bid" in resolved.frame.columns and "ask" in resolved.frame.columns
        backend = EventDrivenDiscoveryBackend(
            bars=resolved.frame,
            require_real_bars=True,
            asset_class="cfd",
            intraday_only=True,
            artifact_dir=tmp_path,
            silver_resolution=resolved.as_dict(),
            wfo_config=WalkForwardConfig(
                train_window_days=2,
                validation_window_days=1,
                step_forward_days=1,
                purge_gap_bars=1,
                embargo_gap_bars=1,
                bars_per_day=1440,
                max_folds=2,
                param_grid={},
            ),
        )
        gen = CandidateGenerator()
        # Known-good generated seed from real-data funnel diagnosis
        cand = gen.generate(seed=41)
        folds, diag = backend.evaluate(cand)
        oos_trades = sum(f.n_trades for f in folds)
        oos_funnels = [
            f for f in (diag.get("fold_trade_funnels") or []) if f.get("phase") == "validation_oos"
        ]
        assert diag.get("signal_source") == "candidate_dsl_trees"
        assert diag.get("evaluation_path") == "dsl_event_driven_wfo"
        assert oos_trades >= 1, (
            f"expected real-data OOS trades, funnels={oos_funnels} totals="
            f"entry={diag.get('signals_entry_count')} fills={diag.get('fills_count')}"
        )
        assert any(int(f.get("entry_true_count", 0)) >= 1 for f in oos_funnels)
        assert any(int(f.get("fills", 0)) >= 1 for f in oos_funnels)
        assert any(int(f.get("positions_closed", 0)) >= 1 for f in oos_funnels)
        # Funnel schema required for ops diagnosis
        for f in oos_funnels:
            for key in (
                "entry_true_count",
                "exit_true_count",
                "orders_submitted",
                "fills",
                "positions_opened",
                "positions_closed",
                "forced_window_closes",
                "rejected_orders",
            ):
                assert key in f

    def test_seed_template_trades_on_bounded_2024_slice(self, yearly_entry, tmp_path: Path) -> None:
        resolved = resolve_bid_ask_bars(yearly_entry, timeframe="1m", max_rows=12_000)
        from config.walk_forward_config import WalkForwardConfig
        from discovery.generator import CandidateGenerator

        backend = EventDrivenDiscoveryBackend(
            bars=resolved.frame,
            require_real_bars=True,
            asset_class="cfd",
            intraday_only=True,
            artifact_dir=tmp_path,
            silver_resolution=resolved.as_dict(),
            wfo_config=WalkForwardConfig(
                train_window_days=2,
                validation_window_days=1,
                step_forward_days=1,
                purge_gap_bars=1,
                embargo_gap_bars=1,
                bars_per_day=1440,
                max_folds=2,
                param_grid={},
            ),
        )
        cand = CandidateGenerator().seed_template_mean_reversion(seed=42)
        folds, diag = backend.evaluate(cand)
        assert sum(f.n_trades for f in folds) >= 1
        assert diag.get("signal_source") == "candidate_dsl_trees"
        with pytest.raises(RuntimeError) as ei:
            EventDrivenDiscoveryBackend(require_real_bars=True, bars=None)
        assert REAL_DATA_BACKEND_UNAVAILABLE in str(ei.value)

    def test_all_precheck_rejected_terminal(self, tmp_path: Path) -> None:
        from control_plane.alpha_results import build_alpha_miner_report
        from discovery.search_controller import DiscoveryRunResult
        from registry.experiment_registry import ExperimentRegistry, TrialStatus

        reg = ExperimentRegistry(tmp_path / "reg")
        for i in range(3):
            reg.create_trial(
                candidate_id=f"c{i}",
                lineage_id=f"l{i}",
                strategy_family="mean_reversion_vwap_bb",
                parameters={},
                config_snapshot={"is_full_event_wfo": True, "backend_kind": "event_driven_wfo"},
                system_version="t",
                data_hash="d",
                random_seed=i,
                cost_model_version="cost_v1",
                code_hash="c",
                gross_metrics={},
                net_metrics={},
                ranking_score=None,
                rejection_reason="precheck:too_complex",
                trial_status=TrialStatus.FAILED,
                failure_reason="too_complex",
            )
        result = DiscoveryRunResult(
            discovery_run_id="d1",
            budget_id="b1",
            stop_reason="max_generated_candidates",
            generations=0,
            evaluated=0,
            registered_trials=3,
            rankings=[],
            finalists=[],
            promoted=[],
            portfolio_pool={},
            clusters=[],
            reproducible_fingerprint="x",
        )
        report = build_alpha_miner_report(
            registry=reg,
            counters=BudgetCounters(generated=3),
            budget=SearchBudget(max_generated_candidates=3, population_size=2),
            result=result,
            elapsed_seconds=1.0,
            evaluation_backend="event_driven_wfo",
        )
        assert report["discovery_result"] == "ALL_CANDIDATES_PRECHECK_REJECTED"
        assert report["terminal_reason"] == "ALL_CANDIDATES_PRECHECK_REJECTED"
        assert report["full_wfo_evaluations"] == 0


class TestStagnationWarmup:
    def test_stagnation_respects_minimum_generations(self) -> None:
        budget = SearchBudget(
            stagnation_limit=1,
            stagnation_generations=1,
            minimum_generations_before_stagnation=4,
            population_size=4,
            max_generated_candidates=100,
            max_evaluated_candidates=100,
            max_runtime_seconds=9999,
        )
        counters = BudgetCounters(stagnant_generations=5, generations_completed=2)
        assert counters.stop_reason(budget) is None
        counters.generations_completed = 4
        assert counters.stop_reason(budget) == "stagnation_limit"


class TestImmutabilityAndAcquisition:
    def test_yearly_2024_id_and_hash_unchanged(self, yearly_entry) -> None:
        assert yearly_entry.dataset_id == YEARLY_ID
        h = yearly_entry.normalized_hash or yearly_entry.raw_hash
        assert h == YEARLY_HASH

    def test_march_2024_unchanged(self) -> None:
        cat = DatasetCatalog(ARCHIVE_CATALOG)
        march = cat.get(MARCH_ID)
        assert march is not None
        assert march.dataset_id == MARCH_ID
        assert march.row_count == 644633 or march.row_count > 600_000

    def test_multiyear_acquisition_still_running(self) -> None:
        if not MULTIYEAR_PARENT.is_file():
            pytest.skip("multiyear parent_job.json missing")
        parent = json.loads(MULTIYEAR_PARENT.read_text(encoding="utf-8"))
        state = str(parent.get("state") or "")
        assert state in {"RUNNING", "PAUSED", "COMPLETED"}
        # Must not have been rewritten to include protected diagnostic year 2024.
        years = parent.get("years") or []
        assert 2024 not in years
        pid = parent.get("pid")
        if state in {"RUNNING", "PAUSED"} and pid:
            alive = False
            try:
                os.kill(int(pid), 0)
                alive = True
            except OSError:
                # Windows: os.kill(pid, 0) is unreliable — fall back to tasklist
                import subprocess

                out = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {int(pid)}"],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                alive = str(pid) in (out.stdout or "")
            assert alive, f"acquisition pid {pid} not running"


class TestSyntheticBackendLabel:
    def test_synthetic_backend_kind(self) -> None:
        b = SyntheticOOSBackend()
        assert b.backend_kind == "synthetic_oos_probe"
        assert b.is_full_event_wfo is False
