"""Job runners — research work executed outside the HTTP request."""

from __future__ import annotations

import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from control_plane.events import (
    EventSeverity,
    EventType,
    ResearchEvent,
    ResearchEventSink,
)
from control_plane.models import RunRecord, RunState, RunType
from discovery.generator import CandidateGenerator
from discovery.multi_family_campaign import (
    MultiFamilyCampaign,
    family_campaign_from_config,
)
from discovery.search_budget import BudgetCounters, resolve_bucket_caps, search_budget_from_config
from discovery.search_controller import DiscoveryRunResult, SearchController
from registry.experiment_registry import ExperimentRegistry
from registry.hashing import sha256_json


CancelCheck = Callable[[], bool]


def _emit(
    sink: ResearchEventSink,
    run: RunRecord,
    et: EventType,
    message: str,
    *,
    stage: str = "",
    progress: float | None = None,
    payload: dict[str, Any] | None = None,
    severity: EventSeverity = EventSeverity.INFO,
) -> None:
    sink.emit(
        ResearchEvent.create(
            run_id=run.run_id,
            event_type=et,
            message=message,
            stage=stage or run.current_stage,
            progress=progress if progress is not None else run.progress_pct,
            payload=payload,
            severity=severity,
        )
    )


def _write_manifest(run: RunRecord, files: list[str]) -> None:
    run.artifact_manifest = list(dict.fromkeys(run.artifact_manifest + files))
    art = Path(run.artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    (art / "artifact_manifest.json").write_text(
        json.dumps({"files": run.artifact_manifest}, indent=2), encoding="utf-8"
    )


def _finalize_completed(run: RunRecord, summary: dict[str, Any]) -> None:
    run.state = RunState.COMPLETED
    run.completed_at = datetime.now(tz=timezone.utc).isoformat()
    run.progress_pct = 100.0
    run.summary = summary
    run.software_success = True
    run.terminal_reason = "completed"
    if run.started_at:
        try:
            t0 = datetime.fromisoformat(run.started_at)
            run.elapsed_seconds = (datetime.now(tz=timezone.utc) - t0).total_seconds()
        except ValueError:
            pass


def run_research_demo(
    run: RunRecord,
    sink: ResearchEventSink,
    *,
    cancel_check: CancelCheck,
) -> RunRecord:
    """Lightweight research demo with progress events (does not use live)."""
    art = Path(run.artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    registry = ExperimentRegistry(art / "registry")

    _emit(sink, run, EventType.RUN_STARTED, "Research demo started", stage="init", progress=5)
    run.current_stage = "data"
    if cancel_check():
        run.state = RunState.CANCELLED
        run.terminal_reason = "cancelled"
        _emit(sink, run, EventType.RUN_CANCELLED, "Cancelled before data", severity=EventSeverity.WARNING)
        return run

    rng = np.random.default_rng(run.random_seed)
    n = 400
    idx = pd.date_range("2024-01-02 08:30", periods=n, freq="5min", tz="America/Chicago")
    px = 4800 + np.cumsum(rng.normal(0, 0.5, n))
    bars = pd.DataFrame(
        {
            "timestamp": idx,
            "open": px,
            "high": px + 0.5,
            "low": px - 0.5,
            "close": px,
            "volume": rng.integers(100, 1000, n).astype(float),
        }
    )
    _emit(sink, run, EventType.DATA_LOADED, f"Loaded {len(bars)} bars", stage="data", progress=15)
    _emit(sink, run, EventType.DATA_VALIDATED, "Synthetic data accepted", stage="data", progress=20)

    hold = float(run.config_snapshot.get("hold_seconds") or 0.0)
    if hold > 0:
        import time

        deadline = time.monotonic() + hold
        run.current_stage = "hold"
        while time.monotonic() < deadline:
            if cancel_check():
                run.state = RunState.CANCELLED
                run.terminal_reason = "cancelled"
                run.completed_at = datetime.now(tz=timezone.utc).isoformat()
                _emit(
                    sink,
                    run,
                    EventType.RUN_CANCELLED,
                    "Cancelled during hold",
                    severity=EventSeverity.WARNING,
                )
                return run
            time.sleep(0.05)
            run.progress_pct = min(40.0, 20.0 + 20.0 * (1.0 - max(0.0, deadline - time.monotonic()) / hold))
            _emit(sink, run, EventType.PROGRESS, "Holding for cooperative cancel", stage="hold", progress=run.progress_pct)

    # Simple equity path for honesty demo — intentionally may be negative
    rets = rng.normal(-0.00005, 0.001, n)
    equity = 100_000 * np.cumprod(1.0 + rets)
    total_return_pct = float((equity[-1] / equity[0] - 1.0) * 100.0)
    run.total_return_pct = total_return_pct
    run.profitability_result = (
        "POSITIVE" if total_return_pct > 0 else ("NEGATIVE" if total_return_pct < 0 else "FLAT")
    )

    eq_path = art / "equity_curve.csv"
    pd.DataFrame({"equity": equity}, index=idx).to_csv(eq_path)
    _write_manifest(run, ["equity_curve.csv", "artifact_manifest.json", "run_summary.json"])
    _emit(
        sink,
        run,
        EventType.ARTIFACT_CREATED,
        "equity_curve.csv",
        stage="artifacts",
        progress=70,
        payload={"file": "equity_curve.csv"},
    )

    # Registry sample trials (include a reject)
    for i in range(3):
        registry.create_trial(
            candidate_id=f"{run.run_id}_c{i}",
            lineage_id=f"{run.run_id}_l{i}",
            strategy_family=run.config_snapshot.get("strategy_family", "demo"),
            parameters={"i": i},
            config_snapshot=run.config_snapshot,
            system_version=run.system_version,
            data_hash="demo",
            random_seed=run.random_seed,
            cost_model_version=run.cost_model_version,
            code_hash="control_plane",
            ranking_score=0.1 * i if i < 2 else None,
            rejection_reason="precheck:demo" if i == 2 else None,
        )
        run.generated_count += 1
        if i == 2:
            run.rejected_count += 1
            _emit(
                sink,
                run,
                EventType.CANDIDATE_REJECTED,
                "Demo reject kept visible",
                stage="registry",
                payload={"candidate_id": f"{run.run_id}_c{i}"},
            )
        else:
            run.evaluated_count += 1
            run.qualified_count += 1

    if cancel_check():
        run.state = RunState.CANCELLED
        run.terminal_reason = "cancelled"
        _emit(sink, run, EventType.RUN_CANCELLED, "Cancelled", severity=EventSeverity.WARNING)
        return run

    summary = {
        "starting_equity": 100_000.0,
        "final_equity": float(equity[-1]),
        "total_return_pct": total_return_pct,
        "software_success": True,
        "software_execution_status": "SUCCESS",
        "discovery_result": "NOT_APPLICABLE",
        "statistical_validation": "INSUFFICIENT_DATA",
        "statistical_result": "INSUFFICIENT_DATA",
        "profitability_result": (
            "POSITIVE" if total_return_pct > 0 else ("NEGATIVE" if total_return_pct < 0 else "FLAT")
        ),
        "portfolio_result": "NOT_APPLICABLE",
        "vault_result": "NOT_SUBMITTED",
        "total_return_status": "AVAILABLE",
        "registry_trials": len(registry.all_trials()),
        "rejected_visible": len(registry.rejected_trials()),
        "wfo_ranking_source": "validation_oos",
    }
    (art / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    run.statistical_validation = summary["statistical_result"]
    run.profitability_result = summary["profitability_result"]
    run.vault_result = summary["vault_result"]
    _finalize_completed(run, summary)
    _emit(
        sink,
        run,
        EventType.RUN_COMPLETED,
        f"Completed — return {total_return_pct:.2f}% ({run.profitability_result})",
        stage="done",
        progress=100,
        payload=summary,
    )
    return run


def run_alpha_miner_job(
    run: RunRecord,
    sink: ResearchEventSink,
    *,
    cancel_check: CancelCheck,
    on_update: Callable[[RunRecord], None] | None = None,
) -> RunRecord:
    from control_plane.alpha_results import build_alpha_miner_report
    from control_plane.backend_resolution import (
        BackendResolutionError,
        resolve_alpha_miner_backend,
    )
    from data.catalog.silver_bar_resolution import (
        REAL_DATA_BACKEND_UNAVAILABLE,
        SilverResolutionError,
        normalize_timeframe,
        resolve_bid_ask_bars,
    )

    art = Path(run.artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    from control_plane.runtime_provenance import (
        assert_dsl_signal_source,
        collect_runtime_provenance,
    )

    provenance = collect_runtime_provenance(
        repo_hint=Path(__file__).resolve().parents[2]
    )
    (art / "runtime_provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str), encoding="utf-8"
    )
    _emit(
        sink,
        run,
        EventType.RUN_STARTED,
        "Alpha Miner started",
        stage="miner",
        progress=3,
        payload={"runtime_provenance": provenance},
    )

    registry = ExperimentRegistry(art / "registry")
    budget_cfg = dict(run.config_snapshot.get("search_budget") or {})
    budget = search_budget_from_config(budget_cfg)
    bucket_caps = resolve_bucket_caps(
        budget_cfg, max_generated_candidates=budget.max_generated_candidates
    )
    canary_cfg = dict(run.config_snapshot.get("ui_canary") or {})
    _emit(sink, run, EventType.RUN_STARTED, "Alpha Miner budget ready", stage="miner", progress=5)

    if cancel_check():
        run.state = RunState.CANCELLED
        run.terminal_reason = "cancelled"
        _emit(sink, run, EventType.RUN_CANCELLED, "Cancelled", severity=EventSeverity.WARNING)
        return run

    event_map = {
        "GENERATION_STARTED": EventType.GENERATION_STARTED,
        "CANDIDATE_GENERATED": EventType.CANDIDATE_GENERATED,
        "CANDIDATE_REJECTED": EventType.CANDIDATE_REJECTED,
        "CANDIDATE_EVALUATION_STARTED": EventType.CANDIDATE_EVALUATION_STARTED,
        "CANDIDATE_EVALUATED": EventType.CANDIDATE_EVALUATED,
        "WFO_FOLD_STARTED": EventType.WFO_FOLD_STARTED,
        "WFO_FOLD_COMPLETED": EventType.WFO_FOLD_COMPLETED,
        "BEHAVIORAL_CLUSTER_UPDATED": EventType.BEHAVIORAL_CLUSTER_UPDATED,
        "FINALIST_SELECTED": EventType.FINALIST_SELECTED,
    }
    t0 = datetime.now(tz=timezone.utc)
    dataset_id = str(run.config_snapshot.get("dataset") or "synthetic_demo")
    smoke_test = bool(run.config_snapshot.get("smoke_test"))
    research_eligible = bool(run.config_snapshot.get("dataset_research_eligible"))
    smoke_test_only = bool(run.config_snapshot.get("dataset_smoke_test_only"))
    strategy_family = str(
        run.config_snapshot.get("strategy_family") or "mean_reversion_vwap_bb"
    )
    timeframe = str(run.config_snapshot.get("timeframe") or "1m")
    silver_meta: dict[str, Any] = {}

    try:
        resolution = resolve_alpha_miner_backend(
            dataset_id=dataset_id,
            smoke_test=smoke_test,
            research_eligible=research_eligible,
            smoke_test_only=smoke_test_only,
            budget_cfg=budget_cfg,
            strategy_family=strategy_family,
            timeframe=timeframe,
        )
    except BackendResolutionError as exc:
        run.state = RunState.FAILED
        run.software_success = False
        run.terminal_reason = REAL_DATA_BACKEND_UNAVAILABLE
        summary = {
            "terminal_reason": REAL_DATA_BACKEND_UNAVAILABLE,
            "error": str(exc),
            "evaluation_backend": None,
            "discovery_result": REAL_DATA_BACKEND_UNAVAILABLE,
            "software_execution_status": "FAILED",
        }
        (art / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        _emit(
            sink,
            run,
            EventType.RUN_FAILED,
            str(exc),
            severity=EventSeverity.ERROR,
            payload=summary,
        )
        return run

    backend_name = resolution.evaluation_backend
    if resolution.requires_silver_bars:
        from config.walk_forward_config import WalkForwardConfig
        from data.catalog.catalog import DatasetCatalog
        from discovery.event_wfo_backend import EventDrivenDiscoveryBackend

        # Resolve catalog entry from registered lake (immutable; no rebuild)
        catalog_roots = [
            Path("data_import/dukascopy_archive/dataset_catalog"),
            Path(__file__).resolve().parents[1]
            / "data_import"
            / "dukascopy_archive"
            / "dataset_catalog",
        ]
        entry = None
        for root in catalog_roots:
            if (root / "dataset_catalog.json").is_file():
                entry = DatasetCatalog(root).get(dataset_id)
                if entry is not None:
                    break
        if entry is None:
            # Control-plane catalog copy (may lack lake_root) — still try
            cp_root = Path(run.artifact_dir).resolve().parents[1] / "dataset_catalog"
            if (cp_root / "dataset_catalog.json").is_file():
                entry = DatasetCatalog(cp_root).get(dataset_id)
        if entry is None:
            run.state = RunState.FAILED
            run.software_success = False
            run.terminal_reason = REAL_DATA_BACKEND_UNAVAILABLE
            msg = f"{REAL_DATA_BACKEND_UNAVAILABLE}: catalog entry {dataset_id!r} not found"
            (art / "run_summary.json").write_text(
                json.dumps({"terminal_reason": REAL_DATA_BACKEND_UNAVAILABLE, "error": msg}, indent=2),
                encoding="utf-8",
            )
            _emit(sink, run, EventType.RUN_FAILED, msg, severity=EventSeverity.ERROR)
            return run
        try:
            canary_max_rows = canary_cfg.get("max_rows")
            max_rows = int(canary_max_rows) if canary_max_rows is not None else None
            resolved = resolve_bid_ask_bars(entry, timeframe=timeframe, max_rows=max_rows)
        except (SilverResolutionError, ValueError) as exc:
            run.state = RunState.FAILED
            run.software_success = False
            run.terminal_reason = REAL_DATA_BACKEND_UNAVAILABLE
            summary = {
                "terminal_reason": REAL_DATA_BACKEND_UNAVAILABLE,
                "error": str(exc),
                "evaluation_backend": None,
                "discovery_result": REAL_DATA_BACKEND_UNAVAILABLE,
                "software_execution_status": "FAILED",
            }
            (art / "run_summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
            _emit(sink, run, EventType.RUN_FAILED, str(exc), severity=EventSeverity.ERROR)
            return run
        silver_meta = resolved.as_dict()
        (art / "silver_resolution.json").write_text(
            json.dumps(silver_meta, indent=2, default=str), encoding="utf-8"
        )
        tf_norm = normalize_timeframe(timeframe)
        bars_per_day = 1440 if tf_norm == "1min" else 288
        wfo_cfg_raw = dict(run.config_snapshot.get("wfo") or {})
        wfo_cfg = WalkForwardConfig(
            train_window_days=int(wfo_cfg_raw.get("train_window_days", 5)),
            validation_window_days=int(wfo_cfg_raw.get("validation_window_days", 2)),
            step_forward_days=int(wfo_cfg_raw.get("step_forward_days", 5)),
            purge_gap_bars=int(wfo_cfg_raw.get("purge_gap_bars", 1)),
            embargo_gap_bars=int(wfo_cfg_raw.get("embargo_gap_bars", 1)),
            bars_per_day=int(wfo_cfg_raw.get("bars_per_day", bars_per_day)),
            max_folds=int(wfo_cfg_raw.get("max_folds", 4)),
            # DSL candidates carry their own parameters; do not inject MR stub grid.
            param_grid=dict(wfo_cfg_raw.get("param_grid") or {}),
        )
        try:
            backend = EventDrivenDiscoveryBackend(
                bars=resolved.frame,
                wfo_config=wfo_cfg,
                require_real_bars=True,
                asset_class="cfd" if entry.asset_class == "cfd" else "futures",
                intraday_only=bool(entry.intraday_only),
                artifact_dir=art,
                silver_resolution=silver_meta,
            )
        except Exception as exc:  # noqa: BLE001
            run.state = RunState.FAILED
            run.software_success = False
            run.terminal_reason = REAL_DATA_BACKEND_UNAVAILABLE
            summary = {
                "terminal_reason": REAL_DATA_BACKEND_UNAVAILABLE,
                "error": str(exc),
                "discovery_result": REAL_DATA_BACKEND_UNAVAILABLE,
                "software_execution_status": "FAILED",
            }
            (art / "run_summary.json").write_text(
                json.dumps(summary, indent=2, default=str), encoding="utf-8"
            )
            _emit(sink, run, EventType.RUN_FAILED, str(exc), severity=EventSeverity.ERROR)
            return run
        backend_name = "event_driven_wfo"
        _emit(
            sink,
            run,
            EventType.DATA_LOADED,
            f"Silver BID/ASK loaded rows={silver_meta.get('row_count')}",
            stage="data",
            progress=8,
            payload=silver_meta,
        )
    else:
        from discovery.evaluator import SyntheticOOSBackend

        backend = SyntheticOOSBackend()
        backend_name = "synthetic_oos_probe"

    # Fail fast unless every full WFO evaluation reports DSL signal_source.
    if getattr(backend, "is_full_event_wfo", False) and hasattr(backend, "evaluate"):
        _orig_backend_evaluate = backend.evaluate

        def _backend_evaluate_with_signal_gate(candidate):  # type: ignore[no-untyped-def]
            folds, train_diag = _orig_backend_evaluate(candidate)
            arts = dict(getattr(backend, "last_run_artifacts") or {})
            src = arts.get("signal_source") or (train_diag or {}).get("signal_source")
            assert_dsl_signal_source(src, where=f"candidate={candidate.candidate_id}")
            if isinstance(train_diag, dict):
                train_diag = {
                    **train_diag,
                    "runtime_provenance": provenance,
                    "signal_source": src,
                }
            return folds, train_diag

        backend.evaluate = _backend_evaluate_with_signal_gate  # type: ignore[method-assign]

    multi_cfg = family_campaign_from_config(run.config_snapshot.get("multi_family"))
    campaign_result = None
    aggregated_records: list[Any] = []

    # Mutable holder so progress_hook can update counters from either path.
    class _CounterProxy:
        generated = 0
        evaluated = 0

    counter_proxy = _CounterProxy()

    def progress_hook(name: str, payload: dict[str, Any]) -> None:
        et = event_map.get(name)
        # Allow multi-family progress even without a mapped EventType.
        if et is None and name not in {
            "MULTI_FAMILY_STARTED",
            "FAMILY_PHASE_COMPLETED",
        }:
            return
        msg_name = name
        if name == "FINALIST_SELECTED" and backend_name != "event_driven_wfo":
            msg_name = "SHORTLIST_UPDATED"
            payload = {**payload, "label": "TOP_RANKED_UNVALIDATED"}
        gen_cap = max(1, budget.max_generated_candidates)
        if multi_cfg is not None:
            gen_cap = max(1, multi_cfg.total_candidate_budget)
        progress = min(95.0, 10.0 + 80.0 * (counter_proxy.generated / gen_cap))
        run.generated_count = counter_proxy.generated
        run.evaluated_count = counter_proxy.evaluated
        run.rejected_count = len(registry.rejected_trials())
        run.qualified_count = len(
            [t for t in registry.accepted_trials() if t.ranking_score is not None]
        )
        run.progress_pct = progress
        run.current_stage = msg_name.lower()
        run.current_message = (
            f"{msg_name}: {payload.get('candidate_id', payload.get('family_id', payload.get('generation', '')))}"
        )
        elapsed = (datetime.now(tz=timezone.utc) - t0).total_seconds()
        run.elapsed_seconds = elapsed
        if et is not None:
            _emit(
                sink,
                run,
                et,
                run.current_message,
                stage=run.current_stage,
                progress=progress,
                payload={
                    **payload,
                    "generated": run.generated_count,
                    "evaluated": run.evaluated_count,
                    "evaluation_backend": backend_name,
                },
            )
        if on_update is not None:
            on_update(run)

    try:
        if multi_cfg is not None:
            from discovery.evaluation_cache import EvaluationCache
            from discovery.search_checkpoint import load_checkpoint
            from discovery.search_program import (
                INCOMPATIBLE_FINGERPRINT,
                SearchMode,
                SearchProgramStore,
                SearchSessionRecord,
                assert_fingerprint_compatible,
                compute_compatibility_fingerprint,
                hash_multi_family_config,
                hash_wfo_config,
                new_search_program_id,
                now_iso,
            )
            from discovery.search_resume import apply_budget_extension
            from registry.hashing import sha256_json

            search_mode = str(
                run.config_snapshot.get("search_mode")
                or (run.config_snapshot.get("multi_family") or {}).get("search_mode")
                or SearchMode.NEW_SEARCH.value
            ).upper()
            mf_raw = dict(run.config_snapshot.get("multi_family") or {})
            source_run_id = run.config_snapshot.get("source_run_id") or mf_raw.get(
                "source_run_id"
            )
            resumed_from_run_id = run.config_snapshot.get("resumed_from_run_id") or mf_raw.get(
                "resumed_from_run_id"
            )
            program_id = (
                run.config_snapshot.get("search_program_id")
                or mf_raw.get("search_program_id")
            )
            add_runtime = float(
                run.config_snapshot.get("additional_runtime_seconds")
                or mf_raw.get("additional_runtime_seconds")
                or 0.0
            )
            add_gen = int(
                run.config_snapshot.get("additional_generated_budget")
                or mf_raw.get("additional_generated_budget")
                or 0
            )
            add_wfo = int(
                run.config_snapshot.get("additional_full_wfo_budget")
                or mf_raw.get("additional_full_wfo_budget")
                or 0
            )
            if search_mode in {
                SearchMode.EXTEND_BUDGET.value,
                SearchMode.RESUME_SEARCH.value,
            }:
                apply_budget_extension(
                    multi_cfg,
                    additional_runtime_seconds=add_runtime,
                    additional_generated_budget=add_gen,
                    additional_full_wfo_budget=add_wfo,
                )
            # Re-align search_budget caps after possible extension.
            budget = budget.with_overrides(
                max_generated_candidates=multi_cfg.total_candidate_budget,
                max_evaluated_candidates=int(
                    multi_cfg.max_evaluated_candidates or multi_cfg.total_candidate_budget
                ),
                max_full_wfo_evaluations=multi_cfg.max_full_wfo,
                min_oos_trades=multi_cfg.min_oos_trades,
                min_oos_trades_per_fold=multi_cfg.min_oos_trades_per_fold,
                max_oos_drawdown=multi_cfg.max_oos_drawdown,
                max_runtime_seconds=multi_cfg.max_runtime_seconds,
                stagnation_generations=multi_cfg.stagnation_generations,
                stagnation_limit=multi_cfg.stagnation_generations,
                population_size=multi_cfg.population_size,
            )

            dataset_hash = str(
                silver_meta.get("content_hash")
                or silver_meta.get("dataset_hash")
                or run.config_snapshot.get("dataset")
                or "synthetic_demo"
            )
            wfo_hash = hash_wfo_config(run.config_snapshot.get("wfo"))
            mf_hash = hash_multi_family_config(mf_raw)
            code_hash = str(
                provenance.get("git_commit_sha")
                or provenance.get("code_hash")
                or "local"
            )
            fp_components = {
                "dataset_hash": dataset_hash,
                "timeframe": str(timeframe),
                "wfo_config_hash": wfo_hash,
                "cost_model_version": str(
                    run.config_snapshot.get("cost_model_version") or run.cost_model_version
                ),
                "risk_model_version": str(
                    run.config_snapshot.get("risk_profile") or "demo"
                ),
                "execution_engine_code_hash": code_hash,
            }
            compatibility_fp = compute_compatibility_fingerprint(
                dataset_hash=fp_components["dataset_hash"],
                timeframe=fp_components["timeframe"],
                wfo_config_hash=fp_components["wfo_config_hash"],
                cost_model_version=fp_components["cost_model_version"],
                risk_model_version=fp_components["risk_model_version"],
                execution_engine_code_hash=fp_components["execution_engine_code_hash"],
                multi_family_config_hash=mf_hash,
                seed=int(multi_cfg.seed),
                family_ids=list(multi_cfg.family_ids) if multi_cfg.family_ids else None,
            )

            program_root = Path(run.artifact_dir).resolve().parents[1] / "search_programs"
            program_store = SearchProgramStore(program_root)
            resume_ckpt = None
            if search_mode == SearchMode.NEW_SEARCH.value or not program_id:
                program_id = program_id or new_search_program_id()
                program_store.create(
                    compatibility_fingerprint=compatibility_fp,
                    seed=int(multi_cfg.seed),
                    family_ids=list(multi_cfg.family_ids) if multi_cfg.family_ids else None,
                    search_program_id=program_id,
                    metadata={"created_by_run_id": run.run_id},
                )
            else:
                existing = program_store.get(str(program_id))
                if existing is None:
                    raise RuntimeError(f"unknown search_program_id={program_id!r}")
                assert_fingerprint_compatible(
                    existing.compatibility_fingerprint,
                    compatibility_fp,
                    mode=SearchMode(search_mode),
                )
                resume_ckpt = load_checkpoint(program_store.checkpoint_path(str(program_id)))
                if resume_ckpt is None and source_run_id:
                    # Bootstrap from prior run artifacts when checkpoint missing.
                    from control_plane.search_bootstrap import (
                        bootstrap_checkpoint_from_run,
                    )

                    art_root = Path(run.artifact_dir).resolve().parent
                    src = art_root / str(source_run_id)
                    resume_ckpt = bootstrap_checkpoint_from_run(
                        src,
                        search_program_id=str(program_id),
                        compatibility_fingerprint=compatibility_fp,
                        session_run_id=run.run_id,
                        search_mode=search_mode,
                        fingerprint_components=fp_components,
                        source_run_id=str(source_run_id),
                        resumed_from_run_id=str(resumed_from_run_id or source_run_id),
                    )
                if resume_ckpt is None and search_mode != SearchMode.NEW_SEARCH.value:
                    raise RuntimeError(
                        f"MISSING_SEARCH_CHECKPOINT: program={program_id!r} "
                        f"mode={search_mode}"
                    )

            program_store.append_session(
                str(program_id),
                SearchSessionRecord(
                    run_id=run.run_id,
                    search_mode=search_mode,
                    source_run_id=str(source_run_id) if source_run_id else None,
                    resumed_from_run_id=(
                        str(resumed_from_run_id) if resumed_from_run_id else None
                    ),
                    started_at=now_iso(),
                ),
            )
            # Freeze search_program metadata into config snapshot for dashboard.
            mf_raw = dict(run.config_snapshot.get("multi_family") or {})
            mf_raw.update(
                {
                    "search_program_id": program_id,
                    "search_mode": search_mode,
                    "source_run_id": source_run_id,
                    "resumed_from_run_id": resumed_from_run_id,
                    "compatibility_fingerprint": compatibility_fp,
                    "fingerprint_components": fp_components,
                }
            )
            run.config_snapshot["multi_family"] = mf_raw
            run.config_snapshot["search_program_id"] = program_id
            run.config_snapshot["search_mode"] = search_mode
            run.config_hash = "cfg_" + sha256_json(run.config_snapshot)[:24]
            if on_update is not None:
                on_update(run)

            eval_cache = EvaluationCache(program_store.evaluation_cache_dir(str(program_id)))
            ckpt_path = program_store.checkpoint_path(str(program_id))

            campaign = MultiFamilyCampaign(
                config=multi_cfg,
                registry=registry,
                backend=backend,
                system_version=run.system_version,
                discovery_run_id=run.run_id,
                progress_hook=progress_hook,
                research_eligible=bool(research_eligible),
                search_program_id=str(program_id),
                search_mode=search_mode,
                compatibility_fingerprint=compatibility_fp,
                fingerprint_components=fp_components,
                checkpoint_path=ckpt_path,
                evaluation_cache=eval_cache,
                resume_checkpoint=resume_ckpt,
                source_run_id=str(source_run_id) if source_run_id else None,
                resumed_from_run_id=(
                    str(resumed_from_run_id) if resumed_from_run_id else None
                ),
                reevaluate_invalidate_cache=(
                    search_mode == SearchMode.REEVALUATE_FROZEN_CANDIDATES.value
                ),
            )

            def _campaign_progress(name: str, payload: dict[str, Any]) -> None:
                # Refresh proxy counters from registry growth.
                counter_proxy.generated = len(registry.all_trials())
                counter_proxy.evaluated = len(
                    [t for t in registry.all_trials() if t.ranking_score is not None or t.rejection_reason]
                )
                progress_hook(name, payload)

            campaign.progress_hook = _campaign_progress
            try:
                campaign_result = campaign.run()
            except ValueError as exc:
                if INCOMPATIBLE_FINGERPRINT in str(exc):
                    run.state = RunState.FAILED
                    run.software_success = False
                    run.terminal_reason = INCOMPATIBLE_FINGERPRINT
                    _emit(
                        sink,
                        run,
                        EventType.RUN_FAILED,
                        str(exc),
                        severity=EventSeverity.ERROR,
                    )
                    return run
                raise
            # Persist cumulative program totals after session.
            if campaign._live_checkpoint is not None:  # noqa: SLF001
                program_store.update_cumulative(
                    str(program_id),
                    generated=campaign._live_checkpoint.campaign_generated,  # noqa: SLF001
                    evaluated=campaign._live_checkpoint.campaign_evaluated,  # noqa: SLF001
                    full_wfo=campaign._live_checkpoint.campaign_full_wfo,  # noqa: SLF001
                    checkpoint_path=str(ckpt_path),
                )
            # Aggregate counters / discovery result for the existing report builder.
            ctrl_counters = BudgetCounters()
            all_records: list[Any] = []
            rankings: list[dict[str, Any]] = []
            finalists: list[str] = []
            clusters: list[dict[str, Any]] = []
            research_shortlist: list[Any] = []
            for dr in campaign_result.discovery_results:
                rankings.extend(dr.rankings)
                finalists.extend(dr.finalists)
                clusters.extend(dr.clusters)
                research_shortlist.extend(dr.research_shortlist)
            if campaign_result.clusters and not clusters:
                clusters = list(campaign_result.clusters)
            if campaign_result.research_shortlist and not research_shortlist:
                research_shortlist = [
                    e.as_dict() if hasattr(e, "as_dict") else e
                    for e in campaign_result.research_shortlist
                ]
            for st in campaign_result.family_stats:
                ctrl_counters.generated += int(st.generated)
                ctrl_counters.unique_generated += int(st.generated)
                ctrl_counters.evaluated += int(st.evaluated)
                ctrl_counters.full_wfo += int(st.full_wfo)
            ctrl_counters.runtime_seconds = 0.0
            counter_proxy.generated = ctrl_counters.generated
            counter_proxy.evaluated = ctrl_counters.evaluated
            result = DiscoveryRunResult(
                discovery_run_id=run.run_id,
                budget_id=budget.budget_id,
                stop_reason=campaign_result.aggregated_stop_reason,
                generations=len(campaign_result.discovery_results),
                evaluated=ctrl_counters.evaluated,
                registered_trials=len(registry.all_trials()),
                rankings=sorted(
                    rankings,
                    key=lambda r: float(r.get("fitness") or float("-inf")),
                    reverse=True,
                ),
                finalists=list(dict.fromkeys(finalists)),
                promoted=[],
                portfolio_pool={"members": [], "size": 0},
                clusters=clusters,
                reproducible_fingerprint=campaign_result.reproducible_fingerprint,
                pipeline_level=campaign_result.pipeline_level,
                post_wfo_pipeline_complete=False,
                research_shortlist=research_shortlist,
                vault_candidates=[],
                paper_candidates=[],
            )
            # Reconstruct evaluation records from registry enrichment path — empty ok.
            aggregated_records = all_records
            ctrl = None  # type: ignore[assignment]
            report_counters = ctrl_counters
        else:
            ctrl = SearchController(
                registry=registry,
                budget=budget,
                seed=run.random_seed,
                system_version=run.system_version,
                discovery_run_id=run.run_id,
                progress_hook=None,
                canary_generate_seed=(
                    int(canary_cfg["generate_seed"])
                    if canary_cfg.get("generate_seed") is not None
                    else None
                ),
                skip_seed_template=bool(
                    canary_cfg.get("skip_seed_template") or canary_cfg.get("enabled")
                ),
            )
            ctrl.research_eligible = bool(research_eligible)
            ctrl.synthetic_stress_forbidden = bool(research_eligible)
            ctrl.bind_evaluation_backend(backend, research_eligible=bool(research_eligible))

            def _single_progress(name: str, payload: dict[str, Any]) -> None:
                counter_proxy.generated = ctrl.counters.generated
                counter_proxy.evaluated = ctrl.counters.evaluated
                progress_hook(name, payload)

            ctrl.progress_hook = _single_progress
            ctrl.evaluator.progress_hook = _single_progress
            result = ctrl.run()
            report_counters = ctrl.counters
            aggregated_records = ctrl.evaluator.records()
    except RuntimeError as exc:
        fail_reasons = (
            "SIGNAL_SOURCE_MISMATCH",
            "FAMILY_GRAMMAR_COLLAPSE",
            "FAMILY_LABEL_COLLAPSE",
            "duplicate FamilySpec",
        )
        if not any(r in str(exc) for r in fail_reasons):
            raise
        run.state = RunState.FAILED
        run.software_success = False
        terminal = (
            "SIGNAL_SOURCE_MISMATCH"
            if "SIGNAL_SOURCE_MISMATCH" in str(exc)
            else "MULTI_FAMILY_FAILURE"
        )
        run.terminal_reason = terminal
        fail_summary = {
            "terminal_reason": terminal,
            "error": str(exc),
            "runtime_provenance": provenance,
            "software_execution_status": "FAILED",
            "discovery_result": "SOFTWARE_FAILURE",
        }
        (art / "run_summary.json").write_text(
            json.dumps(fail_summary, indent=2, default=str), encoding="utf-8"
        )
        _emit(sink, run, EventType.RUN_FAILED, str(exc), severity=EventSeverity.ERROR, payload=fail_summary)
        return run
    elapsed = (datetime.now(tz=timezone.utc) - t0).total_seconds()
    report_counters.runtime_seconds = max(report_counters.runtime_seconds, elapsed)

    report = build_alpha_miner_report(
        registry=registry,
        counters=report_counters,
        budget=budget,
        result=result,
        elapsed_seconds=elapsed,
        cancelled=False,
        evaluation_records=aggregated_records,
        evaluation_backend=backend_name,
    )
    if campaign_result is not None:
        report["multi_family"] = campaign_result.as_dict()
        report["family_funnel"] = [s.as_dict() for s in campaign_result.family_stats]
        report["budget_allocation"] = campaign_result.budget_allocation
        report["best_candidates_per_family"] = campaign_result.as_dict()[
            "best_candidates_per_family"
        ]
        report["clusters"] = list(campaign_result.clusters)
        report["behavioral_clusters"] = len(campaign_result.clusters)
        report["research_shortlist"] = [
            e.as_dict() if hasattr(e, "as_dict") else e
            for e in campaign_result.research_shortlist
        ]
        report["shortlisted"] = len(campaign_result.research_shortlist)
        report["candidate_statistics_summaries"] = [
            s.as_dict() for s in campaign_result.candidate_statistics_summaries
        ]
        report["statistics_accounting"] = (
            campaign_result.statistics_accounting.as_dict()
            if campaign_result.statistics_accounting is not None
            else None
        )
        report["clustering_accounting"] = (
            campaign_result.clustering_accounting.as_dict()
            if campaign_result.clustering_accounting is not None
            else None
        )
        report["shortlist_rejects"] = [
            r.as_dict() if hasattr(r, "as_dict") else r
            for r in campaign_result.shortlist_rejects
        ]
        report["population_stats"] = dict(campaign_result.population_stats)
        report["candidate_status_history"] = [
            e.as_dict() for e in campaign_result.candidate_status_history
        ]
        report["pipeline_level"] = campaign_result.pipeline_level
        report["statistics_pipeline_complete"] = (
            campaign_result.statistics_pipeline_complete
        )
        report["clustering_pipeline_complete"] = (
            campaign_result.clustering_pipeline_complete
        )
        report["research_shortlist_pipeline_complete"] = (
            campaign_result.research_shortlist_pipeline_complete
        )
        report["vault_pipeline_complete"] = campaign_result.vault_pipeline_complete
        report["paper_pipeline_complete"] = campaign_result.paper_pipeline_complete
        report["live_pipeline_complete"] = campaign_result.live_pipeline_complete
        report["post_wfo_blocked_reasons"] = list(
            campaign_result.post_wfo_blocked_reasons
        )
        if campaign_result.population_stats:
            report["statistical_result"] = (
                "PASSED"
                if campaign_result.statistics_accounting
                and campaign_result.statistics_accounting.candidates_statistics_passed > 0
                else "INSUFFICIENT_DATA"
            )
        (art / "multi_family_campaign.json").write_text(
            json.dumps(campaign_result.as_dict(), indent=2, default=str), encoding="utf-8"
        )
    report["silver_resolution"] = silver_meta
    report["execution_banners"] = list(resolution.banners)
    report["is_full_event_wfo"] = bool(resolution.is_full_event_wfo)
    report["evaluation_path"] = backend_name
    report["runtime_provenance"] = provenance
    report["signal_source"] = (
        (getattr(backend, "last_run_artifacts", {}) or {}).get("signal_source")
        if backend_name == "event_driven_wfo"
        else "n/a"
    )
    if backend_name == "event_driven_wfo":
        try:
            assert_dsl_signal_source(report["signal_source"], where="run_summary")
        except RuntimeError as exc:
            run.state = RunState.FAILED
            run.software_success = False
            run.terminal_reason = "SIGNAL_SOURCE_MISMATCH"
            fail_summary = {
                "terminal_reason": "SIGNAL_SOURCE_MISMATCH",
                "error": str(exc),
                "runtime_provenance": provenance,
                "signal_source": report.get("signal_source"),
                "software_execution_status": "FAILED",
                "discovery_result": "SOFTWARE_FAILURE",
            }
            (art / "run_summary.json").write_text(
                json.dumps(fail_summary, indent=2, default=str), encoding="utf-8"
            )
            _emit(sink, run, EventType.RUN_FAILED, str(exc), severity=EventSeverity.ERROR, payload=fail_summary)
            return run
    report["invalid_candidates"] = int(report_counters.invalid)
    report["search_budget_consumed"] = {
        **report.get("search_budget_consumed", {}),
        "generated_attempts": report_counters.generated_attempts,
        "unique_generated_candidates": report_counters.unique_generated,
        "duplicate_attempts": report_counters.duplicate_attempts,
        "evaluated_candidates": report_counters.evaluated,
        "full_wfo_evaluations": report_counters.full_wfo,
        "invalid_candidates": report_counters.invalid,
        "stagnation_generations_config": budget.effective_stagnation_generations(),
        "minimum_generations_before_stagnation": (
            budget.effective_minimum_generations_before_stagnation()
        ),
        "generations_completed": report_counters.generations_completed,
        "stagnant_generations": report_counters.stagnant_generations,
        "max_candidates_per_family_requested": bucket_caps["requested"][
            "max_candidates_per_family"
        ],
        "max_candidates_per_family_effective": budget.max_candidates_per_family,
        "max_candidates_per_complexity_tier_requested": bucket_caps["requested"][
            "max_candidates_per_complexity_tier"
        ],
        "max_candidates_per_complexity_tier_effective": (
            budget.max_candidates_per_complexity_tier
        ),
        "max_candidates_per_feature_family_requested": bucket_caps["requested"][
            "max_candidates_per_feature_family"
        ],
        "max_candidates_per_feature_family_effective": (
            budget.max_candidates_per_feature_family
        ),
        "min_oos_trades": budget.min_oos_trades,
        "min_oos_trades_per_fold": budget.min_oos_trades_per_fold,
        "max_oos_drawdown": budget.max_oos_drawdown,
        "bucket_caps_requested": bucket_caps["requested"],
        "bucket_caps_effective": bucket_caps["effective"],
    }
    report["frozen_search_budget"] = {
        "requested": dict(budget_cfg),
        "effective": {
            **budget.as_dict(),
            "bucket_caps_requested": bucket_caps["requested"],
            "bucket_caps_effective": bucket_caps["effective"],
        },
    }
    run.generated_count = int(report["generated_candidates"])
    run.evaluated_count = int(report["evaluated_candidates"])
    run.rejected_count = int(report["rejected_total"])
    run.qualified_count = int(report["score_qualified"])
    run.finalist_count = int(report["finalist_count"])
    run.portfolio_count = int(report.get("shortlisted") or 0)
    run.elapsed_seconds = elapsed
    run.terminal_reason = report["terminal_reason"]
    run.software_success = True
    run.statistical_validation = report["statistical_result"]
    run.profitability_result = report["profitability_result"]
    run.vault_result = report["vault_result"]
    run.total_return_pct = None  # discovery does not imply portfolio return

    (art / "alpha_miner_report.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    (art / "candidates.json").write_text(
        json.dumps(
            {"candidates": report["candidates"], "tables": report.get("tables")},
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )
    summary = {
        **report,
        "discovery_run_id": result.discovery_run_id,
        "stop_reason": result.stop_reason,
        "rankings": result.rankings[:20],
        # Numeric finalist count — never a list of IDs
        "finalists": int(report["finalist_count"]),
        "finalist_ids": list(report.get("finalist_ids") or []),
        "controller_shortlist_ids": list(result.finalists),
        "wfo_ranking_source": "validation_oos",
        "software_execution_status": report["software_execution_status"],
        "discovery_result": report["discovery_result"],
        "statistical_result": report["statistical_result"],
        "portfolio_result": report["portfolio_result"],
        "total_return_pct": None,
        "total_return_status": "NOT_AVAILABLE",
        "silver_resolution": silver_meta,
        "evaluation_path": backend_name,
        "is_full_event_wfo": bool(resolution.is_full_event_wfo),
        "runtime_provenance": provenance,
        "signal_source": report.get("signal_source"),
        "ui_canary": canary_cfg or None,
        "multi_family": (campaign_result.as_dict() if campaign_result is not None else None),
        "family_funnel": report.get("family_funnel"),
        "budget_allocation": report.get("budget_allocation"),
    }
    (art / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    manifest_files = [
        "run_summary.json",
        "alpha_miner_report.json",
        "candidates.json",
        "runtime_provenance.json",
        "artifact_manifest.json",
    ]
    if silver_meta:
        manifest_files.append("silver_resolution.json")
    if campaign_result is not None:
        manifest_files.append("multi_family_campaign.json")
    _write_manifest(run, manifest_files)
    if cancel_check():
        run.state = RunState.CANCELLED
        run.terminal_reason = "cancelled"
        run.software_success = False
        _emit(sink, run, EventType.RUN_CANCELLED, "Cancelled after miner", severity=EventSeverity.WARNING)
        return run
    _finalize_completed(run, summary)
    _emit(
        sink,
        run,
        EventType.RUN_COMPLETED,
        f"Alpha Miner completed — {report['discovery_result']}",
        stage="done",
        progress=100,
        payload=summary,
    )
    if on_update is not None:
        on_update(run)
    return run


def run_wfo_only(
    run: RunRecord,
    sink: ResearchEventSink,
    *,
    cancel_check: CancelCheck,
) -> RunRecord:
    from config.walk_forward_config import WalkForwardConfig
    from validation.walk_forward import run_walk_forward

    art = Path(run.artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    _emit(sink, run, EventType.RUN_STARTED, "WFO-only started", stage="wfo", progress=5)
    rng = np.random.default_rng(run.random_seed)
    n = 600
    idx = pd.date_range("2024-01-02 08:30", periods=n, freq="5min", tz="America/Chicago")
    close = 4800 + np.cumsum(rng.normal(0, 0.4, n))
    frame = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 1000.0,
        }
    )
    _emit(sink, run, EventType.DATA_LOADED, f"{len(frame)} bars", stage="data", progress=15)

    wfo_cfg_raw = run.config_snapshot.get("wfo") or {}
    wfo_cfg = WalkForwardConfig(
        train_window_days=int(wfo_cfg_raw.get("train_window_days", 3)),
        validation_window_days=int(wfo_cfg_raw.get("validation_window_days", 1)),
        step_forward_days=int(wfo_cfg_raw.get("step_forward_days", 1)),
        purge_gap_bars=int(wfo_cfg_raw.get("purge_gap_bars", 1)),
        embargo_gap_bars=int(wfo_cfg_raw.get("embargo_gap_bars", 1)),
        bars_per_day=78,
        param_grid={"lookback": [10, 15], "z_entry": [1.5]},
    )

    fold_rows: list[dict[str, Any]] = []
    n_folds_est = 5

    def evaluate_fn(window: pd.DataFrame, params: dict) -> float:
        lb = int(params.get("lookback", 10))
        if len(window) < lb + 2:
            return -999.0
        c = window["close"].astype(float)
        z = (c - c.rolling(lb).mean()) / c.rolling(lb).std(ddof=0)
        return float(-z.dropna().abs().mean())

    def persist(trial: dict) -> None:
        if trial.get("phase") == "validation":
            fold_rows.append(
                {
                    "fold": trial.get("fold_id"),
                    "phase": "validation_oos",
                    "metric": trial.get("metric"),
                    "params": trial.get("params"),
                    "is_training": False,
                }
            )
            _emit(
                sink,
                run,
                EventType.WFO_FOLD_COMPLETED,
                f"OOS fold metric={trial.get('metric')}",
                stage="wfo",
                progress=min(90.0, 20 + 10 * len(fold_rows)),
                payload={"fold": trial, "ranking_source": "validation_oos"},
            )

    if cancel_check():
        run.state = RunState.CANCELLED
        run.terminal_reason = "cancelled"
        _emit(sink, run, EventType.RUN_CANCELLED, "Cancelled", severity=EventSeverity.WARNING)
        return run

    _emit(sink, run, EventType.WFO_FOLD_STARTED, "Starting WFO", stage="wfo", progress=25)
    result = run_walk_forward(frame, wfo_cfg, evaluate_fn=evaluate_fn, persist_trial_fn=persist)
    summary = {
        "folds": len(result.folds),
        "aggregated_validation_metric": result.aggregated_validation_metric,
        "fold_oos": fold_rows,
        "ranking_source": "validation_oos",
        "training_metrics_labeled_as_oos": False,
        "software_success": True,
        "statistical_validation": "wfo_oos",
        "profitability_result": "metric_only",
    }
    (art / "wfo_report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (art / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_manifest(run, ["wfo_report.json", "run_summary.json", "artifact_manifest.json"])
    _finalize_completed(run, summary)
    _emit(sink, run, EventType.RUN_COMPLETED, "WFO completed", stage="done", progress=100, payload=summary)
    return run


def run_portfolio_build(
    run: RunRecord,
    sink: ResearchEventSink,
    *,
    cancel_check: CancelCheck,
) -> RunRecord:
    from portfolio import (
        ConstructionMethod,
        MemberMeta,
        PortfolioConstraints,
        PortfolioOptimizer,
        build_correlation_matrix,
        build_portfolio_version,
        freeze_portfolio,
    )

    art = Path(run.artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    _emit(sink, run, EventType.RUN_STARTED, "Portfolio build started", stage="portfolio", progress=10)
    members = [
        MemberMeta("c0", "l0", "mr", ("ES",), "futures", "range", "rth", 1.0, 0.2, 0.05, 0.5),
        MemberMeta("c1", "l1", "trend", ("NQ",), "futures", "trend", "rth", 1.2, 0.2, 0.05, 0.4),
    ]
    pnl = {"c0": (1, 2, 1, 0), "c1": (0, 1, -1, 2)}
    pairwise = build_correlation_matrix(pnl)
    opt = PortfolioOptimizer(constraints=PortfolioConstraints(max_members=5, max_strategy_weight=0.7, max_asset_class_exposure=1.0, max_regime_concentration=1.0, max_session_concentration=1.0))
    result = opt.optimize(members, pairwise=pairwise, method=ConstructionMethod.GREEDY_MARGINAL)
    if not result.accepted:
        run.state = RunState.FAILED
        run.terminal_reason = result.reason
        run.software_success = False
        _emit(sink, run, EventType.RUN_FAILED, result.reason, severity=EventSeverity.ERROR)
        return run
    version = build_portfolio_version(
        weights=result.weights,
        lineage_ids={m.candidate_id: m.lineage_id for m in members},
        construction_method=result.method.value,
        constraints=PortfolioConstraints().as_dict(),
        feature_set_version=run.feature_set_version,
        cost_model_version=run.cost_model_version,
    )
    _emit(sink, run, EventType.PORTFOLIO_BUILT, version.portfolio_id, stage="portfolio", progress=70)
    frozen = freeze_portfolio(version)
    run.portfolio_count = 1
    _emit(sink, run, EventType.PORTFOLIO_FROZEN, frozen.portfolio_version, stage="freeze", progress=90)
    summary = {"portfolio": frozen.as_dict(), "software_success": True, "frozen": True}
    (art / "portfolio_report.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (art / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_manifest(run, ["portfolio_report.json", "run_summary.json", "artifact_manifest.json"])
    _finalize_completed(run, summary)
    _emit(sink, run, EventType.RUN_COMPLETED, "Portfolio frozen", stage="done", progress=100, payload=summary)
    return run


def run_vault_evaluation(
    run: RunRecord,
    sink: ResearchEventSink,
    *,
    cancel_check: CancelCheck,
) -> RunRecord:
    from portfolio import ConstructionMethod, PortfolioConstraints, build_portfolio_version, freeze_portfolio
    from portfolio.versioning import PortfolioPipelineStage
    from validation.candidate_lineage import CandidateLineage
    from validation.vault import ValidationVault, VaultAccessError

    art = Path(run.artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    _emit(sink, run, EventType.VAULT_STARTED, "Vault evaluation started", stage="vault", progress=20)
    version = build_portfolio_version(
        weights={"a": 1.0},
        lineage_ids={"a": "lin_a"},
        construction_method=ConstructionMethod.EQUAL_RISK_CONTRIBUTION.value,
        constraints=PortfolioConstraints().as_dict(),
        feature_set_version=run.feature_set_version,
        cost_model_version=run.cost_model_version,
    )
    frozen = freeze_portfolio(version).with_stage(PortfolioPipelineStage.VAULT_ELIGIBLE)
    idx = pd.date_range("2024-06-01 08:30", periods=40, freq="5min", tz="America/Chicago")
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
    )
    vault = ValidationVault(
        vault_version="vault_v1",
        data=df,
        data_hash=sha256_json({"n": len(df)}),
        store_path=art / "vault",
    )
    lineage = CandidateLineage.create(
        strategy_family="portfolio",
        parameters=dict(frozen.weights),
        config_snapshot={"portfolio_id": frozen.portfolio_id, "portfolio_version": frozen.portfolio_version},
        code_hash=frozen.portfolio_version,
        data_hash="vault_demo",
        cost_model_version=run.cost_model_version,
        candidate_id=frozen.portfolio_id,
    )
    # Safe evaluate — result metrics only; never return raw bars to API
    result = vault.evaluate_frozen_candidate(
        lineage,
        evaluate_fn=lambda d: {"n_bars": len(d), "status": "evaluated", "passed": True},
    )
    # Enforce one-shot
    one_shot_ok = False
    try:
        vault.evaluate_frozen_candidate(lineage, evaluate_fn=lambda d: {"n": len(d)})
    except VaultAccessError:
        one_shot_ok = True
    summary = {
        "portfolio_id": frozen.portfolio_id,
        "lineage_id": lineage.lineage_id,
        "frozen": True,
        "vault_version": result["vault_version"],
        "access_count": result["access_count"],
        "passed": True,
        "one_shot_enforced": one_shot_ok,
        "high_level_metrics": result["result"],
        "raw_vault_exposed": False,
        "software_success": True,
    }
    run.vault_result = "passed"
    (art / "vault_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (art / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_manifest(run, ["vault_summary.json", "run_summary.json", "artifact_manifest.json"])
    _emit(sink, run, EventType.VAULT_COMPLETED, "Vault passed (metadata only)", stage="vault", progress=90)
    _finalize_completed(run, summary)
    _emit(sink, run, EventType.RUN_COMPLETED, "Vault evaluation completed", stage="done", progress=100)
    return run


def run_shadow_or_paper(
    run: RunRecord,
    sink: ResearchEventSink,
    *,
    cancel_check: CancelCheck,
    paper: bool,
) -> RunRecord:
    from brokers import BrokerMode, OrderSide, SimulatedBroker
    from runtime import MarketBar, RuntimeState, Signal, StateStore, TradingRuntime

    art = Path(run.artifact_dir)
    art.mkdir(parents=True, exist_ok=True)
    mode = BrokerMode.PAPER if paper else BrokerMode.SHADOW
    broker = SimulatedBroker(mode=mode)
    broker.set_price("ES", 5000.0)
    state = RuntimeState(run_id=run.run_id, mode=mode.value, cash=100_000.0)
    store = StateStore(art / "runtime_state.json")
    store.save(state)
    rt = TradingRuntime(broker=broker, state=state, store=store)
    _emit(
        sink,
        run,
        EventType.RUN_STARTED,
        f"{mode.value} runtime started",
        stage="runtime",
        progress=10,
    )
    from datetime import timedelta

    now = datetime.now(tz=timezone.utc)
    bar = MarketBar("ES", now - timedelta(seconds=1), 4999, 5001, 5000)
    rt.on_bar(bar)
    order = rt.process_signal(Signal("s1", "ES", OrderSide.BUY, 1.0, bar.timestamp.isoformat()))
    submitted_flag = bool(order and order.meta.get("submitted"))
    if paper and not submitted_flag:
        run.state = RunState.FAILED
        run.terminal_reason = "paper_order_not_submitted"
        _emit(sink, run, EventType.RUN_FAILED, run.terminal_reason, severity=EventSeverity.ERROR)
        return run
    if not paper and submitted_flag:
        run.state = RunState.FAILED
        run.terminal_reason = "shadow_must_not_submit"
        _emit(sink, run, EventType.RUN_FAILED, run.terminal_reason, severity=EventSeverity.ERROR)
        return run
    summary = {
        "mode": mode.value,
        "broker": "simulated",
        "order_status": order.status.value if order else None,
        "submitted": submitted_flag,
        "positions": [p.as_dict() for p in broker.positions()],
        "real_submit_count": broker.real_submit_count() if hasattr(broker, "real_submit_count") else None,
        "live_enabled": False,
        "software_success": True,
        "paper_eligible": paper,
    }
    run.paper_eligible = paper
    (art / "runtime_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    (art / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    _write_manifest(run, ["runtime_summary.json", "run_summary.json", "artifact_manifest.json", "runtime_state.json"])
    _finalize_completed(run, summary)
    _emit(sink, run, EventType.RUN_COMPLETED, f"{mode.value} completed", stage="done", progress=100)
    return run


def execute_run(
    run: RunRecord,
    sink: ResearchEventSink,
    *,
    cancel_check: CancelCheck,
    on_update: Callable[[RunRecord], None] | None = None,
) -> RunRecord:
    try:
        run.state = RunState.RUNNING
        run.started_at = datetime.now(tz=timezone.utc).isoformat()
        if run.run_type is RunType.RESEARCH_DEMO:
            return run_research_demo(run, sink, cancel_check=cancel_check)
        if run.run_type is RunType.ALPHA_MINER:
            return run_alpha_miner_job(
                run, sink, cancel_check=cancel_check, on_update=on_update
            )
        if run.run_type is RunType.WFO_ONLY:
            return run_wfo_only(run, sink, cancel_check=cancel_check)
        if run.run_type is RunType.PORTFOLIO_BUILD:
            return run_portfolio_build(run, sink, cancel_check=cancel_check)
        if run.run_type is RunType.VAULT_EVALUATION:
            return run_vault_evaluation(run, sink, cancel_check=cancel_check)
        if run.run_type is RunType.SHADOW_RUNTIME:
            return run_shadow_or_paper(run, sink, cancel_check=cancel_check, paper=False)
        if run.run_type is RunType.PAPER_RUNTIME:
            return run_shadow_or_paper(run, sink, cancel_check=cancel_check, paper=True)
        run.state = RunState.FAILED
        run.terminal_reason = f"unsupported_run_type:{run.run_type}"
        _emit(sink, run, EventType.RUN_FAILED, run.terminal_reason, severity=EventSeverity.ERROR)
        return run
    except Exception as exc:  # noqa: BLE001
        from discovery.expression_tree import DSLValidationError
        from discovery.typecheck import INVALID_DSL_TYPE, InvalidDslTypeError

        run.state = RunState.FAILED
        run.software_success = False
        run.error_count += 1
        msg = str(exc)
        is_dsl = isinstance(exc, (DSLValidationError, InvalidDslTypeError)) or (
            INVALID_DSL_TYPE in msg or "child[" in msg and " type " in msg
        )
        if is_dsl:
            run.terminal_reason = "DSL_TYPE_VALIDATION_FAILURE"
            discovery = "SOFTWARE_FAILURE"
            software_status = "SOFTWARE_FAILURE"
        else:
            run.terminal_reason = msg
            discovery = "SOFTWARE_FAILURE"
            software_status = "SOFTWARE_FAILURE"
        backend = None
        try:
            backend = (run.config_snapshot.get("search_budget") or {}).get(
                "evaluation_backend"
            )
        except Exception:  # noqa: BLE001
            backend = None
        if not backend:
            # Prefer last known real-data path from snapshot flags
            if run.config_snapshot.get("dataset_research_eligible") and not run.config_snapshot.get(
                "smoke_test"
            ):
                backend = "event_driven_wfo"
            else:
                backend = "unknown"
        summary = {
            "software_execution_status": software_status,
            "discovery_result": discovery,
            "terminal_reason": run.terminal_reason,
            "error": msg,
            "evaluation_backend": backend,
            "evaluation_path": backend,
            "is_full_event_wfo": backend == "event_driven_wfo",
            "proxy_metric_used": False,
            "qualified_candidate_status": "NOT_APPLICABLE",
            "message_no_finalists": None,
            "software_failure_banner": True,
        }
        art = Path(run.artifact_dir)
        art.mkdir(parents=True, exist_ok=True)
        (art / "run_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8"
        )
        run.summary = summary
        _emit(
            sink,
            run,
            EventType.RUN_FAILED,
            msg,
            severity=EventSeverity.ERROR,
            payload={"traceback": traceback.format_exc(), **summary},
        )
        return run
