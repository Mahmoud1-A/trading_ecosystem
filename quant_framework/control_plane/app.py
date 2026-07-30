"""FastAPI research control plane (Phase 12)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any

from control_plane.models import FORBIDDEN_LIVE_MODES, TERMINAL_STATES, RunState
from control_plane.run_manager import RunManager
from control_plane.security import (
    ConfigValidationError,
    CreateRunRequest,
    reject_arbitrary_command_payload,
    safe_artifact_path,
)

ACTIVE_STATES = frozenset(
    {
        RunState.CREATED,
        RunState.QUEUED,
        RunState.STARTING,
        RunState.RUNNING,
        RunState.CANCEL_REQUESTED,
    }
)
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import ValidationError

STATIC_DIR = Path(__file__).resolve().parent / "static"


def create_app(manager: RunManager | None = None, *, root: Path | None = None) -> FastAPI:
    root = Path(root or os.environ.get("QUANT_CONTROL_PLANE_ROOT", Path.cwd() / "artifacts" / "control_plane"))
    mgr = manager or RunManager(root)

    app = FastAPI(
        title="Quant Framework Research Control Plane",
        version="0.12.2-phase12.2",
        description="RESEARCH / BACKTEST ONLY — LIVE TRADING IS NOT ENABLED",
    )
    app.state.manager = mgr
    app.state.root = root

    @app.middleware("http")
    async def research_only_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
        response = await call_next(request)
        response.headers["X-Research-Only"] = "true"
        response.headers["X-Live-Trading-Enabled"] = "false"
        return response

    def _mgr() -> RunManager:
        return app.state.manager  # type: ignore[no-any-return]

    @app.post("/api/runs")
    def create_run(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            reject_arbitrary_command_payload(payload)
            # Hard-block live modes before pydantic (extra keys already forbid)
            for key in ("environment", "trading_mode", "mode", "deployment_mode"):
                val = payload.get(key)
                if isinstance(val, str) and val.upper() in FORBIDDEN_LIVE_MODES:
                    raise HTTPException(status_code=400, detail=f"live mode {val!r} cannot be activated")
            if payload.get("live_trading") is True or payload.get("activate_live") is True:
                raise HTTPException(status_code=400, detail="live trading cannot be activated")
            req = CreateRunRequest.model_validate(payload)
            req.require_confirmations()
            run = _mgr().create_run(req)
            run = _mgr().enqueue(run.run_id)
            return run.as_dict()
        except HTTPException:
            raise
        except (ValidationError, ConfigValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @app.get("/api/runs")
    def list_runs(
        filter: str | None = Query(None, alias="filter"),
        state: str | None = None,
    ) -> dict[str, Any]:
        """List runs. filter=ALL|ACTIVE|QUEUED|COMPLETED|FAILED|CANCELLED|INTERRUPTED|DEFAULT."""
        runs = _mgr().store.list_runs()
        key = (filter or state or "DEFAULT").upper()
        if key == "ACTIVE":
            runs = [r for r in runs if r.state in ACTIVE_STATES]
        elif key == "QUEUED":
            runs = [r for r in runs if r.state is RunState.QUEUED]
        elif key == "COMPLETED":
            runs = [r for r in runs if r.state is RunState.COMPLETED]
        elif key == "FAILED":
            runs = [r for r in runs if r.state is RunState.FAILED]
        elif key == "CANCELLED":
            runs = [r for r in runs if r.state is RunState.CANCELLED]
        elif key == "INTERRUPTED":
            runs = [r for r in runs if r.state is RunState.INTERRUPTED]
        elif key == "DEFAULT":
            active = [r for r in runs if r.state in ACTIVE_STATES]
            terminal = [r for r in runs if r.state in TERMINAL_STATES]
            runs = active + terminal
        elif key != "ALL":
            runs = [r for r in runs if r.state.value == key]
        return {
            "runs": [r.as_dict() for r in runs],
            "count": len(runs),
            "filter": key,
            "cancellable_states": [s.value for s in sorted(ACTIVE_STATES, key=lambda s: s.value)],
        }

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str) -> dict[str, Any]:
        run = _mgr().store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        payload = run.as_dict()
        payload["cancellable"] = run.state in ACTIVE_STATES
        return payload

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str) -> dict[str, Any]:
        try:
            return _mgr().cancel(run_id).as_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        except ConfigValidationError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.get("/api/runs/{run_id}/events")
    def get_events(run_id: str, after_seq: int = Query(0, ge=0)) -> dict[str, Any]:
        try:
            events = _mgr().get_events(run_id, after_seq=after_seq)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="run not found") from exc
        return {"run_id": run_id, "after_seq": after_seq, "events": events}

    @app.get("/api/runs/{run_id}/events/stream")
    async def stream_events(run_id: str, after_seq: int = Query(0, ge=0)) -> StreamingResponse:
        if _mgr().store.get(run_id) is None:
            raise HTTPException(status_code=404, detail="run not found")

        async def gen():  # type: ignore[no-untyped-def]
            cursor = after_seq
            idle = 0
            while True:
                events = _mgr().get_events(run_id, after_seq=cursor)
                for e in events:
                    cursor = max(cursor, int(e.get("seq", 0)))
                    yield f"data: {json.dumps(e, default=str)}\n\n"
                    idle = 0
                run = _mgr().store.get(run_id)
                if run and run.state in {
                    RunState.COMPLETED,
                    RunState.FAILED,
                    RunState.CANCELLED,
                    RunState.INTERRUPTED,
                }:
                    # Final catch-up then close
                    events = _mgr().get_events(run_id, after_seq=cursor)
                    for e in events:
                        yield f"data: {json.dumps(e, default=str)}\n\n"
                    yield f"event: end\ndata: {json.dumps({'run_id': run_id, 'state': run.state.value})}\n\n"
                    break
                idle += 1
                if idle > 600:  # ~60s of idle keepalives then continue anyway
                    idle = 0
                yield ": keepalive\n\n"
                await asyncio.sleep(0.1)

        return StreamingResponse(gen(), media_type="text/event-stream")

    @app.get("/api/runs/{run_id}/summary")
    def get_summary(run_id: str) -> dict[str, Any]:
        run = _mgr().store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        summary = dict(run.summary)
        # Result honesty fields — never imply discovery success from software success
        summary["software_success"] = run.software_success
        summary["software_execution_status"] = summary.get("software_execution_status") or (
            "SUCCESS" if run.software_success else ("FAILED" if run.software_success is False else "NOT_EVALUATED")
        )
        summary["discovery_result"] = summary.get("discovery_result") or "NOT_APPLICABLE"
        summary["statistical_validation"] = run.statistical_validation
        summary["statistical_result"] = summary.get("statistical_result") or run.statistical_validation or "NOT_EVALUATED"
        summary["profitability_result"] = (
            summary.get("profitability_result") or run.profitability_result or "NOT_AVAILABLE"
        )
        summary["portfolio_result"] = summary.get("portfolio_result") or "NOT_APPLICABLE"
        summary["vault_result"] = summary.get("vault_result") or run.vault_result or "NOT_SUBMITTED"
        summary["total_return_pct"] = run.total_return_pct
        summary["total_return_status"] = (
            summary.get("total_return_status")
            if summary.get("total_return_status")
            else ("NOT_AVAILABLE" if run.total_return_pct is None else "AVAILABLE")
        )
        summary["paper_eligible"] = run.paper_eligible
        summary["state"] = run.state.value
        summary["cancellable"] = run.state in ACTIVE_STATES
        summary["terminal_reason"] = run.terminal_reason
        summary["qualified_candidate_status"] = summary.get("qualified_candidate_status") or "NOT_APPLICABLE"
        if run.total_return_pct is not None:
            summary["return_display"] = f"{run.total_return_pct:.2f}%"
            summary["return_is_negative"] = run.total_return_pct < 0
        else:
            summary["return_display"] = summary.get("total_return_status") or "NOT_AVAILABLE"
            summary["return_is_negative"] = False
        # Avoid ambiguous dash in UI — expose semantic blanks
        for key in (
            "profitability_result",
            "statistical_result",
            "vault_result",
            "discovery_result",
            "software_execution_status",
        ):
            if summary.get(key) in (None, "", "-"):
                summary[key] = "NOT_EVALUATED"
        return summary

    @app.get("/api/runs/{run_id}/candidates")
    def get_candidates(run_id: str) -> dict[str, Any]:
        run = _mgr().store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        # Prefer persisted candidate rows from Alpha Miner
        cand_path = Path(run.artifact_dir) / "candidates.json"
        if cand_path.exists():
            data = json.loads(cand_path.read_text(encoding="utf-8"))
            rows = list(data.get("candidates") or [])
            return {
                "candidates": rows,
                "tables": data.get("tables") or {},
                "rejected_visible": True,
                "registry_total": len(rows),
                "rejected_count": sum(1 for c in rows if c.get("rejected")),
                "duplicate_count": sum(
                    1 for c in rows if c.get("evaluation_stage") == "DUPLICATE"
                ),
            }
        from registry.experiment_registry import ExperimentRegistry

        reg_path = Path(run.artifact_dir) / "registry"
        if not reg_path.exists():
            return {
                "candidates": [],
                "tables": {},
                "rejected_visible": True,
                "registry_total": 0,
                "rejected_count": 0,
                "duplicate_count": 0,
            }
        registry = ExperimentRegistry(reg_path)
        # Rebuild a minimal report view when artifacts lack candidates.json
        from control_plane.alpha_results import build_alpha_miner_report
        from discovery.search_budget import BudgetCounters, SearchBudget
        from discovery.search_controller import DiscoveryRunResult

        summary = run.summary or {}
        result = DiscoveryRunResult(
            discovery_run_id=str(summary.get("discovery_run_id") or run.run_id),
            budget_id="unknown",
            stop_reason=str(summary.get("stop_reason") or "completed"),
            generations=int(summary.get("generations") or 0),
            evaluated=int(run.evaluated_count),
            registered_trials=len(registry.all_trials()),
            rankings=list(summary.get("rankings") or []),
            finalists=list(summary.get("controller_shortlist_ids") or summary.get("finalist_ids") or []),
            promoted=[],
            portfolio_pool=dict(summary.get("portfolio_pool") or {}),
            clusters=list(summary.get("clusters") or []),
            reproducible_fingerprint="",
        )
        report = build_alpha_miner_report(
            registry=registry,
            counters=BudgetCounters(
                generated=run.generated_count,
                evaluated=run.evaluated_count,
            ),
            budget=SearchBudget(),
            result=result,
            elapsed_seconds=run.elapsed_seconds,
            evaluation_backend=str(summary.get("evaluation_backend") or "synthetic_oos_probe"),
        )
        rows = report["candidates"]
        return {
            "candidates": rows,
            "tables": report.get("tables") or {},
            "rejected_visible": True,
            "registry_total": len(rows),
            "rejected_count": report["rejected_total"],
            "duplicate_count": report["duplicate_candidates"],
        }

    @app.get("/api/runs/{run_id}/alpha_miner")
    def get_alpha_miner(run_id: str) -> dict[str, Any]:
        run = _mgr().store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        report_path = Path(run.artifact_dir) / "alpha_miner_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
        else:
            report = dict(run.summary or {})
        report["run_id"] = run_id
        report["state"] = run.state.value
        report["progress_pct"] = run.progress_pct
        report["current_stage"] = run.current_stage
        report["cancellable"] = run.state in ACTIVE_STATES
        # Ensure finalists is numeric for UI cards
        if isinstance(report.get("finalists"), list):
            report["finalist_ids"] = list(report["finalists"])
            report["finalists"] = int(report.get("finalist_count") or 0)
        report["finalist_count"] = int(
            report.get("finalist_count")
            if report.get("finalist_count") is not None
            else (report["finalists"] if isinstance(report.get("finalists"), int) else 0)
        )
        if report["finalist_count"] == 0:
            report["message_no_finalists"] = (
                report.get("message_no_finalists")
                or "No candidate passed all mandatory Alpha Miner gates."
            )
            report["qualified_candidate_status"] = report.get(
                "qualified_candidate_status", "NO_QUALIFIED_CANDIDATE"
            )
        return report

    @app.get("/api/runs/{run_id}/wfo")
    def get_wfo(run_id: str) -> dict[str, Any]:
        run = _mgr().store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        path = Path(run.artifact_dir) / "wfo_report.json"
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            data["training_metrics_labeled_as_oos"] = False
            data["ranking_source"] = data.get("ranking_source", "validation_oos")
            return data
        return {
            "folds": [],
            "ranking_source": "validation_oos",
            "training_metrics_labeled_as_oos": False,
            "message": "no WFO report for this run",
        }

    @app.get("/api/runs/{run_id}/portfolio")
    def get_portfolio(run_id: str) -> dict[str, Any]:
        run = _mgr().store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        path = Path(run.artifact_dir) / "portfolio_report.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        return run.summary.get("portfolio") or {"message": "no portfolio for this run"}

    @app.get("/api/runs/{run_id}/vault")
    def get_vault(run_id: str) -> dict[str, Any]:
        """Safe Vault metadata only — never raw bars/features/returns."""
        run = _mgr().store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        path = Path(run.artifact_dir) / "vault_summary.json"
        if not path.exists():
            return {"message": "no vault summary", "raw_vault_exposed": False}
        data = json.loads(path.read_text(encoding="utf-8"))
        # Strip any accidental raw payloads
        for banned in ("raw_bars", "features", "returns", "timestamps", "vault_data", "data"):
            data.pop(banned, None)
        data["raw_vault_exposed"] = False
        return data

    @app.post("/api/runs/{run_id}/vault/reevaluate")
    def vault_reevaluate_blocked(run_id: str) -> JSONResponse:
        raise HTTPException(
            status_code=403,
            detail="Repeated Vault evaluation for the same lineage is forbidden",
        )

    @app.get("/api/runs/{run_id}/artifacts")
    def list_artifacts(run_id: str) -> dict[str, Any]:
        run = _mgr().store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        return {"run_id": run_id, "manifest": list(run.artifact_manifest), "artifact_dir": run.artifact_dir}

    @app.get("/api/runs/{run_id}/artifacts/{name:path}")
    def get_artifact(run_id: str, name: str) -> FileResponse:
        run = _mgr().store.get(run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="run not found")
        try:
            path = safe_artifact_path(Path(run.artifact_dir), name, run.artifact_manifest)
        except ConfigValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="artifact not found") from exc
        return FileResponse(path)

    @app.get("/api/system/health")
    def health() -> dict[str, Any]:
        h = _mgr().health()
        try:
            import psutil  # type: ignore[import-untyped]

            proc = psutil.Process()
            h["cpu_percent"] = proc.cpu_percent(interval=0.0)
            h["memory_mb"] = proc.memory_info().rss / (1024 * 1024)
        except Exception:  # noqa: BLE001
            h["cpu_percent"] = None
            h["memory_mb"] = None
        h["system_version"] = "0.12.1-phase12.1"
        h["banner"] = "RESEARCH / BACKTEST ONLY — LIVE TRADING IS NOT ENABLED"
        return h

    @app.get("/api/system/capabilities")
    def capabilities() -> dict[str, Any]:
        return _mgr().capabilities()

    @app.post("/api/system/activate_live")
    def activate_live_blocked() -> None:
        raise HTTPException(status_code=403, detail="LIVE trading cannot be activated from the dashboard")

    @app.get("/api/overview")
    def overview() -> dict[str, Any]:
        runs = _mgr().store.list_runs()
        by_state: dict[str, int] = {}
        for r in runs:
            by_state[r.state.value] = by_state.get(r.state.value, 0) + 1
        latest_discovery = next((r for r in runs if r.run_type.value == "ALPHA_MINER"), None)
        latest_portfolio = next((r for r in runs if r.run_type.value == "PORTFOLIO_BUILD"), None)
        latest_vault = next((r for r in runs if r.run_type.value == "VAULT_EVALUATION"), None)
        latest_paper = next((r for r in runs if r.run_type.value == "PAPER_RUNTIME"), None)
        return {
            "system_version": "0.12.1-phase12.1",
            "banner": "RESEARCH / BACKTEST ONLY — LIVE TRADING IS NOT ENABLED",
            "research_only": True,
            "live_trading_enabled": False,
            "counts": by_state,
            "active": by_state.get("RUNNING", 0) + by_state.get("STARTING", 0),
            "queued": by_state.get("QUEUED", 0),
            "completed": by_state.get("COMPLETED", 0),
            "failed": by_state.get("FAILED", 0),
            "latest_discovery_run": latest_discovery.run_id if latest_discovery else None,
            "latest_portfolio": latest_portfolio.run_id if latest_portfolio else None,
            "latest_vault": latest_vault.run_id if latest_vault else None,
            "latest_paper": latest_paper.run_id if latest_paper else None,
            "health": _mgr().health(),
            "dataset_count": len(_mgr().catalog.list_entries()),
        }

    # ---------- Dataset Catalog / Import (Phase 12.1) ----------
    def _import_root() -> Path:
        from data.catalog.safe_paths import default_import_root

        override = os.environ.get("QUANT_DATA_IMPORT_ROOT")
        if override:
            root_path = Path(override)
            root_path.mkdir(parents=True, exist_ok=True)
            return root_path
        return default_import_root()

    def _parse_import_request(payload: dict[str, Any]):
        from data.catalog.import_pipeline import ColumnMapping, ImportRequest

        mapping_raw = payload.get("column_mapping") or {}
        mapping = ColumnMapping(
            timestamp=str(mapping_raw.get("timestamp", "timestamp")),
            open=str(mapping_raw.get("open", "open")),
            high=str(mapping_raw.get("high", "high")),
            low=str(mapping_raw.get("low", "low")),
            close=str(mapping_raw.get("close", "close")),
            volume=str(mapping_raw.get("volume", "volume")),
            symbol=mapping_raw.get("symbol"),
            tradable_contract=mapping_raw.get("tradable_contract"),
        )
        return ImportRequest(
            relative_path=str(payload.get("relative_path") or ""),
            asset_class=str(payload.get("asset_class") or "").upper(),
            column_mapping=mapping,
            provider_id=str(payload.get("provider_id") or ""),
            provider_version=str(payload.get("provider_version") or ""),
            source_timezone=str(payload.get("source_timezone") or ""),
            exchange_or_broker=str(payload.get("exchange_or_broker") or ""),
            timeframe=str(payload.get("timeframe") or ""),
            volume_type=str(payload.get("volume_type") or ""),
            root_symbol=str(payload.get("root_symbol") or ""),
            tradable_contract=payload.get("tradable_contract"),
            continuous_series=bool(payload.get("continuous_series") or False),
            rollover_metadata=dict(payload.get("rollover_metadata") or {}),
            multiplier=payload.get("multiplier"),
            tick_size=payload.get("tick_size"),
            tick_value=payload.get("tick_value"),
            contract_expiration=payload.get("contract_expiration"),
            broker_symbol=payload.get("broker_symbol"),
            spread_available=payload.get("spread_available"),
            financing_rate_available=payload.get("financing_rate_available"),
            confirm=bool(payload.get("confirm") or False),
            display_name=payload.get("display_name"),
            calendar=str(payload.get("calendar") or "CME"),
        )

    @app.get("/api/datasets")
    def list_datasets(
        research_eligible_only: bool = False,
        include_smoke: bool = True,
    ) -> dict[str, Any]:
        rows = _mgr().catalog.list_entries(
            research_eligible_only=research_eligible_only,
            include_smoke=include_smoke,
        )
        return {
            "datasets": [e.as_dict() for e in rows],
            "count": len(rows),
            "import_root": str(_import_root()),
        }

    @app.get("/api/datasets/{dataset_id}")
    def get_dataset(dataset_id: str) -> dict[str, Any]:
        entry = _mgr().catalog.get(dataset_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        payload = entry.as_dict()
        # Never expose credentials / secrets
        for banned in ("credentials", "api_key", "password", "token", "secret"):
            payload.pop(banned, None)
            if isinstance(payload.get("provenance"), dict):
                payload["provenance"].pop(banned, None)
        return payload

    @app.get("/api/datasets/{dataset_id}/constraints")
    def dataset_constraints(dataset_id: str) -> dict[str, Any]:
        from control_plane.backend_resolution import preview_backend_for_form
        from data.catalog.silver_bar_resolution import (
            compatible_timeframes_for_entry,
            strategy_allows_timeframe,
        )

        entry = _mgr().catalog.get(dataset_id)
        if entry is None:
            raise HTTPException(status_code=404, detail="dataset not found")
        compat_tfs = compatible_timeframes_for_entry(entry)
        if not compat_tfs and entry.timeframe:
            compat_tfs = [entry.timeframe]
        # Prefer 1m default for mean_reversion_vwap_bb when available
        default_tf = "1m" if "1m" in compat_tfs else (compat_tfs[0] if compat_tfs else entry.timeframe)
        families = list(entry.compatible_strategy_families)
        family_tfs = {
            fam: [tf for tf in compat_tfs if strategy_allows_timeframe(fam, tf)]
            for fam in families
        }
        backend_preview = preview_backend_for_form(
            dataset_id=entry.dataset_id,
            smoke_test=bool(entry.smoke_test_only),
            research_eligible=bool(entry.research_eligible),
            smoke_test_only=bool(entry.smoke_test_only),
            strategy_family=families[0] if families else "mean_reversion_vwap_bb",
            timeframe=default_tf,
        )
        return {
            "dataset_id": entry.dataset_id,
            "symbols": entry.symbols,
            "tradable_contracts": entry.tradable_contracts,
            "timeframe": entry.timeframe,
            "compatible_timeframes": compat_tfs,
            "default_timeframe": default_tf,
            "asset_class": entry.asset_class,
            "research_eligible": entry.research_eligible,
            "smoke_test_only": entry.smoke_test_only,
            "intraday_only": bool(getattr(entry, "intraday_only", False)),
            "compatible_strategy_families": families,
            "strategy_timeframes": family_tfs,
            "compatible_cost_models": entry.compatible_cost_models,
            "compatible_risk_profiles": entry.compatible_risk_profiles,
            "compatible_feature_sets": entry.compatible_feature_sets,
            "quality_status": entry.quality_status,
            "resolved_execution_backend": backend_preview.get("evaluation_backend"),
            "backend_preview": backend_preview,
            "execution_banners": backend_preview.get("banners") or [],
        }

    @app.get("/api/import/files")
    def list_import_files() -> dict[str, Any]:
        from data.catalog.safe_paths import list_importable_files

        root = _import_root()
        return {"import_root": str(root), "files": list_importable_files(root)}

    @app.post("/api/import/preview")
    def import_preview(payload: dict[str, Any]) -> dict[str, Any]:
        from data.catalog.import_pipeline import ColumnMapping, preview_import

        try:
            reject_arbitrary_command_payload(payload)
            mapping_raw = payload.get("column_mapping") or {}
            mapping = ColumnMapping(
                timestamp=str(mapping_raw.get("timestamp", "timestamp")),
                open=str(mapping_raw.get("open", "open")),
                high=str(mapping_raw.get("high", "high")),
                low=str(mapping_raw.get("low", "low")),
                close=str(mapping_raw.get("close", "close")),
                volume=str(mapping_raw.get("volume", "volume")),
                symbol=mapping_raw.get("symbol"),
                tradable_contract=mapping_raw.get("tradable_contract"),
            )
            return preview_import(
                import_root=_import_root(),
                relative_path=str(payload.get("relative_path") or ""),
                mapping=mapping,
                n=int(payload.get("n") or 8),
            )
        except (ConfigValidationError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/import/validate")
    def import_validate(payload: dict[str, Any]) -> dict[str, Any]:
        from data.catalog.import_pipeline import validate_import

        try:
            reject_arbitrary_command_payload(payload)
            req = _parse_import_request(payload)
            return validate_import(import_root=_import_root(), req=req)
        except (ConfigValidationError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/import/commit")
    def import_commit(payload: dict[str, Any]) -> dict[str, Any]:
        from data.catalog.import_pipeline import commit_import

        try:
            reject_arbitrary_command_payload(payload)
            req = _parse_import_request(payload)
            if not req.confirm:
                raise ConfigValidationError("confirm=true required to write immutable Bronze")
            entry = commit_import(
                import_root=_import_root(),
                catalog=_mgr().catalog,
                req=req,
            )
            return entry.as_dict()
        except (ConfigValidationError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # ---------- Phase 12.2: CFD data acquisition ----------

    def _acq():  # type: ignore[no-untyped-def]
        from api.data_acquisition_api import DataAcquisitionAPI
        from data.acquisition.download_job import DownloadJobStore

        existing = getattr(app.state, "acquisition_api", None)
        if existing is not None:
            return existing
        acq_root = Path(app.state.root) / "acquisition"
        api_obj = DataAcquisitionAPI(
            catalog=_mgr().catalog,
            job_store=DownloadJobStore(acq_root / "jobs"),
            storage_root=acq_root,
            import_root=_import_root(),
        )
        app.state.acquisition_api = api_obj
        return api_obj

    @app.get("/api/acquisition/source")
    def acquisition_source() -> dict[str, Any]:
        from dashboard.data_acquisition import data_acquisition_view

        return data_acquisition_view(_acq())

    @app.get("/api/acquisition/jobs")
    def acquisition_jobs() -> dict[str, Any]:
        jobs = _acq().list_jobs()
        return {"jobs": jobs, "count": len(jobs)}

    @app.get("/api/acquisition/jobs/{job_id}")
    def acquisition_job(job_id: str) -> dict[str, Any]:
        from dashboard.data_acquisition import job_progress_view

        try:
            job = _acq().get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="download job not found") from exc
        return {"job": job, "progress": job_progress_view(job)}

    @app.post("/api/acquisition/jobs")
    def acquisition_create_job(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            reject_arbitrary_command_payload(payload)
            return _acq().create_job(
                symbol=str(payload.get("symbol") or ""),
                start=str(payload.get("start") or ""),
                end=str(payload.get("end") or ""),
                granularity=str(payload.get("granularity") or "tick"),
                acquisition_mode=str(
                    payload.get("acquisition_mode") or "MANUAL_EXPORT_IMPORT"
                ),
                stage=payload.get("stage"),
                rate_limit_rps=float(payload.get("rate_limit_rps") or 2.0),
                retry_limit=int(payload.get("retry_limit") or 3),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ConfigValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/acquisition/probe")
    def acquisition_probe(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            reject_arbitrary_command_payload(payload)
            return _acq().probe_one_day(
                symbol=str(payload.get("symbol") or ""),
                day=str(payload.get("day") or payload.get("start") or ""),
                relative_path=payload.get("relative_path"),
                job_id=payload.get("job_id"),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ConfigValidationError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/acquisition/jobs/{job_id}/run")
    def acquisition_run(job_id: str) -> dict[str, Any]:
        try:
            return _acq().run_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="download job not found") from exc
        except (ConfigValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/acquisition/jobs/{job_id}/pause")
    def acquisition_pause(job_id: str) -> dict[str, Any]:
        try:
            return _acq().pause_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="download job not found") from exc
        except (ConfigValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/acquisition/jobs/{job_id}/resume")
    def acquisition_resume(job_id: str) -> dict[str, Any]:
        try:
            return _acq().resume_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="download job not found") from exc
        except (ConfigValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/acquisition/jobs/{job_id}/cancel")
    def acquisition_cancel(job_id: str) -> dict[str, Any]:
        try:
            return _acq().cancel_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="download job not found") from exc
        except (ConfigValidationError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/acquisition/register")
    def acquisition_register(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            reject_arbitrary_command_payload(payload)
            if not payload.get("confirm"):
                raise ConfigValidationError(
                    "confirm=true required to write immutable Bronze and register"
                )
            result = _acq().register_dataset(
                job_id=payload.get("job_id"),
                symbol=payload.get("symbol"),
                relative_path=payload.get("relative_path"),
                display_name=payload.get("display_name"),
            )
            return result.as_dict()
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ConfigValidationError, FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.get("/api/acquisition/source_target")
    def acquisition_source_target() -> dict[str, Any]:
        from dashboard.source_target_comparison import source_target_comparison_view

        acq = _acq()
        return source_target_comparison_view(
            source=acq.source, instrument=acq.instrument, target=acq.target
        )

    @app.get("/api/datasets/{dataset_id}/preview")
    def dataset_preview(dataset_id: str, n: int = Query(10, ge=1, le=50)) -> dict[str, Any]:
        from api.dataset_preview_api import DatasetPreviewAPI

        try:
            return DatasetPreviewAPI(catalog=_mgr().catalog).preview(dataset_id, n=n)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="dataset not found") from exc

    # ---------- Frontend ----------
    if STATIC_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/")
    def index() -> HTMLResponse:
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse(_FALLBACK_HTML)

    @app.get("/{page}")
    def spa_pages(page: str) -> HTMLResponse:
        # Client-side router pages — serve SPA shell
        if page.startswith("api"):
            raise HTTPException(status_code=404)
        index_path = STATIC_DIR / "index.html"
        if index_path.exists():
            return HTMLResponse(index_path.read_text(encoding="utf-8"))
        return HTMLResponse(_FALLBACK_HTML)

    return app


_FALLBACK_HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8"/><title>Research Control Plane</title></head>
<body><h1>Research Control Plane</h1>
<p>RESEARCH / BACKTEST ONLY — LIVE TRADING IS NOT ENABLED</p>
<p>Static dashboard missing — API is available under /api/</p>
</body></html>
"""
