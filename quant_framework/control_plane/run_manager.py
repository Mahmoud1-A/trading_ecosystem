"""Supervised run manager — workers outside the HTTP request lifecycle."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from config.system_config import default_system_config
from control_plane.events import (
    EventType,
    InMemoryEventSink,
    PersistentEventSink,
    ResearchEvent,
    ResearchEventSink,
)
from control_plane.jobs import execute_run
from control_plane.models import (
    FORBIDDEN_LIVE_MODES,
    RunRecord,
    RunState,
    RunType,
    TERMINAL_STATES,
    build_run_record,
    now_iso,
)
from control_plane.run_store import RunStore
from control_plane.security import ConfigValidationError, CreateRunRequest


class RunManager:
    def __init__(
        self,
        root: Path,
        *,
        max_workers: int = 2,
        memory_guard_mb: float = 4096.0,
        catalog: Any | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.store = RunStore(self.root / "runs")
        self.max_workers = max_workers
        self.memory_guard_mb = memory_guard_mb
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="research-worker")
        self._futures: dict[str, Future[RunRecord]] = {}
        self._cancel_flags: dict[str, threading.Event] = {}
        self._mem_sinks: dict[str, InMemoryEventSink] = {}
        self._lock = threading.RLock()
        self.worker_id = f"worker_{os.getpid()}"
        # Dataset catalog (Phase 12.1)
        if catalog is not None:
            self.catalog = catalog
        else:
            from data.catalog.catalog import DatasetCatalog

            self.catalog = DatasetCatalog(self.root / "dataset_catalog")
        # Restart recovery
        self.store.mark_interrupted_non_terminal()

    def _sink_for(self, run: RunRecord) -> ResearchEventSink:
        from control_plane.events import CompositeEventSink  # noqa: F401

        mem = InMemoryEventSink()
        path = Path(run.artifact_dir) / "events.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)

        class PersistThenMemory(ResearchEventSink):
            def __init__(self) -> None:
                self.persistent = PersistentEventSink(path)
                self.memory = mem

            def emit(self, event: ResearchEvent) -> None:
                self.persistent.emit(event)
                self.memory.events.append(event)
                self.memory._seq = max(self.memory._seq, event.seq)
                for cb in list(self.memory._subscribers):
                    try:
                        cb(event)
                    except Exception:  # noqa: BLE001
                        pass

        sink = PersistThenMemory()
        with self._lock:
            self._mem_sinks[run.run_id] = sink.memory
        return sink

    def create_run(self, req: CreateRunRequest) -> RunRecord:
        req.require_confirmations()
        if req.environment and req.environment.upper() in FORBIDDEN_LIVE_MODES:
            raise ConfigValidationError("live mode forbidden")
        if req.trading_mode and req.trading_mode.upper() in FORBIDDEN_LIVE_MODES:
            raise ConfigValidationError("live mode forbidden")
        req.validate_against_catalog(self.catalog)

        cfg = default_system_config("futures")
        entry = self.catalog.get(req.dataset)
        snapshot = {
            "run_type": req.run_type.value,
            "symbols": req.symbols,
            "timeframe": req.timeframe,
            "dataset": req.dataset,
            "smoke_test": req.smoke_test,
            "dataset_research_eligible": bool(entry.research_eligible) if entry else False,
            "dataset_smoke_test_only": bool(entry.smoke_test_only) if entry else True,
            "strategy_family": req.strategy_family,
            "search_budget": req.search_budget,
            "wfo": req.wfo,
            "cost_model_version": req.cost_model_version,
            "risk_profile": req.risk_profile,
            "feature_set_version": req.feature_set_version,
            "grammar_version": req.grammar_version,
            "random_seed": req.random_seed,
            "date_range_start": req.date_range_start,
            "date_range_end": req.date_range_end,
            "hold_seconds": req.hold_seconds,
            "ui_canary": dict(req.ui_canary or {}),
            "multi_family": dict(req.multi_family or {}),
            "environment": "research",
            "live_trading_enabled": False,
        }
        run = build_run_record(
            run_type=req.run_type,
            system_version=cfg.system_version,
            config_snapshot=snapshot,
            artifact_root=str(self.root / "artifacts"),
            random_seed=req.random_seed,
            dataset_ids=[req.dataset],
            feature_set_version=req.feature_set_version,
            grammar_version=req.grammar_version,
            cost_model_version=req.cost_model_version,
        )
        Path(run.artifact_dir).mkdir(parents=True, exist_ok=True)
        self.store.save(run)
        if entry is not None:
            self.catalog.record_dependent_run(req.dataset, run.run_id)
        sink = self._sink_for(run)
        sink.emit(
            ResearchEvent.create(
                run_id=run.run_id,
                event_type=EventType.RUN_CREATED,
                message=f"Created {run.run_type.value}",
                stage="created",
                progress=0,
            )
        )
        return run

    def enqueue(self, run_id: str) -> RunRecord:
        run = self.store.get(run_id)
        if run is None:
            raise KeyError(run_id)
        if run.state not in {RunState.CREATED, RunState.QUEUED}:
            raise ConfigValidationError(f"cannot enqueue run in state {run.state}")
        # Concurrency guard
        active = sum(
            1
            for r in self.store.list_runs()
            if r.state in {RunState.QUEUED, RunState.STARTING, RunState.RUNNING}
        )
        if active >= self.max_workers and run.state is RunState.CREATED:
            run.state = RunState.QUEUED
            run.queued_at = now_iso()
            self.store.save(run)
        else:
            run.state = RunState.QUEUED
            run.queued_at = now_iso()
            self.store.save(run)

        cancel_flag = threading.Event()
        with self._lock:
            self._cancel_flags[run_id] = cancel_flag

        def _work() -> RunRecord:
            current = self.store.get(run_id)
            assert current is not None
            current.state = RunState.STARTING
            current.worker_id = self.worker_id
            current.process_id = os.getpid()
            self.store.save(current)
            sink = self._sink_for(current)

            def cancel_check() -> bool:
                return cancel_flag.is_set() or bool(
                    self.store.get(run_id) and self.store.get(run_id).cancel_requested  # type: ignore[union-attr]
                )

            def on_update(rec: RunRecord) -> None:
                self.store.save(rec)

            finished = execute_run(
                current, sink, cancel_check=cancel_check, on_update=on_update
            )
            # Completion requires summary + manifest + terminal event already emitted by jobs
            if finished.state is RunState.RUNNING:
                finished.state = RunState.FAILED
                finished.terminal_reason = "missing_terminal_event"
            if finished.state is RunState.COMPLETED:
                if not finished.summary or not finished.artifact_manifest:
                    finished.state = RunState.FAILED
                    finished.terminal_reason = "incomplete_completion_requirements"
                    finished.software_success = False
            self.store.save(finished)
            return finished

        fut = self._executor.submit(_work)
        with self._lock:
            self._futures[run_id] = fut
        return self.store.get(run_id)  # type: ignore[return-value]

    def cancel(self, run_id: str) -> RunRecord:
        run = self.store.get(run_id)
        if run is None:
            raise KeyError(run_id)
        if run.state in TERMINAL_STATES:
            raise ConfigValidationError(
                f"cannot cancel terminal run in state {run.state.value}"
            )
        run.cancel_requested = True
        run.state = RunState.CANCEL_REQUESTED
        self.store.save(run)
        with self._lock:
            flag = self._cancel_flags.get(run_id)
            if flag:
                flag.set()
        # Wait briefly for cooperative cancel
        fut = self._futures.get(run_id)
        if fut is not None:
            try:
                fut.result(timeout=5.0)
            except Exception:  # noqa: BLE001
                # Forced termination path — mark cancelled
                run = self.store.get(run_id) or run
                if run.state not in TERMINAL_STATES:
                    run.state = RunState.CANCELLED
                    run.terminal_reason = "forced_cancel_timeout"
                    run.completed_at = now_iso()
                    self.store.save(run)
        else:
            run.state = RunState.CANCELLED
            run.terminal_reason = "cancelled_before_start"
            run.completed_at = now_iso()
            self.store.save(run)
        return self.store.get(run_id)  # type: ignore[return-value]

    def get_events(self, run_id: str, after_seq: int = 0) -> list[dict[str, Any]]:
        run = self.store.get(run_id)
        if run is None:
            raise KeyError(run_id)
        path = Path(run.artifact_dir) / "events.jsonl"
        if path.exists():
            from control_plane.events import PersistentEventSink

            return [e.as_dict() for e in PersistentEventSink(path).since(after_seq)]
        mem = self._mem_sinks.get(run_id)
        if mem:
            return [e.as_dict() for e in mem.since(after_seq)]
        return []

    def subscribe(self, run_id: str, callback: Any) -> None:
        mem = self._mem_sinks.get(run_id)
        if mem is None:
            # Load persistent into memory sink for live updates
            run = self.store.get(run_id)
            if run is None:
                raise KeyError(run_id)
            self._sink_for(run)
            mem = self._mem_sinks[run_id]
        mem.subscribe(callback)

    def health(self) -> dict[str, Any]:
        runs = self.store.list_runs()
        counts: dict[str, int] = {}
        for r in runs:
            counts[r.state.value] = counts.get(r.state.value, 0) + 1
        return {
            "system_ok": True,
            "research_only": True,
            "live_trading_enabled": False,
            "worker_id": self.worker_id,
            "pid": os.getpid(),
            "max_workers": self.max_workers,
            "memory_guard_mb": self.memory_guard_mb,
            "run_counts": counts,
            "active_futures": len([f for f in self._futures.values() if not f.done()]),
        }

    def capabilities(self) -> dict[str, Any]:
        return {
            "run_types": [t.value for t in RunType],
            "forbidden_live_modes": sorted(FORBIDDEN_LIVE_MODES),
            "event_stream": "SSE",
            "vault_raw_data_exposed": False,
            "live_activation": False,
            "dataset_catalog": True,
            "phase": "12.1",
        }
