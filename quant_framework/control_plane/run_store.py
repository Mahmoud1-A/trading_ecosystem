"""Persistent run store — JSON index + per-run directories."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from control_plane.models import RunRecord, RunState


class RunStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "runs_index.json"
        self._lock = threading.RLock()
        self._runs: dict[str, RunRecord] = {}
        self._load()

    def _load(self) -> None:
        if not self.index_path.exists():
            return
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        for item in raw.get("runs", []):
            rec = RunRecord.from_dict(item)
            self._runs[rec.run_id] = rec

    def _persist_index(self) -> None:
        payload = {"runs": [r.as_dict() for r in self._runs.values()]}
        self.index_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    def save(self, record: RunRecord) -> RunRecord:
        with self._lock:
            self._runs[record.run_id] = record
            run_dir = Path(record.artifact_dir)
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "run_record.json").write_text(
                json.dumps(record.as_dict(), indent=2, default=str), encoding="utf-8"
            )
            self._persist_index()
            return record

    def get(self, run_id: str) -> RunRecord | None:
        with self._lock:
            return self._runs.get(run_id)

    def list_runs(self) -> list[RunRecord]:
        with self._lock:
            return sorted(self._runs.values(), key=lambda r: r.created_at, reverse=True)

    def mark_interrupted_non_terminal(self) -> list[str]:
        """On control-plane restart: active jobs → INTERRUPTED unless recovered."""
        changed: list[str] = []
        with self._lock:
            for rec in self._runs.values():
                # Active / in-flight only — CREATED stays visible and is not silently duplicated
                if rec.state in {
                    RunState.QUEUED,
                    RunState.STARTING,
                    RunState.RUNNING,
                    RunState.CANCEL_REQUESTED,
                }:
                    rec.state = RunState.INTERRUPTED
                    rec.terminal_reason = "control_plane_restart"
                    rec.completed_at = rec.completed_at or rec.created_at
                    rec.software_success = False
                    changed.append(rec.run_id)
            if changed:
                self._persist_index()
                for rid in changed:
                    self.save(self._runs[rid])
        return changed

    def events_path(self, run_id: str) -> Path:
        rec = self.get(run_id)
        if rec is None:
            raise KeyError(run_id)
        return Path(rec.artifact_dir) / "events.jsonl"
