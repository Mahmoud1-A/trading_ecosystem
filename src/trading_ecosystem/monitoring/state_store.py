from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any

from trading_ecosystem.common.config import data_dir

logger = logging.getLogger(__name__)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_funnel() -> dict[str, int]:
    """Session discovery funnel (same time scope for all counters)."""
    return {
        "generated": 0,
        "evaluated": 0,
        "preliminary_eligible": 0,
        "score_qualified": 0,
        "behaviorally_unique": 0,
        "book_accepted": 0,
        # Vault pipeline (unique book fingerprints this session)
        "vault_eligible": 0,
        "vault_pending": 0,
        "vault_tested": 0,
        "vault_passed": 0,
        "vault_failed": 0,
        "paper_promoted": 0,
    }


def _default_discovery() -> dict[str, Any]:
    return {
        "status": "idle",  # idle | running | done | error
        "phase": None,
        "message": "No discovery run yet",
        "current": 0,
        "total": 0,
        "percent": 0.0,
        "current_strategy_id": None,
        "current_class": None,
        "current_symbol": None,
        "current_source": None,
        "passed_so_far": 0,  # = preliminary_eligible (session)
        "failed_so_far": 0,  # = evaluated - eligible (session)
        "best_oos_cagr_so_far": None,
        "started_at": None,
        "updated_at": None,
        "finished_at": None,
        "elapsed_sec": 0.0,
        "avg_sec_per_candidate": None,
        "eta_sec": None,
        "eta_human": None,
        "last_summary": None,
        "mode": None,  # batch | continuous
        "generation": None,
        "total_evaluated": 0,
        "best_ever": None,
        "top50": [],
        "best3_by_sector": {},
        "portfolio_metrics": None,
        "selected_portfolio": [],
        "readiness": None,
        "workers": None,
        "funnel": _default_funnel(),
        "fitness_folds_note": "validate + yearly oos only; vault sealed (promotion-only)",
    }


class MonitorState:
    """Thread-safe in-memory + disk snapshot for dashboard / alerts."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self.equity: float | None = None
        self.cash: float | None = None
        self.daily_dd_pct: float = 0.0
        self.total_dd_pct: float = 0.0
        self.halted: bool = False
        self.daily_halted: bool = False
        self.agents: dict[str, Any] = {}
        self.recent_decisions: list[dict[str, Any]] = []
        self.last_bar_ts: str | None = None
        self.alerts: list[dict[str, Any]] = []
        self.discovery: dict[str, Any] = _default_discovery()
        self.paper: dict[str, Any] = {}
        self.path = data_dir() / "processed" / "monitor_state.json"

    def update(self, **kwargs: Any) -> None:
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self, k):
                    setattr(self, k, v)
            self._persist_unlocked()

    def push_decision(self, decision: dict[str, Any]) -> None:
        with self._lock:
            self.recent_decisions.insert(0, decision)
            self.recent_decisions = self.recent_decisions[:100]
            self._persist_unlocked()

    def push_alert(self, level: str, message: str) -> None:
        with self._lock:
            self.alerts.insert(
                0,
                {
                    "ts": _utc_now(),
                    "level": level,
                    "message": message,
                },
            )
            self.alerts = self.alerts[:200]
            self._persist_unlocked()

    def discovery_start(self, *, phase: str, total: int, message: str | None = None) -> None:
        with self._lock:
            last = self.discovery.get("last_summary")
            self.discovery = _default_discovery()
            self.discovery["last_summary"] = last
            self.discovery.update(
                {
                    "status": "running",
                    "phase": phase,
                    "message": message or f"Evaluating candidates ({phase})",
                    "total": int(total),
                    "current": 0,
                    "percent": 0.0,
                    "started_at": _utc_now(),
                    "updated_at": _utc_now(),
                }
            )
            self._persist_unlocked()

    def discovery_progress(
        self,
        *,
        current: int,
        total: int,
        strategy_id: str,
        class_name: str,
        symbols: list[str],
        source: str,
        passed: bool | None = None,
        oos_cagr: float | None = None,
        message: str | None = None,
    ) -> None:
        with self._lock:
            d = self.discovery
            d["status"] = "running"
            d["current"] = int(current)
            d["total"] = int(total)
            d["percent"] = round(100.0 * current / total, 2) if total else 0.0
            d["current_strategy_id"] = strategy_id
            d["current_class"] = class_name
            d["current_symbol"] = ",".join(symbols)
            d["current_source"] = source
            d["updated_at"] = _utc_now()
            if message:
                d["message"] = message
            else:
                d["message"] = f"Trying {class_name} on {d['current_symbol']} ({current}/{total})"

            # Do NOT increment Eligible/Rejected here — parallel progress ticks and
            # book-phase callbacks made counters diverge from total_evaluated.
            # Use discovery_tally_batch() after each finished evaluation batch.

            if oos_cagr is not None:
                prev = d.get("best_oos_cagr_so_far")
                if prev is None or oos_cagr > float(prev):
                    d["best_oos_cagr_so_far"] = float(oos_cagr)

            started = d.get("started_at")
            if started and current > 0:
                start_dt = datetime.fromisoformat(started)
                elapsed = max(0.0, (datetime.now(timezone.utc) - start_dt).total_seconds())
                avg = elapsed / current
                remaining = max(0, total - current)
                eta = remaining * avg
                d["elapsed_sec"] = round(elapsed, 1)
                d["avg_sec_per_candidate"] = round(avg, 2)
                d["eta_sec"] = round(eta, 1)
                d["eta_human"] = _format_duration(eta)
            self._persist_unlocked()

    def discovery_tally_batch(
        self,
        *,
        generated: int,
        evaluated_rows: list[dict[str, Any]],
        behaviorally_unique: int,
        book_members: int | None = None,
    ) -> None:
        """Accumulate session research funnel; Eligible+Rejected == evaluated."""
        with self._lock:
            d = self.discovery
            funnel = dict(d.get("funnel") or _default_funnel())
            n_eval = len(evaluated_rows)
            n_elig = sum(1 for r in evaluated_rows if (r.get("fitness") or {}).get("passed"))
            n_score = sum(
                1
                for r in evaluated_rows
                if (r.get("fitness") or {}).get("passed")
                and (
                    (r.get("fitness") or {}).get("hit_target_cagr")
                    or float((r.get("fitness") or {}).get("oos_cagr") or -1e9) >= 0.30
                )
            )
            funnel["generated"] = int(funnel.get("generated") or 0) + max(0, int(generated))
            funnel["evaluated"] = int(funnel.get("evaluated") or 0) + n_eval
            funnel["preliminary_eligible"] = int(funnel.get("preliminary_eligible") or 0) + n_elig
            funnel["score_qualified"] = int(funnel.get("score_qualified") or 0) + n_score
            funnel["behaviorally_unique"] = int(funnel.get("behaviorally_unique") or 0) + max(
                0, int(behaviorally_unique)
            )
            if book_members is not None:
                funnel["book_accepted"] = int(book_members)
            # Keep vault pending = eligible − tested (never negative)
            elig = int(funnel.get("vault_eligible") or 0)
            tested = int(funnel.get("vault_tested") or 0)
            funnel["vault_pending"] = max(0, elig - tested)

            d["funnel"] = funnel
            d["passed_so_far"] = int(funnel["preliminary_eligible"])
            d["failed_so_far"] = int(funnel["evaluated"]) - int(funnel["preliminary_eligible"])
            d["updated_at"] = _utc_now()
            self._persist_unlocked()

    def record_vault_pipeline(
        self,
        *,
        fingerprint: str,
        eligible: bool,
        tested: bool,
        passed: bool | None = None,
        skipped: bool = False,  # noqa: ARG002 — kept for callers; sets make updates idempotent
    ) -> None:
        """
        Track sealed-vault pipeline for a book fingerprint (once per stage).

        vault_eligible: met stability gate → may enter vault
        vault_pending: eligible − tested
        vault_tested: vault simulation actually ran
        vault_passed / vault_failed: vault test outcome
        """
        fp = (fingerprint or "").strip()
        with self._lock:
            d = self.discovery
            funnel = dict(d.get("funnel") or _default_funnel())
            if not fp:
                funnel["vault_pending"] = max(
                    0, int(funnel.get("vault_eligible") or 0) - int(funnel.get("vault_tested") or 0)
                )
                d["funnel"] = funnel
                d["updated_at"] = _utc_now()
                self._persist_unlocked()
                return

            seen_e = set(d.get("vault_fp_eligible") or [])
            seen_t = set(d.get("vault_fp_tested") or [])
            seen_p = set(d.get("vault_fp_passed") or [])
            seen_f = set(d.get("vault_fp_failed") or [])

            if eligible:
                seen_e.add(fp)
            if tested:
                seen_t.add(fp)
                if passed is True:
                    seen_p.add(fp)
                    seen_f.discard(fp)
                elif passed is False:
                    seen_f.add(fp)
                    seen_p.discard(fp)

            funnel["vault_eligible"] = len(seen_e)
            funnel["vault_tested"] = len(seen_t)
            funnel["vault_passed"] = len(seen_p)
            funnel["vault_failed"] = len(seen_f)
            funnel["vault_pending"] = max(0, len(seen_e) - len(seen_t))

            d["vault_fp_eligible"] = sorted(seen_e)[-200:]
            d["vault_fp_tested"] = sorted(seen_t)[-200:]
            d["vault_fp_passed"] = sorted(seen_p)[-200:]
            d["vault_fp_failed"] = sorted(seen_f)[-200:]
            d["funnel"] = funnel
            d["updated_at"] = _utc_now()
            self._persist_unlocked()

    def record_paper_promotion(self) -> None:
        """Count a real freeze-to-paper event (not readiness alone)."""
        with self._lock:
            d = self.discovery
            funnel = dict(d.get("funnel") or _default_funnel())
            funnel["paper_promoted"] = int(funnel.get("paper_promoted") or 0) + 1
            d["funnel"] = funnel
            d["updated_at"] = _utc_now()
            self._persist_unlocked()

    def discovery_patch(self, **fields: Any) -> None:
        """Merge arbitrary discovery fields (generation, best_ever, mode, …)."""
        with self._lock:
            self.discovery.update(fields)
            self.discovery["updated_at"] = _utc_now()
            self._persist_unlocked()

    def discovery_finish(self, summary: dict[str, Any] | None = None, *, error: str | None = None) -> None:
        with self._lock:
            d = self.discovery
            d["finished_at"] = _utc_now()
            d["updated_at"] = d["finished_at"]
            if error:
                d["status"] = "error"
                d["message"] = error
            else:
                d["status"] = "done"
                d["percent"] = 100.0
                d["message"] = "Discovery finished"
                if summary:
                    d["last_summary"] = summary
                    evaluated = summary.get("candidates_evaluated")
                    if evaluated is None:
                        evaluated = summary.get("total_evaluated")
                    if evaluated is not None:
                        d["current"] = int(evaluated)
                        d["total"] = int(evaluated)
            started = d.get("started_at")
            if started:
                start_dt = datetime.fromisoformat(started)
                d["elapsed_sec"] = round(
                    max(0.0, (datetime.now(timezone.utc) - start_dt).total_seconds()),
                    1,
                )
            d["eta_sec"] = 0.0
            d["eta_human"] = "0s"
            self._persist_unlocked()

    def _to_dict(self) -> dict[str, Any]:
        return {
            "equity": self.equity,
            "cash": self.cash,
            "daily_dd_pct": self.daily_dd_pct,
            "total_dd_pct": self.total_dd_pct,
            "halted": self.halted,
            "daily_halted": self.daily_halted,
            "agents": self.agents,
            "recent_decisions": self.recent_decisions[:20],
            "last_bar_ts": self.last_bar_ts,
            "alerts": self.alerts[:20],
            "discovery": self.discovery,
            "paper": self.paper,
        }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._to_dict()

    def _merge_for_persist(self, mem: dict[str, Any]) -> dict[str, Any]:
        """Merge with on-disk state so paper/discover processes don't wipe each other."""
        disk: dict[str, Any] = {}
        if self.path.exists():
            try:
                disk = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                disk = {}
        out = dict(disk)
        out.update(
            {
                k: v
                for k, v in mem.items()
                if k not in {"discovery", "paper"} and v is not None
            }
        )

        disk_disc = dict(disk.get("discovery") or {})
        mem_disc = dict(mem.get("discovery") or {})
        disk_status = str(disk_disc.get("status") or "idle")
        mem_status = str(mem_disc.get("status") or "idle")
        if mem_status == "running":
            out["discovery"] = mem_disc
        elif disk_status == "running" and mem_status != "running":
            out["discovery"] = disk_disc
        else:
            # Prefer the newer discovery blob when both idle/done.
            disk_ts = str(disk_disc.get("updated_at") or "")
            mem_ts = str(mem_disc.get("updated_at") or "")
            out["discovery"] = mem_disc if mem_ts >= disk_ts else disk_disc

        disk_paper = dict(disk.get("paper") or {})
        mem_paper = dict(mem.get("paper") or {})
        merged_paper = dict(disk_paper)
        merged_paper.update(mem_paper)
        out["paper"] = merged_paper

        # Keep paper equity if memory cleared it accidentally.
        if out.get("equity") is None and disk.get("equity") is not None:
            out["equity"] = disk.get("equity")
            out["cash"] = disk.get("cash")
        return out

    def _persist_unlocked(self) -> None:
        """Atomic-ish write with retries — Windows often locks during monitor+discover."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        merged = self._merge_for_persist(self._to_dict())
        # Keep in-memory discovery if disk won the merge (paper process case).
        if isinstance(merged.get("discovery"), dict):
            self.discovery = merged["discovery"]
        payload = json.dumps(merged, indent=2, default=str)
        for attempt in range(6):
            tmp = self.path.with_name(f"{self.path.stem}.{os.getpid()}.{attempt}.tmp")
            try:
                tmp.write_text(payload, encoding="utf-8")
                os.replace(tmp, self.path)
                return
            except PermissionError:
                time.sleep(0.05 * (attempt + 1))
            except OSError:
                time.sleep(0.05 * (attempt + 1))
            finally:
                try:
                    if tmp.exists():
                        tmp.unlink()
                except OSError:
                    pass
        try:
            self.path.write_text(payload, encoding="utf-8")
        except OSError as exc:
            logger.warning("monitor_state persist skipped: %s", exc)

    def load(self) -> None:
        if not self.path.exists():
            return
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        with self._lock:
            for k, v in raw.items():
                if k == "discovery" and isinstance(v, dict):
                    base = _default_discovery()
                    base.update(v)
                    self.discovery = base
                elif hasattr(self, k):
                    setattr(self, k, v)


def _format_duration(seconds: float) -> str:
    sec = int(max(0, seconds))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


MONITOR = MonitorState()
