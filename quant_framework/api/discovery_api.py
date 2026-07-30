"""Discovery control-plane API — authoritative counters from the Experiment Registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from api.types import CounterKind, TypedCounter
from registry.experiment_registry import ExperimentRegistry, TrialRecord


@dataclass
class DiscoverySessionState:
    """Session-scoped event counters (reset per dashboard session / run view)."""

    generated: int = 0
    precheck_rejected: int = 0
    fully_evaluated: int = 0
    score_qualified: int = 0
    behaviorally_unique: int = 0
    finalists: int = 0

    def bump(self, name: str, n: int = 1) -> None:
        setattr(self, name, getattr(self, name) + n)


@dataclass
class DiscoveryAPI:
    """
    Read model over the registry for the candidate discovery funnel.

    Snapshot values (e.g. current finalist set size) are never mixed into
    funnel event counts.
    """

    registry: ExperimentRegistry
    session: DiscoverySessionState = field(default_factory=DiscoverySessionState)
    score_qualify_threshold: float = 0.0
    _seen_candidate_ids: set[str] = field(default_factory=set)

    def ingest_trial(self, trial: TrialRecord, *, is_finalist: bool = False) -> None:
        """Update session event counters from a newly observed trial."""
        self.session.bump("generated")
        reason = trial.rejection_reason or ""
        if reason.startswith("precheck") or "precheck:" in reason:
            self.session.bump("precheck_rejected")
            return
        if trial.failed and reason:
            # Still counts as generated; may not be fully evaluated
            if "eval_failed" in reason or "wfo" in reason.lower():
                return
        self.session.bump("fully_evaluated")
        score = trial.ranking_score
        if score is not None and score >= self.score_qualify_threshold and not trial.rejection_reason:
            self.session.bump("score_qualified")
            if trial.candidate_id not in self._seen_candidate_ids:
                self._seen_candidate_ids.add(trial.candidate_id)
                self.session.bump("behaviorally_unique")
            if is_finalist:
                self.session.bump("finalists")

    def sync_from_registry(self, *, finalist_ids: set[str] | None = None) -> None:
        """Rebuild session counters by replaying the ledger (idempotent reset)."""
        self.session = DiscoverySessionState()
        self._seen_candidate_ids = set()
        finalist_ids = finalist_ids or set()
        for trial in self.registry.all_trials():
            self.ingest_trial(trial, is_finalist=trial.candidate_id in finalist_ids)

    def funnel_counters(self) -> list[TypedCounter]:
        s = self.session
        trials = self.registry.all_trials()
        accepted = [t for t in trials if not t.rejection_reason and not t.failed]
        return [
            TypedCounter("generated", s.generated, CounterKind.SESSION_EVENT, "candidates observed this session"),
            TypedCounter("valid", len(accepted), CounterKind.CUMULATIVE_EVENT, "registry accepted trials"),
            TypedCounter("precheck_rejected", s.precheck_rejected, CounterKind.SESSION_EVENT),
            TypedCounter("full_wfo_evaluated", s.fully_evaluated, CounterKind.SESSION_EVENT),
            TypedCounter("score_qualified", s.score_qualified, CounterKind.SESSION_EVENT),
            TypedCounter("behaviorally_unique", s.behaviorally_unique, CounterKind.SESSION_EVENT),
            TypedCounter(
                "finalists",
                s.finalists,
                CounterKind.SESSION_EVENT,
                "finalist promotions this session (event)",
            ),
            TypedCounter(
                "current_finalist_set_size",
                len({t.candidate_id for t in accepted if t.ranking_score is not None}),
                CounterKind.SNAPSHOT,
                "distinct accepted scored candidates now (snapshot — not a funnel stage)",
            ),
            TypedCounter(
                "registry_total_trials",
                len(trials),
                CounterKind.CUMULATIVE_EVENT,
                "full ledger size including rejects",
            ),
        ]

    def as_dict(self) -> dict[str, Any]:
        return {"counters": [c.as_dict() for c in self.funnel_counters()]}
