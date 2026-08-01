"""Atomic Multi-Family search checkpoints for crash-safe resume."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from discovery.search_program import PipelinePhase, now_iso
from registry.hashing import sha256_json


CHECKPOINT_VERSION = 1


@dataclass
class FamilyCheckpointState:
    family_id: str
    generated_count: int = 0
    generation_0_count: int = 0
    descendant_count: int = 0
    mutation_count: int = 0
    crossover_count: int = 0
    full_wfo_count: int = 0
    highest_generation: int = 0
    structural_parents_found: int = 0
    family_stop_reason: str | None = None
    family_gen_cap: int = 0
    family_wfo_cap: int = 0
    initial_alloc: int = 0
    evolutionary_alloc: int = 0
    best_score_so_far: float | None = None
    stagnation_count: int = 0
    current_generation: int = 0
    generation_0_complete: bool = False
    # Queues: generation -> candidate_ids (pending or full)
    gen_queues: dict[str, list[str]] = field(default_factory=dict)
    # Unevaluated candidate ids still in the active generation queue
    pending_eval_ids: list[str] = field(default_factory=list)
    evaluated_ids: list[str] = field(default_factory=list)
    parent_pool_ids: list[str] = field(default_factory=list)
    population_ids: list[str] = field(default_factory=list)
    descendant_reject_counts: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "generated_count": self.generated_count,
            "generation_0_count": self.generation_0_count,
            "descendant_count": self.descendant_count,
            "mutation_count": self.mutation_count,
            "crossover_count": self.crossover_count,
            "full_wfo_count": self.full_wfo_count,
            "highest_generation": self.highest_generation,
            "structural_parents_found": self.structural_parents_found,
            "family_stop_reason": self.family_stop_reason,
            "family_gen_cap": self.family_gen_cap,
            "family_wfo_cap": self.family_wfo_cap,
            "initial_alloc": self.initial_alloc,
            "evolutionary_alloc": self.evolutionary_alloc,
            "best_score_so_far": self.best_score_so_far,
            "stagnation_count": self.stagnation_count,
            "current_generation": self.current_generation,
            "generation_0_complete": self.generation_0_complete,
            "gen_queues": {k: list(v) for k, v in self.gen_queues.items()},
            "pending_eval_ids": list(self.pending_eval_ids),
            "evaluated_ids": list(self.evaluated_ids),
            "parent_pool_ids": list(self.parent_pool_ids),
            "population_ids": list(self.population_ids),
            "descendant_reject_counts": dict(self.descendant_reject_counts),
        }

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> FamilyCheckpointState:
        return FamilyCheckpointState(
            family_id=str(raw["family_id"]),
            generated_count=int(raw.get("generated_count") or 0),
            generation_0_count=int(raw.get("generation_0_count") or 0),
            descendant_count=int(raw.get("descendant_count") or 0),
            mutation_count=int(raw.get("mutation_count") or 0),
            crossover_count=int(raw.get("crossover_count") or 0),
            full_wfo_count=int(raw.get("full_wfo_count") or 0),
            highest_generation=int(raw.get("highest_generation") or 0),
            structural_parents_found=int(raw.get("structural_parents_found") or 0),
            family_stop_reason=raw.get("family_stop_reason"),
            family_gen_cap=int(raw.get("family_gen_cap") or 0),
            family_wfo_cap=int(raw.get("family_wfo_cap") or 0),
            initial_alloc=int(raw.get("initial_alloc") or 0),
            evolutionary_alloc=int(raw.get("evolutionary_alloc") or 0),
            best_score_so_far=(
                float(raw["best_score_so_far"])
                if raw.get("best_score_so_far") is not None
                else None
            ),
            stagnation_count=int(raw.get("stagnation_count") or 0),
            current_generation=int(raw.get("current_generation") or 0),
            generation_0_complete=bool(raw.get("generation_0_complete")),
            gen_queues={str(k): list(v) for k, v in (raw.get("gen_queues") or {}).items()},
            pending_eval_ids=list(raw.get("pending_eval_ids") or []),
            evaluated_ids=list(raw.get("evaluated_ids") or []),
            parent_pool_ids=list(raw.get("parent_pool_ids") or []),
            population_ids=list(raw.get("population_ids") or []),
            descendant_reject_counts={
                str(k): int(v) for k, v in (raw.get("descendant_reject_counts") or {}).items()
            },
        )


@dataclass
class SearchCheckpoint:
    """Full resumable Multi-Family search state.

    Atomic write after every completed candidate evaluation and generation so a
    crash loses at most the currently running candidate.
    """

    version: int
    search_program_id: str
    compatibility_fingerprint: str
    session_run_id: str
    source_run_id: str | None
    resumed_from_run_id: str | None
    search_mode: str
    pipeline_phase: str
    updated_at: str
    seed: int
    config: dict[str, Any]
    family_specs: list[dict[str, Any]]
    candidates: dict[str, dict[str, Any]]
    candidate_gates: dict[str, str]
    evaluation_records: dict[str, dict[str, Any]]
    family_states: dict[str, FamilyCheckpointState]
    generation_records: list[dict[str, Any]]
    status_history: list[dict[str, Any]]
    # Cumulative program counters
    campaign_generated: int = 0
    campaign_evaluated: int = 0
    campaign_full_wfo: int = 0
    # Current session deltas
    session_generated: int = 0
    session_evaluated: int = 0
    session_full_wfo: int = 0
    evo_reallocation_pool: int = 0
    completed_generation_count: int = 0
    stop_reason: str | None = None
    # Pending gate queues (deterministic order)
    pending_stress_ids: list[str] = field(default_factory=list)
    pending_robustness_ids: list[str] = field(default_factory=list)
    pending_statistics_ids: list[str] = field(default_factory=list)
    pending_eval_ids: list[str] = field(default_factory=list)
    # Completed gate artifacts
    stress_summaries: list[dict[str, Any]] = field(default_factory=list)
    robustness_summaries: list[dict[str, Any]] = field(default_factory=list)
    statistics_summaries: list[dict[str, Any]] = field(default_factory=list)
    clusters: list[dict[str, Any]] = field(default_factory=list)
    behavioral_signatures: list[dict[str, Any]] = field(default_factory=list)
    shortlist_rejects: list[dict[str, Any]] = field(default_factory=list)
    research_shortlist: list[dict[str, Any]] = field(default_factory=list)
    population_stats: dict[str, Any] = field(default_factory=dict)
    # Full trial population for cumulative DSR/PBO (all sessions)
    trial_population_candidate_ids: list[str] = field(default_factory=list)
    historical_evaluation_records: list[dict[str, Any]] = field(default_factory=list)
    rng_state: dict[str, Any] = field(default_factory=dict)
    budget_caps: dict[str, Any] = field(default_factory=dict)
    fingerprint_components: dict[str, str] = field(default_factory=dict)
    status_seq: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "search_program_id": self.search_program_id,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "session_run_id": self.session_run_id,
            "source_run_id": self.source_run_id,
            "resumed_from_run_id": self.resumed_from_run_id,
            "search_mode": self.search_mode,
            "pipeline_phase": self.pipeline_phase,
            "updated_at": self.updated_at,
            "seed": self.seed,
            "config": dict(self.config),
            "family_specs": list(self.family_specs),
            "candidates": dict(self.candidates),
            "candidate_gates": dict(self.candidate_gates),
            "evaluation_records": dict(self.evaluation_records),
            "family_states": {k: v.as_dict() for k, v in self.family_states.items()},
            "generation_records": list(self.generation_records),
            "status_history": list(self.status_history),
            "campaign_generated": self.campaign_generated,
            "campaign_evaluated": self.campaign_evaluated,
            "campaign_full_wfo": self.campaign_full_wfo,
            "session_generated": self.session_generated,
            "session_evaluated": self.session_evaluated,
            "session_full_wfo": self.session_full_wfo,
            "evo_reallocation_pool": self.evo_reallocation_pool,
            "completed_generation_count": self.completed_generation_count,
            "stop_reason": self.stop_reason,
            "pending_stress_ids": list(self.pending_stress_ids),
            "pending_robustness_ids": list(self.pending_robustness_ids),
            "pending_statistics_ids": list(self.pending_statistics_ids),
            "pending_eval_ids": list(self.pending_eval_ids),
            "stress_summaries": list(self.stress_summaries),
            "robustness_summaries": list(self.robustness_summaries),
            "statistics_summaries": list(self.statistics_summaries),
            "clusters": list(self.clusters),
            "behavioral_signatures": list(self.behavioral_signatures),
            "shortlist_rejects": list(self.shortlist_rejects),
            "research_shortlist": list(self.research_shortlist),
            "population_stats": dict(self.population_stats),
            "trial_population_candidate_ids": list(self.trial_population_candidate_ids),
            "historical_evaluation_records": list(self.historical_evaluation_records),
            "rng_state": dict(self.rng_state),
            "budget_caps": dict(self.budget_caps),
            "fingerprint_components": dict(self.fingerprint_components),
            "status_seq": self.status_seq,
            "content_hash": self.content_hash(),
        }

    def content_hash(self) -> str:
        payload = {
            "search_program_id": self.search_program_id,
            "pipeline_phase": self.pipeline_phase,
            "campaign_generated": self.campaign_generated,
            "campaign_evaluated": self.campaign_evaluated,
            "campaign_full_wfo": self.campaign_full_wfo,
            "candidate_ids": sorted(self.candidates.keys()),
            "pending_stress_ids": list(self.pending_stress_ids),
            "pending_robustness_ids": list(self.pending_robustness_ids),
            "pending_statistics_ids": list(self.pending_statistics_ids),
            "pending_eval_ids": list(self.pending_eval_ids),
            "gates": dict(self.candidate_gates),
        }
        return "ckpt_" + sha256_json(payload)[:24]

    @staticmethod
    def from_dict(raw: dict[str, Any]) -> SearchCheckpoint:
        family_states = {
            str(k): FamilyCheckpointState.from_dict(v)
            for k, v in (raw.get("family_states") or {}).items()
        }
        return SearchCheckpoint(
            version=int(raw.get("version") or CHECKPOINT_VERSION),
            search_program_id=str(raw["search_program_id"]),
            compatibility_fingerprint=str(raw["compatibility_fingerprint"]),
            session_run_id=str(raw.get("session_run_id") or ""),
            source_run_id=raw.get("source_run_id"),
            resumed_from_run_id=raw.get("resumed_from_run_id"),
            search_mode=str(raw.get("search_mode") or "NEW_SEARCH"),
            pipeline_phase=str(raw.get("pipeline_phase") or PipelinePhase.GENERATION_0.value),
            updated_at=str(raw.get("updated_at") or ""),
            seed=int(raw.get("seed") or 42),
            config=dict(raw.get("config") or {}),
            family_specs=list(raw.get("family_specs") or []),
            candidates=dict(raw.get("candidates") or {}),
            candidate_gates=dict(raw.get("candidate_gates") or {}),
            evaluation_records=dict(raw.get("evaluation_records") or {}),
            family_states=family_states,
            generation_records=list(raw.get("generation_records") or []),
            status_history=list(raw.get("status_history") or []),
            campaign_generated=int(raw.get("campaign_generated") or 0),
            campaign_evaluated=int(raw.get("campaign_evaluated") or 0),
            campaign_full_wfo=int(raw.get("campaign_full_wfo") or 0),
            session_generated=int(raw.get("session_generated") or 0),
            session_evaluated=int(raw.get("session_evaluated") or 0),
            session_full_wfo=int(raw.get("session_full_wfo") or 0),
            evo_reallocation_pool=int(raw.get("evo_reallocation_pool") or 0),
            completed_generation_count=int(raw.get("completed_generation_count") or 0),
            stop_reason=raw.get("stop_reason"),
            pending_stress_ids=list(raw.get("pending_stress_ids") or []),
            pending_robustness_ids=list(raw.get("pending_robustness_ids") or []),
            pending_statistics_ids=list(raw.get("pending_statistics_ids") or []),
            pending_eval_ids=list(raw.get("pending_eval_ids") or []),
            stress_summaries=list(raw.get("stress_summaries") or []),
            robustness_summaries=list(raw.get("robustness_summaries") or []),
            statistics_summaries=list(raw.get("statistics_summaries") or []),
            clusters=list(raw.get("clusters") or []),
            behavioral_signatures=list(raw.get("behavioral_signatures") or []),
            shortlist_rejects=list(raw.get("shortlist_rejects") or []),
            research_shortlist=list(raw.get("research_shortlist") or []),
            population_stats=dict(raw.get("population_stats") or {}),
            trial_population_candidate_ids=list(raw.get("trial_population_candidate_ids") or []),
            historical_evaluation_records=list(raw.get("historical_evaluation_records") or []),
            rng_state=dict(raw.get("rng_state") or {}),
            budget_caps=dict(raw.get("budget_caps") or {}),
            fingerprint_components=dict(raw.get("fingerprint_components") or {}),
            status_seq=int(raw.get("status_seq") or 0),
        )

    def pending_by_gate(self) -> dict[str, int]:
        return {
            "SCORE_QUALIFIED_AWAITING_STRESS": len(self.pending_stress_ids),
            "STRESS_PASSED_AWAITING_ROBUSTNESS": len(self.pending_robustness_ids),
            "ROBUSTNESS_PASSED_AWAITING_STATISTICS": len(self.pending_statistics_ids),
            "GENERATED_AWAITING_EVAL": len(self.pending_eval_ids),
        }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically (temp + fsync + replace). Crash loses at most in-flight work."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(payload, indent=2, default=str)
    with tmp.open("w", encoding="utf-8") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def save_checkpoint(path: Path, checkpoint: SearchCheckpoint) -> None:
    checkpoint.updated_at = now_iso()
    atomic_write_json(path, checkpoint.as_dict())


def load_checkpoint(path: Path) -> SearchCheckpoint | None:
    path = Path(path)
    if not path.is_file():
        return None
    return SearchCheckpoint.from_dict(json.loads(path.read_text(encoding="utf-8")))
