"""Resume orchestration helpers for Multi-Family search programs."""

from __future__ import annotations

from typing import Any

from discovery.candidate import StrategyCandidate
from discovery.evaluator import EvalOutcome, EvaluationRecord
from discovery.family_spec import FamilySpec
from discovery.search_checkpoint import (
    CHECKPOINT_VERSION,
    SearchCheckpoint,
)
from discovery.search_program import PipelinePhase, now_iso

# Gate status strings (must match multi_family_campaign constants).
SCORE_QUALIFIED = "SCORE_QUALIFIED"
STRESS_PASSED = "STRESS_PASSED"
STRESS_FAILED = "STRESS_FAILED"
STRESS_TESTED = "STRESS_TESTED"
ROBUSTNESS_PASSED = "ROBUSTNESS_PASSED"
ROBUSTNESS_FAILED = "ROBUSTNESS_FAILED"
FULL_WFO_COMPLETED = "FULL_WFO_COMPLETED"
STATISTICALLY_PASSED = "STATISTICALLY_PASSED"
STATISTICALLY_REJECTED = "STATISTICALLY_REJECTED"
RESEARCH_SHORTLISTED = "RESEARCH_SHORTLISTED"


def empty_checkpoint(
    *,
    search_program_id: str,
    compatibility_fingerprint: str,
    session_run_id: str,
    search_mode: str,
    seed: int,
    config: dict[str, Any],
    fingerprint_components: dict[str, str] | None = None,
    source_run_id: str | None = None,
    resumed_from_run_id: str | None = None,
    budget_caps: dict[str, Any] | None = None,
) -> SearchCheckpoint:
    return SearchCheckpoint(
        version=CHECKPOINT_VERSION,
        search_program_id=search_program_id,
        compatibility_fingerprint=compatibility_fingerprint,
        session_run_id=session_run_id,
        source_run_id=source_run_id,
        resumed_from_run_id=resumed_from_run_id,
        search_mode=search_mode,
        pipeline_phase=PipelinePhase.GENERATION_0.value,
        updated_at=now_iso(),
        seed=seed,
        config=config,
        family_specs=[],
        candidates={},
        candidate_gates={},
        evaluation_records={},
        family_states={},
        generation_records=[],
        status_history=[],
        fingerprint_components=dict(fingerprint_components or {}),
        budget_caps=dict(budget_caps or {}),
    )


def rebuild_candidates(checkpoint: SearchCheckpoint) -> dict[str, StrategyCandidate]:
    return {
        cid: StrategyCandidate.from_dict(payload)
        for cid, payload in checkpoint.candidates.items()
    }


def rebuild_evaluation_records(
    checkpoint: SearchCheckpoint,
) -> dict[str, EvaluationRecord]:
    return {
        cid: EvaluationRecord.from_dict(payload)
        for cid, payload in checkpoint.evaluation_records.items()
    }


def rebuild_families(checkpoint: SearchCheckpoint) -> list[FamilySpec]:
    return [FamilySpec.from_dict(f) for f in checkpoint.family_specs]


def rebuild_records_by_family(
    checkpoint: SearchCheckpoint,
    families: list[FamilySpec],
    cand_by_id: dict[str, StrategyCandidate],
) -> dict[str, list[EvaluationRecord]]:
    records = rebuild_evaluation_records(checkpoint)
    out: dict[str, list[EvaluationRecord]] = {f.family_id: [] for f in families}
    placed: set[str] = set()
    for fid, fst in checkpoint.family_states.items():
        for cid in fst.evaluated_ids:
            rec = records.get(cid)
            if rec is None:
                continue
            out.setdefault(fid, []).append(rec)
            placed.add(cid)
    for cid, rec in records.items():
        if cid in placed:
            continue
        cand = cand_by_id.get(cid)
        fid = None
        if cand is not None:
            fid = (cand.family_provenance or {}).get("family_id") or cand.strategy_family
        if fid is None:
            continue
        out.setdefault(str(fid), []).append(rec)
    return out


def sync_pending_gate_queues(checkpoint: SearchCheckpoint) -> None:
    """Recompute pending gate queues from gates + completed summaries."""
    stressed = {
        s.get("candidate_id")
        for s in checkpoint.stress_summaries
        if s.get("final_decision") in {STRESS_PASSED, STRESS_FAILED}
    }
    robusted = {
        s.get("candidate_id")
        for s in checkpoint.robustness_summaries
        if s.get("final_decision") in {ROBUSTNESS_PASSED, ROBUSTNESS_FAILED}
    }
    statted = {
        s.get("candidate_id")
        for s in checkpoint.statistics_summaries
        if s.get("final_decision")
        in {STATISTICALLY_PASSED, STATISTICALLY_REJECTED, "DSR_FAILED", "PBO_FAILED"}
    }
    pending_stress: list[str] = []
    pending_rob: list[str] = []
    pending_stat: list[str] = []
    for cid, gate in sorted(checkpoint.candidate_gates.items()):
        if gate == SCORE_QUALIFIED and cid not in stressed:
            pending_stress.append(cid)
        elif gate == STRESS_PASSED and cid not in robusted:
            pending_rob.append(cid)
        elif gate == ROBUSTNESS_PASSED and cid not in statted:
            pending_stat.append(cid)
    checkpoint.pending_stress_ids = pending_stress
    checkpoint.pending_robustness_ids = pending_rob
    checkpoint.pending_statistics_ids = pending_stat
    pending_eval: list[str] = []
    seen: set[str] = set()
    for fst in checkpoint.family_states.values():
        for cid in fst.pending_eval_ids:
            if cid not in seen and cid not in checkpoint.evaluation_records:
                pending_eval.append(cid)
                seen.add(cid)
    for cid in checkpoint.pending_eval_ids:
        if cid not in seen and cid not in checkpoint.evaluation_records:
            pending_eval.append(cid)
            seen.add(cid)
    checkpoint.pending_eval_ids = pending_eval


def choose_resume_phase(checkpoint: SearchCheckpoint) -> PipelinePhase:
    """Resume priority: Stress → Robustness → Statistics → pending eval → evolution."""
    sync_pending_gate_queues(checkpoint)
    if checkpoint.pending_stress_ids:
        return PipelinePhase.STRESS
    if checkpoint.pending_robustness_ids:
        return PipelinePhase.ROBUSTNESS
    if checkpoint.pending_statistics_ids:
        return PipelinePhase.STATISTICS
    if checkpoint.pending_eval_ids:
        return PipelinePhase.PENDING_EVAL
    if any(
        fst.family_stop_reason is None
        or fst.family_stop_reason
        in {
            "max_runtime_seconds",
            "budget_exhausted_before_breed",
            "empty_generation_queue",
        }
        for fst in checkpoint.family_states.values()
    ):
        return PipelinePhase.EVOLUTION
    if checkpoint.pipeline_phase == PipelinePhase.COMPLETE.value:
        return PipelinePhase.COMPLETE
    return PipelinePhase.EVOLUTION


def mark_score_qualified_pending(checkpoint: SearchCheckpoint) -> None:
    """Stamp SCORE_QUALIFIED gates from evaluation records and fill pending stress."""
    for cid, raw in checkpoint.evaluation_records.items():
        outcome = raw.get("outcome")
        fit_raw = raw.get("fitness")
        if outcome != EvalOutcome.REGISTERED.value or not fit_raw:
            continue
        if fit_raw.get("rejected"):
            continue
        gate = checkpoint.candidate_gates.get(cid)
        if gate in {
            STRESS_PASSED,
            STRESS_FAILED,
            STRESS_TESTED,
            ROBUSTNESS_PASSED,
            ROBUSTNESS_FAILED,
            STATISTICALLY_PASSED,
            RESEARCH_SHORTLISTED,
        }:
            continue
        checkpoint.candidate_gates[cid] = SCORE_QUALIFIED
    sync_pending_gate_queues(checkpoint)


def apply_budget_extension(
    config: Any,
    *,
    additional_runtime_seconds: float = 0.0,
    additional_generated_budget: int = 0,
    additional_full_wfo_budget: int = 0,
) -> Any:
    """EXTEND_BUDGET: raise generated/WFO caps; session runtime is replaced.

    ``additional_runtime_seconds`` is the runtime limit for the new session
    (not ``original_runtime_cap + additional_runtime_seconds``).
    """
    add_rt = float(additional_runtime_seconds or 0.0)
    if add_rt > 0:
        config.max_runtime_seconds = add_rt
    config.total_candidate_budget = int(config.total_candidate_budget) + int(
        additional_generated_budget
    )
    config.max_full_wfo = int(config.max_full_wfo) + int(additional_full_wfo_budget)
    if config.max_evaluated_candidates is not None:
        config.max_evaluated_candidates = int(config.max_evaluated_candidates) + int(
            additional_generated_budget
        )
    return config


def checkpoint_totals_payload(checkpoint: SearchCheckpoint) -> dict[str, Any]:
    return {
        "search_program_id": checkpoint.search_program_id,
        "compatibility_fingerprint": checkpoint.compatibility_fingerprint,
        "pipeline_phase": checkpoint.pipeline_phase,
        "source_run_id": checkpoint.source_run_id,
        "resumed_from_run_id": checkpoint.resumed_from_run_id,
        "session_run_id": checkpoint.session_run_id,
        "search_mode": checkpoint.search_mode,
        "cumulative": {
            "generated": checkpoint.campaign_generated,
            "evaluated": checkpoint.campaign_evaluated,
            "full_wfo": checkpoint.campaign_full_wfo,
        },
        "session": {
            "generated": checkpoint.session_generated,
            "evaluated": checkpoint.session_evaluated,
            "full_wfo": checkpoint.session_full_wfo,
        },
        "pending_by_gate": checkpoint.pending_by_gate(),
        "trial_population_size": len(checkpoint.trial_population_candidate_ids),
    }
