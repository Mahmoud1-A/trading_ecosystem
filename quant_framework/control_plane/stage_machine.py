"""Strict Alpha Miner candidate stage machine (control-plane honesty layer)."""

from __future__ import annotations

from enum import Enum
from typing import Any


class CandidateStage(str, Enum):
    GENERATED = "GENERATED"
    VALIDATED = "VALIDATED"
    PRECHECK_PASSED = "PRECHECK_PASSED"
    FULL_WFO_STARTED = "FULL_WFO_STARTED"
    FULL_WFO_EVALUATED = "FULL_WFO_EVALUATED"
    SCORE_QUALIFIED = "SCORE_QUALIFIED"
    STRESS_EVALUATED = "STRESS_EVALUATED"
    STRESS_PASSED = "STRESS_PASSED"
    STATISTICALLY_EVALUATED = "STATISTICALLY_EVALUATED"
    BEHAVIORALLY_CLUSTERED = "BEHAVIORALLY_CLUSTERED"
    BEHAVIORALLY_UNIQUE = "BEHAVIORALLY_UNIQUE"
    FINALIST = "FINALIST"
    # Terminal / side states
    INVALID = "INVALID"
    DUPLICATE = "DUPLICATE"
    PRECHECK_REJECTED = "PRECHECK_REJECTED"
    WFO_FAILED = "WFO_FAILED"
    SCORE_REJECTED = "SCORE_REJECTED"
    STRESS_REJECTED = "STRESS_REJECTED"
    STATISTICAL_REJECTED = "STATISTICAL_REJECTED"
    BEHAVIORAL_DUPLICATE = "BEHAVIORAL_DUPLICATE"
    EVALUATION_ERROR = "EVALUATION_ERROR"
    TOP_RANKED_UNVALIDATED = "TOP_RANKED_UNVALIDATED"
    SHORTLISTED = "SHORTLISTED"
    RESEARCH_SHORTLISTED = "RESEARCH_SHORTLISTED"


MANDATORY_FINALIST_STAGES = (
    CandidateStage.GENERATED,
    CandidateStage.VALIDATED,
    CandidateStage.PRECHECK_PASSED,
    CandidateStage.FULL_WFO_STARTED,
    CandidateStage.FULL_WFO_EVALUATED,
    CandidateStage.SCORE_QUALIFIED,
    CandidateStage.STRESS_EVALUATED,
    CandidateStage.STRESS_PASSED,
    CandidateStage.STATISTICALLY_EVALUATED,
    CandidateStage.BEHAVIORALLY_CLUSTERED,
    CandidateStage.BEHAVIORALLY_UNIQUE,
)

TERMINAL_STOP_REASONS = {
    "max_generated_candidates": "GENERATED_BUDGET_EXHAUSTED",
    "max_evaluated_candidates": "EVALUATED_BUDGET_EXHAUSTED",
    "max_full_wfo_evaluations": "FULL_WFO_BUDGET_EXHAUSTED",
    "max_runtime_seconds": "RUNTIME_EXHAUSTED",
    "max_cpu_seconds": "RUNTIME_EXHAUSTED",
    "stagnation_limit": "STAGNATION_LIMIT",
    "empty_population": "STAGNATION_LIMIT",
    "completed": "CONVERGENCE",
    "cancelled": "CANCELLED",
}


def map_terminal_reason(stop_reason: str | None, *, cancelled: bool = False) -> str:
    if cancelled:
        return "CANCELLED"
    if not stop_reason:
        return "ERROR"
    if stop_reason in TERMINAL_STOP_REASONS:
        return TERMINAL_STOP_REASONS[stop_reason]
    if stop_reason.startswith("max_"):
        return "EVALUATED_BUDGET_EXHAUSTED"
    if "error" in stop_reason.lower():
        return "ERROR"
    return stop_reason.upper() if stop_reason else "ERROR"


def next_missing_gate(completed: CandidateStage) -> str:
    """Return the next mandatory gate after the last completed stage."""
    if completed in {
        CandidateStage.FINALIST,
        CandidateStage.INVALID,
        CandidateStage.DUPLICATE,
        CandidateStage.PRECHECK_REJECTED,
        CandidateStage.WFO_FAILED,
        CandidateStage.SCORE_REJECTED,
        CandidateStage.STRESS_REJECTED,
        CandidateStage.STATISTICAL_REJECTED,
        CandidateStage.BEHAVIORAL_DUPLICATE,
        CandidateStage.EVALUATION_ERROR,
    }:
        return "NONE"
    try:
        idx = MANDATORY_FINALIST_STAGES.index(completed)
    except ValueError:
        # Unvalidated shortlist path
        if completed in {
            CandidateStage.TOP_RANKED_UNVALIDATED,
            CandidateStage.SHORTLISTED,
            CandidateStage.RESEARCH_SHORTLISTED,
        }:
            return CandidateStage.FULL_WFO_EVALUATED.value
        return CandidateStage.VALIDATED.value
    if idx + 1 >= len(MANDATORY_FINALIST_STAGES):
        return CandidateStage.FINALIST.value
    return MANDATORY_FINALIST_STAGES[idx + 1].value


def classify_candidate(
    *,
    rejection_reason: str | None,
    ranking_score: float | None,
    is_full_event_wfo: bool,
    fold_count: int,
    completed_fold_count: int,
    stress_status: str,
    dsr_status: str,
    pbo_status: str,
    behavioral_cluster: str | None,
    is_cluster_representative: bool,
    controller_shortlist: bool,
    allow_research_shortlist: bool = True,
) -> dict[str, Any]:
    """
    Derive institutional stage labels.

    SearchController shortlist members are NEVER auto-FINALIST when the
    evaluation path is synthetic / incomplete.
    """
    reason = rejection_reason or ""

    if "duplicate" in reason.lower():
        stage = CandidateStage.DUPLICATE
        return _pack(stage, promotion="NONE", vault=False, paper=False)

    if reason == "INVALID_DSL_TYPE" or reason.startswith("INVALID_DSL_TYPE"):
        stage = CandidateStage.INVALID
        return _pack(stage, promotion="NONE", vault=False, paper=False, note="invalid_dsl_type")

    if reason.startswith("precheck") or "precheck" in reason.lower():
        stage = CandidateStage.PRECHECK_REJECTED
        return _pack(stage, promotion="NONE", vault=False, paper=False)

    if reason.startswith("eval_failed") or "invalid" in reason.lower():
        stage = CandidateStage.INVALID if "invalid" in reason.lower() else CandidateStage.EVALUATION_ERROR
        return _pack(stage, promotion="NONE", vault=False, paper=False)

    if "wfo" in reason.lower() and reason:
        stage = CandidateStage.WFO_FAILED
        return _pack(stage, promotion="NONE", vault=False, paper=False)

    # Pipeline progress
    last = CandidateStage.GENERATED
    last = CandidateStage.VALIDATED
    if not reason.startswith("precheck"):
        last = CandidateStage.PRECHECK_PASSED

    if is_full_event_wfo and fold_count > 0:
        last = CandidateStage.FULL_WFO_STARTED
        if completed_fold_count >= fold_count and completed_fold_count > 0:
            last = CandidateStage.FULL_WFO_EVALUATED
        else:
            stage = CandidateStage.WFO_FAILED if reason else last
            if completed_fold_count < fold_count:
                return _pack(
                    last,
                    promotion="NONE",
                    vault=False,
                    paper=False,
                    note="incomplete_wfo_folds",
                )
    elif ranking_score is not None and not is_full_event_wfo:
        # Synthetic / proxy probe — may shortlist but not qualify for FINALIST
        if reason:
            stage = CandidateStage.SCORE_REJECTED
            return _pack(stage, promotion="NONE", vault=False, paper=False)
        if controller_shortlist:
            stage = CandidateStage.TOP_RANKED_UNVALIDATED
            return _pack(
                stage,
                promotion="TOP_RANKED_UNVALIDATED",
                vault=False,
                paper=False,
                note="synthetic_oos_probe_not_institutional_wfo",
            )
        stage = CandidateStage.SHORTLISTED
        return _pack(
            stage,
            promotion="SHORTLISTED",
            vault=False,
            paper=False,
            note="missing_full_event_wfo",
        )

    if not is_full_event_wfo or completed_fold_count == 0:
        # No completed institutional WFO → no score qualification
        if controller_shortlist:
            return _pack(
                CandidateStage.TOP_RANKED_UNVALIDATED,
                promotion="TOP_RANKED_UNVALIDATED",
                vault=False,
                paper=False,
                note="no_full_wfo",
            )
        return _pack(
            last,
            promotion="NONE",
            vault=False,
            paper=False,
            note="no_full_wfo",
        )

    # Full WFO completed
    if ranking_score is None or reason:
        stage = CandidateStage.SCORE_REJECTED
        return _pack(stage, promotion="NONE", vault=False, paper=False)
    last = CandidateStage.SCORE_QUALIFIED

    if stress_status in {"NOT_EVALUATED", "", None}:
        if controller_shortlist:
            return _pack(
                last,
                promotion="SHORTLISTED",
                vault=False,
                paper=False,
                note="stress_not_evaluated",
            )
        return _pack(last, promotion="NONE", vault=False, paper=False)

    last = CandidateStage.STRESS_EVALUATED
    if stress_status != "PASSED":
        stage = CandidateStage.STRESS_REJECTED
        return _pack(stage, promotion="NONE", vault=False, paper=False)
    last = CandidateStage.STRESS_PASSED

    # Statistical gate
    if dsr_status in {"NOT_EVALUATED"} or pbo_status in {"NOT_EVALUATED"}:
        # Should not happen after evaluation attempt — treat as insufficient
        dsr_status = "INSUFFICIENT_DATA" if dsr_status == "NOT_EVALUATED" else dsr_status
        pbo_status = "INSUFFICIENT_DATA" if pbo_status == "NOT_EVALUATED" else pbo_status

    if dsr_status == "INSUFFICIENT_DATA" or pbo_status == "INSUFFICIENT_DATA":
        last = CandidateStage.STATISTICALLY_EVALUATED
        if allow_research_shortlist and controller_shortlist:
            return _pack(
                CandidateStage.RESEARCH_SHORTLISTED,
                promotion="RESEARCH_SHORTLISTED",
                vault=False,
                paper=False,
                note="insufficient_dsr_pbo",
            )
        stage = CandidateStage.STATISTICAL_REJECTED
        return _pack(stage, promotion="NONE", vault=False, paper=False)

    if dsr_status in {"FAILED", "REJECTED"} or pbo_status in {"FAILED", "REJECTED"}:
        stage = CandidateStage.STATISTICAL_REJECTED
        return _pack(stage, promotion="NONE", vault=False, paper=False)

    last = CandidateStage.STATISTICALLY_EVALUATED

    if not behavioral_cluster or behavioral_cluster in {"NOT_EVALUATED", "NONE"}:
        return _pack(
            last,
            promotion="SHORTLISTED" if controller_shortlist else "NONE",
            vault=False,
            paper=False,
            note="missing_behavioral_cluster",
        )
    last = CandidateStage.BEHAVIORALLY_CLUSTERED

    if not is_cluster_representative:
        stage = CandidateStage.BEHAVIORAL_DUPLICATE
        return _pack(stage, promotion="NONE", vault=False, paper=False)
    last = CandidateStage.BEHAVIORALLY_UNIQUE

    # All mandatory gates passed
    return _pack(
        CandidateStage.FINALIST,
        promotion="FINALIST",
        vault=True,
        paper=False,  # Paper still requires separate Phase 9 eligibility
    )


def _pack(
    stage: CandidateStage,
    *,
    promotion: str,
    vault: bool,
    paper: bool,
    note: str | None = None,
) -> dict[str, Any]:
    return {
        "evaluation_stage": stage.value,
        "last_completed_stage": stage.value,
        "next_missing_gate": next_missing_gate(stage),
        "promotion_label": promotion,
        "vault_eligible": vault,
        "paper_eligible": paper,
        "stage_note": note,
    }
