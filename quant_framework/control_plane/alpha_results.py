"""Alpha Miner result shaping — Registry-backed counters and stage honesty."""

from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Any

from control_plane.stage_machine import CandidateStage, classify_candidate, map_terminal_reason
from discovery.evaluator import EvaluationRecord
from discovery.search_budget import BudgetCounters, SearchBudget
from discovery.search_controller import DiscoveryRunResult
from registry.experiment_registry import ExperimentRegistry, TrialRecord

STATUS_NOT_EVALUATED = "NOT_EVALUATED"
STATUS_INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
STATUS_NO_QUALIFIED = "NO_QUALIFIED_CANDIDATE"
STATUS_NOT_APPLICABLE = "NOT_APPLICABLE"


def _latest_by_candidate(trials: list[TrialRecord]) -> dict[str, TrialRecord]:
    """Keep the richest trial per candidate, not a later duplicate shell.

    A later duplicate registry entry must not erase completed WFO fold evidence
    from the original evaluation of the same deterministic candidate ID.
    """

    def richness(t: TrialRecord) -> tuple[int, int, int, int]:
        reason = (t.rejection_reason or "").lower()
        return (
            0 if "duplicate" in reason else 1,
            1 if t.fold_records else 0,
            1 if t.ranking_score is not None else 0,
            len(t.fold_records or []),
        )

    out: dict[str, TrialRecord] = {}
    for t in trials:
        prev = out.get(t.candidate_id)
        if prev is None or richness(t) >= richness(prev):
            out[t.candidate_id] = t
    return out


def _is_full_event_wfo(trial: TrialRecord) -> bool:
    snap = trial.config_snapshot or {}
    if snap.get("is_full_event_wfo") is True:
        return True
    if snap.get("evaluation_path") == "event_driven_wfo":
        return True
    if snap.get("backend_kind") == "event_driven_wfo":
        return True
    return False


def _wfo_fold_info(trial: TrialRecord) -> dict[str, Any]:
    snap = trial.config_snapshot or {}
    folds = list(trial.fold_records or [])
    wfo_folds = list(snap.get("wfo_folds") or [])
    fold_count = int(snap.get("wfo_fold_count") or len(folds) or len(wfo_folds) or 0)
    completed = int(snap.get("wfo_completed_folds") or (len(folds) if folds else 0))
    if _is_full_event_wfo(trial) and folds:
        completed = len(folds)
        fold_count = max(fold_count, completed)
    return {
        "fold_count": fold_count,
        "completed_fold_count": completed,
        "training_ranges": [
            {"start": f.get("train_start"), "end": f.get("train_end")}
            for f in wfo_folds
            if isinstance(f, dict)
        ],
        "oos_ranges": [
            {"start": f.get("oos_start"), "end": f.get("oos_end")}
            for f in wfo_folds
            if isinstance(f, dict)
        ],
        "aggregate_oos_metrics": {
            "ranking_score": trial.ranking_score,
            "fold_metrics": [
                f["expectancy"] if "expectancy" in f else f.get("oos_metric")
                for f in (folds or wfo_folds)
                if isinstance(f, dict)
            ],
        },
        "wfo_terminal_status": snap.get("wfo_terminal_status")
        or (
            "COMPLETED"
            if _is_full_event_wfo(trial) and completed > 0
            else ("NOT_APPLICABLE" if not _is_full_event_wfo(trial) else "FAILED")
        ),
    }


def compute_population_stats(registry: ExperimentRegistry) -> dict[str, Any]:
    """DSR/PBO over the full trial population — never fabricate OK."""
    scores = registry.trial_history_for_dsr()
    if len(registry.all_trials()) < 2 or len(scores) < 2:
        return {
            "dsr_status": STATUS_INSUFFICIENT_DATA,
            "pbo_status": STATUS_INSUFFICIENT_DATA,
            "dsr": None,
            "pbo": None,
        }
    try:
        observed = float(max(scores))
        result = registry.evaluate_overfitting(
            observed_sharpe=observed,
            n_observations=max(20, len(scores) * 5),
            performance_matrix=None,
        )
        dsr = result.get("dsr") or {}
        dsr_status = str(dsr.get("status") or STATUS_INSUFFICIENT_DATA)
        pbo = result.get("pbo")
        pbo_status = (
            STATUS_INSUFFICIENT_DATA
            if pbo is None
            else str(pbo.get("status") or STATUS_INSUFFICIENT_DATA)
        )
        return {
            "dsr_status": dsr_status,
            "pbo_status": pbo_status,
            "dsr": dsr,
            "pbo": pbo,
        }
    except Exception:  # noqa: BLE001
        return {
            "dsr_status": STATUS_INSUFFICIENT_DATA,
            "pbo_status": STATUS_INSUFFICIENT_DATA,
            "dsr": None,
            "pbo": None,
        }


def enrich_from_records(
    records: list[EvaluationRecord],
) -> dict[str, dict[str, Any]]:
    """Map candidate_id -> stress / cluster enrichment from in-memory eval records."""
    out: dict[str, dict[str, Any]] = {}
    for rec in records:
        stress_status = STATUS_NOT_EVALUATED
        if rec.stress_results:
            passed = sum(
                1
                for v in rec.stress_results.values()
                if isinstance(v, dict) and v.get("passed") is True
            )
            total = len(rec.stress_results)
            stress_status = "PASSED" if total > 0 and passed == total else (
                "PASSED" if total > 0 and passed / total >= 0.5 else "FAILED"
            )
            if total > 0 and passed / total >= 0.5:
                stress_status = "PASSED"
            elif total > 0:
                stress_status = "FAILED"
        out[rec.candidate_id] = {
            "stress_status": stress_status,
            "stress_results": dict(rec.stress_results or {}),
            "behavioral_cluster": rec.behavioral_cluster,
            "outcome": rec.outcome.value if rec.outcome else None,
        }
    return out


def build_candidate_row(
    trial: TrialRecord,
    *,
    controller_shortlist_ids: set[str],
    cluster_by_id: dict[str, str],
    representative_ids: set[str],
    enrichment: dict[str, dict[str, Any]],
    pop_stats: dict[str, Any],
) -> dict[str, Any]:
    snap = trial.config_snapshot or {}
    net = trial.net_metrics or {}
    folds = list(trial.fold_records or [])
    expectancies = [
        float(f["expectancy"]) for f in folds if isinstance(f, dict) and "expectancy" in f
    ]
    pfs = [
        float(f["profit_factor"]) for f in folds if isinstance(f, dict) and "profit_factor" in f
    ]
    dds = [
        float(f["max_drawdown"]) for f in folds if isinstance(f, dict) and "max_drawdown" in f
    ]
    calmars = [float(f["calmar"]) for f in folds if isinstance(f, dict) and "calmar" in f]
    trade_counts = [int(f["n_trades"]) for f in folds if isinstance(f, dict) and f.get("n_trades") is not None]
    total_oos_trades = sum(max(0, n) for n in trade_counts)
    if "total_oos_trades" in net and net.get("total_oos_trades") is not None:
        total_oos_trades = int(net["total_oos_trades"])
    if "fold_trade_counts" in net and isinstance(net.get("fold_trade_counts"), list):
        trade_counts = [int(x) for x in net["fold_trade_counts"]]
        total_oos_trades = sum(max(0, n) for n in trade_counts)
    min_oos_trades = int(net.get("min_oos_trades") or snap.get("min_oos_trades") or 8)
    wfo = _wfo_fold_info(trial)
    enr = enrichment.get(trial.candidate_id, {})
    stress_status = (
        enr.get("stress_status")
        or net.get("stress_status")
        or snap.get("stress_status")
        or STATUS_NOT_EVALUATED
    )
    cluster = (
        enr.get("behavioral_cluster")
        or cluster_by_id.get(trial.candidate_id)
        or STATUS_NOT_EVALUATED
    )
    dsr_status = pop_stats.get("dsr_status", STATUS_INSUFFICIENT_DATA)
    pbo_status = pop_stats.get("pbo_status", STATUS_INSUFFICIENT_DATA)
    # Per-candidate: without population evidence, insufficient
    if trial.ranking_score is None:
        dsr_status = STATUS_NOT_EVALUATED if trial.rejection_reason else dsr_status
        pbo_status = STATUS_NOT_EVALUATED if trial.rejection_reason else pbo_status
    # Closed-trade floors: DSR/PBO are not evaluable without enough OOS trades.
    if (
        total_oos_trades <= 0
        or total_oos_trades < min_oos_trades
        or (trial.rejection_reason or "")
        in {
            "NO_OOS_TRADES",
            "INSUFFICIENT_OOS_TRADES",
            "NEGATIVE_EXPECTANCY",
            "PF_BELOW_ONE",
            "MAX_DRAWDOWN_EXCEEDED",
        }
    ):
        dsr_status = STATUS_NOT_EVALUATED
        pbo_status = STATUS_NOT_EVALUATED

    classified = classify_candidate(
        rejection_reason=trial.rejection_reason,
        ranking_score=trial.ranking_score,
        is_full_event_wfo=_is_full_event_wfo(trial),
        fold_count=wfo["fold_count"],
        completed_fold_count=wfo["completed_fold_count"],
        stress_status=str(stress_status),
        dsr_status=str(dsr_status),
        pbo_status=str(pbo_status),
        behavioral_cluster=None if cluster in {STATUS_NOT_EVALUATED, None} else str(cluster),
        is_cluster_representative=trial.candidate_id in representative_ids,
        controller_shortlist=trial.candidate_id in controller_shortlist_ids,
        total_oos_trades=int(total_oos_trades),
    )

    median_oos: Any = (
        float(sorted(expectancies)[len(expectancies) // 2])
        if expectancies
        else STATUS_NOT_EVALUATED
    )
    fitness: Any = trial.ranking_score
    if fitness is None:
        fitness = (
            STATUS_NOT_APPLICABLE
            if classified["evaluation_stage"]
            in {CandidateStage.DUPLICATE.value, CandidateStage.PRECHECK_REJECTED.value}
            else STATUS_NOT_EVALUATED
        )

    # Zero-trade hygiene for dashboard/report: never surface manufactured
    # expectancy / PF / MaxDD / Calmar (or leave them as NOT_EVALUATED).
    zero_trade = total_oos_trades <= 0 or (trial.rejection_reason or "") == "NO_OOS_TRADES"
    if zero_trade:
        median_oos = 0.0
        profit_factor: Any = 0.0
        max_drawdown: Any = 0.0
        max_drawdown_pct: Any = 0.0
        calmar: Any = 0.0
        metrics_basis_note = (
            "n_trades==0: expectancy/PF/MaxDD/Calmar forced to 0 (no manufactured equity-path metrics)"
        )
        if fitness not in {STATUS_NOT_APPLICABLE}:
            fitness = STATUS_NOT_EVALUATED
    else:
        profit_factor = float(median(pfs)) if pfs else STATUS_NOT_EVALUATED
        max_drawdown = float(min(dds)) if dds else STATUS_NOT_EVALUATED
        max_drawdown_pct = float(min(dds) * 100.0) if dds else STATUS_NOT_EVALUATED
        calmar = float(median(calmars)) if calmars else STATUS_NOT_EVALUATED
        metrics_basis_note = net.get("metrics_basis_note") or STATUS_NOT_APPLICABLE

    train_diag = {}
    if isinstance(net.get("train_diagnostic"), dict):
        train_diag = dict(net["train_diagnostic"])
    elif isinstance(getattr(trial, "gross_metrics", None), dict):
        gd = trial.gross_metrics.get("train_diagnostic")
        if isinstance(gd, dict):
            train_diag = dict(gd)
    oos_funnels = [
        f
        for f in list(train_diag.get("fold_trade_funnels") or [])
        if isinstance(f, dict) and f.get("phase") == "validation_oos"
    ]
    entry_true_count = int(
        sum(int(f.get("entry_true_count", 0) or 0) for f in oos_funnels)
        or train_diag.get("signals_entry_count")
        or 0
    )
    fills_count = int(
        sum(int(f.get("fills", 0) or 0) for f in oos_funnels)
        or train_diag.get("fills_count")
        or 0
    )
    orders_submitted = int(sum(int(f.get("orders_submitted", 0) or 0) for f in oos_funnels))
    signal_source = (
        train_diag.get("signal_source")
        or snap.get("signal_source")
        or net.get("signal_source")
        or STATUS_NOT_EVALUATED
    )

    return {
        "candidate_id": trial.candidate_id,
        "lineage_id": trial.lineage_id,
        "family": trial.strategy_family,
        "strategy_family": trial.strategy_family,
        "generation": snap.get("generation", STATUS_NOT_EVALUATED),
        "parent_ids": list(snap.get("parent_ids") or []),
        "complexity": snap.get("complexity", STATUS_NOT_EVALUATED),
        "fitness": fitness,
        "median_oos_expectancy": median_oos,
        "profit_factor": profit_factor,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "calmar": calmar,
        "calmar_mar": calmar,
        "total_oos_trades": int(total_oos_trades) if trade_counts or total_oos_trades else 0,
        "entry_true_count": entry_true_count,
        "fills": fills_count,
        "orders_submitted": orders_submitted,
        "signal_source": signal_source,
        "fold_trade_counts": trade_counts,
        "min_oos_trades": min_oos_trades,
        "dsr": dsr_status,
        "pbo": pbo_status,
        "stress_status": stress_status,
        "behavioral_cluster": cluster,
        "rejection_reason": trial.rejection_reason or STATUS_NOT_APPLICABLE,
        "evaluation_error_reason": (
            trial.rejection_reason
            if classified["evaluation_stage"] == CandidateStage.EVALUATION_ERROR.value
            or (trial.rejection_reason or "").startswith("eval_failed")
            else STATUS_NOT_APPLICABLE
        ),
        "metrics_basis_note": metrics_basis_note,
        "evaluation_stage": classified["evaluation_stage"],
        "last_completed_stage": classified["last_completed_stage"],
        "next_missing_gate": classified["next_missing_gate"],
        "promotion_label": classified["promotion_label"],
        "vault_eligible": classified["vault_eligible"],
        "paper_eligible": classified["paper_eligible"],
        "stage_note": classified.get("stage_note"),
        "ranking_source": trial.ranking_source or "validation_oos",
        "rejected": bool(trial.rejection_reason),
        "ranking_score": trial.ranking_score,
        "parameters": trial.parameters,
        "backend_kind": snap.get("backend_kind", STATUS_NOT_EVALUATED),
        "is_full_event_wfo": _is_full_event_wfo(trial),
        "evaluation_path": snap.get("evaluation_path", snap.get("backend_kind")),
        "family_provenance": dict(snap.get("family_provenance") or {}),
        "wfo": wfo,
    }


def funnel_counters(rows: list[dict[str, Any]], *, cluster_count: int) -> dict[str, Any]:
    """Distinct-candidate counters for the dashboard."""
    by_id = {r["candidate_id"]: r for r in rows}
    vals = list(by_id.values())

    def count_stage(*stages: str) -> int:
        return sum(1 for r in vals if r.get("evaluation_stage") in stages)

    generated = len(vals)
    duplicates = count_stage(CandidateStage.DUPLICATE.value)
    invalid = count_stage(CandidateStage.INVALID.value, CandidateStage.EVALUATION_ERROR.value)
    precheck = count_stage(CandidateStage.PRECHECK_REJECTED.value)
    evaluated = sum(
        1
        for r in vals
        if r.get("evaluation_stage")
        not in {
            CandidateStage.DUPLICATE.value,
            CandidateStage.PRECHECK_REJECTED.value,
            CandidateStage.GENERATED.value,
            CandidateStage.INVALID.value,
        }
        or r.get("ranking_score") is not None
        or (r.get("wfo") or {}).get("completed_fold_count", 0) > 0
    )
    # Prefer explicit evaluated: not duplicate/precheck-only
    evaluated = sum(
        1
        for r in vals
        if r.get("evaluation_stage")
        not in {
            CandidateStage.DUPLICATE.value,
            CandidateStage.PRECHECK_REJECTED.value,
        }
    )
    full_wfo = sum(1 for r in vals if r.get("is_full_event_wfo") and (r.get("wfo") or {}).get("completed_fold_count", 0) > 0)
    score_qualified = sum(
        1
        for r in vals
        if r.get("evaluation_stage")
        in {
            CandidateStage.SCORE_QUALIFIED.value,
            CandidateStage.STRESS_EVALUATED.value,
            CandidateStage.STRESS_PASSED.value,
            CandidateStage.STATISTICALLY_EVALUATED.value,
            CandidateStage.BEHAVIORALLY_CLUSTERED.value,
            CandidateStage.BEHAVIORALLY_UNIQUE.value,
            CandidateStage.FINALIST.value,
            CandidateStage.RESEARCH_SHORTLISTED.value,
            CandidateStage.STRESS_REJECTED.value,
            CandidateStage.STATISTICAL_REJECTED.value,
            CandidateStage.BEHAVIORAL_DUPLICATE.value,
        }
        or (
            r.get("is_full_event_wfo")
            and r.get("ranking_score") is not None
            and not r.get("rejected")
        )
    )
    stress_passed = sum(1 for r in vals if r.get("stress_status") == "PASSED")
    statistically_qualified = sum(
        1
        for r in vals
        if r.get("dsr") not in {STATUS_NOT_EVALUATED, STATUS_INSUFFICIENT_DATA, "FAILED", "REJECTED"}
        and r.get("pbo") not in {STATUS_NOT_EVALUATED, STATUS_INSUFFICIENT_DATA, "FAILED", "REJECTED"}
        and r.get("promotion_label") == "FINALIST"
    )
    behaviorally_unique = count_stage(
        CandidateStage.BEHAVIORALLY_UNIQUE.value, CandidateStage.FINALIST.value
    )
    shortlisted = sum(
        1
        for r in vals
        if r.get("promotion_label")
        in {"SHORTLISTED", "TOP_RANKED_UNVALIDATED", "RESEARCH_SHORTLISTED"}
    )
    finalists = sum(1 for r in vals if r.get("promotion_label") == "FINALIST")

    return {
        "generated": generated,
        "generated_candidates": generated,
        "invalid": invalid,
        "invalid_candidates": invalid,
        "duplicates": duplicates,
        "duplicate_candidates": duplicates,
        "precheck_rejected": precheck,
        "evaluated": evaluated,
        "evaluated_candidates": evaluated,
        "full_wfo_evaluated": full_wfo,
        "full_wfo_evaluations": full_wfo,
        "score_qualified": score_qualified,
        "stress_passed": stress_passed,
        "statistically_qualified": statistically_qualified,
        "behavioral_clusters": cluster_count,
        "behaviorally_unique": behaviorally_unique,
        "shortlisted": shortlisted,
        "finalists": finalists,
        "finalist_count": finalists,
        "rejected_total": sum(1 for r in vals if r.get("rejected")),
        "registry_trials": len(vals),
    }


def build_alpha_miner_report(
    *,
    registry: ExperimentRegistry,
    counters: BudgetCounters,
    budget: SearchBudget,
    result: DiscoveryRunResult,
    elapsed_seconds: float,
    cancelled: bool = False,
    evaluation_records: list[EvaluationRecord] | None = None,
    evaluation_backend: str = "synthetic_oos_probe",
) -> dict[str, Any]:
    unique_trials = list(_latest_by_candidate(registry.all_trials()).values())
    cluster_by_id: dict[str, str] = {}
    representative_ids: set[str] = set()
    for cl in result.clusters:
        cid = str(cl.get("cluster_id"))
        rep = cl.get("representative_id")
        if rep:
            representative_ids.add(str(rep))
            cluster_by_id[str(rep)] = cid
        for mid in cl.get("member_ids") or []:
            cluster_by_id[str(mid)] = cid

    controller_shortlist = set(result.finalists)
    enrichment = enrich_from_records(evaluation_records or [])
    # Attach cluster ids from controller onto enrichment
    for cid, cl in cluster_by_id.items():
        enrichment.setdefault(cid, {})
        enrichment[cid].setdefault("behavioral_cluster", cl)

    pop_stats = compute_population_stats(registry)
    rows = [
        build_candidate_row(
            t,
            controller_shortlist_ids=controller_shortlist,
            cluster_by_id=cluster_by_id,
            representative_ids=representative_ids,
            enrichment=enrichment,
            pop_stats=pop_stats,
        )
        for t in unique_trials
    ]
    funnel = funnel_counters(rows, cluster_count=len(result.clusters))

    true_finalist_ids = [r["candidate_id"] for r in rows if r["promotion_label"] == "FINALIST"]
    shortlist_ids = [
        r["candidate_id"]
        for r in rows
        if r["promotion_label"] in {"SHORTLISTED", "TOP_RANKED_UNVALIDATED", "RESEARCH_SHORTLISTED"}
    ]

    all_precheck = (
        funnel["generated"] > 0
        and funnel["precheck_rejected"] == funnel["generated"]
        and funnel["full_wfo_evaluations"] == 0
    )
    if cancelled:
        discovery = "CANCELLED"
        terminal = map_terminal_reason(result.stop_reason, cancelled=True)
    elif true_finalist_ids:
        discovery = "FINALISTS_FOUND"
        terminal = map_terminal_reason(result.stop_reason, cancelled=False)
    elif all_precheck:
        discovery = "ALL_CANDIDATES_PRECHECK_REJECTED"
        terminal = "ALL_CANDIDATES_PRECHECK_REJECTED"
    else:
        discovery = "NO_QUALIFIED_CANDIDATE"
        terminal = map_terminal_reason(result.stop_reason, cancelled=False)

    # Never claim profitability from synthetic probe when backend is not full WFO
    is_synthetic_backend = evaluation_backend == "synthetic_oos_probe"
    scores = [r.get("fitness") for r in result.rankings if isinstance(r.get("fitness"), (int, float))]
    if is_synthetic_backend or not scores:
        profitability = "NOT_AVAILABLE"
    else:
        best = max(scores)
        profitability = "POSITIVE" if best > 1e-9 else ("NEGATIVE" if best < -1e-9 else "FLAT")

    if pop_stats["dsr_status"] == STATUS_INSUFFICIENT_DATA:
        statistical = STATUS_INSUFFICIENT_DATA
    elif true_finalist_ids:
        statistical = "PASSED"
    else:
        statistical = STATUS_INSUFFICIENT_DATA

    throughput = (funnel["evaluated"] / elapsed_seconds) if elapsed_seconds > 0 else 0.0
    why_short_of_cap = (
        f"Search stopped with terminal_reason={terminal} after generating "
        f"{funnel['generated']} of max {budget.max_generated_candidates} candidates "
        f"(stop_reason={result.stop_reason!r}). Generation stops when any budget "
        f"dimension is exhausted (evaluated/full_wfo/runtime/stagnation), not only "
        f"the generated cap."
    )

    return {
        **funnel,
        "software_execution_status": "SUCCESS" if not cancelled else "FAILED",
        "discovery_result": discovery,
        "profitability_result": profitability,
        "statistical_result": statistical,
        "portfolio_result": "NOT_APPLICABLE" if funnel["finalists"] == 0 else "CANDIDATES_READY",
        "vault_result": "NOT_ELIGIBLE" if not true_finalist_ids else "NOT_SUBMITTED",
        "qualified_candidate_status": (
            "FINALISTS_FOUND" if true_finalist_ids else STATUS_NO_QUALIFIED
        ),
        "elapsed_time": elapsed_seconds,
        "throughput_per_second": throughput,
        "search_budget_consumed": {
            "generated": counters.generated,
            "generated_cap": budget.max_generated_candidates,
            "evaluated": counters.evaluated,
            "evaluated_cap": budget.max_evaluated_candidates,
            "full_wfo_budget_counter": counters.full_wfo,
            "full_wfo_cap": budget.max_full_wfo_evaluations,
            "runtime_seconds": counters.runtime_seconds,
            "runtime_cap": budget.max_runtime_seconds,
            "controller_stop_reason": result.stop_reason,
            "generated_attempts": counters.generated_attempts,
            "unique_generated_candidates": counters.unique_generated,
            "duplicate_attempts": counters.duplicate_attempts,
            "max_candidates_per_family_effective": budget.max_candidates_per_family,
            "max_candidates_per_complexity_tier_effective": (
                budget.max_candidates_per_complexity_tier
            ),
            "max_candidates_per_feature_family_effective": (
                budget.max_candidates_per_feature_family
            ),
            "bucket_caps_effective": {
                "max_candidates_per_family": budget.max_candidates_per_family,
                "max_candidates_per_complexity_tier": (
                    budget.max_candidates_per_complexity_tier
                ),
                "max_candidates_per_feature_family": (
                    budget.max_candidates_per_feature_family
                ),
            },
        },
        "terminal_reason": terminal,
        "controller_stop_reason": result.stop_reason,
        "generation_cap_explanation": why_short_of_cap,
        "evaluation_backend": evaluation_backend,
        "proxy_metric_used": False,
        "generations": result.generations,
        "controller_shortlist_ids": list(result.finalists),
        "finalist_ids": true_finalist_ids,
        "shortlist_ids": shortlist_ids,
        "clusters": list(result.clusters),
        "rankings": list(result.rankings),
        "portfolio_pool": dict(result.portfolio_pool or {}),
        "rejection_reason_distribution": dict(
            Counter(
                (t.rejection_reason or "unknown")
                for t in unique_trials
                if t.rejection_reason
            )
        ),
        "candidates": rows,
        "tables": {
            "finalists": [r for r in rows if r["promotion_label"] == "FINALIST"],
            "unvalidated_shortlist": [
                r
                for r in rows
                if r["promotion_label"]
                in {"SHORTLISTED", "TOP_RANKED_UNVALIDATED", "RESEARCH_SHORTLISTED"}
            ],
            "rejected": [
                r
                for r in rows
                if r.get("rejected")
                and r.get("evaluation_stage") != CandidateStage.DUPLICATE.value
            ],
            "duplicates": [
                r for r in rows if r.get("evaluation_stage") == CandidateStage.DUPLICATE.value
            ],
            "evaluation_failures": [
                r
                for r in rows
                if r.get("evaluation_stage")
                in {
                    CandidateStage.EVALUATION_ERROR.value,
                    CandidateStage.WFO_FAILED.value,
                    CandidateStage.INVALID.value,
                }
            ],
        },
        "population_stats": pop_stats,
        "wfo_summary": {
            "full_wfo_evaluations": funnel["full_wfo_evaluations"],
            "ranking_source": "validation_oos",
            "training_metrics_labeled_as_oos": False,
            "status": "EVALUATED" if funnel["full_wfo_evaluations"] else "NOT_EVALUATED",
            "evaluation_backend": evaluation_backend,
        },
        "stress_summary": {
            "stress_evaluations": counters.stress,
            "stress_passed": funnel["stress_passed"],
            "status": "EVALUATED" if counters.stress else STATUS_NOT_EVALUATED,
        },
        "behavioral_cluster_summary": {
            "cluster_count": len(result.clusters),
            "clusters": list(result.clusters),
            "status": "UPDATED" if result.clusters else STATUS_NOT_EVALUATED,
        },
        "message_no_finalists": (
            None
            if true_finalist_ids
            else "No candidate passed all mandatory Alpha Miner gates."
        ),
        "software_success": not cancelled,
        "statistical_validation": statistical,
        "discovery_run_id": result.discovery_run_id,
        "stop_reason": result.stop_reason,
        "finalists": funnel["finalist_count"],
    }
