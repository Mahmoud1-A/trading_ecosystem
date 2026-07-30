"""Candidate evaluation pipeline — full registry visibility, OOS ranking only."""

from __future__ import annotations

import copy
import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol

import numpy as np

from discovery.candidate import StrategyCandidate
from discovery.expression_tree import DSLValidationError
from discovery.feature_domains import (
    INVALID_FEATURE_THRESHOLD_DOMAIN,
    collect_candidate_threshold_violations,
)
from discovery.fitness import (
    RANKING_SOURCE_OOS,
    FoldOOSMetrics,
    FitnessResult,
    RobustFitness,
)
from discovery.grammar import Grammar
from discovery.prechecks import structural_precheck
from discovery.search_budget import BudgetCounters, SearchBudget
from discovery.typecheck import (
    INVALID_DSL_TYPE,
    InvalidDslTypeError,
    check_strategy_trees,
    parse_dsl_validation_error,
)
from registry.experiment_registry import ExperimentRegistry, TrialStatus


FEATURE_UNAVAILABLE = "FEATURE_UNAVAILABLE"
FEATURE_CAPABILITY_RESOLUTION_FAILED = "FEATURE_CAPABILITY_RESOLUTION_FAILED"
PERMISSIVE_NON_RESEARCH_FALLBACK = "permissive_non_research_fallback"


class EvalOutcome(str, Enum):
    REGISTERED = "REGISTERED"
    DUPLICATE_SKIPPED = "DUPLICATE_SKIPPED"
    PRECHECK_FAILED = "PRECHECK_FAILED"
    INVALID_DSL_TYPE = "INVALID_DSL_TYPE"
    FEATURE_UNAVAILABLE = "FEATURE_UNAVAILABLE"
    FEATURE_CAPABILITY_RESOLUTION_FAILED = "FEATURE_CAPABILITY_RESOLUTION_FAILED"
    INVALID_FEATURE_THRESHOLD_DOMAIN = "INVALID_FEATURE_THRESHOLD_DOMAIN"
    EVAL_FAILED = "EVAL_FAILED"
    RISK_FAILED = "RISK_FAILED"
    WFO_FAILED = "WFO_FAILED"
    REJECTED = "REJECTED"
    FINALIST = "FINALIST"


BASELINE_WFO_ARTIFACT_KEYS: tuple[str, ...] = (
    "candidate_id",
    "signal_source",
    "backend_kind",
    "is_full_event_wfo",
    "wfo_completed_folds",
    "closed_trades",
    "orders_count",
    "fills_count",
    "trades_count",
    "oos_ranges",
)


def snapshot_baseline_wfo_artifacts(
    *,
    candidate_id: str,
    backend: Any,
    train_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Immutable Stress baseline subset from the original qualifying WFO.

    Captures only fields required by Stress (removed_best_* / integrity proof).
    Never aliases mutable ``backend.last_run_artifacts``.
    """
    arts = dict(getattr(backend, "last_run_artifacts", None) or {})
    train = dict(train_metrics or {})
    closed_raw = arts.get("closed_trades")
    closed_trades = copy.deepcopy(list(closed_raw)) if closed_raw else []
    oos_raw = arts.get("oos_ranges")
    if oos_raw is None:
        oos_raw = train.get("oos_ranges")
    oos_ranges = copy.deepcopy(list(oos_raw)) if oos_raw else []
    return {
        "candidate_id": str(arts.get("candidate_id") or candidate_id),
        "signal_source": str(
            arts.get("signal_source") or train.get("signal_source") or ""
        ),
        "backend_kind": str(
            arts.get("backend_kind")
            or train.get("backend_kind")
            or getattr(backend, "backend_kind", type(backend).__name__)
        ),
        "is_full_event_wfo": bool(
            arts.get(
                "is_full_event_wfo",
                train.get(
                    "is_full_event_wfo",
                    getattr(backend, "is_full_event_wfo", False),
                ),
            )
        ),
        "wfo_completed_folds": int(
            arts.get("wfo_completed_folds")
            or train.get("wfo_completed_folds")
            or 0
        ),
        "closed_trades": closed_trades,
        "orders_count": int(arts.get("orders_count") or train.get("orders_count") or 0),
        "fills_count": int(arts.get("fills_count") or train.get("fills_count") or 0),
        "trades_count": int(arts.get("trades_count") or train.get("trades_count") or 0),
        "oos_ranges": oos_ranges,
    }


@dataclass
class EvaluationRecord:
    outcome: EvalOutcome
    candidate_id: str
    lineage_id: str
    trial_id: str | None
    fitness: FitnessResult | None
    rejection_reason: str | None
    oos_folds: list[FoldOOSMetrics] = field(default_factory=list)
    train_metrics: dict[str, float] = field(default_factory=dict)
    runtime_seconds: float = 0.0
    memory_mb: float = 0.0
    stress_results: dict[str, Any] = field(default_factory=dict)
    robustness_results: dict[str, Any] = field(default_factory=dict)
    behavioral_cluster: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "outcome": self.outcome.value,
            "candidate_id": self.candidate_id,
            "lineage_id": self.lineage_id,
            "trial_id": self.trial_id,
            "fitness": self.fitness.as_dict() if self.fitness else None,
            "rejection_reason": self.rejection_reason,
            "oos_folds": [f.as_dict() for f in self.oos_folds],
            "train_metrics": dict(self.train_metrics),
            "runtime_seconds": self.runtime_seconds,
            "memory_mb": self.memory_mb,
            "stress_results": dict(self.stress_results),
            "robustness_results": dict(self.robustness_results),
            "behavioral_cluster": self.behavioral_cluster,
            "meta": dict(self.meta),
        }


class BacktestBackend(Protocol):
    """Produces OOS fold metrics — never used for IS ranking."""

    def evaluate(self, candidate: StrategyCandidate) -> tuple[list[FoldOOSMetrics], dict[str, float]]:
        """Return (oos_folds, train_metrics_diagnostic_only)."""
        ...


@dataclass
class SyntheticOOSBackend:
    """
    Deterministic synthetic OOS backend for unit tests and cheap pre-ranking probes.

    Produces fold-level OOS metrics from the candidate identity hash so identical
    candidates reproduce identical rankings. Not a shortcut for final production
    ranking — production must inject a full event-driven WFO backend.

    Control-plane honesty: ``is_full_event_wfo=False`` — synthetic probes must not
    promote dashboard FINALIST / Vault eligibility.
    """

    n_folds: int = 3
    seed_salt: int = 0
    backend_kind: str = "synthetic_oos_probe"
    is_full_event_wfo: bool = False

    def evaluate(self, candidate: StrategyCandidate) -> tuple[list[FoldOOSMetrics], dict[str, float]]:
        digest = hashlib.sha256(
            f"{candidate.candidate_id}:{self.seed_salt}".encode()
        ).hexdigest()
        rng = np.random.default_rng(int(digest[:8], 16) % (2**31 - 1))
        folds: list[FoldOOSMetrics] = []
        for i in range(self.n_folds):
            base = float(rng.normal(0.05, 0.15))
            # Slight complexity drag so simpler templates are not always worse
            base -= 0.01 * candidate.complexity_score
            folds.append(
                FoldOOSMetrics(
                    fold_id=i,
                    expectancy=base,
                    sharpe=base * 4.0 + float(rng.normal(0, 0.2)),
                    profit_factor=max(0.5, 1.0 + base),
                    calmar=base * 2.0,
                    max_drawdown=-abs(float(rng.uniform(0.02, 0.25))),
                    drawdown_duration=float(rng.uniform(1, 20)),
                    worst_day=-abs(float(rng.uniform(0.01, 0.08))),
                    turnover=float(rng.uniform(0.1, 2.0)),
                    prop_breach_prob=float(rng.uniform(0.0, 0.2)),
                    regime_entropy=float(rng.uniform(0.4, 1.0)),
                    n_trades=int(rng.integers(5, 80)),
                )
            )
        # Inflated IS metrics that must never promote
        train = {
            "expectancy": float(max(f.expectancy for f in folds) + 1.0),
            "sharpe": float(max(f.sharpe for f in folds) + 2.0),
        }
        return folds, train


@dataclass
class CandidateEvaluator:
    """
    Evaluate a candidate: structural precheck -> OOS backend -> robust fitness -> registry.

    Every outcome is registered, including duplicates (as skipped) and rejections.
    """

    registry: ExperimentRegistry
    budget: SearchBudget
    counters: BudgetCounters
    fitness_model: RobustFitness = field(default_factory=RobustFitness)
    backend: BacktestBackend = field(default_factory=SyntheticOOSBackend)
    system_version: str = "0.6.3-phase6d"
    data_hash: str = "discovery_data_v1"
    code_hash: str = "discovery_code_v1"
    discovery_run_id: str = "run_unknown"
    grammar: Grammar = field(default_factory=Grammar)
    progress_hook: Callable[[str, dict[str, Any]], None] | None = None
    # When set, reject candidates whose feature_ids are not subset before Full WFO.
    available_feature_ids: frozenset[str] | None = None
    dataset_capabilities: tuple[str, ...] = ()
    # Research-eligible execution must fail closed when feature-capability
    # resolution failed; non-research smoke runs may remain permissive but
    # the fact is always recorded on the evaluation record (never hidden).
    research_eligible: bool = False
    _seen_candidate_ids: set[str] = field(default_factory=set)
    _records: list[EvaluationRecord] = field(default_factory=list)

    def records(self) -> list[EvaluationRecord]:
        return list(self._records)

    def register_invalid_dsl(
        self,
        exc: Exception,
        *,
        seed: int,
        operation: str,
        parent_ids: tuple[str, ...] = (),
        candidate: StrategyCandidate | None = None,
    ) -> EvaluationRecord:
        """Persist INVALID_DSL_TYPE without running the evaluation backend."""
        from discovery.expression_tree import constant_node, feature_node, op_node
        from discovery.operators import OperatorId
        from discovery.types import CreationMethod, ValueType
        from discovery.candidate import build_candidate

        err = parse_dsl_validation_error(
            exc,
            operation=operation,
            candidate_id=candidate.candidate_id if candidate else None,
            parent_ids=parent_ids,
        )
        if candidate is None:
            # Unique placeholder tree so registry identity does not collide
            z = feature_node("price.rolling_z_20", ValueType.ZSCORE)
            entry = op_node(
                OperatorId.ENTRY_LONG,
                op_node(OperatorId.LESS_THAN, z, constant_node(float(seed % 10_000) / 1000.0)),
            )
            candidate = build_candidate(
                entry_tree=entry,
                strategy_family="invalid_dsl_placeholder",
                creation_method=CreationMethod.RANDOM,
                grammar_version="strategy_dsl_v1",
                feature_set_version=self.system_version,
                cost_model_version="cost_v1",
                random_seed=seed,
                parent_ids=parent_ids,
            )
            err.candidate_id = candidate.candidate_id

        self.counters.invalid += 1
        self.counters.generated_attempts += 1
        meta = err.as_dict()
        trial_id = self._register(
            candidate,
            rejection_reason=INVALID_DSL_TYPE,
            ranking_score=None,
            net_metrics={"invalid_dsl": True},
            trial_status=TrialStatus.FAILED,
            failure_reason=err.message,
            extra_snapshot={"invalid_dsl_type": meta, "dsl_operation": operation},
        )
        rec = EvaluationRecord(
            outcome=EvalOutcome.INVALID_DSL_TYPE,
            candidate_id=candidate.candidate_id,
            lineage_id=candidate.lineage_id,
            trial_id=trial_id,
            fitness=None,
            rejection_reason=INVALID_DSL_TYPE,
            runtime_seconds=0.0,
            meta=meta,
        )
        self._records.append(rec)
        self._emit_progress(
            "CANDIDATE_REJECTED",
            {
                "candidate_id": candidate.candidate_id,
                "reason": INVALID_DSL_TYPE,
                "invalid_dsl_type": meta,
            },
        )
        return rec

    def _emit_progress(self, event_name: str, payload: dict[str, Any] | None = None) -> None:
        if self.progress_hook is not None:
            self.progress_hook(event_name, payload or {})

    def _register(
        self,
        candidate: StrategyCandidate,
        *,
        rejection_reason: str | None,
        ranking_score: float | None,
        net_metrics: dict[str, Any],
        gross_metrics: dict[str, Any] | None = None,
        fold_records: list[dict[str, Any]] | None = None,
        trial_status: TrialStatus = TrialStatus.COMPLETED,
        failure_reason: str | None = None,
        extra_snapshot: dict[str, Any] | None = None,
    ) -> str:
        backend_kind = getattr(self.backend, "backend_kind", type(self.backend).__name__)
        is_full_event_wfo = bool(getattr(self.backend, "is_full_event_wfo", False))
        snapshot = {
            "discovery_run_id": self.discovery_run_id,
            "system_version": self.system_version,
            "search_budget_id": self.budget.budget_id,
            "search_budget_version": self.budget.version,
            "grammar_version": candidate.grammar_version,
            "feature_set_version": candidate.feature_set_version,
            "creation_method": candidate.creation_method.value,
            "generation": candidate.generation,
            "parent_ids": list(candidate.parent_ids),
            "expression_tree": candidate.entry_tree.as_dict(),
            "complexity": candidate.complexity_score,
            "backend_kind": backend_kind,
            "is_full_event_wfo": is_full_event_wfo,
            "evaluation_path": (
                "event_driven_wfo" if is_full_event_wfo else str(backend_kind)
            ),
            "family_provenance": dict(candidate.family_provenance or {}),
            **(extra_snapshot or {}),
        }
        trial = self.registry.create_trial(
            candidate_id=candidate.candidate_id,
            lineage_id=candidate.lineage_id,
            strategy_family=candidate.strategy_family,
            parameters=dict(candidate.parameters),
            config_snapshot=snapshot,
            system_version=self.system_version,
            data_hash=self.data_hash,
            random_seed=candidate.random_seed,
            cost_model_version=candidate.cost_model_version,
            code_hash=self.code_hash,
            gross_metrics=gross_metrics or {},
            net_metrics=net_metrics,
            ranking_score=ranking_score,
            rejection_reason=rejection_reason,
            trial_status=trial_status,
            failure_reason=failure_reason,
            fold_records=fold_records,
            ranking_source=RANKING_SOURCE_OOS,
            execution_assumptions={"pipeline": "discovery_evaluator_v1"},
        )
        return trial.trial_id

    def evaluate(self, candidate: StrategyCandidate) -> EvaluationRecord:
        t0 = time.perf_counter()
        self._emit_progress(
            "CANDIDATE_EVALUATION_STARTED",
            {"candidate_id": candidate.candidate_id, "family": candidate.strategy_family},
        )

        # Duplicates: register visibility, do not re-run backend
        if candidate.candidate_id in self._seen_candidate_ids:
            trial_id = self._register(
                candidate,
                rejection_reason="duplicate_not_reevaluated",
                ranking_score=None,
                net_metrics={"duplicate": True},
                extra_snapshot={"duplicate_of": candidate.candidate_id},
            )
            rec = EvaluationRecord(
                outcome=EvalOutcome.DUPLICATE_SKIPPED,
                candidate_id=candidate.candidate_id,
                lineage_id=candidate.lineage_id,
                trial_id=trial_id,
                fitness=None,
                rejection_reason="duplicate_not_reevaluated",
                runtime_seconds=time.perf_counter() - t0,
            )
            self._records.append(rec)
            self._emit_progress(
                "CANDIDATE_REJECTED",
                {"candidate_id": candidate.candidate_id, "reason": "duplicate_not_reevaluated"},
            )
            return rec

        self._seen_candidate_ids.add(candidate.candidate_id)

        try:
            check_strategy_trees(
                candidate.entry_tree,
                candidate.exit_tree,
                candidate.stop,
                candidate.target,
                *candidate.regime_gates,
                operation="evaluate",
                candidate_id=candidate.candidate_id,
                parent_ids=candidate.parent_ids,
            )
        except (DSLValidationError, InvalidDslTypeError) as exc:
            return self.register_invalid_dsl(
                exc,
                seed=candidate.random_seed,
                operation="evaluate",
                parent_ids=candidate.parent_ids,
                candidate=candidate,
            )

        pre = structural_precheck(
            entry=candidate.entry_tree,
            exit=candidate.exit_tree,
            regime_gates=candidate.regime_gates,
            grammar=self.grammar,
            seed=candidate.random_seed,
        )
        if not pre.accepted:
            trial_id = self._register(
                candidate,
                rejection_reason=f"precheck:{pre.reason}",
                ranking_score=None,
                net_metrics={},
                trial_status=TrialStatus.FAILED,
                failure_reason=pre.reason,
            )
            rec = EvaluationRecord(
                outcome=EvalOutcome.PRECHECK_FAILED,
                candidate_id=candidate.candidate_id,
                lineage_id=candidate.lineage_id,
                trial_id=trial_id,
                fitness=None,
                rejection_reason=pre.reason,
                runtime_seconds=time.perf_counter() - t0,
            )
            self._records.append(rec)
            self._emit_progress(
                "CANDIDATE_REJECTED",
                {"candidate_id": candidate.candidate_id, "reason": f"precheck:{pre.reason}"},
            )
            return rec

        # Semantic threshold-domain validation — runs regardless of creation
        # path (generation, mutation, crossover, import, manual construction,
        # or robustness perturbation) since generation-time checks alone can
        # be bypassed by all of those. Rejects before Full WFO; zero budget
        # consumed.
        violations = collect_candidate_threshold_violations(candidate, self.grammar)
        if violations:
            artifact = {"violations": [v.as_dict() for v in violations]}
            trial_id = self._register(
                candidate,
                rejection_reason=INVALID_FEATURE_THRESHOLD_DOMAIN,
                ranking_score=None,
                net_metrics={"invalid_feature_threshold_domain": True, **artifact},
                trial_status=TrialStatus.FAILED,
                failure_reason=INVALID_FEATURE_THRESHOLD_DOMAIN,
                extra_snapshot={"threshold_domain_violations": artifact},
            )
            rec = EvaluationRecord(
                outcome=EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN,
                candidate_id=candidate.candidate_id,
                lineage_id=candidate.lineage_id,
                trial_id=trial_id,
                fitness=None,
                rejection_reason=INVALID_FEATURE_THRESHOLD_DOMAIN,
                runtime_seconds=time.perf_counter() - t0,
                meta=artifact,
            )
            self._records.append(rec)
            self._emit_progress(
                "CANDIDATE_REJECTED",
                {
                    "candidate_id": candidate.candidate_id,
                    "reason": INVALID_FEATURE_THRESHOLD_DOMAIN,
                    **artifact,
                },
            )
            return rec

        # Feature-capability resolution must fail closed for research-eligible
        # execution — an unresolved capability set is not a green light to
        # skip validation, it is a stop condition. Non-research smoke runs may
        # remain permissive, but the fact is always labeled on the record.
        backend_resolution_status = getattr(self.backend, "feature_resolution_status", None)
        feature_resolution_note: dict[str, Any] = {}
        if backend_resolution_status == "failed":
            resolution_error = getattr(self.backend, "feature_resolution_error", None)
            if self.research_eligible:
                artifact = {
                    "research_eligible": True,
                    "feature_resolution_status": FEATURE_CAPABILITY_RESOLUTION_FAILED,
                    "resolution_error": resolution_error,
                    "available_features": None,
                    "dataset_capabilities": list(self.dataset_capabilities),
                    "candidate_id": candidate.candidate_id,
                    "full_wfo_consumed": False,
                }
                trial_id = self._register(
                    candidate,
                    rejection_reason=FEATURE_CAPABILITY_RESOLUTION_FAILED,
                    ranking_score=None,
                    net_metrics={"feature_capability_resolution_failed": True, **artifact},
                    trial_status=TrialStatus.FAILED,
                    failure_reason=resolution_error or FEATURE_CAPABILITY_RESOLUTION_FAILED,
                    extra_snapshot={"feature_capability_resolution": artifact},
                )
                rec = EvaluationRecord(
                    outcome=EvalOutcome.FEATURE_CAPABILITY_RESOLUTION_FAILED,
                    candidate_id=candidate.candidate_id,
                    lineage_id=candidate.lineage_id,
                    trial_id=trial_id,
                    fitness=None,
                    rejection_reason=FEATURE_CAPABILITY_RESOLUTION_FAILED,
                    runtime_seconds=time.perf_counter() - t0,
                    meta=artifact,
                )
                self._records.append(rec)
                self._emit_progress(
                    "CANDIDATE_REJECTED",
                    {
                        "candidate_id": candidate.candidate_id,
                        "reason": FEATURE_CAPABILITY_RESOLUTION_FAILED,
                        **artifact,
                    },
                )
                return rec
            # Explicit non-research smoke fallback — permissive, but labeled so
            # it can never be confused with a real research evaluation.
            feature_resolution_note = {
                "feature_resolution_status": PERMISSIVE_NON_RESEARCH_FALLBACK,
                "resolution_error": resolution_error,
            }

        # Fail before Full WFO when required features are unavailable.
        avail = self.available_feature_ids
        if avail is None:
            avail = getattr(self.backend, "available_feature_ids", None)
        if avail is not None:
            required = tuple(candidate.feature_ids)
            missing = sorted(set(required) - set(avail))
            if missing:
                artifact = {
                    "required_features": list(required),
                    "available_features": sorted(avail),
                    "missing_features": missing,
                    "dataset_capabilities": list(self.dataset_capabilities),
                }
                # Do NOT consume Full WFO budget for feature-unavailable candidates.
                trial_id = self._register(
                    candidate,
                    rejection_reason=FEATURE_UNAVAILABLE,
                    ranking_score=None,
                    net_metrics={"feature_unavailable": True, **artifact},
                    trial_status=TrialStatus.FAILED,
                    failure_reason=FEATURE_UNAVAILABLE,
                    extra_snapshot={"feature_availability": artifact},
                )
                rec = EvaluationRecord(
                    outcome=EvalOutcome.FEATURE_UNAVAILABLE,
                    candidate_id=candidate.candidate_id,
                    lineage_id=candidate.lineage_id,
                    trial_id=trial_id,
                    fitness=None,
                    rejection_reason=FEATURE_UNAVAILABLE,
                    runtime_seconds=time.perf_counter() - t0,
                    meta=artifact,
                )
                self._records.append(rec)
                self._emit_progress(
                    "CANDIDATE_REJECTED",
                    {
                        "candidate_id": candidate.candidate_id,
                        "reason": FEATURE_UNAVAILABLE,
                        **artifact,
                    },
                )
                return rec

        try:
            self._emit_progress(
                "WFO_FOLD_STARTED",
                {"candidate_id": candidate.candidate_id, "backend": type(self.backend).__name__},
            )
            oos_folds, train_metrics = self.backend.evaluate(candidate)
            for fold in oos_folds:
                self._emit_progress(
                    "WFO_FOLD_COMPLETED",
                    {
                        "candidate_id": candidate.candidate_id,
                        "fold_id": fold.fold_id,
                        "expectancy": fold.expectancy,
                        "phase": "validation_oos",
                    },
                )
            baseline_wfo_artifacts = snapshot_baseline_wfo_artifacts(
                candidate_id=candidate.candidate_id,
                backend=self.backend,
                train_metrics=train_metrics if isinstance(train_metrics, dict) else {},
            )
        except Exception as exc:  # noqa: BLE001 — register evaluation failures
            trial_id = self._register(
                candidate,
                rejection_reason=f"eval_failed:{exc}",
                ranking_score=None,
                net_metrics={},
                trial_status=TrialStatus.FAILED,
                failure_reason=str(exc),
            )
            rec = EvaluationRecord(
                outcome=EvalOutcome.EVAL_FAILED,
                candidate_id=candidate.candidate_id,
                lineage_id=candidate.lineage_id,
                trial_id=trial_id,
                fitness=None,
                rejection_reason=str(exc),
                runtime_seconds=time.perf_counter() - t0,
            )
            self._records.append(rec)
            self._emit_progress(
                "CANDIDATE_REJECTED",
                {"candidate_id": candidate.candidate_id, "reason": f"eval_failed:{exc}"},
            )
            return rec

        self.counters.evaluated += 1
        # Full WFO budget only advances when a real WFO evaluation actually
        # completed and the backend's own returned artifacts prove it — being
        # "an event-driven backend" is not sufficient on its own. All prior
        # early-exit paths (duplicate, precheck, invalid DSL, feature
        # unavailable, eval exception) already returned before this line and
        # therefore never consume Full WFO budget.
        completed_fold_count = int(train_metrics.get("wfo_completed_folds", len(oos_folds)))
        is_full_wfo_completion = (
            bool(getattr(self.backend, "is_full_event_wfo", False))
            and train_metrics.get("signal_source") == "candidate_dsl_trees"
            and bool(train_metrics.get("is_full_event_wfo", False))
            and completed_fold_count > 0
        )
        if is_full_wfo_completion:
            self.counters.full_wfo += 1

        fit = self.fitness_model.score(
            oos_folds,
            ranking_source=RANKING_SOURCE_OOS,
            complexity=candidate.complexity_score,
            train_metrics=train_metrics,
        )
        fold_trade_counts = [max(0, int(f.n_trades)) for f in oos_folds]
        total_oos_trades = int(sum(fold_trade_counts))
        trade_meta = {
            "total_oos_trades": total_oos_trades,
            "fold_trade_counts": fold_trade_counts,
            "min_oos_trades": int(self.budget.min_oos_trades),
            "min_oos_trades_per_fold": int(self.budget.min_oos_trades_per_fold),
            "metrics_basis_note": (
                "expectancy_and_profit_factor_are_trade_based; "
                "max_drawdown_and_calmar_are_equity_curve_based"
            ),
        }

        # Prop-risk soft reject
        if any(f.prop_breach_prob > 0.5 for f in oos_folds):
            trial_id = self._register(
                candidate,
                rejection_reason="risk_failed:prop_breach",
                ranking_score=fit.fitness,
                net_metrics={"fitness": fit.fitness, **fit.components, **trade_meta},
                fold_records=[f.as_dict() for f in oos_folds],
                trial_status=TrialStatus.FAILED,
                failure_reason="prop_breach",
            )
            rec = EvaluationRecord(
                outcome=EvalOutcome.RISK_FAILED,
                candidate_id=candidate.candidate_id,
                lineage_id=candidate.lineage_id,
                trial_id=trial_id,
                fitness=fit,
                rejection_reason="prop_breach",
                oos_folds=oos_folds,
                train_metrics=train_metrics,
                runtime_seconds=time.perf_counter() - t0,
                meta={
                    "is_full_wfo_completion": is_full_wfo_completion,
                    "baseline_wfo_artifacts": baseline_wfo_artifacts,
                    **feature_resolution_note,
                },
            )
            self._records.append(rec)
            self._emit_progress(
                "CANDIDATE_REJECTED",
                {"candidate_id": candidate.candidate_id, "reason": "prop_breach"},
            )
            return rec

        if fit.rejected:
            trial_id = self._register(
                candidate,
                rejection_reason=fit.rejection_reason,
                ranking_score=None,
                net_metrics={
                    "train_diagnostic": train_metrics,
                    **fit.components,
                    **trade_meta,
                },
                fold_records=[f.as_dict() for f in oos_folds],
            )
            rec = EvaluationRecord(
                outcome=EvalOutcome.REJECTED,
                candidate_id=candidate.candidate_id,
                lineage_id=candidate.lineage_id,
                trial_id=trial_id,
                fitness=fit,
                rejection_reason=fit.rejection_reason,
                oos_folds=oos_folds,
                train_metrics=train_metrics,
                runtime_seconds=time.perf_counter() - t0,
                meta={
                    "is_full_wfo_completion": is_full_wfo_completion,
                    "baseline_wfo_artifacts": baseline_wfo_artifacts,
                    **feature_resolution_note,
                },
            )
            self._records.append(rec)
            self._emit_progress(
                "CANDIDATE_REJECTED",
                {
                    "candidate_id": candidate.candidate_id,
                    "reason": fit.rejection_reason or "fitness_rejected",
                },
            )
            return rec

        wfo_meta = {
            k: train_metrics[k]
            for k in (
                "wfo_fold_count",
                "wfo_completed_folds",
                "wfo_terminal_status",
                "wfo_folds",
                "is_full_event_wfo",
                "backend_kind",
                "proxy_metric_used",
            )
            if k in train_metrics
        }
        wfo_meta["is_full_wfo_completion"] = is_full_wfo_completion
        trial_id = self._register(
            candidate,
            rejection_reason=None,
            ranking_score=fit.fitness,
            net_metrics={"fitness": fit.fitness, **fit.components, **trade_meta},
            gross_metrics={"train_diagnostic": train_metrics},
            fold_records=[f.as_dict() for f in oos_folds],
            extra_snapshot={"ranking_source": RANKING_SOURCE_OOS, **wfo_meta, **trade_meta},
        )
        rec = EvaluationRecord(
            outcome=EvalOutcome.REGISTERED,
            candidate_id=candidate.candidate_id,
            lineage_id=candidate.lineage_id,
            trial_id=trial_id,
            fitness=fit,
            rejection_reason=None,
            oos_folds=oos_folds,
            train_metrics=train_metrics,
            runtime_seconds=time.perf_counter() - t0,
            meta={
                "is_full_wfo_completion": is_full_wfo_completion,
                "baseline_wfo_artifacts": baseline_wfo_artifacts,
                **feature_resolution_note,
            },
        )
        self._records.append(rec)
        self.counters.runtime_seconds += rec.runtime_seconds
        self._emit_progress(
            "CANDIDATE_EVALUATED",
            {
                "candidate_id": candidate.candidate_id,
                "fitness": fit.fitness,
                "ranking_source": RANKING_SOURCE_OOS,
                "evaluated": self.counters.evaluated,
            },
        )
        return rec
