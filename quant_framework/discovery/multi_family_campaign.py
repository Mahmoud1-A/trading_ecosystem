"""Multi-family Alpha Miner campaign — allocate budget across FamilySpecs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable
from uuid import uuid4

from discovery.candidate import StrategyCandidate
from discovery.crossover import Crossover, CrossoverError
from discovery.evaluator import (
    CandidateEvaluator,
    EvalOutcome,
    EvaluationRecord,
    SyntheticOOSBackend,
)
from discovery.expression_tree import ExprNode
from discovery.family_generator import StrategyFamilyGenerator
from discovery.family_spec import FamilySpec, assert_diverse_family_grammars
from discovery.feature_domains import validate_tree_threshold_domains
from discovery.fitness import RobustFitness
from discovery.generator import CandidateGenerator
from discovery.grammar import Grammar
from discovery.mutation import MutationRejected, Mutator
from discovery.operators import OperatorId
from discovery.search_budget import BudgetCounters, SearchBudget
from discovery.search_controller import DiscoveryRunResult
from discovery.stable_hash import stable_seed
from discovery.parameter_robustness import (
    PARAMETER_BUDGET_NOT_REACHED,
    PARAMETER_INSUFFICIENT_NEIGHBORHOOD,
    PARAMETER_INTEGRITY_FAILED,
    PARAMETER_NOT_BOUND,
    PARAMETER_ROBUST,
    PARAMETER_UNSTABLE,
    ParameterRobustness,
    ParameterRobustnessSummary,
    REAL_ROBUSTNESS_BACKEND_REQUIRED,
    ROBUSTNESS_BACKEND_KIND_INVALID,
    ROBUSTNESS_BUDGET_EXHAUSTED,
    ROBUSTNESS_INSUFFICIENT_VALID_NEIGHBORHOOD,
    ROBUSTNESS_INTEGRITY_FAILED,
    ROBUSTNESS_PARAMETER_NOT_BOUND,
    ROBUSTNESS_SIGNAL_SOURCE_INVALID,
    ROBUSTNESS_WFO_INCOMPLETE,
    SYNTHETIC_ROBUSTNESS_FORBIDDEN,
)
from discovery.research_shortlist_pipeline import (
    AlignedPerformanceMatrix,
    BEHAVIORALLY_CLUSTERED,
    CLUSTERING_NOT_ENTERED,
    CampaignClusteringAccounting,
    CampaignStatisticsAccounting,
    CandidateStatisticsSummary,
    DSR_FAILED,
    DSR_INSUFFICIENT_DATA,
    DSR_PASSED,
    NO_BEHAVIORAL_SIGNATURE,
    NO_ROBUSTNESS_PASSED_FOR_STATISTICS,
    PBO_FAILED,
    PBO_INSUFFICIENT_DATA,
    PBO_PASSED,
    PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_RESEARCH_SHORTLIST,
    RESEARCH_SHORTLISTED,
    RESEARCH_SHORTLISTED_DOES_NOT_MEAN,
    ResearchShortlistConfig,
    ResearchShortlistEntry,
    ResearchShortlistPhaseResult,
    SHORTLIST_NOT_ENTERED,
    SHORTLIST_REJECTED,
    STATISTICS_NOT_ENTERED,
    STATISTICS_REQUIRES_FULL_WFO,
    STATISTICS_REQUIRES_ROBUSTNESS_PASSED,
    STATISTICS_REQUIRES_STRESS_PASSED,
    STATISTICS_TESTED,
    STATISTICALLY_PASSED,
    STATISTICALLY_REJECTED,
    ShortlistRejectRecord,
    align_performance_matrix,
    build_behavioral_signature,
    build_full_wfo_trial_population,
    evaluate_candidate_dsr_pbo,
    select_best_per_cluster,
    BehavioralDeduper,
)
from discovery.stress import StressResult, StressTester, attach_stress
from discovery.stress_backend import (
    STRESS_BACKEND_KIND,
    SYNTHETIC_STRESS_FORBIDDEN,
    make_stress_backend_factory,
)
from discovery.types import NodeKind
from registry.experiment_registry import ExperimentRegistry
from registry.hashing import sha256_json

STRUCTURAL_PARENT_ELIGIBLE = "STRUCTURAL_PARENT_ELIGIBLE"
SCORE_QUALIFIED = "SCORE_QUALIFIED"
NOT_PARENT_ELIGIBLE = "NOT_PARENT_ELIGIBLE"
FAMILY_DIRECTION_INCOHERENT = "FAMILY_DIRECTION_INCOHERENT"
NO_STRUCTURAL_PARENTS = "NO_STRUCTURAL_PARENTS"

# Phase 3B.1 candidate status events (sequence-ordered; not wall-clock).
FULL_WFO_COMPLETED = "FULL_WFO_COMPLETED"
STRESS_TESTED = "STRESS_TESTED"
STRESS_PASSED = "STRESS_PASSED"
STRESS_FAILED = "STRESS_FAILED"
STRESS_NOT_ENTERED = "STRESS_NOT_ENTERED"
ROBUSTNESS_TESTED = "ROBUSTNESS_TESTED"
ROBUSTNESS_PASSED = "ROBUSTNESS_PASSED"
ROBUSTNESS_FAILED = "ROBUSTNESS_FAILED"
ROBUSTNESS_NOT_ENTERED = "ROBUSTNESS_NOT_ENTERED"

# Explicit Stress integrity / entry failures.
REAL_STRESS_BACKEND_REQUIRED = "REAL_STRESS_BACKEND_REQUIRED"
STRESS_SIGNAL_SOURCE_INVALID = "STRESS_SIGNAL_SOURCE_INVALID"
STRESS_WFO_INCOMPLETE = "STRESS_WFO_INCOMPLETE"
STRESS_BACKEND_KIND_INVALID = "STRESS_BACKEND_KIND_INVALID"
STRESS_INTEGRITY_FAILED = "STRESS_INTEGRITY_FAILED"
STRESS_BUDGET_EXHAUSTED = "STRESS_BUDGET_EXHAUSTED"
STRESS_NO_EXECUTED_SCENARIOS = "STRESS_NO_EXECUTED_SCENARIOS"
STRESS_UNSUPPORTED_REQUIRED = "STRESS_UNSUPPORTED_REQUIRED"
STRESS_BASELINE_UNAVAILABLE_REQUIRED = "STRESS_BASELINE_UNAVAILABLE_REQUIRED"
STRESS_PASS_RATE_LOW = "STRESS_PASS_RATE_LOW"

# Explicit Robustness integrity / entry failures.
ROBUSTNESS_INCOMPLETE = "ROBUSTNESS_INCOMPLETE"
ROBUSTNESS_PARAMETER_UNSTABLE = "ROBUSTNESS_PARAMETER_UNSTABLE"

_REQUIRED_STRESS_SIGNAL_SOURCE = "candidate_dsl_trees"
_REQUIRED_STRESS_BACKEND_KIND = STRESS_BACKEND_KIND
_REQUIRED_ROBUSTNESS_SIGNAL_SOURCE = "candidate_dsl_trees"
_REQUIRED_ROBUSTNESS_BACKEND_KIND = "event_driven_wfo"
BASELINE_ARTIFACT_SOURCE_ORIGINAL_WFO = "original_qualifying_wfo"

DEFAULT_MULTIFAMILY_STRESS_SCENARIOS: tuple[str, ...] = (
    "base_costs",
    "costs_2x",
    "costs_4x",
    "wider_spread",
    "worse_slippage",
    "delayed_execution",
    "conservative_intrabar",
    "removed_best_day",
    "removed_best_trades",
    "symbol_exclusion",
)

# Families whose entry condition sign must economically match ENTRY_LONG/SHORT.
_DIRECTION_LINKED_FAMILY_IDS = frozenset(
    {
        "mean_reversion",
        "VWAP_reversion",
        "momentum",
        "breakout",
        "gap_fade",
    }
)

_FADE_FEATURES = frozenset(
    {
        "price.rolling_z_20",
        "liq.dist_session_vwap",
        "price.dist_rolling_mean_20",
        "price.close_to_open",
    }
)
_FOLLOW_FEATURES = frozenset(
    {
        "price.return_5",
        "price.simple_return_1",
        "price.log_return_1",
        "price.breakout_distance_20",
        "price.breakdown_distance_20",
    }
)
# Upside breakout distance proves LONG only; downside breakdown proves SHORT only.
# Negative distance from the prior maximum is not a downside-breakout proof.
_FOLLOW_UP_ONLY_FEATURES = frozenset({"price.breakout_distance_20"})
_FOLLOW_DOWN_ONLY_FEATURES = frozenset({"price.breakdown_distance_20"})
# Volatility / liquidity / participation context — never casts a direction vote.
_CONTEXT_ONLY_FEATURES = frozenset(
    {
        "vol.range_compression_20",
        "vol.prior_range_compression_20",
        "liq.volume_pct_20",
    }
)
_UP_OPS = frozenset(
    {
        OperatorId.GREATER_THAN.value,
        OperatorId.CROSS_ABOVE.value,
    }
)
_DOWN_OPS = frozenset(
    {
        OperatorId.LESS_THAN.value,
        OperatorId.CROSS_BELOW.value,
    }
)
_CMP_OPS = _UP_OPS | _DOWN_OPS

# Honest capability declaration.
PIPELINE_LEVEL_MULTI_FAMILY_WFO_SCREENING = "MULTI_FAMILY_WFO_SCREENING"
PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_WFO_SCREENING = (
    "MULTI_FAMILY_EVOLUTIONARY_WFO_SCREENING"
)
PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_STRESS_SCREENING = (
    "MULTI_FAMILY_EVOLUTIONARY_STRESS_SCREENING"
)
PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_ROBUSTNESS_SCREENING = (
    "MULTI_FAMILY_EVOLUTIONARY_ROBUSTNESS_SCREENING"
)
SCORE_QUALIFIED_MEANING = "passed economic WFO qualification"
SCORE_QUALIFIED_DOES_NOT_MEAN = (
    "finalist",
    "promoted",
    "stress passed",
    "research shortlisted",
    "vault eligible",
    "paper eligible",
)
STRESS_PASSED_DOES_NOT_MEAN = (
    "finalist",
    "promoted",
    "research shortlisted",
    "vault eligible",
    "paper eligible",
)
ROBUSTNESS_PASSED_DOES_NOT_MEAN = (
    "finalist",
    "promoted",
    "research shortlisted",
    "vault eligible",
    "paper eligible",
    "live eligible",
)

POST_WFO_PIPELINE_NOT_RUN = "POST_WFO_PIPELINE_NOT_RUN"
STRESS_NOT_RUN = "STRESS_NOT_RUN"
ROBUSTNESS_NOT_RUN = "ROBUSTNESS_NOT_RUN"
STATISTICS_NOT_RUN = "STATISTICS_NOT_RUN"
CLUSTERING_NOT_RUN = "CLUSTERING_NOT_RUN"
NOT_RESEARCH_SHORTLISTED = "NOT_RESEARCH_SHORTLISTED"
VAULT_NOT_RUN = "VAULT_NOT_RUN"
PAPER_NOT_RUN = "PAPER_NOT_RUN"
LIVE_NOT_RUN = "LIVE_NOT_RUN"
FAMILY_LOCAL_EVOLUTION_NOT_READY = "FAMILY_LOCAL_EVOLUTION_NOT_READY"
CROSS_FAMILY_CROSSOVER_UNSUPPORTED = "CROSS_FAMILY_CROSSOVER_UNSUPPORTED"
FAMILY_STAGNATION = "FAMILY_STAGNATION"


@dataclass(frozen=True)
class DirectionCoherenceResult:
    """Outcome of :func:`validate_family_direction_coherence`."""

    coherent: bool
    rejection_reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.coherent


def _numeric_threshold(node: ExprNode | None) -> float | None:
    if node is None:
        return None
    if node.kind is NodeKind.CONSTANT:
        return float(node.meta.get("value", 0.0))
    if node.kind is NodeKind.PARAMETER:
        return float(node.meta.get("default", 0.0))
    return None


def _unwrap_abs_feature(node: ExprNode) -> tuple[str | None, bool]:
    """Return ``(feature_id, wrapped_in_abs)`` for a comparison left-hand side."""
    if node.kind is NodeKind.FEATURE:
        return node.name, False
    if (
        node.kind is NodeKind.OPERATOR
        and node.name == OperatorId.ABS.value
        and node.children
        and node.children[0].kind is NodeKind.FEATURE
    ):
        return node.children[0].name, True
    return None, False


def _mechanism_for_feature(family_id: str, feature_id: str) -> str | None:
    """Return ``'fade'``, ``'follow'``, or ``None`` if feature is not directional.

    Context-only features (compression, volume participation, etc.) never vote.
    """
    _ = family_id
    if feature_id in _CONTEXT_ONLY_FEATURES:
        return None
    if feature_id in _FADE_FEATURES:
        # Gap and displacement features are fade economics regardless of family label.
        return "fade"
    if feature_id in _FOLLOW_FEATURES:
        return "follow"
    return None


def _expected_direction_from_comparison(
    *,
    mechanism: str,
    op_name: str,
    threshold: float | None,
    feature_id: str | None = None,
) -> tuple[str | None, str]:
    """Map a signed comparison onto the economically required entry direction."""
    is_up = op_name in _UP_OPS
    if feature_id in _FOLLOW_UP_ONLY_FEATURES:
        if not is_up:
            return None, "breakout_distance_does_not_prove_short"
        if threshold is not None and threshold < 0:
            return None, "follow_long_requires_non_negative_threshold"
        return "ENTRY_LONG", "follow_up_signed_condition"
    if feature_id in _FOLLOW_DOWN_ONLY_FEATURES:
        if is_up:
            return None, "breakdown_distance_does_not_prove_long"
        if threshold is not None and threshold > 0:
            return None, "follow_short_requires_non_positive_threshold"
        return "ENTRY_SHORT", "follow_down_signed_condition"
    if mechanism == "fade":
        expected = "ENTRY_SHORT" if is_up else "ENTRY_LONG"
        if threshold is not None:
            if expected == "ENTRY_LONG" and threshold > 0:
                return None, "fade_long_requires_non_positive_threshold"
            if expected == "ENTRY_SHORT" and threshold < 0:
                return None, "fade_short_requires_non_negative_threshold"
        return expected, "fade_signed_condition"
    if mechanism == "follow":
        expected = "ENTRY_LONG" if is_up else "ENTRY_SHORT"
        if threshold is not None:
            if expected == "ENTRY_LONG" and threshold < 0:
                return None, "follow_long_requires_non_negative_threshold"
            if expected == "ENTRY_SHORT" and threshold > 0:
                return None, "follow_short_requires_non_positive_threshold"
        return expected, "follow_signed_condition"
    return None, "unknown_mechanism"


def validate_family_direction_coherence(
    candidate: StrategyCandidate,
    family_spec: FamilySpec,
) -> DirectionCoherenceResult:
    """Prove ENTRY_LONG/SHORT remains economically coherent with the entry condition.

    Direction-linked families (mean reversion, VWAP reversion, momentum, breakout,
    gap fade) must keep wrapper direction aligned with the signed feature condition.
    Ambiguous evidence (e.g. ABS(gap) without a signed directional condition) is
    rejected rather than guessed. Non-direction-linked families pass through.
    """
    family_id = str(family_spec.family_id)
    prov = dict(candidate.family_provenance or {})
    creation = (
        candidate.creation_method.value
        if hasattr(candidate.creation_method, "value")
        else str(candidate.creation_method)
    )
    base_details: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "family_id": family_id,
        "creation_method": creation,
        "entry_direction": None,
        "condition_feature": None,
        "comparison_operator": None,
        "threshold_value": None,
        "expected_direction": None,
        "directional_evidence": [],
        "context_evidence": [],
        "coherence_reason": None,
    }

    if family_id not in _DIRECTION_LINKED_FAMILY_IDS:
        base_details["coherence_reason"] = "family_not_direction_linked"
        return DirectionCoherenceResult(coherent=True, details=base_details)

    entry = candidate.entry_tree
    if entry is None or entry.kind is not NodeKind.OPERATOR:
        base_details["coherence_reason"] = "missing_entry_wrapper"
        return DirectionCoherenceResult(
            coherent=False,
            rejection_reason=FAMILY_DIRECTION_INCOHERENT,
            details=base_details,
        )
    if entry.name not in {
        OperatorId.ENTRY_LONG.value,
        OperatorId.ENTRY_SHORT.value,
    }:
        base_details["coherence_reason"] = "missing_entry_wrapper"
        return DirectionCoherenceResult(
            coherent=False,
            rejection_reason=FAMILY_DIRECTION_INCOHERENT,
            details=base_details,
        )

    entry_direction = entry.name
    base_details["entry_direction"] = entry_direction
    if not entry.children:
        base_details["coherence_reason"] = "missing_entry_condition"
        return DirectionCoherenceResult(
            coherent=False,
            rejection_reason=FAMILY_DIRECTION_INCOHERENT,
            details=base_details,
        )

    condition = entry.children[0]
    # Provenance pattern is advisory; family_id + features are authoritative.
    _ = prov.get("entry_pattern") or prov.get("pattern")

    expected_votes: list[str] = []
    directional_evidence: list[dict[str, Any]] = []
    context_evidence: list[dict[str, Any]] = []
    abs_only_directional = False

    for node in condition.walk():
        if node.kind is not NodeKind.OPERATOR or node.name not in _CMP_OPS:
            continue
        if len(node.children) < 2:
            continue
        left, right = node.children[0], node.children[1]
        feature_id, wrapped_abs = _unwrap_abs_feature(left)
        if feature_id is None:
            continue
        thr = _numeric_threshold(right)
        # Context features may be recorded but never cast a direction vote.
        if feature_id in _CONTEXT_ONLY_FEATURES:
            context_evidence.append(
                {
                    "condition_feature": feature_id,
                    "comparison_operator": node.name,
                    "threshold_value": thr,
                    "wrapped_abs": wrapped_abs,
                    "role": "context",
                }
            )
            continue
        mechanism = _mechanism_for_feature(family_id, feature_id)
        if mechanism is None:
            continue
        if wrapped_abs:
            # ABS(signed_feature) alone cannot prove trade direction.
            abs_only_directional = True
            directional_evidence.append(
                {
                    "condition_feature": feature_id,
                    "comparison_operator": node.name,
                    "threshold_value": thr,
                    "coherence_reason": "abs_signed_feature_ambiguous",
                }
            )
            continue
        expected, reason = _expected_direction_from_comparison(
            mechanism=mechanism,
            op_name=node.name,
            threshold=thr,
            feature_id=feature_id,
        )
        directional_evidence.append(
            {
                "condition_feature": feature_id,
                "comparison_operator": node.name,
                "threshold_value": thr,
                "expected_direction": expected,
                "coherence_reason": reason,
            }
        )
        if expected is None:
            details = {
                **base_details,
                **directional_evidence[-1],
                "directional_evidence": directional_evidence,
                "context_evidence": context_evidence,
                "coherence_reason": reason,
            }
            return DirectionCoherenceResult(
                coherent=False,
                rejection_reason=FAMILY_DIRECTION_INCOHERENT,
                details=details,
            )
        expected_votes.append(expected)

    base_details["directional_evidence"] = directional_evidence
    base_details["context_evidence"] = context_evidence

    if not expected_votes:
        reason = (
            "abs_signed_feature_without_directional_proof"
            if abs_only_directional
            else "no_provable_directional_condition"
        )
        details = {**base_details, "coherence_reason": reason}
        if directional_evidence:
            details.update(
                {
                    k: v
                    for k, v in directional_evidence[0].items()
                    if k != "coherence_reason"
                }
            )
            details["coherence_reason"] = reason
        elif context_evidence:
            details.update(
                {
                    k: v
                    for k, v in context_evidence[0].items()
                    if k not in {"role", "wrapped_abs"}
                }
            )
            details["coherence_reason"] = reason
        return DirectionCoherenceResult(
            coherent=False,
            rejection_reason=FAMILY_DIRECTION_INCOHERENT,
            details=details,
        )

    if len(set(expected_votes)) != 1:
        details = {
            **base_details,
            **(directional_evidence[0] if directional_evidence else {}),
            "expected_direction": sorted(set(expected_votes)),
            "directional_evidence": directional_evidence,
            "context_evidence": context_evidence,
            "coherence_reason": "conflicting_directional_conditions",
        }
        return DirectionCoherenceResult(
            coherent=False,
            rejection_reason=FAMILY_DIRECTION_INCOHERENT,
            details=details,
        )

    expected_direction = expected_votes[0]
    primary = directional_evidence[0] if directional_evidence else {}
    details = {
        **base_details,
        **primary,
        "expected_direction": expected_direction,
        "directional_evidence": directional_evidence,
        "context_evidence": context_evidence,
    }
    if entry_direction != expected_direction:
        details["coherence_reason"] = (
            f"entry_{entry_direction}_conflicts_with_expected_{expected_direction}"
        )
        return DirectionCoherenceResult(
            coherent=False,
            rejection_reason=FAMILY_DIRECTION_INCOHERENT,
            details=details,
        )

    details["coherence_reason"] = "direction_coherent"
    return DirectionCoherenceResult(coherent=True, details=details)

# Outcomes that must never be used as evolutionary parents.
_NOT_PARENT_OUTCOMES = frozenset(
    {
        EvalOutcome.INVALID_DSL_TYPE,
        EvalOutcome.PRECHECK_FAILED,
        EvalOutcome.INVALID_FEATURE_THRESHOLD_DOMAIN,
        EvalOutcome.FEATURE_UNAVAILABLE,
        EvalOutcome.FEATURE_CAPABILITY_RESOLUTION_FAILED,
        EvalOutcome.DUPLICATE_SKIPPED,
        EvalOutcome.EVAL_FAILED,
    }
)

POST_WFO_BLOCKED_REASONS: tuple[str, ...] = (
    POST_WFO_PIPELINE_NOT_RUN,
    STRESS_NOT_RUN,
    ROBUSTNESS_NOT_RUN,
    STATISTICS_NOT_RUN,
    CLUSTERING_NOT_RUN,
    NOT_RESEARCH_SHORTLISTED,
    VAULT_NOT_RUN,
    PAPER_NOT_RUN,
    LIVE_NOT_RUN,
)

# After Phase 3B.1 Stress orchestration: Stress ran; later stages still blocked.
POST_STRESS_BLOCKED_REASONS: tuple[str, ...] = (
    POST_WFO_PIPELINE_NOT_RUN,
    ROBUSTNESS_NOT_RUN,
    STATISTICS_NOT_RUN,
    CLUSTERING_NOT_RUN,
    NOT_RESEARCH_SHORTLISTED,
    VAULT_NOT_RUN,
    PAPER_NOT_RUN,
    LIVE_NOT_RUN,
)

# After Phase 3B.2 Robustness orchestration: Robustness ran; later stages blocked.
POST_ROBUSTNESS_BLOCKED_REASONS: tuple[str, ...] = (
    POST_WFO_PIPELINE_NOT_RUN,
    STATISTICS_NOT_RUN,
    CLUSTERING_NOT_RUN,
    NOT_RESEARCH_SHORTLISTED,
    VAULT_NOT_RUN,
    PAPER_NOT_RUN,
    LIVE_NOT_RUN,
)

# After Phase 3C Research Shortlist: statistics/clustering/shortlist ran; Vault/Paper/Live blocked.
POST_SHORTLIST_BLOCKED_REASONS: tuple[str, ...] = (
    POST_WFO_PIPELINE_NOT_RUN,
    VAULT_NOT_RUN,
    PAPER_NOT_RUN,
    LIVE_NOT_RUN,
)

EMPTY_COLLECTIONS_REASONS: dict[str, list[str]] = {
    "finalists": [
        POST_WFO_PIPELINE_NOT_RUN,
        STRESS_NOT_RUN,
        ROBUSTNESS_NOT_RUN,
        CLUSTERING_NOT_RUN,
    ],
    "promoted": [POST_WFO_PIPELINE_NOT_RUN, STRESS_NOT_RUN, ROBUSTNESS_NOT_RUN],
    "clusters": [POST_WFO_PIPELINE_NOT_RUN, CLUSTERING_NOT_RUN],
    "research_shortlist": [POST_WFO_PIPELINE_NOT_RUN, NOT_RESEARCH_SHORTLISTED],
    "vault_candidates": [POST_WFO_PIPELINE_NOT_RUN, NOT_RESEARCH_SHORTLISTED, VAULT_NOT_RUN],
    "paper_candidates": [POST_WFO_PIPELINE_NOT_RUN, NOT_RESEARCH_SHORTLISTED, PAPER_NOT_RUN],
}

EMPTY_COLLECTIONS_REASONS_AFTER_STRESS: dict[str, list[str]] = {
    "finalists": [
        POST_WFO_PIPELINE_NOT_RUN,
        ROBUSTNESS_NOT_RUN,
        CLUSTERING_NOT_RUN,
        NOT_RESEARCH_SHORTLISTED,
    ],
    "promoted": [POST_WFO_PIPELINE_NOT_RUN, ROBUSTNESS_NOT_RUN],
    "clusters": [POST_WFO_PIPELINE_NOT_RUN, CLUSTERING_NOT_RUN],
    "research_shortlist": [POST_WFO_PIPELINE_NOT_RUN, NOT_RESEARCH_SHORTLISTED],
    "vault_candidates": [POST_WFO_PIPELINE_NOT_RUN, NOT_RESEARCH_SHORTLISTED, VAULT_NOT_RUN],
    "paper_candidates": [POST_WFO_PIPELINE_NOT_RUN, NOT_RESEARCH_SHORTLISTED, PAPER_NOT_RUN],
}

EMPTY_COLLECTIONS_REASONS_AFTER_ROBUSTNESS: dict[str, list[str]] = {
    "finalists": [
        POST_WFO_PIPELINE_NOT_RUN,
        STATISTICS_NOT_RUN,
        CLUSTERING_NOT_RUN,
        NOT_RESEARCH_SHORTLISTED,
    ],
    "promoted": [POST_WFO_PIPELINE_NOT_RUN, STATISTICS_NOT_RUN, NOT_RESEARCH_SHORTLISTED],
    "clusters": [POST_WFO_PIPELINE_NOT_RUN, CLUSTERING_NOT_RUN],
    "research_shortlist": [POST_WFO_PIPELINE_NOT_RUN, NOT_RESEARCH_SHORTLISTED],
    "vault_candidates": [POST_WFO_PIPELINE_NOT_RUN, NOT_RESEARCH_SHORTLISTED, VAULT_NOT_RUN],
    "paper_candidates": [POST_WFO_PIPELINE_NOT_RUN, NOT_RESEARCH_SHORTLISTED, PAPER_NOT_RUN],
}

EMPTY_COLLECTIONS_REASONS_AFTER_SHORTLIST: dict[str, list[str]] = {
    "finalists": [POST_WFO_PIPELINE_NOT_RUN, VAULT_NOT_RUN],
    "promoted": [POST_WFO_PIPELINE_NOT_RUN, VAULT_NOT_RUN],
    "clusters": [],
    "research_shortlist": [],
    "vault_candidates": [NOT_RESEARCH_SHORTLISTED, VAULT_NOT_RUN],
    "paper_candidates": [NOT_RESEARCH_SHORTLISTED, PAPER_NOT_RUN],
}


@dataclass
class FamilyCampaignConfig:
    """Multi-family campaign configuration.

    ``initial_candidates_per_family`` is the **exact** generation-0 population
    per family under family-local evolution. When omitted, ``min_candidates_per_family``
    maps to that exact initial size for compatibility (it is no longer only a
    floor that lets the full ``total_candidate_budget`` expand generation 0).
    """

    requested_family_count: int = 6
    min_candidates_per_family: int = 10
    initial_candidates_per_family: int | None = None
    total_candidate_budget: int = 60
    max_full_wfo: int = 18
    adaptive_reallocation: bool = True
    seed: int = 42
    family_ids: list[str] | None = None
    max_evaluated_candidates: int | None = None
    max_runtime_seconds: float = 600.0
    min_oos_trades: int = 1
    min_oos_trades_per_fold: int = 1
    max_oos_drawdown: float = 0.20
    population_size: int = 2
    stagnation_generations: int = 99
    family_local_evolution: bool = False
    evolution_generations: int = 2
    allow_cross_family_crossover: bool = False
    minimum_improvement: float = 1e-4
    # Phase 3B.1 Stress orchestration (evolutionary path).
    max_stress_evaluations: int = 24
    max_stress_scenarios_per_candidate: int = 10
    min_stress_pass_rate: float = 0.5
    stress_scenarios: tuple[str, ...] | None = None
    fail_closed_unsupported_stress: bool = True
    # Phase 3B.2 Robustness orchestration (evolutionary path).
    max_robustness_candidates: int = 2
    max_robustness_evaluations: int = 30
    max_parameters_per_candidate: int = 2
    max_points_per_parameter: int = 5
    allow_one_sided_neighborhood: bool = False
    min_valid_neighborhood_points: int = 3
    # Phase 3C: DSR/PBO → Behavioral Clustering → Research Shortlist.
    min_dsr: float = 0.95
    max_pbo: float = 0.50
    pbo_n_splits: int = 4
    behavioral_similarity_threshold: float = 0.85
    min_oos_observations_for_dsr: int = 20

    def effective_initial_candidates_per_family(self) -> int:
        """Exact generation-0 size per family (never expanded to exhaust total budget)."""
        if self.initial_candidates_per_family is not None:
            return max(1, int(self.initial_candidates_per_family))
        return max(1, int(self.min_candidates_per_family))

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_family_count": self.requested_family_count,
            "min_candidates_per_family": self.min_candidates_per_family,
            "initial_candidates_per_family": self.effective_initial_candidates_per_family(),
            "total_candidate_budget": self.total_candidate_budget,
            "max_full_wfo": self.max_full_wfo,
            "adaptive_reallocation": self.adaptive_reallocation,
            "seed": self.seed,
            "family_ids": list(self.family_ids) if self.family_ids else None,
            "max_evaluated_candidates": self.max_evaluated_candidates,
            "max_runtime_seconds": self.max_runtime_seconds,
            "min_oos_trades": self.min_oos_trades,
            "min_oos_trades_per_fold": self.min_oos_trades_per_fold,
            "max_oos_drawdown": self.max_oos_drawdown,
            "population_size": self.population_size,
            "stagnation_generations": self.stagnation_generations,
            "family_local_evolution": self.family_local_evolution,
            "evolution_generations": self.evolution_generations,
            "allow_cross_family_crossover": self.allow_cross_family_crossover,
            "minimum_improvement": self.minimum_improvement,
            "max_stress_evaluations": self.max_stress_evaluations,
            "max_stress_scenarios_per_candidate": self.max_stress_scenarios_per_candidate,
            "min_stress_pass_rate": self.min_stress_pass_rate,
            "stress_scenarios": (
                list(self.stress_scenarios) if self.stress_scenarios is not None else None
            ),
            "fail_closed_unsupported_stress": self.fail_closed_unsupported_stress,
            "max_robustness_candidates": self.max_robustness_candidates,
            "max_robustness_evaluations": self.max_robustness_evaluations,
            "max_parameters_per_candidate": self.max_parameters_per_candidate,
            "max_points_per_parameter": self.max_points_per_parameter,
            "allow_one_sided_neighborhood": self.allow_one_sided_neighborhood,
            "min_valid_neighborhood_points": self.min_valid_neighborhood_points,
            "min_dsr": self.min_dsr,
            "max_pbo": self.max_pbo,
            "pbo_n_splits": self.pbo_n_splits,
            "behavioral_similarity_threshold": self.behavioral_similarity_threshold,
            "min_oos_observations_for_dsr": self.min_oos_observations_for_dsr,
        }


@dataclass
class FamilyStats:
    family_id: str
    hypothesis: str
    canonical_hash: str
    grammar_fingerprint: str
    allocation_generated: int
    allocation_wfo: int
    generated: int = 0
    evaluated: int = 0
    full_wfo: int = 0
    score_qualified: int = 0
    stress_passed: int = 0
    robustness_passed: int = 0
    statistically_passed: int = 0
    research_shortlisted: int = 0
    best_fitness: float | None = None
    median_oos_expectancy: float | None = None
    median_pf: float | None = None
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    best_candidate_ids: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    parent_status_counts: dict[str, int] = field(default_factory=dict)
    allocation_detail: dict[str, Any] = field(default_factory=dict)
    generation_0_generated: int = 0
    descendants_generated: int = 0
    mutation_children: int = 0
    crossover_children: int = 0
    highest_generation_reached: int = 0
    structural_parents_found: int = 0
    family_stop_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "hypothesis": self.hypothesis,
            "canonical_hash": self.canonical_hash,
            "grammar_fingerprint": self.grammar_fingerprint,
            "allocation_generated": self.allocation_generated,
            "allocation_wfo": self.allocation_wfo,
            "generated": self.generated,
            "evaluated": self.evaluated,
            "full_wfo": self.full_wfo,
            "score_qualified": self.score_qualified,
            "stress_passed": self.stress_passed,
            "robustness_passed": self.robustness_passed,
            "statistically_passed": self.statistically_passed,
            "research_shortlisted": self.research_shortlisted,
            "best_fitness": self.best_fitness,
            "median_oos_expectancy": self.median_oos_expectancy,
            "median_pf": self.median_pf,
            "rejection_reasons": dict(self.rejection_reasons),
            "best_candidate_ids": list(self.best_candidate_ids),
            "constraints": dict(self.constraints),
            "parent_status_counts": dict(self.parent_status_counts),
            "allocation_detail": dict(self.allocation_detail),
            "generation_0_generated": self.generation_0_generated,
            "descendants_generated": self.descendants_generated,
            "mutation_children": self.mutation_children,
            "crossover_children": self.crossover_children,
            "highest_generation_reached": self.highest_generation_reached,
            "structural_parents_found": self.structural_parents_found,
            "family_stop_reason": self.family_stop_reason,
        }


@dataclass
class GenerationRecord:
    """Per-family, per-generation evolution accounting."""

    family_id: str
    generation: int
    input_population_ids: list[str] = field(default_factory=list)
    evaluated_candidate_ids: list[str] = field(default_factory=list)
    selected_parent_ids: list[str] = field(default_factory=list)
    mutation_candidate_ids: list[str] = field(default_factory=list)
    crossover_candidate_ids: list[str] = field(default_factory=list)
    completed_full_wfo_candidate_ids: list[str] = field(default_factory=list)
    score_qualified_candidate_ids: list[str] = field(default_factory=list)
    best_generation_score: float | None = None
    best_score_so_far: float | None = None
    generated_count: int = 0
    evaluated_count: int = 0
    completed_full_wfo_count: int = 0
    stagnation_count: int = 0
    stop_reason: str | None = None
    previous_best_score: float | None = None
    generation_best_score: float | None = None
    improvement: float | None = None
    minimum_required_improvement: float | None = None
    parent_selection_reasons: dict[str, str] = field(default_factory=dict)
    parent_eligibility: dict[str, str] = field(default_factory=dict)
    rejected_descendants: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "generation": self.generation,
            "input_population_ids": list(self.input_population_ids),
            "evaluated_candidate_ids": list(self.evaluated_candidate_ids),
            "selected_parent_ids": list(self.selected_parent_ids),
            "mutation_candidate_ids": list(self.mutation_candidate_ids),
            "crossover_candidate_ids": list(self.crossover_candidate_ids),
            "completed_full_wfo_candidate_ids": list(self.completed_full_wfo_candidate_ids),
            "score_qualified_candidate_ids": list(self.score_qualified_candidate_ids),
            "best_generation_score": self.best_generation_score,
            "best_score_so_far": self.best_score_so_far,
            "generated_count": self.generated_count,
            "evaluated_count": self.evaluated_count,
            "completed_full_wfo_count": self.completed_full_wfo_count,
            "stagnation_count": self.stagnation_count,
            "stop_reason": self.stop_reason,
            "previous_best_score": self.previous_best_score,
            "generation_best_score": self.generation_best_score,
            "improvement": self.improvement,
            "minimum_required_improvement": self.minimum_required_improvement,
            "parent_selection_reasons": dict(self.parent_selection_reasons),
            "parent_eligibility": dict(self.parent_eligibility),
            "rejected_descendants": [dict(r) for r in self.rejected_descendants],
        }


@dataclass
class CandidateStatusEvent:
    """Deterministic sequence-ordered candidate status transition."""

    sequence: int
    candidate_id: str
    family_id: str
    generation: int
    prior_status: str
    new_status: str
    reason: str
    artifact_refs: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "generation": self.generation,
            "prior_status": self.prior_status,
            "new_status": self.new_status,
            "reason": self.reason,
            "artifact_refs": dict(self.artifact_refs),
        }


@dataclass
class CandidateStressSummary:
    """Deterministic candidate-level Stress summary (Phase 3B.1)."""

    candidate_id: str
    family_id: str
    generation: int
    lineage: dict[str, Any]
    score_qualified_proof: dict[str, Any]
    backend_kind: str
    research_eligible: bool
    scenario_names: list[str] = field(default_factory=list)
    scenario_statuses: dict[str, str] = field(default_factory=dict)
    scenario_artifact_refs: dict[str, Any] = field(default_factory=dict)
    total_scenarios_configured: int = 0
    total_scenarios_executed: int = 0
    passed_count: int = 0
    failed_count: int = 0
    not_applicable_count: int = 0
    unsupported_count: int = 0
    denominator: int = 0
    pass_rate: float = 0.0
    required_pass_rate: float = 0.5
    worst_scenario: str | None = None
    worst_expectancy: float | None = None
    worst_pf: float | None = None
    worst_drawdown: float | None = None
    signal_source_proof: dict[str, Any] = field(default_factory=dict)
    completed_fold_proof: dict[str, Any] = field(default_factory=dict)
    final_decision: str = STRESS_NOT_ENTERED
    final_reason: str = ""
    stress_budget_consumed: int = 0
    integrity_failures: list[dict[str, Any]] = field(default_factory=list)
    baseline_artifact_source: str | None = None
    baseline_candidate_id: str | None = None
    hidden_baseline_rerun: bool = False
    scenario_backend_calls: int = 0
    stress_counter_delta: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "generation": self.generation,
            "lineage": dict(self.lineage),
            "score_qualified_proof": dict(self.score_qualified_proof),
            "backend_kind": self.backend_kind,
            "research_eligible": self.research_eligible,
            "scenario_names": list(self.scenario_names),
            "scenario_statuses": dict(self.scenario_statuses),
            "scenario_artifact_refs": dict(self.scenario_artifact_refs),
            "total_scenarios_configured": self.total_scenarios_configured,
            "total_scenarios_executed": self.total_scenarios_executed,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "not_applicable_count": self.not_applicable_count,
            "unsupported_count": self.unsupported_count,
            "denominator": self.denominator,
            "pass_rate": self.pass_rate,
            "required_pass_rate": self.required_pass_rate,
            "worst_scenario": self.worst_scenario,
            "worst_expectancy": self.worst_expectancy,
            "worst_pf": self.worst_pf,
            "worst_drawdown": self.worst_drawdown,
            "signal_source_proof": dict(self.signal_source_proof),
            "completed_fold_proof": dict(self.completed_fold_proof),
            "final_decision": self.final_decision,
            "final_reason": self.final_reason,
            "stress_budget_consumed": self.stress_budget_consumed,
            "integrity_failures": [dict(x) for x in self.integrity_failures],
            "baseline_artifact_source": self.baseline_artifact_source,
            "baseline_candidate_id": self.baseline_candidate_id,
            "hidden_baseline_rerun": self.hidden_baseline_rerun,
            "scenario_backend_calls": self.scenario_backend_calls,
            "stress_counter_delta": self.stress_counter_delta,
        }


@dataclass
class CampaignStressAccounting:
    """Campaign-level Stress budget and scenario counters."""

    max_stress_evaluations: int = 0
    stress_evaluations_consumed: int = 0
    candidates_score_qualified: int = 0
    candidates_stress_entered: int = 0
    candidates_stress_passed: int = 0
    candidates_stress_failed: int = 0
    candidates_stress_not_entered: int = 0
    scenarios_configured: list[str] = field(default_factory=list)
    scenarios_executed: int = 0
    scenarios_passed: int = 0
    scenarios_failed: int = 0
    scenarios_not_applicable: int = 0
    scenarios_unsupported: int = 0
    stop_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_stress_evaluations": self.max_stress_evaluations,
            "stress_evaluations_consumed": self.stress_evaluations_consumed,
            "candidates_score_qualified": self.candidates_score_qualified,
            "candidates_stress_entered": self.candidates_stress_entered,
            "candidates_stress_passed": self.candidates_stress_passed,
            "candidates_stress_failed": self.candidates_stress_failed,
            "candidates_stress_not_entered": self.candidates_stress_not_entered,
            "scenarios_configured": list(self.scenarios_configured),
            "scenarios_executed": self.scenarios_executed,
            "scenarios_passed": self.scenarios_passed,
            "scenarios_failed": self.scenarios_failed,
            "scenarios_not_applicable": self.scenarios_not_applicable,
            "scenarios_unsupported": self.scenarios_unsupported,
            "stop_reason": self.stop_reason,
        }


@dataclass
class CandidateRobustnessSummary:
    """Deterministic candidate-level Robustness summary (Phase 3B.2)."""

    candidate_id: str
    family_id: str
    generation: int
    lineage: dict[str, Any]
    stress_passed_proof: dict[str, Any]
    backend_kind: str
    research_eligible: bool
    selected_parameter_names: list[str] = field(default_factory=list)
    parameter_selection_reason: str = ""
    parameter_summaries: list[dict[str, Any]] = field(default_factory=list)
    total_planned_points: int = 0
    real_evaluated_points: int = 0
    not_applicable_points: int = 0
    integrity_failed_points: int = 0
    robustness_backend_calls: int = 0
    robustness_counter_delta: int = 0
    budget_consumed: int = 0
    max_robustness_evaluations: int = 0
    robustness_budget_remaining: int = 0
    accepted_parameter_count: int = 0
    failed_parameter_count: int = 0
    final_decision: str = ROBUSTNESS_NOT_ENTERED
    final_reason: str = ""
    stop_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "family_id": self.family_id,
            "generation": self.generation,
            "lineage": dict(self.lineage),
            "stress_passed_proof": dict(self.stress_passed_proof),
            "backend_kind": self.backend_kind,
            "research_eligible": self.research_eligible,
            "selected_parameter_names": list(self.selected_parameter_names),
            "parameter_selection_reason": self.parameter_selection_reason,
            "parameter_summaries": [dict(p) for p in self.parameter_summaries],
            "total_planned_points": self.total_planned_points,
            "real_evaluated_points": self.real_evaluated_points,
            "not_applicable_points": self.not_applicable_points,
            "integrity_failed_points": self.integrity_failed_points,
            "robustness_backend_calls": self.robustness_backend_calls,
            "robustness_counter_delta": self.robustness_counter_delta,
            "budget_consumed": self.budget_consumed,
            "max_robustness_evaluations": self.max_robustness_evaluations,
            "robustness_budget_remaining": self.robustness_budget_remaining,
            "accepted_parameter_count": self.accepted_parameter_count,
            "failed_parameter_count": self.failed_parameter_count,
            "final_decision": self.final_decision,
            "final_reason": self.final_reason,
            "stop_reason": self.stop_reason,
        }


@dataclass
class CampaignRobustnessAccounting:
    """Campaign-level Robustness budget and evaluation counters."""

    max_robustness_candidates: int = 0
    max_robustness_evaluations: int = 0
    max_parameters_per_candidate: int = 0
    max_points_per_parameter: int = 0
    robustness_evaluations_consumed: int = 0
    candidates_stress_passed: int = 0
    candidates_robustness_entered: int = 0
    candidates_robustness_passed: int = 0
    candidates_robustness_failed: int = 0
    candidates_robustness_not_entered: int = 0
    robustness_backend_calls: int = 0
    robustness_counter_delta: int = 0
    stop_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "max_robustness_candidates": self.max_robustness_candidates,
            "max_robustness_evaluations": self.max_robustness_evaluations,
            "max_parameters_per_candidate": self.max_parameters_per_candidate,
            "max_points_per_parameter": self.max_points_per_parameter,
            "robustness_evaluations_consumed": self.robustness_evaluations_consumed,
            "candidates_stress_passed": self.candidates_stress_passed,
            "candidates_robustness_entered": self.candidates_robustness_entered,
            "candidates_robustness_passed": self.candidates_robustness_passed,
            "candidates_robustness_failed": self.candidates_robustness_failed,
            "candidates_robustness_not_entered": self.candidates_robustness_not_entered,
            "robustness_backend_calls": self.robustness_backend_calls,
            "robustness_counter_delta": self.robustness_counter_delta,
            "stop_reason": self.stop_reason,
        }


@dataclass
class MultiFamilyCampaignResult:
    campaign_id: str
    families: list[FamilySpec]
    family_stats: list[FamilyStats]
    budget_allocation: dict[str, Any]
    discovery_results: list[DiscoveryRunResult]
    aggregated_stop_reason: str
    reproducible_fingerprint: str
    pipeline_level: str = PIPELINE_LEVEL_MULTI_FAMILY_WFO_SCREENING
    post_wfo_pipeline_complete: bool = False
    stress_pipeline_complete: bool = False
    robustness_pipeline_complete: bool = False
    statistics_pipeline_complete: bool = False
    clustering_pipeline_complete: bool = False
    research_shortlist_pipeline_complete: bool = False
    vault_pipeline_complete: bool = False
    paper_pipeline_complete: bool = False
    live_pipeline_complete: bool = False
    score_qualified_meaning: str = SCORE_QUALIFIED_MEANING
    score_qualified_does_not_mean: tuple[str, ...] = SCORE_QUALIFIED_DOES_NOT_MEAN
    stress_passed_does_not_mean: tuple[str, ...] = STRESS_PASSED_DOES_NOT_MEAN
    robustness_passed_does_not_mean: tuple[str, ...] = ROBUSTNESS_PASSED_DOES_NOT_MEAN
    post_wfo_blocked_reasons: list[str] = field(
        default_factory=lambda: list(POST_WFO_BLOCKED_REASONS)
    )
    empty_collections_reasons: dict[str, list[str]] = field(
        default_factory=lambda: {k: list(v) for k, v in EMPTY_COLLECTIONS_REASONS.items()}
    )
    research_shortlist: list[Any] = field(default_factory=list)
    vault_candidates: list[Any] = field(default_factory=list)
    paper_candidates: list[Any] = field(default_factory=list)
    generation_records: list[GenerationRecord] = field(default_factory=list)
    candidate_status_history: list[CandidateStatusEvent] = field(default_factory=list)
    candidate_stress_summaries: list[CandidateStressSummary] = field(default_factory=list)
    stress_accounting: CampaignStressAccounting | None = None
    candidate_robustness_summaries: list[CandidateRobustnessSummary] = field(
        default_factory=list
    )
    robustness_accounting: CampaignRobustnessAccounting | None = None
    candidate_statistics_summaries: list[CandidateStatisticsSummary] = field(
        default_factory=list
    )
    statistics_accounting: CampaignStatisticsAccounting | None = None
    clusters: list[dict[str, Any]] = field(default_factory=list)
    behavioral_signatures: list[dict[str, Any]] = field(default_factory=list)
    clustering_accounting: CampaignClusteringAccounting | None = None
    shortlist_rejects: list[ShortlistRejectRecord] = field(default_factory=list)
    population_stats: dict[str, Any] = field(default_factory=dict)
    research_shortlisted_does_not_mean: tuple[str, ...] = RESEARCH_SHORTLISTED_DOES_NOT_MEAN

    def as_dict(self) -> dict[str, Any]:
        empty_shortlist_reasons = list(
            self.empty_collections_reasons.get("research_shortlist") or []
        )
        if self.research_shortlist_pipeline_complete and not self.research_shortlist:
            if not empty_shortlist_reasons:
                empty_shortlist_reasons = [NOT_RESEARCH_SHORTLISTED]
        clusters_out = list(self.clusters)
        shortlist_out = [
            e.as_dict() if hasattr(e, "as_dict") else e for e in self.research_shortlist
        ]
        return {
            "campaign_id": self.campaign_id,
            "families": [f.as_dict() for f in self.families],
            "family_stats": [s.as_dict() for s in self.family_stats],
            "family_funnel": [s.as_dict() for s in self.family_stats],
            "budget_allocation": dict(self.budget_allocation),
            "discovery_results": [r.as_dict() for r in self.discovery_results],
            "aggregated_stop_reason": self.aggregated_stop_reason,
            "reproducible_fingerprint": self.reproducible_fingerprint,
            "best_candidates_per_family": {
                s.family_id: list(s.best_candidate_ids) for s in self.family_stats
            },
            "pipeline_level": self.pipeline_level,
            "post_wfo_pipeline_complete": self.post_wfo_pipeline_complete,
            "stress_pipeline_complete": self.stress_pipeline_complete,
            "robustness_pipeline_complete": self.robustness_pipeline_complete,
            "statistics_pipeline_complete": self.statistics_pipeline_complete,
            "clustering_pipeline_complete": self.clustering_pipeline_complete,
            "research_shortlist_pipeline_complete": self.research_shortlist_pipeline_complete,
            "vault_pipeline_complete": self.vault_pipeline_complete,
            "paper_pipeline_complete": self.paper_pipeline_complete,
            "live_pipeline_complete": self.live_pipeline_complete,
            "score_qualified_meaning": self.score_qualified_meaning,
            "score_qualified_does_not_mean": list(self.score_qualified_does_not_mean),
            "stress_passed_does_not_mean": list(self.stress_passed_does_not_mean),
            "robustness_passed_does_not_mean": list(self.robustness_passed_does_not_mean),
            "research_shortlisted_does_not_mean": list(self.research_shortlisted_does_not_mean),
            "post_wfo_blocked_reasons": list(self.post_wfo_blocked_reasons),
            "empty_collections_reasons": {
                k: list(v) for k, v in self.empty_collections_reasons.items()
            },
            "research_shortlist": shortlist_out,
            "vault_candidates": list(self.vault_candidates),
            "paper_candidates": list(self.paper_candidates),
            "generation_records": [g.as_dict() for g in self.generation_records],
            "candidate_status_history": [e.as_dict() for e in self.candidate_status_history],
            "candidate_stress_summaries": [s.as_dict() for s in self.candidate_stress_summaries],
            "stress_accounting": (
                self.stress_accounting.as_dict() if self.stress_accounting is not None else None
            ),
            "candidate_robustness_summaries": [
                s.as_dict() for s in self.candidate_robustness_summaries
            ],
            "robustness_accounting": (
                self.robustness_accounting.as_dict()
                if self.robustness_accounting is not None
                else None
            ),
            "candidate_statistics_summaries": [
                s.as_dict() for s in self.candidate_statistics_summaries
            ],
            "statistics_accounting": (
                self.statistics_accounting.as_dict()
                if self.statistics_accounting is not None
                else None
            ),
            "behavioral_signatures": list(self.behavioral_signatures),
            "clustering_accounting": (
                self.clustering_accounting.as_dict()
                if self.clustering_accounting is not None
                else None
            ),
            "shortlist_rejects": [
                r.as_dict() if hasattr(r, "as_dict") else r for r in self.shortlist_rejects
            ],
            "population_stats": dict(self.population_stats),
            # Explicit post-WFO surfaces — Vault/Paper/Live never silent.
            "finalists": [],
            "promoted": [],
            "clusters": clusters_out,
            "research_shortlist_empty_reasons": empty_shortlist_reasons,
        }


def equal_initial_allocation(
    *,
    family_ids: list[str],
    total_budget: int,
    min_per_family: int,
) -> dict[str, int]:
    n = len(family_ids)
    if n == 0:
        return {}
    if total_budget < n * min_per_family:
        raise ValueError(
            f"total_candidate_budget={total_budget} cannot satisfy "
            f"min_candidates_per_family={min_per_family} for {n} families"
        )
    base = total_budget // n
    alloc = {fid: max(min_per_family, base) for fid in family_ids}
    used = sum(alloc.values())
    if used > total_budget:
        overflow = used - total_budget
        for fid in reversed(family_ids):
            if overflow <= 0:
                break
            surplus = alloc[fid] - min_per_family
            take = min(surplus, overflow)
            alloc[fid] -= take
            overflow -= take
        if overflow > 0:
            raise ValueError("cannot allocate without starving a family minimum quota")
    elif used < total_budget:
        rem = total_budget - used
        i = 0
        while rem > 0:
            alloc[family_ids[i % n]] += 1
            rem -= 1
            i += 1
    return alloc


def evolution_budget_split(
    *,
    family_ids: list[str],
    total_budget: int,
    initial_per_family: int,
) -> dict[str, Any]:
    """Split total candidate budget into exact initial population + evolutionary reserve.

    Generation 0 is never expanded merely to exhaust ``total_budget``. Remaining
    slots are reserved for mutation/crossover descendants.
    """
    n = len(family_ids)
    if n == 0:
        return {
            "initial_alloc": {},
            "evolutionary_alloc": {},
            "generated_cap_alloc": {},
            "initial_population_budget": 0,
            "evolutionary_candidate_budget": 0,
            "generated_cap": int(total_budget),
        }
    initial_per = max(1, int(initial_per_family))
    initial_population_budget = n * initial_per
    if total_budget < initial_population_budget:
        raise ValueError(
            f"total_candidate_budget={total_budget} cannot satisfy "
            f"initial_candidates_per_family={initial_per} for {n} families "
            f"(need >={initial_population_budget})"
        )
    evolutionary_candidate_budget = int(total_budget) - initial_population_budget
    initial_alloc = {fid: initial_per for fid in family_ids}
    if evolutionary_candidate_budget <= 0:
        evolutionary_alloc = {fid: 0 for fid in family_ids}
    else:
        evolutionary_alloc = equal_initial_allocation(
            family_ids=family_ids,
            total_budget=evolutionary_candidate_budget,
            min_per_family=0,
        )
    generated_cap_alloc = {
        fid: int(initial_alloc[fid]) + int(evolutionary_alloc[fid]) for fid in family_ids
    }
    return {
        "initial_alloc": initial_alloc,
        "evolutionary_alloc": evolutionary_alloc,
        "generated_cap_alloc": generated_cap_alloc,
        "initial_population_budget": initial_population_budget,
        "evolutionary_candidate_budget": evolutionary_candidate_budget,
        "generated_cap": int(total_budget),
    }


def equal_wfo_allocation(
    *,
    family_ids: list[str],
    max_full_wfo: int,
    min_per_family: int = 0,
) -> dict[str, int]:
    n = len(family_ids)
    if n == 0:
        return {}
    floor = min_per_family if n * min_per_family <= max_full_wfo else 0
    return equal_initial_allocation(
        family_ids=family_ids,
        total_budget=max_full_wfo,
        min_per_family=floor,
    )


def adaptive_reallocate_wfo(
    *,
    family_ids: list[str],
    max_full_wfo: int,
    early_scores: dict[str, float],
    min_quota: int,
    early_counts: dict[str, int] | None = None,
    temperature: float = 1.0,
) -> dict[str, int]:
    """
    Deterministic softmax-weighted allocation of remaining WFO slots.

    Every family keeps ``min_quota``. Remainder favors stronger but uncertain
    families (UCB-style bonus for low early sample counts).
    """
    import math

    n = len(family_ids)
    if n == 0:
        return {}
    reserved = n * min_quota
    if reserved > max_full_wfo:
        return equal_wfo_allocation(family_ids=family_ids, max_full_wfo=max_full_wfo, min_per_family=0)
    alloc = {fid: min_quota for fid in family_ids}
    remainder = max_full_wfo - reserved
    if remainder <= 0:
        return alloc

    counts = early_counts or {fid: min_quota for fid in family_ids}
    finite_scores = [early_scores.get(fid, 0.0) for fid in family_ids]
    # Replace -inf with slightly below min finite score.
    finite_vals = [s for s in finite_scores if math.isfinite(s)]
    floor = (min(finite_vals) - 1.0) if finite_vals else 0.0
    raw: dict[str, float] = {}
    total_n = max(1, sum(max(1, int(counts.get(fid, 1))) for fid in family_ids))
    for fid in family_ids:
        score = early_scores.get(fid, float("-inf"))
        if not math.isfinite(score):
            score = floor
        n_i = max(1, int(counts.get(fid, 1)))
        # UCB exploration bonus — prevents one noisy early candidate from killing a family.
        uncertainty = math.sqrt(2.0 * math.log(total_n + 1.0) / n_i)
        raw[fid] = float(score) + uncertainty

    # Softmax weights (deterministic).
    tmax = max(1e-6, float(temperature))
    peak = max(raw.values())
    exps = {fid: math.exp((raw[fid] - peak) / tmax) for fid in family_ids}
    z = sum(exps.values()) or 1.0
    weights = {fid: exps[fid] / z for fid in family_ids}

    # Largest-remainder method for integer slots.
    exact = {fid: remainder * weights[fid] for fid in family_ids}
    base = {fid: int(math.floor(exact[fid])) for fid in family_ids}
    used = sum(base.values())
    leftover = remainder - used
    order = sorted(
        family_ids,
        key=lambda fid: (exact[fid] - base[fid], weights[fid], fid),
        reverse=True,
    )
    for i in range(leftover):
        base[order[i % n]] += 1
    for fid in family_ids:
        alloc[fid] += base[fid]
    return alloc


def adaptive_allocation_report(
    *,
    family_ids: list[str],
    initial_quota: dict[str, int],
    final_quota: dict[str, int],
    early_scores: dict[str, float],
    early_candidate_ids: dict[str, list[str]],
    early_counts: dict[str, int],
) -> dict[str, dict[str, Any]]:
    import math

    total_n = max(1, sum(max(1, int(early_counts.get(fid, 1))) for fid in family_ids))
    out: dict[str, dict[str, Any]] = {}
    for fid in family_ids:
        score = early_scores.get(fid, float("-inf"))
        n_i = max(1, int(early_counts.get(fid, 1)))
        unc = math.sqrt(2.0 * math.log(total_n + 1.0) / n_i) if math.isfinite(score) else 1.0
        out[fid] = {
            "initial_quota": int(initial_quota.get(fid, 0)),
            "early_candidate_ids": list(early_candidate_ids.get(fid, [])),
            "early_scores": float(score) if math.isfinite(score) else None,
            "uncertainty": float(unc),
            "allocation_weight": None,
            "final_wfo_quota": int(final_quota.get(fid, 0)),
            "allocation_reason": (
                "softmax_ucb_remainder"
                if final_quota.get(fid, 0) > initial_quota.get(fid, 0)
                else "min_quota_floor"
            ),
        }
    # Fill weights consistent with adaptive_reallocate_wfo.
    final = adaptive_reallocate_wfo(
        family_ids=family_ids,
        max_full_wfo=sum(final_quota.values()) or 1,
        early_scores=early_scores,
        min_quota=0,
        early_counts=early_counts,
    )
    total = sum(final.values()) or 1
    for fid in family_ids:
        out[fid]["allocation_weight"] = float(final.get(fid, 0)) / float(total)
    return out


class MultiFamilyCampaign:
    """
    Generate diverse FamilySpecs, produce family-constrained DSL candidates, then
    run Full WFO only up to the allocated per-family quota (never starving minima).
    """

    def __init__(
        self,
        *,
        config: FamilyCampaignConfig,
        registry: ExperimentRegistry,
        backend: Any,
        system_version: str = "0.12.1-phase12.1",
        discovery_run_id: str | None = None,
        progress_hook: Callable[[str, dict[str, Any]], None] | None = None,
        research_eligible: bool = True,
        synthetic_stress_forbidden: bool | None = None,
        stress_backend_factory: Callable[[str], Any] | None = None,
        synthetic_robustness_forbidden: bool | None = None,
        search_program_id: str | None = None,
        search_mode: str = "NEW_SEARCH",
        compatibility_fingerprint: str | None = None,
        fingerprint_components: dict[str, str] | None = None,
        checkpoint_path: Any | None = None,
        evaluation_cache: Any | None = None,
        resume_checkpoint: Any | None = None,
        source_run_id: str | None = None,
        resumed_from_run_id: str | None = None,
        reevaluate_invalidate_cache: bool = False,
    ) -> None:
        self.config = config
        self.registry = registry
        self.backend = backend
        self.system_version = system_version
        self.discovery_run_id = discovery_run_id or ("mfc_" + uuid4().hex[:16])
        self.progress_hook = progress_hook
        self.family_generator = StrategyFamilyGenerator(seed=config.seed)
        self.research_eligible = bool(research_eligible)
        self.synthetic_stress_forbidden = (
            bool(synthetic_stress_forbidden)
            if synthetic_stress_forbidden is not None
            else bool(research_eligible)
        )
        self.synthetic_robustness_forbidden = (
            bool(synthetic_robustness_forbidden)
            if synthetic_robustness_forbidden is not None
            else bool(research_eligible)
        )
        self.stress_backend_factory = stress_backend_factory
        self._status_seq = 0
        self.candidate_status_history: list[CandidateStatusEvent] = []
        self.candidate_stress_summaries: list[CandidateStressSummary] = []
        self.stress_accounting: CampaignStressAccounting | None = None
        self.candidate_robustness_summaries: list[CandidateRobustnessSummary] = []
        self.robustness_accounting: CampaignRobustnessAccounting | None = None
        self.candidate_statistics_summaries: list[CandidateStatisticsSummary] = []
        self.statistics_accounting: CampaignStatisticsAccounting | None = None
        self.clusters: list[dict[str, Any]] = []
        self.behavioral_signatures: list[dict[str, Any]] = []
        self.clustering_accounting: CampaignClusteringAccounting | None = None
        self.shortlist_rejects: list[ShortlistRejectRecord] = []
        self.research_shortlist_entries: list[ResearchShortlistEntry] = []
        self.population_stats: dict[str, Any] = {}
        # Persistent search-program resume state
        from discovery.search_program import SearchMode, new_search_program_id

        self.search_program_id = search_program_id or new_search_program_id()
        self.search_mode = str(search_mode or SearchMode.NEW_SEARCH.value)
        self.compatibility_fingerprint = compatibility_fingerprint or ""
        self.fingerprint_components = dict(fingerprint_components or {})
        self.checkpoint_path = checkpoint_path
        self.evaluation_cache = evaluation_cache
        self.resume_checkpoint = resume_checkpoint
        self.source_run_id = source_run_id
        self.resumed_from_run_id = resumed_from_run_id
        self.reevaluate_invalidate_cache = bool(reevaluate_invalidate_cache)
        self._live_checkpoint = resume_checkpoint
        self._session_generated = 0
        self._session_evaluated = 0
        self._session_full_wfo = 0
        self._cache_hits = 0
        self._cache_misses = 0
        if self.reevaluate_invalidate_cache and self.evaluation_cache is not None:
            self.evaluation_cache.invalidate_all()

    def _emit(self, name: str, payload: dict[str, Any] | None = None) -> None:
        if self.progress_hook is not None:
            self.progress_hook(name, payload or {})

    def _make_evaluator(self, *, spec: FamilySpec, wfo_cap: int, gen_cap: int) -> CandidateEvaluator:
        cfg = self.config
        budget = SearchBudget(
            max_generated_candidates=int(gen_cap),
            max_evaluated_candidates=int(gen_cap),
            max_full_wfo_evaluations=int(max(wfo_cap, 1)),
            max_stress_evaluations=max(1, min(4, max(wfo_cap, 1))),
            population_size=min(cfg.population_size, max(1, gen_cap)),
            elite_count=1,
            stagnation_limit=int(cfg.stagnation_generations),
            stagnation_generations=int(cfg.stagnation_generations),
            minimum_generations_before_stagnation=int(cfg.stagnation_generations),
            max_runtime_seconds=float(cfg.max_runtime_seconds),
            max_candidates_per_family=int(gen_cap),
            max_candidates_per_complexity_tier=int(gen_cap),
            max_candidates_per_feature_family=int(gen_cap),
            min_oos_trades=int(cfg.min_oos_trades),
            min_oos_trades_per_fold=int(cfg.min_oos_trades_per_fold),
            max_oos_drawdown=float(cfg.max_oos_drawdown),
        )
        counters = BudgetCounters()
        fitness = RobustFitness(
            min_total_oos_trades=int(cfg.min_oos_trades),
            min_oos_trades_per_fold=int(cfg.min_oos_trades_per_fold),
            max_oos_drawdown=float(cfg.max_oos_drawdown),
        )
        ev = CandidateEvaluator(
            registry=self.registry,
            budget=budget,
            counters=counters,
            fitness_model=fitness,
            backend=self.backend,
            system_version=self.system_version,
            discovery_run_id=self.discovery_run_id,
            grammar=spec.to_grammar(),
            progress_hook=self.progress_hook,
        )
        return ev

    def _stats_from_records(
        self,
        *,
        spec: FamilySpec,
        records: list[Any],
        generated: int,
        allocation_generated: int,
        allocation_wfo: int,
        full_wfo: int,
    ) -> FamilyStats:
        expectancies: list[float] = []
        pfs: list[float] = []
        fitnesses: list[tuple[float, str]] = []
        rejections: dict[str, int] = {}
        score_qualified = 0
        stress_passed = 0
        evaluated = 0
        parent_status: dict[str, int] = {
            STRUCTURAL_PARENT_ELIGIBLE: 0,
            SCORE_QUALIFIED: 0,
            NOT_PARENT_ELIGIBLE: 0,
        }
        for rec in records:
            evaluated += 1
            if rec.rejection_reason:
                rejections[rec.rejection_reason] = rejections.get(rec.rejection_reason, 0) + 1
            status = NOT_PARENT_ELIGIBLE
            is_score_qualified = False
            if rec.outcome in _NOT_PARENT_OUTCOMES:
                status = NOT_PARENT_ELIGIBLE
            elif rec.outcome is EvalOutcome.REGISTERED and rec.fitness is not None and not rec.fitness.rejected:
                is_score_qualified = True
                score_qualified += 1
                status = STRUCTURAL_PARENT_ELIGIBLE
                fitnesses.append((float(rec.fitness.fitness), rec.candidate_id))
                comps = rec.fitness.components or {}
                if "pos_median_oos_expectancy" in comps:
                    expectancies.append(float(comps["pos_median_oos_expectancy"]))
                if "pos_oos_profit_factor" in comps:
                    pfs.append(float(comps["pos_oos_profit_factor"]))
            elif rec.fitness is not None:
                # Structurally evaluated (may be economically rejected) — parent-eligible
                # without implying Score Qualified.
                status = STRUCTURAL_PARENT_ELIGIBLE
                if not rec.fitness.rejected:
                    fitnesses.append((float(rec.fitness.fitness), rec.candidate_id))
                else:
                    fitnesses.append((float(rec.fitness.fitness), rec.candidate_id))
            elif rec.outcome is EvalOutcome.REGISTERED:
                status = STRUCTURAL_PARENT_ELIGIBLE
            elif rec.rejection_reason in {
                "NEGATIVE_EXPECTANCY",
                "PF_BELOW_ONE",
                "INSUFFICIENT_OOS_TRADES",
                "MAX_DRAWDOWN_EXCEEDED",
            }:
                status = STRUCTURAL_PARENT_ELIGIBLE
            parent_status[status] = parent_status.get(status, 0) + 1
            if is_score_qualified:
                parent_status[SCORE_QUALIFIED] = parent_status.get(SCORE_QUALIFIED, 0) + 1
            if rec.stress_results:
                executed = [
                    v
                    for v in rec.stress_results.values()
                    if isinstance(v, dict) and v.get("status", "executed") == "executed"
                ]
                if executed and all(v.get("passed") is True for v in executed):
                    stress_passed += 1
        fitnesses.sort(key=lambda x: x[0], reverse=True)
        return FamilyStats(
            family_id=spec.family_id,
            hypothesis=spec.hypothesis,
            canonical_hash=spec.canonical_hash(),
            grammar_fingerprint=spec.effective_grammar_fingerprint(),
            allocation_generated=allocation_generated,
            allocation_wfo=allocation_wfo,
            generated=generated,
            evaluated=evaluated,
            full_wfo=full_wfo,
            score_qualified=score_qualified,
            stress_passed=stress_passed,
            best_fitness=fitnesses[0][0] if fitnesses else None,
            median_oos_expectancy=float(median(expectancies)) if expectancies else None,
            median_pf=float(median(pfs)) if pfs else None,
            rejection_reasons=rejections,
            best_candidate_ids=[cid for _, cid in fitnesses[:3]],
            constraints={
                "allowed_features": list(spec.allowed_features),
                "allowed_operators": list(spec.allowed_operators),
                "entry_patterns": list(spec.entry_patterns),
                "exit_patterns": list(spec.exit_patterns),
                "regime_constraints": list(spec.regime_constraints),
                "parameter_ranges": {
                    k: [float(v[0]), float(v[1])] for k, v in spec.parameter_ranges.items()
                },
                "complexity_limits": dict(spec.complexity_limits),
            },
            parent_status_counts=parent_status,
        )

    @staticmethod
    def classify_parent_eligibility(
        rec: EvaluationRecord,
    ) -> tuple[str, bool, str]:
        """Return (parent_status, is_score_qualified, reason).

        STRUCTURAL_PARENT_ELIGIBLE never implies Score Qualified / Stress /
        finalist / promoted / shortlist / Vault / Paper.
        """
        if rec.outcome in _NOT_PARENT_OUTCOMES:
            return (
                NOT_PARENT_ELIGIBLE,
                False,
                f"not_parent:{rec.outcome.value}:{rec.rejection_reason or 'n/a'}",
            )
        is_sq = (
            rec.outcome is EvalOutcome.REGISTERED
            and rec.fitness is not None
            and not rec.fitness.rejected
        )
        if rec.fitness is not None or rec.outcome is EvalOutcome.REGISTERED:
            reason = (
                "structural_parent_after_wfo_score_qualified"
                if is_sq
                else "structural_parent_after_wfo_not_score_qualified"
            )
            return STRUCTURAL_PARENT_ELIGIBLE, is_sq, reason
        if rec.rejection_reason in {
            "NEGATIVE_EXPECTANCY",
            "PF_BELOW_ONE",
            "INSUFFICIENT_OOS_TRADES",
            "MAX_DRAWDOWN_EXCEEDED",
        }:
            return (
                STRUCTURAL_PARENT_ELIGIBLE,
                False,
                f"structural_parent_economic_reject:{rec.rejection_reason}",
            )
        return (
            NOT_PARENT_ELIGIBLE,
            False,
            f"not_parent:unclassified:{rec.outcome.value}:{rec.rejection_reason or 'n/a'}",
        )

    @staticmethod
    def full_wfo_proof(rec: EvaluationRecord) -> dict[str, Any]:
        train = dict(rec.train_metrics or {})
        completed = int(train.get("wfo_completed_folds", len(rec.oos_folds) if rec.oos_folds else 0))
        signal_source = str(train.get("signal_source") or "signal_source_missing")
        is_full = bool(train.get("is_full_event_wfo", False))
        return {
            "signal_source": signal_source,
            "is_full_event_wfo": is_full,
            "completed_fold_count": completed,
            "outcome": rec.outcome.value,
            "full_wfo_ok": (
                signal_source == _REQUIRED_STRESS_SIGNAL_SOURCE
                and is_full
                and completed > 0
            ),
        }

    @classmethod
    def stress_entry_eligibility(
        cls, rec: EvaluationRecord
    ) -> tuple[bool, str, dict[str, Any]]:
        """Return (may_enter_stress, reason, score_qualified_proof).

        Only Score Qualified candidates with proven Full WFO may enter Stress.
        """
        proof = cls.full_wfo_proof(rec)
        parent_status, is_sq, parent_reason = cls.classify_parent_eligibility(rec)
        proof["parent_status"] = parent_status
        proof["parent_reason"] = parent_reason
        proof["is_score_qualified"] = is_sq
        proof["fitness_rejected"] = bool(rec.fitness.rejected) if rec.fitness is not None else None
        proof["fitness_exists"] = rec.fitness is not None
        proof["rejection_reason"] = rec.rejection_reason

        if rec.outcome in _NOT_PARENT_OUTCOMES:
            return False, f"{STRESS_NOT_ENTERED}:{rec.outcome.value}", proof
        if not proof["full_wfo_ok"]:
            if proof["signal_source"] != _REQUIRED_STRESS_SIGNAL_SOURCE:
                return False, f"{STRESS_NOT_ENTERED}:{STRESS_SIGNAL_SOURCE_INVALID}", proof
            return False, f"{STRESS_NOT_ENTERED}:{STRESS_WFO_INCOMPLETE}", proof
        if rec.fitness is None:
            return False, f"{STRESS_NOT_ENTERED}:fitness_missing", proof
        if rec.fitness.rejected:
            reason = rec.rejection_reason or rec.fitness.rejection_reason or "fitness_rejected"
            return False, f"{STRESS_NOT_ENTERED}:{reason}", proof
        if not is_sq:
            return False, f"{STRESS_NOT_ENTERED}:not_score_qualified", proof
        return True, SCORE_QUALIFIED, proof

    def _append_status(
        self,
        *,
        candidate_id: str,
        family_id: str,
        generation: int,
        prior_status: str,
        new_status: str,
        reason: str,
        artifact_refs: dict[str, Any] | None = None,
    ) -> CandidateStatusEvent:
        self._status_seq += 1
        event = CandidateStatusEvent(
            sequence=self._status_seq,
            candidate_id=candidate_id,
            family_id=family_id,
            generation=generation,
            prior_status=prior_status,
            new_status=new_status,
            reason=reason,
            artifact_refs=dict(artifact_refs or {}),
        )
        self.candidate_status_history.append(event)
        return event

    def _resolve_stress_tester(
        self, *, budget: SearchBudget, counters: BudgetCounters
    ) -> tuple[StressTester | None, str | None, str]:
        """Return (tester, bind_error, backend_kind_label)."""
        forbid = bool(self.synthetic_stress_forbidden or self.research_eligible)
        fitness = RobustFitness(
            min_total_oos_trades=int(self.config.min_oos_trades),
            min_oos_trades_per_fold=int(self.config.min_oos_trades_per_fold),
            max_oos_drawdown=float(self.config.max_oos_drawdown),
        )
        if self.stress_backend_factory is not None:
            factory = self.stress_backend_factory
            kind = str(getattr(factory, "backend_kind", "injected_stress_factory"))
            tester = StressTester(
                budget=budget,
                counters=counters,
                fitness_model=fitness,
                backend_factory=factory,
                research_eligible=bool(self.research_eligible),
                synthetic_stress_forbidden=forbid,
            )
            return tester, None, kind

        if forbid and isinstance(self.backend, SyntheticOOSBackend):
            return None, SYNTHETIC_STRESS_FORBIDDEN, "synthetic_oos_probe"

        from discovery.event_wfo_backend import EventDrivenDiscoveryBackend

        if forbid and not isinstance(self.backend, EventDrivenDiscoveryBackend):
            return None, REAL_STRESS_BACKEND_REQUIRED, type(self.backend).__name__

        factory = make_stress_backend_factory(
            self.backend,
            research_eligible=bool(self.research_eligible),
            synthetic_stress_forbidden=forbid,
        )
        kind = STRESS_BACKEND_KIND
        tester = StressTester(
            budget=budget,
            counters=counters,
            fitness_model=fitness,
            backend_factory=factory,
            research_eligible=bool(self.research_eligible),
            synthetic_stress_forbidden=forbid,
        )
        return tester, None, kind

    def _decide_stress_outcome(
        self,
        results: list[StressResult],
        *,
        required_pass_rate: float,
        fail_closed_unsupported: bool,
    ) -> tuple[str, str, dict[str, Any]]:
        executed = [r for r in results if r.status == "executed"]
        not_applicable = [r for r in results if r.status == "not_applicable"]
        unsupported = [r for r in results if r.status == "unsupported"]
        baseline_unavail = [r for r in results if r.status == "baseline_unavailable"]
        passed = [r for r in executed if r.passed]
        failed = [r for r in executed if not r.passed]
        denom = len(executed)
        pass_rate = (len(passed) / denom) if denom > 0 else 0.0

        integrity_failures: list[dict[str, Any]] = []
        integrity_reason: str | None = None
        for r in executed:
            reasons: list[str] = []
            if r.signal_source != _REQUIRED_STRESS_SIGNAL_SOURCE:
                reasons.append(STRESS_SIGNAL_SOURCE_INVALID)
            if r.backend_kind != _REQUIRED_STRESS_BACKEND_KIND:
                reasons.append(STRESS_BACKEND_KIND_INVALID)
            if int(r.completed_fold_count) <= 0:
                reasons.append(STRESS_WFO_INCOMPLETE)
            if not r.integrity_ok and not reasons:
                reasons.append(STRESS_INTEGRITY_FAILED)
            if reasons or not r.integrity_ok:
                primary = reasons[0] if reasons else STRESS_INTEGRITY_FAILED
                if integrity_reason is None:
                    integrity_reason = primary
                integrity_failures.append(
                    {
                        "scenario": r.scenario,
                        "integrity_ok": bool(r.integrity_ok),
                        "signal_source": r.signal_source,
                        "backend_kind": r.backend_kind,
                        "completed_fold_count": int(r.completed_fold_count),
                        "failure_reason": primary,
                    }
                )

        stats = {
            "configured": [r.scenario for r in results],
            "executed": [r.scenario for r in executed],
            "passed": [r.scenario for r in passed],
            "failed": [r.scenario for r in failed],
            "not_applicable": [r.scenario for r in not_applicable],
            "unsupported": [r.scenario for r in unsupported],
            "baseline_unavailable": [r.scenario for r in baseline_unavail],
            "denominator": denom,
            "pass_rate": pass_rate,
            "required_pass_rate": required_pass_rate,
            "integrity_failures": integrity_failures,
        }
        if fail_closed_unsupported and unsupported:
            return STRESS_FAILED, STRESS_UNSUPPORTED_REQUIRED, stats
        if fail_closed_unsupported and baseline_unavail:
            return STRESS_FAILED, STRESS_BASELINE_UNAVAILABLE_REQUIRED, stats
        if denom <= 0:
            return STRESS_FAILED, STRESS_NO_EXECUTED_SCENARIOS, stats
        # Integrity failures are hard research failures — pass-rate cannot override.
        if integrity_failures:
            return STRESS_FAILED, integrity_reason or STRESS_INTEGRITY_FAILED, stats
        if pass_rate < float(required_pass_rate):
            return STRESS_FAILED, STRESS_PASS_RATE_LOW, stats
        return STRESS_PASSED, "stress_pass_rate_ok", stats

    def _build_stress_summary(
        self,
        *,
        cand: StrategyCandidate,
        family_id: str,
        rec: EvaluationRecord,
        proof: dict[str, Any],
        results: list[StressResult],
        backend_kind: str,
        decision: str,
        reason: str,
        stats: dict[str, Any],
        required_pass_rate: float,
        budget_before: int,
        budget_after: int,
        accounting_extra: dict[str, Any] | None = None,
    ) -> CandidateStressSummary:
        executed = [r for r in results if r.status == "executed"]
        worst_scenario = None
        worst_exp = None
        worst_pf = None
        worst_dd = None
        if executed:
            worst = min(executed, key=lambda r: (r.median_expectancy, r.fitness))
            worst_scenario = worst.scenario
            worst_exp = float(worst.median_expectancy)
            worst_dd = float(worst.max_drawdown)
            pfs: list[float] = []
            for r in executed:
                for fm in r.fold_metrics:
                    if isinstance(fm, dict) and "profit_factor" in fm:
                        pfs.append(float(fm["profit_factor"]))
            worst_pf = float(min(pfs)) if pfs else None

        signal_proof = {
            r.scenario: r.signal_source for r in results if r.status == "executed"
        }
        fold_proof = {
            r.scenario: int(r.completed_fold_count)
            for r in results
            if r.status == "executed"
        }
        acct = dict(accounting_extra or {})
        return CandidateStressSummary(
            candidate_id=cand.candidate_id,
            family_id=family_id,
            generation=int(cand.generation),
            lineage={
                "lineage_id": cand.lineage_id,
                "parent_ids": list(cand.parent_ids),
                "creation_method": (
                    cand.creation_method.value
                    if hasattr(cand.creation_method, "value")
                    else str(cand.creation_method)
                ),
            },
            score_qualified_proof=dict(proof),
            backend_kind=backend_kind,
            research_eligible=bool(self.research_eligible),
            scenario_names=[r.scenario for r in results],
            scenario_statuses={r.scenario: r.status for r in results},
            scenario_artifact_refs={
                r.scenario: {
                    "signal_source": r.signal_source,
                    "backend_kind": r.backend_kind,
                    "completed_fold_count": r.completed_fold_count,
                    "passed": r.passed,
                    "failure_reason": r.failure_reason,
                    "integrity_ok": r.integrity_ok,
                }
                for r in results
            },
            total_scenarios_configured=len(results),
            total_scenarios_executed=len(executed),
            passed_count=len(stats.get("passed") or []),
            failed_count=len(stats.get("failed") or []),
            not_applicable_count=len(stats.get("not_applicable") or []),
            unsupported_count=len(stats.get("unsupported") or []),
            denominator=int(stats.get("denominator") or 0),
            pass_rate=float(stats.get("pass_rate") or 0.0),
            required_pass_rate=float(required_pass_rate),
            worst_scenario=worst_scenario,
            worst_expectancy=worst_exp,
            worst_pf=worst_pf,
            worst_drawdown=worst_dd,
            signal_source_proof=signal_proof,
            completed_fold_proof=fold_proof,
            final_decision=decision,
            final_reason=reason,
            stress_budget_consumed=max(0, int(budget_after) - int(budget_before)),
            integrity_failures=list(stats.get("integrity_failures") or []),
            baseline_artifact_source=acct.get("baseline_artifact_source"),
            baseline_candidate_id=acct.get("baseline_candidate_id"),
            hidden_baseline_rerun=bool(acct.get("hidden_baseline_rerun", False)),
            scenario_backend_calls=int(acct.get("scenario_backend_calls") or 0),
            stress_counter_delta=int(acct.get("stress_counter_delta") or 0),
        )

    def _run_stress_phase(
        self,
        *,
        families: list[FamilySpec],
        records_by_family: dict[str, list[EvaluationRecord]],
        cand_by_id: dict[str, StrategyCandidate],
        t0: float,
        only_candidate_ids: set[str] | None = None,
        skip_candidate_ids: set[str] | None = None,
    ) -> CampaignStressAccounting:
        """Connect Score Qualified candidates to the real Stress pipeline."""
        cfg = self.config
        scenarios = tuple(
            cfg.stress_scenarios
            if cfg.stress_scenarios is not None
            else DEFAULT_MULTIFAMILY_STRESS_SCENARIOS
        )
        per_cand_cap = max(1, int(cfg.max_stress_scenarios_per_candidate))
        chosen = scenarios[:per_cand_cap]
        accounting = CampaignStressAccounting(
            max_stress_evaluations=int(cfg.max_stress_evaluations),
            scenarios_configured=list(chosen),
        )
        self.stress_accounting = accounting
        budget = SearchBudget(
            max_generated_candidates=max(1, cfg.total_candidate_budget),
            max_evaluated_candidates=max(1, cfg.total_candidate_budget),
            max_full_wfo_evaluations=max(1, cfg.max_full_wfo),
            max_stress_evaluations=int(cfg.max_stress_evaluations),
            max_runtime_seconds=float(cfg.max_runtime_seconds),
            min_oos_trades=int(cfg.min_oos_trades),
            min_oos_trades_per_fold=int(cfg.min_oos_trades_per_fold),
            max_oos_drawdown=float(cfg.max_oos_drawdown),
        )
        counters = BudgetCounters()
        tester, bind_error, backend_kind = self._resolve_stress_tester(
            budget=budget, counters=counters
        )

        already_done = set(skip_candidate_ids or set())
        already_done.update(
            s.candidate_id
            for s in self.candidate_stress_summaries
            if s.final_decision in {STRESS_PASSED, STRESS_FAILED}
        )

        # Deterministic order: family_id then candidate_id.
        ordered: list[tuple[str, EvaluationRecord, StrategyCandidate]] = []
        for spec in sorted(families, key=lambda s: s.family_id):
            for rec in records_by_family.get(spec.family_id, []):
                cand = cand_by_id.get(rec.candidate_id)
                if cand is None:
                    continue
                if rec.candidate_id in already_done:
                    continue
                if only_candidate_ids is not None and rec.candidate_id not in only_candidate_ids:
                    continue
                ordered.append((spec.family_id, rec, cand))
        ordered.sort(key=lambda t: (t[0], t[1].candidate_id))

        self._emit(
            "MULTI_FAMILY_STRESS_STARTED",
            {
                "scenarios": list(chosen),
                "max_stress_evaluations": accounting.max_stress_evaluations,
                "bind_error": bind_error,
                "pending_count": len(ordered),
                "resume_filtered": only_candidate_ids is not None,
            },
        )

        for family_id, rec, cand in ordered:
            if time.perf_counter() - t0 >= float(cfg.max_runtime_seconds):
                accounting.stop_reason = "max_runtime_seconds"
                break

            eligible, entry_reason, proof = self.stress_entry_eligibility(rec)
            prior = FULL_WFO_COMPLETED if proof.get("full_wfo_ok") else "EVALUATED"
            gen = int(cand.generation)

            if not eligible:
                accounting.candidates_stress_not_entered += 1
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    prior_status=prior,
                    new_status=STRESS_NOT_ENTERED,
                    reason=entry_reason,
                    artifact_refs={"score_qualified_proof": proof},
                )
                continue

            accounting.candidates_score_qualified += 1
            self._append_status(
                candidate_id=cand.candidate_id,
                family_id=family_id,
                generation=gen,
                prior_status=prior,
                new_status=SCORE_QUALIFIED,
                reason=SCORE_QUALIFIED,
                artifact_refs={"score_qualified_proof": proof},
            )

            if bind_error is not None:
                accounting.candidates_stress_entered += 1
                accounting.candidates_stress_failed += 1
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    prior_status=SCORE_QUALIFIED,
                    new_status=STRESS_TESTED,
                    reason=bind_error,
                    artifact_refs={"backend_bind_error": bind_error},
                )
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    prior_status=STRESS_TESTED,
                    new_status=STRESS_FAILED,
                    reason=bind_error,
                    artifact_refs={"backend_bind_error": bind_error},
                )
                summary = CandidateStressSummary(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    lineage={
                        "lineage_id": cand.lineage_id,
                        "parent_ids": list(cand.parent_ids),
                    },
                    score_qualified_proof=dict(proof),
                    backend_kind=backend_kind,
                    research_eligible=bool(self.research_eligible),
                    scenario_names=list(chosen),
                    total_scenarios_configured=len(chosen),
                    required_pass_rate=float(cfg.min_stress_pass_rate),
                    final_decision=STRESS_FAILED,
                    final_reason=bind_error,
                    stress_budget_consumed=0,
                )
                self.candidate_stress_summaries.append(summary)
                continue

            assert tester is not None
            if counters.stress >= budget.max_stress_evaluations:
                accounting.candidates_stress_not_entered += 1
                accounting.stop_reason = STRESS_BUDGET_EXHAUSTED
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    prior_status=SCORE_QUALIFIED,
                    new_status=STRESS_NOT_ENTERED,
                    reason=STRESS_BUDGET_EXHAUSTED,
                    artifact_refs={
                        "stress_consumed": counters.stress,
                        "max_stress_evaluations": budget.max_stress_evaluations,
                    },
                )
                continue

            budget_before = int(counters.stress)
            baseline_arts = None
            if isinstance(rec.meta, dict):
                raw_baseline = rec.meta.get("baseline_wfo_artifacts")
                if isinstance(raw_baseline, dict):
                    baseline_arts = dict(raw_baseline)
            try:
                results = tester.run(
                    cand,
                    base_fitness=float(rec.fitness.fitness) if rec.fitness else 0.0,
                    scenarios=chosen,
                    baseline_artifacts=baseline_arts,
                )
            except RuntimeError as exc:
                msg = str(exc)
                fail_reason = (
                    SYNTHETIC_STRESS_FORBIDDEN
                    if SYNTHETIC_STRESS_FORBIDDEN in msg
                    else msg
                )
                accounting.candidates_stress_entered += 1
                accounting.candidates_stress_failed += 1
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    prior_status=SCORE_QUALIFIED,
                    new_status=STRESS_TESTED,
                    reason=fail_reason,
                    artifact_refs={},
                )
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    prior_status=STRESS_TESTED,
                    new_status=STRESS_FAILED,
                    reason=fail_reason,
                    artifact_refs={},
                )
                self.candidate_stress_summaries.append(
                    CandidateStressSummary(
                        candidate_id=cand.candidate_id,
                        family_id=family_id,
                        generation=gen,
                        lineage={
                            "lineage_id": cand.lineage_id,
                            "parent_ids": list(cand.parent_ids),
                        },
                        score_qualified_proof=dict(proof),
                        backend_kind=backend_kind,
                        research_eligible=bool(self.research_eligible),
                        scenario_names=list(chosen),
                        total_scenarios_configured=len(chosen),
                        required_pass_rate=float(cfg.min_stress_pass_rate),
                        final_decision=STRESS_FAILED,
                        final_reason=fail_reason,
                        stress_budget_consumed=max(0, int(counters.stress) - budget_before),
                        hidden_baseline_rerun=False,
                    )
                )
                continue

            attach_stress(rec, results)
            budget_after = int(counters.stress)
            run_acct = dict(getattr(tester, "last_run_accounting", {}) or {})
            if baseline_arts is not None and not run_acct.get("baseline_artifact_source"):
                run_acct["baseline_artifact_source"] = BASELINE_ARTIFACT_SOURCE_ORIGINAL_WFO
            if baseline_arts is not None and not run_acct.get("baseline_candidate_id"):
                run_acct["baseline_candidate_id"] = str(
                    baseline_arts.get("candidate_id") or cand.candidate_id
                )
            run_acct.setdefault("hidden_baseline_rerun", False)
            decision, reason, stats = self._decide_stress_outcome(
                results,
                required_pass_rate=float(cfg.min_stress_pass_rate),
                fail_closed_unsupported=bool(cfg.fail_closed_unsupported_stress),
            )
            summary = self._build_stress_summary(
                cand=cand,
                family_id=family_id,
                rec=rec,
                proof=proof,
                results=results,
                backend_kind=backend_kind,
                decision=decision,
                reason=reason,
                stats=stats,
                required_pass_rate=float(cfg.min_stress_pass_rate),
                budget_before=budget_before,
                budget_after=budget_after,
                accounting_extra=run_acct,
            )
            self.candidate_stress_summaries.append(summary)
            accounting.candidates_stress_entered += 1
            accounting.scenarios_executed += int(summary.total_scenarios_executed)
            accounting.scenarios_passed += int(summary.passed_count)
            accounting.scenarios_failed += int(summary.failed_count)
            accounting.scenarios_not_applicable += int(summary.not_applicable_count)
            accounting.scenarios_unsupported += int(summary.unsupported_count)
            if decision == STRESS_PASSED:
                accounting.candidates_stress_passed += 1
            else:
                accounting.candidates_stress_failed += 1

            self._append_status(
                candidate_id=cand.candidate_id,
                family_id=family_id,
                generation=gen,
                prior_status=SCORE_QUALIFIED,
                new_status=STRESS_TESTED,
                reason="stress_scenarios_completed",
                artifact_refs={
                    "scenarios": list(summary.scenario_names),
                    "denominator": summary.denominator,
                },
            )
            self._append_status(
                candidate_id=cand.candidate_id,
                family_id=family_id,
                generation=gen,
                prior_status=STRESS_TESTED,
                new_status=decision,
                reason=reason,
                artifact_refs={
                    "pass_rate": summary.pass_rate,
                    "required_pass_rate": summary.required_pass_rate,
                    "summary": summary.as_dict(),
                },
            )

        accounting.stress_evaluations_consumed = int(counters.stress)
        if (
            accounting.stop_reason is None
            and counters.stress >= budget.max_stress_evaluations
            and accounting.candidates_stress_not_entered > 0
        ):
            accounting.stop_reason = STRESS_BUDGET_EXHAUSTED
        self._emit(
            "MULTI_FAMILY_STRESS_COMPLETED",
            accounting.as_dict(),
        )
        return accounting

    @classmethod
    def robustness_entry_eligibility(
        cls,
        cand: StrategyCandidate,
        stress_summary: CandidateStressSummary | None,
        *,
        stress_pipeline_complete: bool,
    ) -> tuple[bool, str, dict[str, Any]]:
        """Return (may_enter_robustness, reason, stress_passed_proof)."""
        proof: dict[str, Any] = {
            "stress_pipeline_complete": bool(stress_pipeline_complete),
            "candidate_id": cand.candidate_id,
        }
        if stress_summary is None:
            proof["final_decision"] = STRESS_NOT_ENTERED
            return False, f"{ROBUSTNESS_NOT_ENTERED}:did_not_enter_stress", proof

        proof.update(
            {
                "final_decision": stress_summary.final_decision,
                "total_scenarios_executed": stress_summary.total_scenarios_executed,
                "denominator": stress_summary.denominator,
                "integrity_failures": list(stress_summary.integrity_failures),
                "scenario_backend_calls": stress_summary.scenario_backend_calls,
                "stress_counter_delta": stress_summary.stress_counter_delta,
                "baseline_candidate_id": stress_summary.baseline_candidate_id,
                "hidden_baseline_rerun": stress_summary.hidden_baseline_rerun,
                "baseline_artifact_source": stress_summary.baseline_artifact_source,
            }
        )
        if stress_summary.final_decision != STRESS_PASSED:
            return (
                False,
                f"{ROBUSTNESS_NOT_ENTERED}:{stress_summary.final_decision}",
                proof,
            )
        if not stress_pipeline_complete:
            return False, f"{ROBUSTNESS_NOT_ENTERED}:stress_pipeline_incomplete", proof
        if int(stress_summary.total_scenarios_executed) <= 0:
            return False, f"{ROBUSTNESS_NOT_ENTERED}:{STRESS_NO_EXECUTED_SCENARIOS}", proof
        if int(stress_summary.denominator) <= 0:
            return False, f"{ROBUSTNESS_NOT_ENTERED}:{STRESS_NO_EXECUTED_SCENARIOS}", proof
        if stress_summary.integrity_failures:
            return False, f"{ROBUSTNESS_NOT_ENTERED}:{STRESS_INTEGRITY_FAILED}", proof
        if int(stress_summary.scenario_backend_calls) != int(stress_summary.stress_counter_delta):
            return False, f"{ROBUSTNESS_NOT_ENTERED}:stress_budget_accounting_mismatch", proof
        baseline_cid = stress_summary.baseline_candidate_id
        if baseline_cid is not None and str(baseline_cid) != str(cand.candidate_id):
            return False, f"{ROBUSTNESS_NOT_ENTERED}:baseline_candidate_mismatch", proof
        if bool(stress_summary.hidden_baseline_rerun):
            return False, f"{ROBUSTNESS_NOT_ENTERED}:hidden_baseline_rerun", proof
        proof["stress_passed"] = True
        return True, STRESS_PASSED, proof

    def _resolve_robustness_probe(
        self, *, counters: BudgetCounters
    ) -> tuple[ParameterRobustness | None, str | None, str]:
        """Return (probe, bind_error, backend_kind_label)."""
        forbid = bool(self.synthetic_robustness_forbidden or self.research_eligible)
        cfg = self.config
        default_steps = (-0.2, -0.1, 0.0, 0.1, 0.2)
        steps = default_steps[: max(1, int(cfg.max_points_per_parameter))]
        fitness = RobustFitness(
            min_total_oos_trades=int(cfg.min_oos_trades),
            min_oos_trades_per_fold=int(cfg.min_oos_trades_per_fold),
            max_oos_drawdown=float(cfg.max_oos_drawdown),
        )
        if forbid and isinstance(self.backend, SyntheticOOSBackend):
            return None, SYNTHETIC_ROBUSTNESS_FORBIDDEN, "synthetic_oos_probe"

        from discovery.event_wfo_backend import EventDrivenDiscoveryBackend

        kind = str(getattr(self.backend, "backend_kind", type(self.backend).__name__))
        if forbid and not isinstance(self.backend, EventDrivenDiscoveryBackend):
            if kind != _REQUIRED_ROBUSTNESS_BACKEND_KIND:
                return None, REAL_ROBUSTNESS_BACKEND_REQUIRED, kind

        probe = ParameterRobustness(
            relative_steps=steps,
            backend=self.backend,
            fitness_model=fitness,
            research_eligible=bool(self.research_eligible),
            synthetic_robustness_forbidden=forbid,
            expected_backend_kind=_REQUIRED_ROBUSTNESS_BACKEND_KIND,
            min_valid_neighborhood_points=int(cfg.min_valid_neighborhood_points),
            allow_one_sided_neighborhood=bool(cfg.allow_one_sided_neighborhood),
        )
        _ = counters
        return probe, None, kind if kind else _REQUIRED_ROBUSTNESS_BACKEND_KIND

    def _select_robustness_parameters(
        self, cand: StrategyCandidate
    ) -> tuple[list[str], str]:
        names = sorted(str(n) for n in cand.parameters.keys())
        cap = max(0, int(self.config.max_parameters_per_candidate))
        selected = names[:cap] if cap else names
        reason = f"deterministic_sorted_name_cap_{cap}"
        return selected, reason

    def _decide_candidate_robustness(
        self,
        *,
        param_summaries: list[ParameterRobustnessSummary],
        selected_names: list[str],
        budget_truncated: bool,
    ) -> tuple[str, str]:
        if not selected_names:
            return ROBUSTNESS_FAILED, ROBUSTNESS_INCOMPLETE
        by_name = {p.parameter: p for p in param_summaries}
        if any(n not in by_name for n in selected_names):
            return ROBUSTNESS_FAILED, ROBUSTNESS_INCOMPLETE
        for name in selected_names:
            p = by_name[name]
            if p.final_parameter_decision == PARAMETER_NOT_BOUND:
                return ROBUSTNESS_FAILED, ROBUSTNESS_PARAMETER_NOT_BOUND
            if p.final_parameter_decision == PARAMETER_INTEGRITY_FAILED:
                return ROBUSTNESS_FAILED, ROBUSTNESS_INTEGRITY_FAILED
            if p.final_parameter_decision == PARAMETER_INSUFFICIENT_NEIGHBORHOOD:
                return ROBUSTNESS_FAILED, ROBUSTNESS_INSUFFICIENT_VALID_NEIGHBORHOOD
            if p.final_parameter_decision == PARAMETER_BUDGET_NOT_REACHED:
                return ROBUSTNESS_FAILED, ROBUSTNESS_BUDGET_EXHAUSTED
            if p.final_parameter_decision == PARAMETER_UNSTABLE:
                return ROBUSTNESS_FAILED, ROBUSTNESS_PARAMETER_UNSTABLE
            if p.final_parameter_decision != PARAMETER_ROBUST:
                return ROBUSTNESS_FAILED, p.final_reason or ROBUSTNESS_INCOMPLETE
        if budget_truncated:
            return ROBUSTNESS_FAILED, ROBUSTNESS_BUDGET_EXHAUSTED
        return ROBUSTNESS_PASSED, "all_required_parameters_robust"

    def _run_robustness_phase(
        self,
        *,
        families: list[FamilySpec],
        records_by_family: dict[str, list[EvaluationRecord]],
        cand_by_id: dict[str, StrategyCandidate],
        t0: float,
        stress_pipeline_complete: bool,
        only_candidate_ids: set[str] | None = None,
        skip_candidate_ids: set[str] | None = None,
    ) -> CampaignRobustnessAccounting:
        """Connect STRESS_PASSED candidates to real Parameter Robustness."""
        cfg = self.config
        accounting = CampaignRobustnessAccounting(
            max_robustness_candidates=int(cfg.max_robustness_candidates),
            max_robustness_evaluations=int(cfg.max_robustness_evaluations),
            max_parameters_per_candidate=int(cfg.max_parameters_per_candidate),
            max_points_per_parameter=int(cfg.max_points_per_parameter),
        )
        self.robustness_accounting = accounting
        counters = BudgetCounters()
        probe, bind_error, backend_kind = self._resolve_robustness_probe(counters=counters)
        stress_by_id = {s.candidate_id: s for s in self.candidate_stress_summaries}
        family_by_id = {f.family_id: f for f in families}
        _ = family_by_id

        already_done = set(skip_candidate_ids or set())
        already_done.update(
            s.candidate_id
            for s in self.candidate_robustness_summaries
            if s.final_decision in {ROBUSTNESS_PASSED, ROBUSTNESS_FAILED}
        )

        ordered: list[tuple[str, EvaluationRecord, StrategyCandidate]] = []
        for spec in sorted(families, key=lambda s: s.family_id):
            for rec in records_by_family.get(spec.family_id, []):
                cand = cand_by_id.get(rec.candidate_id)
                if cand is None:
                    continue
                if rec.candidate_id in already_done:
                    continue
                if only_candidate_ids is not None and rec.candidate_id not in only_candidate_ids:
                    continue
                ordered.append((spec.family_id, rec, cand))
        ordered.sort(key=lambda t: (t[0], t[1].candidate_id))

        self._emit(
            "MULTI_FAMILY_ROBUSTNESS_STARTED",
            {
                "max_robustness_candidates": accounting.max_robustness_candidates,
                "max_robustness_evaluations": accounting.max_robustness_evaluations,
                "bind_error": bind_error,
                "pending_count": len(ordered),
                "resume_filtered": only_candidate_ids is not None,
            },
        )

        for family_id, rec, cand in ordered:
            if time.perf_counter() - t0 >= float(cfg.max_runtime_seconds):
                accounting.stop_reason = "max_runtime_seconds"
                break

            stress_sum = stress_by_id.get(cand.candidate_id)
            eligible, entry_reason, proof = self.robustness_entry_eligibility(
                cand,
                stress_sum,
                stress_pipeline_complete=stress_pipeline_complete,
            )
            gen = int(cand.generation)
            prior = (
                STRESS_PASSED
                if stress_sum is not None and stress_sum.final_decision == STRESS_PASSED
                else (
                    STRESS_FAILED
                    if stress_sum is not None
                    and stress_sum.final_decision == STRESS_FAILED
                    else STRESS_NOT_ENTERED
                )
            )

            if not eligible:
                accounting.candidates_robustness_not_entered += 1
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    prior_status=prior,
                    new_status=ROBUSTNESS_NOT_ENTERED,
                    reason=entry_reason,
                    artifact_refs={"stress_passed_proof": proof},
                )
                continue

            accounting.candidates_stress_passed += 1
            if accounting.candidates_robustness_entered >= accounting.max_robustness_candidates:
                accounting.candidates_robustness_not_entered += 1
                accounting.stop_reason = accounting.stop_reason or ROBUSTNESS_BUDGET_EXHAUSTED
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    prior_status=STRESS_PASSED,
                    new_status=ROBUSTNESS_NOT_ENTERED,
                    reason=f"{ROBUSTNESS_NOT_ENTERED}:{ROBUSTNESS_BUDGET_EXHAUSTED}",
                    artifact_refs={
                        "max_robustness_candidates": accounting.max_robustness_candidates,
                        "entered": accounting.candidates_robustness_entered,
                    },
                )
                continue

            if int(counters.robustness) >= int(cfg.max_robustness_evaluations):
                accounting.candidates_robustness_not_entered += 1
                accounting.stop_reason = ROBUSTNESS_BUDGET_EXHAUSTED
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    prior_status=STRESS_PASSED,
                    new_status=ROBUSTNESS_NOT_ENTERED,
                    reason=f"{ROBUSTNESS_NOT_ENTERED}:{ROBUSTNESS_BUDGET_EXHAUSTED}",
                    artifact_refs={
                        "robustness_consumed": counters.robustness,
                        "max_robustness_evaluations": cfg.max_robustness_evaluations,
                    },
                )
                continue

            selected, sel_reason = self._select_robustness_parameters(cand)
            accounting.candidates_robustness_entered += 1
            self._append_status(
                candidate_id=cand.candidate_id,
                family_id=family_id,
                generation=gen,
                prior_status=STRESS_PASSED,
                new_status=ROBUSTNESS_TESTED,
                reason="robustness_probe_started",
                artifact_refs={
                    "selected_parameters": selected,
                    "parameter_selection_reason": sel_reason,
                },
            )

            if bind_error is not None:
                accounting.candidates_robustness_failed += 1
                summary = CandidateRobustnessSummary(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    lineage={
                        "lineage_id": cand.lineage_id,
                        "parent_ids": list(cand.parent_ids),
                    },
                    stress_passed_proof=dict(proof),
                    backend_kind=backend_kind,
                    research_eligible=bool(self.research_eligible),
                    selected_parameter_names=list(selected),
                    parameter_selection_reason=sel_reason,
                    final_decision=ROBUSTNESS_FAILED,
                    final_reason=bind_error,
                    max_robustness_evaluations=int(cfg.max_robustness_evaluations),
                    robustness_budget_remaining=max(
                        0, int(cfg.max_robustness_evaluations) - int(counters.robustness)
                    ),
                )
                self.candidate_robustness_summaries.append(summary)
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    prior_status=ROBUSTNESS_TESTED,
                    new_status=ROBUSTNESS_FAILED,
                    reason=bind_error,
                    artifact_refs={"backend_bind_error": bind_error},
                )
                continue

            assert probe is not None
            family_spec = family_by_id.get(family_id)
            if family_spec is not None:
                probe.grammar = family_spec.to_grammar()
            budget_before = int(counters.robustness)
            budget_truncated = False
            try:
                results = probe.probe(
                    cand,
                    parameter_names=selected,
                    family_spec=family_spec,
                    counters=counters,
                    max_evaluations=int(cfg.max_robustness_evaluations),
                    require_integrity=True,
                )
            except RuntimeError as exc:
                msg = str(exc)
                fail_reason = (
                    SYNTHETIC_ROBUSTNESS_FORBIDDEN
                    if SYNTHETIC_ROBUSTNESS_FORBIDDEN in msg
                    else (
                        REAL_ROBUSTNESS_BACKEND_REQUIRED
                        if REAL_ROBUSTNESS_BACKEND_REQUIRED in msg
                        else msg
                    )
                )
                accounting.candidates_robustness_failed += 1
                delta = max(0, int(counters.robustness) - budget_before)
                accounting.robustness_backend_calls += delta
                accounting.robustness_counter_delta += delta
                self.candidate_robustness_summaries.append(
                    CandidateRobustnessSummary(
                        candidate_id=cand.candidate_id,
                        family_id=family_id,
                        generation=gen,
                        lineage={
                            "lineage_id": cand.lineage_id,
                            "parent_ids": list(cand.parent_ids),
                        },
                        stress_passed_proof=dict(proof),
                        backend_kind=backend_kind,
                        research_eligible=bool(self.research_eligible),
                        selected_parameter_names=list(selected),
                        parameter_selection_reason=sel_reason,
                        robustness_backend_calls=delta,
                        robustness_counter_delta=delta,
                        budget_consumed=delta,
                        max_robustness_evaluations=int(cfg.max_robustness_evaluations),
                        robustness_budget_remaining=max(
                            0, int(cfg.max_robustness_evaluations) - int(counters.robustness)
                        ),
                        final_decision=ROBUSTNESS_FAILED,
                        final_reason=fail_reason,
                    )
                )
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    prior_status=ROBUSTNESS_TESTED,
                    new_status=ROBUSTNESS_FAILED,
                    reason=fail_reason,
                    artifact_refs={},
                )
                continue

            acct = dict(probe.last_probe_accounting or {})
            delta = int(acct.get("robustness_backend_calls") or 0)
            # Prefer counter delta for this candidate.
            counter_delta = max(0, int(counters.robustness) - budget_before)
            if counter_delta != delta:
                # Integrity: backend calls must match counter increments.
                delta = counter_delta
            accounting.robustness_backend_calls += delta
            accounting.robustness_counter_delta += counter_delta
            if acct.get("stop_reason") == ROBUSTNESS_BUDGET_EXHAUSTED:
                budget_truncated = True
                accounting.stop_reason = ROBUSTNESS_BUDGET_EXHAUSTED

            param_summaries = list(probe.last_parameter_summaries)
            decision, reason = self._decide_candidate_robustness(
                param_summaries=param_summaries,
                selected_names=selected,
                budget_truncated=budget_truncated,
            )
            # All-required must pass; never "best parameter passed".
            accepted = sum(
                1
                for p in param_summaries
                if p.final_parameter_decision == PARAMETER_ROBUST
            )
            failed = len(param_summaries) - accepted
            real_pts = sum(len(p.valid_evaluated_steps) for p in param_summaries)
            na_pts = sum(len(p.not_applicable_steps) for p in param_summaries)
            integ_pts = sum(len(p.integrity_failed_steps) for p in param_summaries)
            planned = len(selected) * len(probe.relative_steps)

            summary = CandidateRobustnessSummary(
                candidate_id=cand.candidate_id,
                family_id=family_id,
                generation=gen,
                lineage={
                    "lineage_id": cand.lineage_id,
                    "parent_ids": list(cand.parent_ids),
                    "creation_method": (
                        cand.creation_method.value
                        if hasattr(cand.creation_method, "value")
                        else str(cand.creation_method)
                    ),
                },
                stress_passed_proof=dict(proof),
                backend_kind=backend_kind,
                research_eligible=bool(self.research_eligible),
                selected_parameter_names=list(selected),
                parameter_selection_reason=sel_reason,
                parameter_summaries=[p.as_dict() for p in param_summaries],
                total_planned_points=planned,
                real_evaluated_points=real_pts,
                not_applicable_points=na_pts,
                integrity_failed_points=integ_pts,
                robustness_backend_calls=delta,
                robustness_counter_delta=counter_delta,
                budget_consumed=counter_delta,
                max_robustness_evaluations=int(cfg.max_robustness_evaluations),
                robustness_budget_remaining=max(
                    0, int(cfg.max_robustness_evaluations) - int(counters.robustness)
                ),
                accepted_parameter_count=accepted,
                failed_parameter_count=failed,
                final_decision=decision,
                final_reason=reason,
                stop_reason=acct.get("stop_reason"),
            )
            self.candidate_robustness_summaries.append(summary)
            if decision == ROBUSTNESS_PASSED:
                accounting.candidates_robustness_passed += 1
            else:
                accounting.candidates_robustness_failed += 1

            # Persist on evaluation record when present.
            if isinstance(rec.meta, dict):
                rec.meta["robustness_summary"] = summary.as_dict()
            rec.robustness_results = {p.parameter: p.as_dict() for p in results}

            self._append_status(
                candidate_id=cand.candidate_id,
                family_id=family_id,
                generation=gen,
                prior_status=ROBUSTNESS_TESTED,
                new_status=decision,
                reason=reason,
                artifact_refs={"summary": summary.as_dict()},
            )

        accounting.robustness_evaluations_consumed = int(counters.robustness)
        # Campaign-level invariant.
        if accounting.robustness_backend_calls != accounting.robustness_counter_delta:
            accounting.stop_reason = accounting.stop_reason or "robustness_budget_invariant_failed"
        self._emit("MULTI_FAMILY_ROBUSTNESS_COMPLETED", accounting.as_dict())
        return accounting

    def _run_statistics_clustering_shortlist_phase(
        self,
        *,
        families: list[FamilySpec],
        records_by_family: dict[str, list[EvaluationRecord]],
        cand_by_id: dict[str, StrategyCandidate],
        robustness_pipeline_complete: bool,
    ) -> ResearchShortlistPhaseResult:
        """Phase 3C: DSR/PBO → behavioral clustering → research shortlist."""
        cfg = self.config
        rs_cfg = ResearchShortlistConfig(
            min_dsr=float(cfg.min_dsr),
            max_pbo=float(cfg.max_pbo),
            pbo_n_splits=int(cfg.pbo_n_splits),
            behavioral_similarity_threshold=float(cfg.behavioral_similarity_threshold),
            min_oos_observations_for_dsr=int(cfg.min_oos_observations_for_dsr),
        )
        stats_acct = CampaignStatisticsAccounting()
        cluster_acct = CampaignClusteringAccounting()
        summaries: list[CandidateStatisticsSummary] = []
        shortlist: list[ResearchShortlistEntry] = []
        rejects: list[ShortlistRejectRecord] = []
        clusters_out: list[dict[str, Any]] = []
        signatures_out: list[dict[str, Any]] = []

        all_records: list[EvaluationRecord] = []
        for spec in families:
            all_records.extend(records_by_family.get(spec.family_id, []))

        robustness_passed_ids = {
            s.candidate_id
            for s in self.candidate_robustness_summaries
            if s.final_decision == ROBUSTNESS_PASSED
        }
        stress_passed_ids = {
            s.candidate_id
            for s in self.candidate_stress_summaries
            if s.final_decision == STRESS_PASSED
        }
        score_qualified_ids: set[str] = set()
        for rec in all_records:
            _status, is_sq, _reason = self.classify_parent_eligibility(rec)
            if is_sq:
                score_qualified_ids.add(rec.candidate_id)

        stats_acct.candidates_robustness_passed = len(robustness_passed_ids)
        population, pop_ids, series_list = build_full_wfo_trial_population(all_records)
        del series_list  # DSR uses per-record extractors; PBO uses time alignment.
        stats_acct.population_total_trials = population.total_trials
        stats_acct.population_scored_trials = population.scored_trials
        # Time/partition-aligned PBO matrix (never trade-ordinal stacking).
        ordered_pop_records: list[EvaluationRecord] = []
        seen_pop: set[str] = set()
        for cid in pop_ids:
            if cid in seen_pop:
                continue
            seen_pop.add(cid)
            for r in all_records:
                if r.candidate_id == cid:
                    ordered_pop_records.append(r)
                    break
        matrix = align_performance_matrix(
            ordered_pop_records, candidate_ids=pop_ids
        )
        if (
            isinstance(matrix, AlignedPerformanceMatrix)
            and matrix.is_usable
            and matrix.matrix is not None
        ):
            stats_acct.pbo_matrix_shape = (
                int(matrix.matrix.shape[0]),
                int(matrix.matrix.shape[1]),
            )
        col_index = {cid: i for i, cid in enumerate(pop_ids)}

        self._emit(
            "MULTI_FAMILY_STATISTICS_STARTED",
            {
                "robustness_passed": len(robustness_passed_ids),
                "population_total_trials": population.total_trials,
                "pbo_matrix_shape": stats_acct.pbo_matrix_shape,
            },
        )

        ordered: list[tuple[str, EvaluationRecord, StrategyCandidate]] = []
        for spec in sorted(families, key=lambda s: s.family_id):
            for rec in records_by_family.get(spec.family_id, []):
                cand = cand_by_id.get(rec.candidate_id)
                if cand is None:
                    continue
                ordered.append((spec.family_id, rec, cand))
        ordered.sort(key=lambda t: (t[0], t[1].candidate_id))

        if not robustness_pipeline_complete:
            stats_acct.stop_reason = STATISTICS_REQUIRES_ROBUSTNESS_PASSED
            stats_acct.candidates_statistics_not_entered = len(ordered)
            for family_id, rec, cand in ordered:
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=int(cand.generation),
                    prior_status=ROBUSTNESS_NOT_ENTERED,
                    new_status=STATISTICS_NOT_ENTERED,
                    reason=f"{STATISTICS_NOT_ENTERED}:{STATISTICS_REQUIRES_ROBUSTNESS_PASSED}",
                    artifact_refs={},
                )
            phase = ResearchShortlistPhaseResult(
                statistics_summaries=summaries,
                statistics_accounting=stats_acct,
                clusters=clusters_out,
                signatures=signatures_out,
                clustering_accounting=cluster_acct,
                research_shortlist=shortlist,
                shortlist_rejects=rejects,
                population_stats=population.as_dict(),
                reproducible_fingerprint_payload={"statistics": "skipped"},
            )
            self.statistics_accounting = stats_acct
            self.clustering_accounting = cluster_acct
            return phase

        if not robustness_passed_ids:
            stats_acct.stop_reason = NO_ROBUSTNESS_PASSED_FOR_STATISTICS

        for family_id, rec, cand in ordered:
            gen = int(cand.generation)
            in_rob = cand.candidate_id in robustness_passed_ids
            in_stress = cand.candidate_id in stress_passed_ids
            prior = (
                ROBUSTNESS_PASSED
                if in_rob
                else (
                    ROBUSTNESS_FAILED
                    if any(
                        s.candidate_id == cand.candidate_id
                        and s.final_decision == ROBUSTNESS_FAILED
                        for s in self.candidate_robustness_summaries
                    )
                    else ROBUSTNESS_NOT_ENTERED
                )
            )
            if not in_rob:
                stats_acct.candidates_statistics_not_entered += 1
                reason = (
                    STATISTICS_REQUIRES_ROBUSTNESS_PASSED
                    if robustness_pipeline_complete
                    else STATISTICS_REQUIRES_ROBUSTNESS_PASSED
                )
                if not in_stress:
                    reason = STATISTICS_REQUIRES_STRESS_PASSED
                proof = self.full_wfo_proof(rec)
                if not proof.get("full_wfo_ok"):
                    reason = STATISTICS_REQUIRES_FULL_WFO
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=gen,
                    prior_status=prior,
                    new_status=STATISTICS_NOT_ENTERED,
                    reason=f"{STATISTICS_NOT_ENTERED}:{reason}",
                    artifact_refs={"full_wfo_proof": proof},
                )
                continue

            stats_acct.candidates_statistics_entered += 1
            self._append_status(
                candidate_id=cand.candidate_id,
                family_id=family_id,
                generation=gen,
                prior_status=ROBUSTNESS_PASSED,
                new_status=STATISTICS_TESTED,
                reason="dsr_pbo_evaluation_started",
                artifact_refs={
                    "population_total_trials": population.total_trials,
                    "artifact_source": "completed_full_wfo",
                },
            )
            dsr, pbo, meta = evaluate_candidate_dsr_pbo(
                rec=rec,
                population=population,
                performance_matrix=matrix,
                trial_column_index=col_index.get(cand.candidate_id),
                config=rs_cfg,
            )
            dsr_decision, dsr_reason = (
                (DSR_PASSED, "dsr_above_minimum")
                if (
                    dsr.status.value == "OK"
                    and dsr.deflated_sharpe is not None
                    and float(dsr.deflated_sharpe) >= float(rs_cfg.min_dsr)
                )
                else (
                    (DSR_INSUFFICIENT_DATA, dsr.reason or DSR_INSUFFICIENT_DATA)
                    if dsr.status.value == "INSUFFICIENT_DATA"
                    else (
                        DSR_FAILED,
                        dsr.reason
                        or (
                            f"DSR_BELOW_MINIMUM:dsr={dsr.deflated_sharpe}"
                            if dsr.deflated_sharpe is not None
                            else dsr.status.value
                        ),
                    )
                )
            )
            if pbo is None:
                pbo_decision, pbo_reason = PBO_INSUFFICIENT_DATA, "PBO_MATRIX_INSUFFICIENT"
                pbo_payload: dict[str, Any] = {}
            elif pbo.status.value == "INSUFFICIENT_DATA":
                pbo_decision, pbo_reason = PBO_INSUFFICIENT_DATA, pbo.reason or PBO_INSUFFICIENT_DATA
                pbo_payload = pbo.as_dict()
            elif (
                pbo.status.value == "OK"
                and pbo.pbo is not None
                and float(pbo.pbo) <= float(rs_cfg.max_pbo)
            ):
                pbo_decision, pbo_reason = PBO_PASSED, "pbo_below_maximum"
                pbo_payload = pbo.as_dict()
            else:
                pbo_decision, pbo_reason = (
                    PBO_FAILED,
                    pbo.reason
                    or (
                        f"PBO_ABOVE_MAXIMUM:pbo={pbo.pbo}"
                        if pbo.pbo is not None
                        else pbo.status.value
                    ),
                )
                pbo_payload = pbo.as_dict()

            if dsr_decision == DSR_PASSED and pbo_decision == PBO_PASSED:
                final_decision = STATISTICALLY_PASSED
                final_reason = "dsr_and_pbo_passed"
                stats_acct.candidates_statistics_passed += 1
            elif dsr_decision == DSR_INSUFFICIENT_DATA or pbo_decision == PBO_INSUFFICIENT_DATA:
                final_decision = STATISTICALLY_REJECTED
                final_reason = (
                    dsr_reason
                    if dsr_decision == DSR_INSUFFICIENT_DATA
                    else pbo_reason
                )
                stats_acct.candidates_statistics_insufficient += 1
            else:
                final_decision = STATISTICALLY_REJECTED
                final_reason = dsr_reason if dsr_decision != DSR_PASSED else pbo_reason
                stats_acct.candidates_statistics_failed += 1

            summary = CandidateStatisticsSummary(
                candidate_id=cand.candidate_id,
                family_id=family_id,
                generation=gen,
                observed_sharpe=dsr.observed_sharpe,
                n_observations=int(dsr.n_observations),
                dsr_status=dsr_decision,
                dsr_value=dsr.deflated_sharpe,
                dsr_reason=dsr_reason,
                pbo_status=pbo_decision,
                pbo_value=None if pbo is None else pbo.pbo,
                pbo_reason=pbo_reason,
                final_decision=final_decision,
                final_reason=final_reason,
                artifact_refs=dict(meta),
                dsr_payload=dsr.as_dict(),
                pbo_payload=pbo_payload,
                trial_population_refs=population.as_dict(),
            )
            summaries.append(summary)
            if isinstance(rec.meta, dict):
                rec.meta["statistics_summary"] = summary.as_dict()

            self._append_status(
                candidate_id=cand.candidate_id,
                family_id=family_id,
                generation=gen,
                prior_status=STATISTICS_TESTED,
                new_status=dsr_decision,
                reason=dsr_reason,
                artifact_refs={"dsr": dsr.as_dict()},
            )
            self._append_status(
                candidate_id=cand.candidate_id,
                family_id=family_id,
                generation=gen,
                prior_status=dsr_decision,
                new_status=pbo_decision,
                reason=pbo_reason,
                artifact_refs={"pbo": pbo_payload},
            )
            self._append_status(
                candidate_id=cand.candidate_id,
                family_id=family_id,
                generation=gen,
                prior_status=pbo_decision,
                new_status=final_decision,
                reason=final_reason,
                artifact_refs={"summary": summary.as_dict()},
            )

        # Behavioral clustering on robustness-passed candidates with Full WFO artifacts.
        self._emit("MULTI_FAMILY_CLUSTERING_STARTED", {"pool": len(robustness_passed_ids)})
        sigs = []
        sig_rec: dict[str, EvaluationRecord] = {}
        sig_cand: dict[str, StrategyCandidate] = {}
        for family_id, rec, cand in ordered:
            if cand.candidate_id not in robustness_passed_ids:
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=int(cand.generation),
                    prior_status=STATISTICS_NOT_ENTERED,
                    new_status=CLUSTERING_NOT_ENTERED,
                    reason=f"{CLUSTERING_NOT_ENTERED}:{STATISTICS_REQUIRES_ROBUSTNESS_PASSED}",
                    artifact_refs={},
                )
                continue
            sig = build_behavioral_signature(
                cand, rec, n_bins=rs_cfg.min_signature_bins
            )
            if sig is None:
                self._append_status(
                    candidate_id=cand.candidate_id,
                    family_id=family_id,
                    generation=int(cand.generation),
                    prior_status=STATISTICALLY_REJECTED,
                    new_status=CLUSTERING_NOT_ENTERED,
                    reason=f"{CLUSTERING_NOT_ENTERED}:{NO_BEHAVIORAL_SIGNATURE}",
                    artifact_refs={},
                )
                continue
            sigs.append(sig)
            sig_rec[cand.candidate_id] = rec
            sig_cand[cand.candidate_id] = cand
            signatures_out.append(sig.as_dict())
            rec.behavioral_cluster = None  # set after clustering

        deduper = BehavioralDeduper(
            similarity_threshold=float(rs_cfg.behavioral_similarity_threshold)
        )
        clusters = deduper.cluster(sigs) if sigs else []
        cluster_acct.candidates_clustered = len(sigs)
        cluster_acct.cluster_count = len(clusters)
        clusters_out = [c.as_dict() for c in clusters]
        cluster_by_member: dict[str, str] = {}
        for cl in clusters:
            for mid in cl.member_ids:
                cluster_by_member[mid] = cl.cluster_id
                if mid in sig_rec:
                    sig_rec[mid].behavioral_cluster = cl.cluster_id
                self._append_status(
                    candidate_id=mid,
                    family_id=next(
                        (
                            fid
                            for fid, r, c in ordered
                            if c.candidate_id == mid
                        ),
                        "",
                    ),
                    generation=int(sig_cand[mid].generation) if mid in sig_cand else 0,
                    prior_status=STATISTICALLY_PASSED
                    if any(
                        s.candidate_id == mid and s.final_decision == STATISTICALLY_PASSED
                        for s in summaries
                    )
                    else STATISTICALLY_REJECTED,
                    new_status=BEHAVIORALLY_CLUSTERED,
                    reason=f"cluster={cl.cluster_id}",
                    artifact_refs={
                        "cluster_id": cl.cluster_id,
                        "representative_id": cl.representative_id,
                        "member_ids": list(cl.member_ids),
                    },
                )

        stats_by_id = {s.candidate_id: s for s in summaries}
        shortlist, rejects = select_best_per_cluster(
            clusters=clusters,
            candidates=sig_cand,
            records=sig_rec,
            stats_by_id=stats_by_id,
            score_qualified_ids=score_qualified_ids,
            stress_passed_ids=stress_passed_ids,
            robustness_passed_ids=robustness_passed_ids,
        )
        cluster_acct.shortlist_count = len(shortlist)
        cluster_acct.reject_count = len(rejects)

        shortlist_ids = {e.candidate_id for e in shortlist}
        for entry in shortlist:
            self._append_status(
                candidate_id=entry.candidate_id,
                family_id=entry.family_id,
                generation=entry.generation,
                prior_status=BEHAVIORALLY_CLUSTERED,
                new_status=RESEARCH_SHORTLISTED,
                reason="cluster_best_passed_hard_gates",
                artifact_refs=entry.as_dict(),
            )
        for rej in rejects:
            if rej.candidate_id in shortlist_ids:
                continue
            if rej.reason == "CLUSTER_NO_GATE_PASSER":
                continue
            fam = rej.family_id or next(
                (fid for fid, r, c in ordered if c.candidate_id == rej.candidate_id),
                "",
            )
            gen = int(sig_cand[rej.candidate_id].generation) if rej.candidate_id in sig_cand else 0
            self._append_status(
                candidate_id=rej.candidate_id,
                family_id=fam,
                generation=gen,
                prior_status=BEHAVIORALLY_CLUSTERED
                if rej.candidate_id in cluster_by_member
                else STATISTICS_NOT_ENTERED,
                new_status=SHORTLIST_REJECTED,
                reason=rej.reason,
                artifact_refs=rej.as_dict(),
            )

        self._emit(
            "MULTI_FAMILY_RESEARCH_SHORTLIST_COMPLETED",
            {
                "clusters": len(clusters_out),
                "shortlist": len(shortlist),
                "rejects": len(rejects),
            },
        )

        pop_stats = {
            "trial_population": population.as_dict(),
            "pbo_matrix_shape": stats_acct.pbo_matrix_shape,
            "min_dsr": rs_cfg.min_dsr,
            "max_pbo": rs_cfg.max_pbo,
        }
        phase = ResearchShortlistPhaseResult(
            statistics_summaries=summaries,
            statistics_accounting=stats_acct,
            clusters=clusters_out,
            signatures=signatures_out,
            clustering_accounting=cluster_acct,
            research_shortlist=shortlist,
            shortlist_rejects=rejects,
            population_stats=pop_stats,
            reproducible_fingerprint_payload={
                "statistics_summaries": [s.as_dict() for s in summaries],
                "clusters": clusters_out,
                "research_shortlist": [e.as_dict() for e in shortlist],
                "shortlist_rejects": [r.as_dict() for r in rejects],
            },
        )
        self.candidate_statistics_summaries = summaries
        self.statistics_accounting = stats_acct
        self.clusters = clusters_out
        self.behavioral_signatures = signatures_out
        self.clustering_accounting = cluster_acct
        self.shortlist_rejects = rejects
        self.research_shortlist_entries = shortlist
        self.population_stats = pop_stats
        return phase

    @staticmethod
    def candidate_within_family_grammar(cand: StrategyCandidate, grammar: Grammar) -> bool:
        allowed_features = {leaf.feature_id for leaf in grammar.feature_leaves}
        for fid in cand.feature_ids:
            if fid not in allowed_features:
                return False
        trees: list[ExprNode | None] = [
            cand.entry_tree,
            cand.exit_tree,
            cand.stop,
            cand.target,
            cand.sizing,
            *cand.regime_gates,
        ]
        for tree in trees:
            if tree is None:
                continue
            for node in tree.walk():
                if node.kind is NodeKind.OPERATOR:
                    try:
                        oid = OperatorId(node.name)
                    except ValueError:
                        return False
                    if not grammar.allows_operator(oid):
                        return False
        return True

    def _record_rejected_descendant(
        self,
        *,
        greg: GenerationRecord,
        child: StrategyCandidate,
        kind: str,
        rejection_reason: str,
        details: dict[str, Any],
        reject_counts: dict[str, int],
    ) -> None:
        # Provenance for diagnostics — strip run-specific keys so campaign
        # fingerprints stay reproducible across discovery_run_id values.
        prov = {
            k: v
            for k, v in dict(child.family_provenance or {}).items()
            if k
            not in {
                "campaign_id",
                "campaign_seed",
                "ordinal",
            }
        }
        enriched = {
            "candidate_id": child.candidate_id,
            "generation": child.generation,
            "parent_ids": list(child.parent_ids),
            "creation_method": (
                child.creation_method.value
                if hasattr(child.creation_method, "value")
                else str(child.creation_method)
            ),
            "expression_tree": child.entry_tree.as_dict() if child.entry_tree is not None else None,
            "family_provenance": prov,
            **dict(details),
        }
        payload = {
            "rejection_reason": rejection_reason,
            "creation_kind": kind,
            **enriched,
        }
        greg.rejected_descendants.append(payload)
        reject_counts[rejection_reason] = reject_counts.get(rejection_reason, 0) + 1
        self._emit("DESCENDANT_REJECTED", payload)

    def _bind_evaluator_features(self, ev: CandidateEvaluator) -> None:
        avail = getattr(self.backend, "available_feature_ids", None)
        if avail is not None:
            ev.available_feature_ids = frozenset(avail)
            ev.dataset_capabilities = tuple(
                getattr(self.backend, "dataset_capabilities", ()) or ()
            )

    def _evaluate_candidate(
        self,
        *,
        ev: CandidateEvaluator,
        cand: StrategyCandidate,
        full_wfo_counts: dict[str, int],
        fid: str,
    ) -> EvaluationRecord:
        from discovery.evaluation_cache import (
            EvaluationCacheEntry,
            build_evaluation_cache_key,
        )
        from discovery.search_program import SearchMode, now_iso

        cache_key = None
        if (
            self.evaluation_cache is not None
            and self.search_mode != SearchMode.REEVALUATE_FROZEN_CANDIDATES.value
            and self.fingerprint_components
        ):
            cache_key = build_evaluation_cache_key(
                candidate_canonical_hash=cand.candidate_id,
                dataset_hash=str(self.fingerprint_components.get("dataset_hash") or ""),
                timeframe=str(self.fingerprint_components.get("timeframe") or ""),
                wfo_config_hash=str(
                    self.fingerprint_components.get("wfo_config_hash") or ""
                ),
                cost_model_version=str(
                    self.fingerprint_components.get("cost_model_version") or ""
                ),
                risk_model_version=str(
                    self.fingerprint_components.get("risk_model_version") or ""
                ),
                execution_engine_code_hash=str(
                    self.fingerprint_components.get("execution_engine_code_hash") or ""
                ),
            )
            hit = self.evaluation_cache.get(cache_key)
            if hit is not None:
                self._cache_hits += 1
                rec = EvaluationRecord.from_dict(hit.evaluation_record)
                # Restore Full-WFO counter accounting without re-running WFO.
                meta = dict(rec.meta or {})
                if meta.get("is_full_event_wfo") or (
                    isinstance(rec.train_metrics, dict)
                    and rec.train_metrics.get("is_full_event_wfo")
                ):
                    full_wfo_counts[fid] = full_wfo_counts.get(fid, 0) + 1
                elif rec.outcome is EvalOutcome.REGISTERED and rec.oos_folds:
                    # Honest Full-WFO completed trials always carry fold evidence.
                    proof_ok = False
                    if isinstance(rec.meta, dict):
                        baseline = rec.meta.get("baseline_wfo_artifacts") or {}
                        proof_ok = bool(baseline.get("is_full_event_wfo"))
                    if proof_ok or (rec.fitness is not None):
                        # Prefer explicit proof; fall back for synthetic tests.
                        if proof_ok or getattr(self.backend, "is_full_event_wfo", False):
                            full_wfo_counts[fid] = full_wfo_counts.get(fid, 0) + 1
                return rec

        prev_full_wfo = ev.counters.full_wfo
        rec = ev.evaluate(cand)
        if ev.counters.full_wfo > prev_full_wfo:
            full_wfo_counts[fid] = full_wfo_counts.get(fid, 0) + 1
        self._cache_misses += 1
        if cache_key is not None and self.evaluation_cache is not None:
            self.evaluation_cache.put(
                EvaluationCacheEntry(
                    cache_key=cache_key,
                    candidate_id=cand.candidate_id,
                    candidate_canonical_hash=cand.candidate_id,
                    evaluation_record=rec.as_dict(),
                    created_at=now_iso(),
                    fingerprint_components=dict(self.fingerprint_components),
                )
            )
        return rec

    def _persist_live_checkpoint(self) -> None:
        if self.checkpoint_path is None or self._live_checkpoint is None:
            return
        from discovery.search_checkpoint import save_checkpoint

        save_checkpoint(self.checkpoint_path, self._live_checkpoint)
        self._emit(
            "SEARCH_CHECKPOINT_SAVED",
            {
                "search_program_id": self.search_program_id,
                "pipeline_phase": self._live_checkpoint.pipeline_phase,
                "campaign_evaluated": self._live_checkpoint.campaign_evaluated,
                "pending_by_gate": self._live_checkpoint.pending_by_gate(),
            },
        )

    def _ensure_live_checkpoint(self, *, families: list[FamilySpec]) -> Any:
        from discovery.search_resume import empty_checkpoint

        if self._live_checkpoint is not None:
            return self._live_checkpoint
        ckpt = empty_checkpoint(
            search_program_id=self.search_program_id,
            compatibility_fingerprint=self.compatibility_fingerprint,
            session_run_id=self.discovery_run_id,
            search_mode=self.search_mode,
            seed=int(self.config.seed),
            config=self.config.as_dict(),
            fingerprint_components=self.fingerprint_components,
            source_run_id=self.source_run_id,
            resumed_from_run_id=self.resumed_from_run_id,
            budget_caps={
                "total_candidate_budget": self.config.total_candidate_budget,
                "max_full_wfo": self.config.max_full_wfo,
                "max_runtime_seconds": self.config.max_runtime_seconds,
                "max_evaluated_candidates": self.config.max_evaluated_candidates,
            },
        )
        ckpt.family_specs = [f.as_dict() for f in families]
        self._live_checkpoint = ckpt
        return ckpt

    def _checkpoint_after_candidate(
        self,
        *,
        families: list[FamilySpec],
        cand: StrategyCandidate,
        rec: EvaluationRecord,
        fid: str,
        gate: str,
        campaign_generated: int,
        campaign_evaluated: int,
        campaign_full_wfo: int,
        family_states_payload: dict[str, Any] | None = None,
        pending_eval_ids: list[str] | None = None,
    ) -> None:
        from discovery.search_resume import mark_score_qualified_pending

        ckpt = self._ensure_live_checkpoint(families=families)
        ckpt.candidates[cand.candidate_id] = cand.as_dict()
        ckpt.evaluation_records[cand.candidate_id] = rec.as_dict()
        ckpt.candidate_gates[cand.candidate_id] = gate
        ckpt.campaign_generated = int(campaign_generated)
        ckpt.campaign_evaluated = int(campaign_evaluated)
        ckpt.campaign_full_wfo = int(campaign_full_wfo)
        ckpt.session_generated = int(self._session_generated)
        ckpt.session_evaluated = int(self._session_evaluated)
        ckpt.session_full_wfo = int(self._session_full_wfo)
        ckpt.status_history = [e.as_dict() for e in self.candidate_status_history]
        ckpt.status_seq = int(self._status_seq)
        if cand.candidate_id not in ckpt.trial_population_candidate_ids:
            ckpt.trial_population_candidate_ids.append(cand.candidate_id)
        if family_states_payload:
            from discovery.search_checkpoint import FamilyCheckpointState

            for fam_id, payload in family_states_payload.items():
                if isinstance(payload, FamilyCheckpointState):
                    ckpt.family_states[fam_id] = payload
                elif isinstance(payload, dict):
                    ckpt.family_states[fam_id] = FamilyCheckpointState.from_dict(payload)
        if pending_eval_ids is not None:
            ckpt.pending_eval_ids = list(pending_eval_ids)
        mark_score_qualified_pending(ckpt)
        self._persist_live_checkpoint()

    def _force_generation_zero(self, cand: StrategyCandidate) -> StrategyCandidate:
        if cand.generation == 0 and not cand.parent_ids:
            return cand
        from discovery.candidate import build_candidate

        return build_candidate(
            entry_tree=cand.entry_tree,
            exit_tree=cand.exit_tree,
            stop=cand.stop,
            target=cand.target,
            sizing=cand.sizing,
            regime_gates=cand.regime_gates,
            strategy_family=cand.strategy_family,
            creation_method=cand.creation_method,
            generation=0,
            parent_ids=(),
            grammar_version=cand.grammar_version,
            feature_set_version=cand.feature_set_version,
            cost_model_version=cand.cost_model_version,
            asset_universe=cand.asset_universe,
            random_seed=cand.random_seed,
            family_provenance=dict(cand.family_provenance or {}),
        )

    def _stamp_candidate_budget_metadata(
        self,
        cand: StrategyCandidate,
        *,
        family_id: str,
        generation_budget_consumed: int,
        initial_or_descendant: str,
    ) -> StrategyCandidate:
        """Attach required generation-budget metadata without claiming false lineage."""
        from discovery.candidate import build_candidate

        creation = (
            cand.creation_method.value
            if hasattr(cand.creation_method, "value")
            else str(cand.creation_method)
        )
        prov = dict(cand.family_provenance or {})
        prov.update(
            {
                "family_id": family_id or prov.get("family_id") or cand.strategy_family,
                "generation": int(cand.generation),
                "creation_method": creation,
                "parent_ids": list(cand.parent_ids),
                "initial_or_descendant": initial_or_descendant,
                "generation_budget_consumed": int(generation_budget_consumed),
            }
        )
        return build_candidate(
            entry_tree=cand.entry_tree,
            exit_tree=cand.exit_tree,
            stop=cand.stop,
            target=cand.target,
            sizing=cand.sizing,
            regime_gates=cand.regime_gates,
            strategy_family=cand.strategy_family,
            creation_method=cand.creation_method,
            generation=int(cand.generation),
            parent_ids=tuple(cand.parent_ids),
            grammar_version=cand.grammar_version,
            feature_set_version=cand.feature_set_version,
            cost_model_version=cand.cost_model_version,
            asset_universe=cand.asset_universe,
            random_seed=cand.random_seed,
            lineage_id=cand.lineage_id,
            family_provenance=prov,
        )

    def _generate_initial_population(
        self,
        *,
        spec: FamilySpec,
        target: int,
        seen: set[str],
        budget_consumed_start: int = 0,
    ) -> list[StrategyCandidate]:
        cfg = self.config
        generator = CandidateGenerator.from_family_spec(spec)
        pool: list[StrategyCandidate] = []
        consumed = int(budget_consumed_start)

        # Deterministic grammar-derived seed templates first (family coverage).
        seed_base = int(stable_seed(cfg.seed, spec.family_id, 0, salt=331) % (2**31 - 1))
        for tmpl in generator.grammar_seed_templates(seed=seed_base):
            if len(pool) >= target:
                break
            cand = self._force_generation_zero(tmpl)
            if cand.strategy_family == "dsl_generated":
                raise RuntimeError(
                    f"FAMILY_LABEL_COLLAPSE: seed template dsl_generated under {spec.family_id!r}"
                )
            if cand.candidate_id in seen:
                continue
            # Seed templates must still pass the same direction / domain gates.
            coherence = validate_family_direction_coherence(cand, spec)
            if not coherence.coherent:
                continue
            grammar = spec.to_grammar()
            domain_reason = validate_tree_threshold_domains(cand.entry_tree, grammar)
            if domain_reason is not None:
                continue
            consumed += 1
            cand = self._stamp_candidate_budget_metadata(
                cand,
                family_id=spec.family_id,
                generation_budget_consumed=consumed,
                initial_or_descendant="initial",
            )
            seen.add(cand.candidate_id)
            pool.append(cand)
            self._emit(
                "CANDIDATE_GENERATED",
                {
                    "candidate_id": cand.candidate_id,
                    "family": cand.strategy_family,
                    "family_id": spec.family_id,
                    "generation": 0,
                    "creation_method": "SEED_TEMPLATE",
                    "parent_ids": [],
                    "initial_or_descendant": "initial",
                    "generation_budget_consumed": consumed,
                    "entry_pattern": (cand.family_provenance or {}).get("entry_pattern"),
                },
            )

        attempts = 0
        while len(pool) < target and attempts < target * 8:
            attempts += 1
            seed_i = stable_seed(cfg.seed, spec.family_id, attempts, salt=997)
            try:
                cand = generator.generate(seed=seed_i)
            except Exception:  # noqa: BLE001
                continue
            if cand.strategy_family == "dsl_generated":
                raise RuntimeError(
                    f"FAMILY_LABEL_COLLAPSE: generated dsl_generated under {spec.family_id!r}"
                )
            cand = self._force_generation_zero(cand)
            if cand.candidate_id in seen:
                continue
            consumed += 1
            cand = self._stamp_candidate_budget_metadata(
                cand,
                family_id=spec.family_id,
                generation_budget_consumed=consumed,
                initial_or_descendant="initial",
            )
            seen.add(cand.candidate_id)
            pool.append(cand)
            creation = (
                cand.creation_method.value
                if hasattr(cand.creation_method, "value")
                else str(cand.creation_method)
            )
            self._emit(
                "CANDIDATE_GENERATED",
                {
                    "candidate_id": cand.candidate_id,
                    "family": cand.strategy_family,
                    "family_id": spec.family_id,
                    "generation": 0,
                    "creation_method": creation,
                    "parent_ids": [],
                    "initial_or_descendant": "initial",
                    "generation_budget_consumed": consumed,
                },
            )
        return pool

    def _select_parents_deterministic(
        self,
        *,
        eligible: list[tuple[StrategyCandidate, EvaluationRecord, float]],
        n_parents: int,
    ) -> tuple[list[StrategyCandidate], dict[str, str]]:
        """Rank by fitness then candidate_id; record deterministic selection reasons."""
        ranked = sorted(
            eligible,
            key=lambda t: (-float(t[2]), t[0].candidate_id),
        )
        selected = [t[0] for t in ranked[: max(0, n_parents)]]
        reasons: dict[str, str] = {}
        for rank, cand in enumerate(selected):
            reasons[cand.candidate_id] = (
                f"deterministic_structural_fitness_rank_{rank}"
            )
        return selected, reasons

    def _run_family_local_evolution(
        self,
        *,
        families: list[FamilySpec],
        gen_alloc: dict[str, int],
        wfo_alloc: dict[str, int],
        initial_alloc: dict[str, int] | None = None,
        evolutionary_alloc: dict[str, int] | None = None,
        initial_population_budget: int | None = None,
        evolutionary_candidate_budget: int | None = None,
    ) -> MultiFamilyCampaignResult:
        """Deterministic per-family generation queues with mutation/crossover."""
        cfg = self.config
        t0 = time.perf_counter()
        pipeline = PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_STRESS_SCREENING
        family_ids = [f.family_id for f in families]
        initial_per = cfg.effective_initial_candidates_per_family()
        if initial_alloc is None:
            initial_alloc = {fid: initial_per for fid in family_ids}
        if evolutionary_alloc is None:
            evolutionary_alloc = {
                fid: max(0, int(gen_alloc[fid]) - int(initial_alloc[fid])) for fid in family_ids
            }
        if initial_population_budget is None:
            initial_population_budget = sum(initial_alloc.values())
        if evolutionary_candidate_budget is None:
            evolutionary_candidate_budget = max(
                0, int(cfg.total_candidate_budget) - int(initial_population_budget)
            )
        # Mutable per-family generated caps (evolutionary remainder may reallocate).
        family_gen_caps: dict[str, int] = {
            fid: int(initial_alloc[fid]) + int(evolutionary_alloc[fid]) for fid in family_ids
        }
        evo_reallocation_pool = 0
        records_by_family: dict[str, list[EvaluationRecord]] = {fid: [] for fid in family_ids}
        full_wfo_counts: dict[str, int] = {fid: 0 for fid in family_ids}
        generated_counts: dict[str, int] = {fid: 0 for fid in family_ids}
        generation_0_counts: dict[str, int] = {fid: 0 for fid in family_ids}
        descendant_counts: dict[str, int] = {fid: 0 for fid in family_ids}
        mutation_counts: dict[str, int] = {fid: 0 for fid in family_ids}
        crossover_counts: dict[str, int] = {fid: 0 for fid in family_ids}
        highest_generation: dict[str, int] = {fid: 0 for fid in family_ids}
        structural_parents_found: dict[str, int] = {fid: 0 for fid in family_ids}
        family_stop_reasons: dict[str, str | None] = {fid: None for fid in family_ids}
        descendant_reject_counts: dict[str, dict[str, int]] = {fid: {} for fid in family_ids}
        cand_by_id: dict[str, StrategyCandidate] = {}
        generation_records: list[GenerationRecord] = []
        seen_ids: set[str] = set()
        campaign_generated = 0
        campaign_evaluated = 0
        campaign_full_wfo = 0
        max_evaluated = (
            int(cfg.max_evaluated_candidates)
            if cfg.max_evaluated_candidates is not None
            else int(cfg.total_candidate_budget)
        )
        stop_reason = "multi_family_evolutionary_robustness_screening_completed"
        completed_generation_count = 0
        skip_evolution_loop = False
        resume_pending_stress: set[str] | None = None
        resume_pending_robustness: set[str] | None = None

        from discovery.search_program import PipelinePhase, SearchMode
        from discovery.search_resume import (
            choose_resume_phase,
            mark_score_qualified_pending,
            rebuild_candidates,
            rebuild_families,
            rebuild_records_by_family,
            sync_pending_gate_queues,
        )
        from discovery.search_checkpoint import FamilyCheckpointState

        resuming = (
            self.resume_checkpoint is not None
            and self.search_mode
            in {
                SearchMode.RESUME_SEARCH.value,
                SearchMode.EXTEND_BUDGET.value,
                SearchMode.REEVALUATE_FROZEN_CANDIDATES.value,
            }
        )
        if resuming:
            ckpt = self.resume_checkpoint
            assert ckpt is not None
            self._live_checkpoint = ckpt
            ckpt.session_run_id = self.discovery_run_id
            ckpt.resumed_from_run_id = self.resumed_from_run_id or ckpt.session_run_id
            ckpt.source_run_id = self.source_run_id or ckpt.source_run_id
            ckpt.search_mode = self.search_mode
            ckpt.session_generated = 0
            ckpt.session_evaluated = 0
            ckpt.session_full_wfo = 0
            # Never regenerate Generation 0 — restore candidates and lineage.
            restored_families = rebuild_families(ckpt) if ckpt.family_specs else families
            if restored_families:
                families = restored_families
                family_ids = [f.family_id for f in families]
            cand_by_id = rebuild_candidates(ckpt)
            seen_ids = set(cand_by_id.keys())
            records_by_family = rebuild_records_by_family(ckpt, families, cand_by_id)
            for fid in family_ids:
                records_by_family.setdefault(fid, [])
            campaign_generated = int(ckpt.campaign_generated)
            campaign_evaluated = int(ckpt.campaign_evaluated)
            campaign_full_wfo = int(ckpt.campaign_full_wfo)
            evo_reallocation_pool = int(ckpt.evo_reallocation_pool)
            completed_generation_count = int(ckpt.completed_generation_count)
            generation_records = [
                GenerationRecord(
                    family_id=str(g["family_id"]),
                    generation=int(g.get("generation") or 0),
                    input_population_ids=list(g.get("input_population_ids") or []),
                    evaluated_candidate_ids=list(g.get("evaluated_candidate_ids") or []),
                    selected_parent_ids=list(g.get("selected_parent_ids") or []),
                    mutation_candidate_ids=list(g.get("mutation_candidate_ids") or []),
                    crossover_candidate_ids=list(g.get("crossover_candidate_ids") or []),
                    completed_full_wfo_candidate_ids=list(
                        g.get("completed_full_wfo_candidate_ids") or []
                    ),
                    score_qualified_candidate_ids=list(
                        g.get("score_qualified_candidate_ids") or []
                    ),
                    best_generation_score=g.get("best_generation_score"),
                    best_score_so_far=g.get("best_score_so_far"),
                    generated_count=int(g.get("generated_count") or 0),
                    evaluated_count=int(g.get("evaluated_count") or 0),
                    completed_full_wfo_count=int(g.get("completed_full_wfo_count") or 0),
                    stagnation_count=int(g.get("stagnation_count") or 0),
                    stop_reason=g.get("stop_reason"),
                    previous_best_score=g.get("previous_best_score"),
                    generation_best_score=g.get("generation_best_score"),
                    improvement=g.get("improvement"),
                    minimum_required_improvement=g.get("minimum_required_improvement"),
                    parent_selection_reasons=dict(g.get("parent_selection_reasons") or {}),
                    parent_eligibility=dict(g.get("parent_eligibility") or {}),
                    rejected_descendants=list(g.get("rejected_descendants") or []),
                )
                for g in ckpt.generation_records
            ]
            if ckpt.status_history and not self.candidate_status_history:
                self._status_seq = int(ckpt.status_seq)
                self.candidate_status_history = [
                    CandidateStatusEvent(
                        sequence=int(e["sequence"]),
                        candidate_id=str(e["candidate_id"]),
                        family_id=str(e["family_id"]),
                        generation=int(e.get("generation") or 0),
                        prior_status=str(e.get("prior_status") or ""),
                        new_status=str(e.get("new_status") or ""),
                        reason=str(e.get("reason") or ""),
                        artifact_refs=dict(e.get("artifact_refs") or {}),
                    )
                    for e in ckpt.status_history
                ]
            for fid, fst in ckpt.family_states.items():
                generated_counts[fid] = int(fst.generated_count)
                generation_0_counts[fid] = int(fst.generation_0_count)
                descendant_counts[fid] = int(fst.descendant_count)
                mutation_counts[fid] = int(fst.mutation_count)
                crossover_counts[fid] = int(fst.crossover_count)
                full_wfo_counts[fid] = int(fst.full_wfo_count)
                highest_generation[fid] = int(fst.highest_generation)
                structural_parents_found[fid] = int(fst.structural_parents_found)
                family_stop_reasons[fid] = fst.family_stop_reason
                family_gen_caps[fid] = int(fst.family_gen_cap or family_gen_caps.get(fid, 0))
                if fst.family_wfo_cap:
                    wfo_alloc[fid] = int(fst.family_wfo_cap)
                descendant_reject_counts[fid] = dict(fst.descendant_reject_counts)
            # Restore prior gate summaries so DSR/PBO population stays cumulative.
            if ckpt.stress_summaries and not self.candidate_stress_summaries:
                for s in ckpt.stress_summaries:
                    self.candidate_stress_summaries.append(
                        CandidateStressSummary(
                            candidate_id=str(s["candidate_id"]),
                            family_id=str(s.get("family_id") or ""),
                            generation=int(s.get("generation") or 0),
                            lineage=dict(s.get("lineage") or {}),
                            score_qualified_proof=dict(s.get("score_qualified_proof") or {}),
                            backend_kind=str(s.get("backend_kind") or ""),
                            research_eligible=bool(s.get("research_eligible")),
                            final_decision=str(s.get("final_decision") or STRESS_NOT_ENTERED),
                            final_reason=str(s.get("final_reason") or ""),
                        )
                    )
            if ckpt.robustness_summaries and not self.candidate_robustness_summaries:
                for s in ckpt.robustness_summaries:
                    self.candidate_robustness_summaries.append(
                        CandidateRobustnessSummary(
                            candidate_id=str(s["candidate_id"]),
                            family_id=str(s.get("family_id") or ""),
                            generation=int(s.get("generation") or 0),
                            lineage=dict(s.get("lineage") or {}),
                            stress_passed_proof=dict(s.get("stress_passed_proof") or {}),
                            backend_kind=str(s.get("backend_kind") or ""),
                            research_eligible=bool(s.get("research_eligible")),
                            final_decision=str(
                                s.get("final_decision") or ROBUSTNESS_NOT_ENTERED
                            ),
                            final_reason=str(s.get("final_reason") or ""),
                        )
                    )
            mark_score_qualified_pending(ckpt)
            phase = choose_resume_phase(ckpt)
            ckpt.pipeline_phase = phase.value
            resume_pending_stress = set(ckpt.pending_stress_ids)
            resume_pending_robustness = set(ckpt.pending_robustness_ids)
            self._emit(
                "SEARCH_RESUME_STARTED",
                {
                    "search_program_id": self.search_program_id,
                    "search_mode": self.search_mode,
                    "phase": phase.value,
                    "pending_by_gate": ckpt.pending_by_gate(),
                    "generation_0_regenerated": False,
                    "cache_reuse": self.search_mode
                    != SearchMode.REEVALUATE_FROZEN_CANDIDATES.value,
                },
            )
            # Priority: Stress → Robustness → Statistics before new evolution.
            if resume_pending_stress:
                self._run_stress_phase(
                    families=families,
                    records_by_family=records_by_family,
                    cand_by_id=cand_by_id,
                    t0=t0,
                    only_candidate_ids=resume_pending_stress,
                )
                for s in self.candidate_stress_summaries:
                    ckpt.candidate_gates[s.candidate_id] = s.final_decision
                ckpt.stress_summaries = [s.as_dict() for s in self.candidate_stress_summaries]
                sync_pending_gate_queues(ckpt)
                resume_pending_robustness = set(ckpt.pending_robustness_ids)
                self._persist_live_checkpoint()
            if resume_pending_robustness or ckpt.pending_robustness_ids:
                self._run_robustness_phase(
                    families=families,
                    records_by_family=records_by_family,
                    cand_by_id=cand_by_id,
                    t0=t0,
                    stress_pipeline_complete=True,
                    only_candidate_ids=set(ckpt.pending_robustness_ids)
                    or resume_pending_robustness,
                )
                for s in self.candidate_robustness_summaries:
                    ckpt.candidate_gates[s.candidate_id] = s.final_decision
                ckpt.robustness_summaries = [
                    s.as_dict() for s in self.candidate_robustness_summaries
                ]
                sync_pending_gate_queues(ckpt)
                self._persist_live_checkpoint()
            if ckpt.pending_statistics_ids or any(
                s.final_decision == ROBUSTNESS_PASSED
                for s in self.candidate_robustness_summaries
            ):
                shortlist_phase = self._run_statistics_clustering_shortlist_phase(
                    families=families,
                    records_by_family=records_by_family,
                    cand_by_id=cand_by_id,
                    robustness_pipeline_complete=True,
                )
                ckpt.statistics_summaries = [
                    s.as_dict() for s in shortlist_phase.statistics_summaries
                ]
                ckpt.clusters = list(shortlist_phase.clusters)
                ckpt.behavioral_signatures = list(shortlist_phase.behavioral_signatures)
                ckpt.research_shortlist = [
                    e.as_dict() if hasattr(e, "as_dict") else e
                    for e in shortlist_phase.research_shortlist
                ]
                ckpt.shortlist_rejects = [
                    r.as_dict() if hasattr(r, "as_dict") else r
                    for r in shortlist_phase.shortlist_rejects
                ]
                ckpt.population_stats = dict(shortlist_phase.population_stats or {})
                # Merge historical trials into cumulative DSR population ids.
                for cid in list(cand_by_id.keys()):
                    if cid not in ckpt.trial_population_candidate_ids:
                        ckpt.trial_population_candidate_ids.append(cid)
                sync_pending_gate_queues(ckpt)
                self._persist_live_checkpoint()
            # Complete generated-but-unevaluated before breeding further.
            if ckpt.pending_eval_ids:
                for cid in list(ckpt.pending_eval_ids):
                    if time.perf_counter() - t0 >= float(cfg.max_runtime_seconds):
                        stop_reason = "max_runtime_seconds"
                        break
                    if campaign_evaluated >= max_evaluated:
                        stop_reason = "max_evaluated_candidates"
                        break
                    if campaign_full_wfo >= cfg.max_full_wfo:
                        stop_reason = "max_full_wfo"
                        break
                    cand = cand_by_id.get(cid)
                    if cand is None:
                        continue
                    fid = str(
                        (cand.family_provenance or {}).get("family_id") or cand.strategy_family
                    )
                    spec = next((f for f in families if f.family_id == fid), None)
                    if spec is None:
                        continue
                    ev = self._make_evaluator(
                        spec=spec,
                        wfo_cap=max(1, int(wfo_alloc.get(fid, 1)) - full_wfo_counts.get(fid, 0)),
                        gen_cap=max(1, family_gen_caps.get(fid, 1)),
                    )
                    self._bind_evaluator_features(ev)
                    prev_wfo = full_wfo_counts.get(fid, 0)
                    rec = self._evaluate_candidate(
                        ev=ev, cand=cand, full_wfo_counts=full_wfo_counts, fid=fid
                    )
                    records_by_family.setdefault(fid, []).append(rec)
                    campaign_evaluated += 1
                    self._session_evaluated += 1
                    if full_wfo_counts.get(fid, 0) > prev_wfo:
                        campaign_full_wfo += full_wfo_counts[fid] - prev_wfo
                        self._session_full_wfo += full_wfo_counts[fid] - prev_wfo
                    parent_status, is_sq, _ = self.classify_parent_eligibility(rec)
                    gate = SCORE_QUALIFIED if is_sq else parent_status
                    if pending_eval_ids := [
                        x for x in ckpt.pending_eval_ids if x != cid
                    ]:
                        pass
                    ckpt.pending_eval_ids = [x for x in ckpt.pending_eval_ids if x != cid]
                    if fid in ckpt.family_states:
                        ckpt.family_states[fid].pending_eval_ids = [
                            x
                            for x in ckpt.family_states[fid].pending_eval_ids
                            if x != cid
                        ]
                        if cid not in ckpt.family_states[fid].evaluated_ids:
                            ckpt.family_states[fid].evaluated_ids.append(cid)
                    self._checkpoint_after_candidate(
                        families=families,
                        cand=cand,
                        rec=rec,
                        fid=fid,
                        gate=gate,
                        campaign_generated=campaign_generated,
                        campaign_evaluated=campaign_evaluated,
                        campaign_full_wfo=campaign_full_wfo,
                    )
                    _ = pending_eval_ids
                # After completing pending evals, run any newly score-qualified stress.
                mark_score_qualified_pending(ckpt)
                if ckpt.pending_stress_ids:
                    self._run_stress_phase(
                        families=families,
                        records_by_family=records_by_family,
                        cand_by_id=cand_by_id,
                        t0=t0,
                        only_candidate_ids=set(ckpt.pending_stress_ids),
                    )
                    ckpt.stress_summaries = [
                        s.as_dict() for s in self.candidate_stress_summaries
                    ]
                    self._persist_live_checkpoint()
            # Continue family-local evolution only after pending gates/evals.
            skip_evolution_loop = False
            # Families that still have room continue below with restored queues.
            self._emit(
                "SEARCH_RESUME_GATES_DRAINED",
                {
                    "pending_by_gate": ckpt.pending_by_gate(),
                    "campaign_evaluated": campaign_evaluated,
                    "session_evaluated": self._session_evaluated,
                    "cache_hits": self._cache_hits,
                    "cache_misses": self._cache_misses,
                },
            )

        self._emit(
            "MULTI_FAMILY_EVOLUTION_STARTED",
            {
                "families": family_ids,
                "gen_alloc": dict(family_gen_caps),
                "wfo_alloc": wfo_alloc,
                "initial_alloc": dict(initial_alloc),
                "evolutionary_alloc": dict(evolutionary_alloc),
                "initial_population_budget": initial_population_budget,
                "evolutionary_candidate_budget": evolutionary_candidate_budget,
                "resuming": resuming,
                "search_program_id": self.search_program_id,
            },
        )

        if skip_evolution_loop:
            pass  # gates-only resume already handled
        else:
            self._ensure_live_checkpoint(families=families)
        for spec in families:
            fid = spec.family_id
            grammar = spec.to_grammar()
            mutator = Mutator(grammar=grammar)
            crossover = Crossover(grammar=grammar)
            family_gen_cap = int(family_gen_caps[fid])
            family_wfo_cap = int(wfo_alloc[fid])
            # Exact initial population — never expand gen-0 to exhaust total budget.
            initial_n = int(initial_alloc[fid])
            # Reserve Full-WFO slots for evolutionary descendants when possible.
            # Do not spend the entire family WFO quota on generation 0 alone.
            wfo_reserve_for_descendants = 0
            if int(evolutionary_alloc.get(fid, 0)) > 0 and max(0, int(cfg.evolution_generations) - 1) >= 1:
                if family_wfo_cap >= 2:
                    wfo_reserve_for_descendants = max(1, family_wfo_cap // 2)
                elif family_wfo_cap == 1 and int(evolutionary_alloc.get(fid, 0)) > 0:
                    # Prefer leaving the single slot for a descendant after one gen-0 parent.
                    wfo_reserve_for_descendants = 0

            # --- Generation 0 queue (initial population only) ---
            gen_queues: dict[int, list[StrategyCandidate]] = {}
            restored_fst = (
                self._live_checkpoint.family_states.get(fid)
                if self._live_checkpoint is not None
                else None
            )
            existing_gen0 = [
                c
                for c in cand_by_id.values()
                if c.generation == 0
                and (
                    (c.family_provenance or {}).get("family_id") == fid
                    or c.strategy_family == fid
                )
            ]
            if resuming and (
                (restored_fst is not None and restored_fst.generation_0_complete)
                or existing_gen0
            ):
                # Never regenerate Generation 0 on resume.
                gen_queues[0] = []
                if restored_fst is not None:
                    for gid_str, ids in restored_fst.gen_queues.items():
                        gen = int(gid_str)
                        gen_queues[gen] = [
                            cand_by_id[cid] for cid in ids if cid in cand_by_id
                        ]
                if not gen_queues.get(0):
                    gen_queues[0] = list(existing_gen0)
                    if restored_fst is not None:
                        for cid in restored_fst.pending_eval_ids:
                            c = cand_by_id.get(cid)
                            if c is not None and c.generation == 0 and c not in gen_queues[0]:
                                gen_queues[0].append(c)
                if restored_fst is not None:
                    generated_counts[fid] = int(restored_fst.generated_count)
                    generation_0_counts[fid] = int(
                        restored_fst.generation_0_count or len(gen_queues[0])
                    )
                    highest_generation[fid] = int(restored_fst.highest_generation)
                else:
                    generated_counts[fid] = max(
                        generated_counts.get(fid, 0), len(existing_gen0)
                    )
                    generation_0_counts[fid] = len(gen_queues[0])
                    highest_generation[fid] = max(
                        (c.generation for c in cand_by_id.values()), default=0
                    )
            else:
                gen_queues[0] = self._generate_initial_population(
                    spec=spec,
                    target=initial_n,
                    seen=seen_ids,
                    budget_consumed_start=campaign_generated,
                )
                if len(gen_queues[0]) < initial_n:
                    raise RuntimeError(
                        f"FAMILY_GENERATION_SHORTFALL: {fid} produced {len(gen_queues[0])} "
                        f"< initial_candidates_per_family={initial_n}"
                    )
                generated_counts[fid] = len(gen_queues[0])
                generation_0_counts[fid] = len(gen_queues[0])
                campaign_generated += len(gen_queues[0])
                self._session_generated += len(gen_queues[0])
                for c in gen_queues[0]:
                    cand_by_id[c.candidate_id] = c
                highest_generation[fid] = 0

            if not (resuming and restored_fst and restored_fst.generation_0_complete):
                pass  # counts already set above for fresh gen-0
            else:
                for c in gen_queues.get(0, []):
                    cand_by_id.setdefault(c.candidate_id, c)

            best_score_so_far = float("-inf")
            if restored_fst is not None and restored_fst.best_score_so_far is not None:
                best_score_so_far = float(restored_fst.best_score_so_far)
            stagnation_count = int(restored_fst.stagnation_count) if restored_fst else 0
            family_stop: str | None = (
                restored_fst.family_stop_reason if restored_fst else None
            )
            # If runtime-exhausted previously, clear stop so EXTEND/RESUME can continue.
            if family_stop == "max_runtime_seconds" and resuming:
                family_stop = None
                family_stop_reasons[fid] = None
            population: list[StrategyCandidate] = []
            if restored_fst and restored_fst.population_ids:
                population = [
                    cand_by_id[cid]
                    for cid in restored_fst.population_ids
                    if cid in cand_by_id
                ]
            if not population:
                population = list(gen_queues.get(0, []))
            rec_by_id: dict[str, EvaluationRecord] = {
                r.candidate_id: r for r in records_by_family.get(fid, [])
            }

            max_gen_index = max(0, int(cfg.evolution_generations) - 1)
            start_gen = int(restored_fst.current_generation) if restored_fst else 0
            # Skip families that already completed evolution (unless runtime-exhausted).
            if (
                family_stop
                and family_stop
                not in {
                    None,
                    "max_runtime_seconds",
                    "budget_exhausted_before_breed",
                    "empty_generation_queue",
                }
            ):
                family_stop_reasons[fid] = family_stop
                continue
            for gen in range(start_gen, max_gen_index + 1):
                runtime = time.perf_counter() - t0
                if runtime >= float(cfg.max_runtime_seconds):
                    family_stop = "max_runtime_seconds"
                    stop_reason = "max_runtime_seconds"
                    break
                if campaign_generated > cfg.total_candidate_budget:
                    family_stop = "total_candidate_budget"
                    stop_reason = "total_candidate_budget"
                    break
                if campaign_evaluated >= max_evaluated:
                    family_stop = "max_evaluated_candidates"
                    stop_reason = "max_evaluated_candidates"
                    break
                if campaign_full_wfo >= cfg.max_full_wfo:
                    family_stop = "max_full_wfo"
                    stop_reason = "max_full_wfo"
                    break
                if generated_counts[fid] > family_gen_cap:
                    family_stop = "family_generated_allocation"
                    stop_reason = "family_generated_allocation"
                    break
                if full_wfo_counts[fid] >= family_wfo_cap and gen > 0:
                    # Allow finishing current queue only if WFO remains; else stop breeding.
                    pass

                # Breed descendants into the NEXT generation queue before evaluating
                # the current generation only when gen>0 was already filled; for gen0
                # the queue is the initial population. For gen>=1, create the queue
                # from parents of the previous generation first if missing.
                if gen > 0 and gen not in gen_queues:
                    # Should have been filled at end of previous iteration.
                    break

                queue = list(gen_queues.get(gen, []))
                if not queue:
                    family_stop = "empty_generation_queue"
                    break

                # Evaluate ONLY this generation's queue (never append behind older randoms).
                remaining_wfo = family_wfo_cap - full_wfo_counts[fid]
                remaining_campaign_wfo = cfg.max_full_wfo - campaign_full_wfo
                remaining_eval = max_evaluated - campaign_evaluated
                eval_cap = min(len(queue), remaining_eval)
                if remaining_wfo <= 0 or remaining_campaign_wfo <= 0:
                    if gen == 0:
                        # Still need at least structural evaluation attempts within WFO cap 0?
                        # With zero WFO allocation, stop.
                        family_stop = "family_full_wfo_allocation"
                        stop_reason = "max_full_wfo"
                        break
                    family_stop = "family_full_wfo_allocation"
                    stop_reason = "max_full_wfo"
                    break

                wfo_room = min(remaining_wfo, remaining_campaign_wfo, eval_cap)
                if gen == 0 and wfo_reserve_for_descendants > 0:
                    # Keep reserved Full-WFO capacity for later generations.
                    gen0_wfo_limit = max(0, family_wfo_cap - wfo_reserve_for_descendants)
                    wfo_room = min(wfo_room, max(0, gen0_wfo_limit - full_wfo_counts[fid]))
                    if wfo_room <= 0:
                        # Still evaluate at least one gen-0 candidate when family has WFO.
                        wfo_room = min(1, remaining_wfo, remaining_campaign_wfo, eval_cap)
                ev = self._make_evaluator(
                    spec=spec,
                    wfo_cap=max(wfo_room, 1),
                    gen_cap=max(family_gen_cap, len(queue)),
                )
                self._bind_evaluator_features(ev)

                greg = GenerationRecord(
                    family_id=fid,
                    generation=gen,
                    input_population_ids=[c.candidate_id for c in population],
                    minimum_required_improvement=float(cfg.minimum_improvement),
                    previous_best_score=(
                        None if best_score_so_far == float("-inf") else float(best_score_so_far)
                    ),
                )
                # Cap how many queue members we attempt when gen-0 WFO is reserved.
                take_n = eval_cap
                if gen == 0 and wfo_reserve_for_descendants > 0:
                    take_n = min(eval_cap, max(wfo_room, 1))
                take = queue[:take_n]
                greg.generated_count = len(queue)
                valid_eval_count = 0
                gen_scores: list[float] = []

                for cand in take:
                    if campaign_evaluated >= max_evaluated:
                        family_stop = "max_evaluated_candidates"
                        stop_reason = "max_evaluated_candidates"
                        break
                    if gen == 0 and wfo_reserve_for_descendants > 0:
                        gen0_wfo_limit = max(1, family_wfo_cap - wfo_reserve_for_descendants)
                        if full_wfo_counts[fid] >= gen0_wfo_limit:
                            break
                    if full_wfo_counts[fid] >= family_wfo_cap and campaign_full_wfo >= 0:
                        # Still allow non-WFO rejects (invalid) but evaluator may no-op WFO.
                        if ev.counters.full_wfo >= ev.budget.max_full_wfo_evaluations:
                            # Rebuild evaluator with zero remaining room — skip further WFO.
                            if full_wfo_counts[fid] >= family_wfo_cap:
                                family_stop = family_stop or "family_full_wfo_allocation"
                                break
                    if time.perf_counter() - t0 >= float(cfg.max_runtime_seconds):
                        family_stop = "max_runtime_seconds"
                        stop_reason = "max_runtime_seconds"
                        break

                    assert cand.generation == gen, (
                        f"candidate {cand.candidate_id} generation={cand.generation} "
                        f"!= queue generation={gen}"
                    )
                    # Never repeat a completed evaluation when resume/cache already holds it.
                    if cand.candidate_id in rec_by_id:
                        rec = rec_by_id[cand.candidate_id]
                        if cand.candidate_id not in greg.evaluated_candidate_ids:
                            greg.evaluated_candidate_ids.append(cand.candidate_id)
                            greg.evaluated_count += 1
                            parent_status, is_sq, reason = self.classify_parent_eligibility(rec)
                            greg.parent_eligibility[cand.candidate_id] = parent_status
                            if is_sq:
                                greg.score_qualified_candidate_ids.append(cand.candidate_id)
                            if parent_status != NOT_PARENT_ELIGIBLE and rec.fitness is not None:
                                score_val = float(rec.fitness.fitness)
                                if score_val == score_val and abs(score_val) != float("inf"):
                                    valid_eval_count += 1
                                    gen_scores.append(score_val)
                                elif parent_status == STRUCTURAL_PARENT_ELIGIBLE:
                                    valid_eval_count += 1
                            _ = reason
                        continue

                    prev_campaign_wfo = campaign_full_wfo
                    prev_family_wfo = full_wfo_counts[fid]
                    rec = self._evaluate_candidate(
                        ev=ev,
                        cand=cand,
                        full_wfo_counts=full_wfo_counts,
                        fid=fid,
                    )
                    records_by_family[fid].append(rec)
                    rec_by_id[cand.candidate_id] = rec
                    campaign_evaluated += 1
                    self._session_evaluated += 1
                    greg.evaluated_candidate_ids.append(cand.candidate_id)
                    greg.evaluated_count += 1

                    if full_wfo_counts[fid] > prev_family_wfo:
                        delta = full_wfo_counts[fid] - prev_family_wfo
                        campaign_full_wfo += delta
                        self._session_full_wfo += delta
                        greg.completed_full_wfo_candidate_ids.append(cand.candidate_id)
                        greg.completed_full_wfo_count += delta

                    parent_status, is_sq, reason = self.classify_parent_eligibility(rec)
                    greg.parent_eligibility[cand.candidate_id] = parent_status
                    if is_sq:
                        greg.score_qualified_candidate_ids.append(cand.candidate_id)
                    if parent_status != NOT_PARENT_ELIGIBLE and rec.fitness is not None:
                        score_val = float(rec.fitness.fitness)
                        if score_val == score_val and abs(score_val) != float("inf"):  # finite
                            valid_eval_count += 1
                            gen_scores.append(score_val)
                        elif parent_status == STRUCTURAL_PARENT_ELIGIBLE:
                            # Structurally valid but non-finite fitness still counts as a
                            # completed valid evaluation for stagnation gating.
                            valid_eval_count += 1
                    _ = reason
                    _ = prev_campaign_wfo
                    gate = SCORE_QUALIFIED if is_sq else (
                        FULL_WFO_COMPLETED if full_wfo_counts[fid] > prev_family_wfo else parent_status
                    )
                    pending_left = [
                        c.candidate_id
                        for c in take
                        if c.candidate_id not in rec_by_id
                    ]
                    fst_payload = FamilyCheckpointState(
                        family_id=fid,
                        generated_count=int(generated_counts[fid]),
                        generation_0_count=int(generation_0_counts[fid]),
                        descendant_count=int(descendant_counts[fid]),
                        mutation_count=int(mutation_counts[fid]),
                        crossover_count=int(crossover_counts[fid]),
                        full_wfo_count=int(full_wfo_counts[fid]),
                        highest_generation=int(highest_generation[fid]),
                        structural_parents_found=int(structural_parents_found[fid]),
                        family_stop_reason=family_stop,
                        family_gen_cap=int(family_gen_cap),
                        family_wfo_cap=int(family_wfo_cap),
                        initial_alloc=int(initial_alloc[fid]),
                        evolutionary_alloc=int(evolutionary_alloc.get(fid, 0)),
                        best_score_so_far=(
                            None
                            if best_score_so_far == float("-inf")
                            else float(best_score_so_far)
                        ),
                        stagnation_count=int(stagnation_count),
                        current_generation=int(gen),
                        generation_0_complete=True,
                        gen_queues={
                            str(g): [c.candidate_id for c in qs]
                            for g, qs in gen_queues.items()
                        },
                        pending_eval_ids=pending_left,
                        evaluated_ids=list(rec_by_id.keys()),
                        parent_pool_ids=[p.candidate_id for p in population],
                        population_ids=[p.candidate_id for p in population],
                        descendant_reject_counts=dict(
                            descendant_reject_counts.get(fid) or {}
                        ),
                    )
                    self._checkpoint_after_candidate(
                        families=families,
                        cand=cand,
                        rec=rec,
                        fid=fid,
                        gate=gate,
                        campaign_generated=campaign_generated,
                        campaign_evaluated=campaign_evaluated,
                        campaign_full_wfo=campaign_full_wfo,
                        family_states_payload={fid: fst_payload},
                        pending_eval_ids=pending_left,
                    )

                generation_best = max(gen_scores) if gen_scores else None
                greg.best_generation_score = generation_best
                greg.generation_best_score = generation_best
                min_imp = float(cfg.minimum_improvement)
                if valid_eval_count > 0:
                    prev = best_score_so_far
                    if prev == float("-inf"):
                        # Establish baseline after the first completed generation.
                        best_score_so_far = (
                            float(generation_best) if generation_best is not None else 0.0
                        )
                        stagnation_count = 0
                        greg.improvement = None
                    else:
                        if generation_best is None:
                            improvement = 0.0
                        else:
                            improvement = float(generation_best - prev)
                        greg.improvement = float(improvement)
                        if improvement < min_imp:
                            stagnation_count += 1
                        else:
                            stagnation_count = 0
                            best_score_so_far = float(generation_best)
                    greg.best_score_so_far = float(best_score_so_far)
                else:
                    greg.improvement = None
                    greg.best_score_so_far = (
                        None
                        if best_score_so_far == float("-inf")
                        else float(best_score_so_far)
                    )

                greg.stagnation_count = stagnation_count
                completed_generation_count += 1

                if (
                    valid_eval_count > 0
                    and stagnation_count >= int(cfg.stagnation_generations)
                ):
                    family_stop = FAMILY_STAGNATION
                    stop_reason = FAMILY_STAGNATION
                    greg.stop_reason = FAMILY_STAGNATION
                    generation_records.append(greg)
                    break

                # Update population from evaluated structural parents in this generation.
                eligible_scored: list[tuple[StrategyCandidate, EvaluationRecord, float]] = []
                for cid in greg.evaluated_candidate_ids:
                    rec = rec_by_id[cid]
                    status, _, sel_reason = self.classify_parent_eligibility(rec)
                    if status is NOT_PARENT_ELIGIBLE:
                        continue
                    fit = float(rec.fitness.fitness) if rec.fitness is not None else float("-inf")
                    eligible_scored.append((cand_by_id[cid], rec, fit))
                    _ = sel_reason

                # Also keep prior structural parents that remain eligible.
                for prev_c in population:
                    if prev_c.candidate_id in {e[0].candidate_id for e in eligible_scored}:
                        continue
                    prev_rec = rec_by_id.get(prev_c.candidate_id)
                    if prev_rec is None:
                        continue
                    status, _, _ = self.classify_parent_eligibility(prev_rec)
                    if status is NOT_PARENT_ELIGIBLE:
                        continue
                    fit = (
                        float(prev_rec.fitness.fitness)
                        if prev_rec.fitness is not None
                        else float("-inf")
                    )
                    eligible_scored.append((prev_c, prev_rec, fit))

                structural_parents_found[fid] = max(
                    structural_parents_found[fid], len(eligible_scored)
                )

                # No structural parents after gen-0: reserve unused evo budget; never
                # refill with extra random generation-0 candidates.
                if gen == 0 and not eligible_scored:
                    family_stop = NO_STRUCTURAL_PARENTS
                    stop_reason = NO_STRUCTURAL_PARENTS
                    greg.stop_reason = NO_STRUCTURAL_PARENTS
                    generation_records.append(greg)
                    unused_evo = max(0, family_gen_cap - generated_counts[fid])
                    if cfg.adaptive_reallocation and unused_evo > 0:
                        evo_reallocation_pool += unused_evo
                        family_gen_caps[fid] = generated_counts[fid]
                        family_gen_cap = generated_counts[fid]
                    break

                n_parents = min(max(1, int(cfg.population_size)), len(eligible_scored))
                # Prefer at least 2 parents when available so family-local crossover can run.
                if len(eligible_scored) >= 2:
                    n_parents = max(n_parents, min(2, len(eligible_scored)))
                parents, parent_reasons = self._select_parents_deterministic(
                    eligible=eligible_scored, n_parents=n_parents
                )
                greg.selected_parent_ids = [p.candidate_id for p in parents]
                greg.parent_selection_reasons = dict(parent_reasons)
                population = list(parents) if parents else list(population)

                # Active evolving family may receive unused evolutionary remainder.
                if (
                    parents
                    and cfg.adaptive_reallocation
                    and evo_reallocation_pool > 0
                ):
                    family_gen_caps[fid] = int(family_gen_caps[fid]) + int(
                        evo_reallocation_pool
                    )
                    family_gen_cap = int(family_gen_caps[fid])
                    evo_reallocation_pool = 0

                # If this was the last generation index, do not breed further.
                if gen >= max_gen_index or family_stop:
                    greg.stop_reason = family_stop or "evolution_generations_completed"
                    generation_records.append(greg)
                    break

                # --- Breed into a SEPARATE next-generation queue ---
                next_gen = gen + 1
                next_queue: list[StrategyCandidate] = []
                remaining_gen_budget = family_gen_cap - generated_counts[fid]
                remaining_campaign_gen = cfg.total_candidate_budget - campaign_generated
                breed_budget = min(
                    remaining_gen_budget,
                    remaining_campaign_gen,
                    max(int(cfg.population_size), 2),
                )

                if breed_budget <= 0 or len(parents) < 1:
                    greg.stop_reason = family_stop or (
                        NO_STRUCTURAL_PARENTS
                        if not parents
                        else "budget_exhausted_before_breed"
                    )
                    if not parents:
                        family_stop = NO_STRUCTURAL_PARENTS
                        stop_reason = NO_STRUCTURAL_PARENTS
                    generation_records.append(greg)
                    unused_evo = max(0, family_gen_cap - generated_counts[fid])
                    if cfg.adaptive_reallocation and unused_evo > 0:
                        evo_reallocation_pool += unused_evo
                        family_gen_caps[fid] = generated_counts[fid]
                        family_gen_cap = generated_counts[fid]
                    break

                breed_seed_base = int(stable_seed(cfg.seed, fid, next_gen, salt=4242) % (2**31 - 1))
                parent_id_set = {p.candidate_id for p in parents}

                def _accept_descendant(
                    child: StrategyCandidate,
                    *,
                    kind: str,
                    family_ref: str,
                ) -> bool:
                    nonlocal campaign_generated
                    if len(next_queue) >= breed_budget:
                        return False
                    if child.candidate_id in seen_ids:
                        return False
                    if child.generation != next_gen:
                        return False
                    if child.strategy_family != family_ref:
                        return False
                    # Never claim a crossover/mutation descendant that is an unchanged parent.
                    if child.candidate_id in parent_id_set:
                        self._record_rejected_descendant(
                            greg=greg,
                            child=child,
                            kind=kind,
                            rejection_reason="UNCHANGED_PARENT_CLONE",
                            details={
                                "candidate_id": child.candidate_id,
                                "family_id": fid,
                                "parent_ids": list(child.parent_ids),
                            },
                            reject_counts=descendant_reject_counts[fid],
                        )
                        return False
                    if kind == "mutation" and len(child.parent_ids) != 1:
                        return False
                    if kind == "crossover" and len(child.parent_ids) != 2:
                        return False
                    # Type / structural validity already enforced by Mutator/Crossover
                    # strict mode. Re-check semantic domain, family grammar, and
                    # family-direction coherence before enqueue; never rewrite.
                    if not self.candidate_within_family_grammar(child, grammar):
                        self._record_rejected_descendant(
                            greg=greg,
                            child=child,
                            kind=kind,
                            rejection_reason="FAMILY_GRAMMAR_VIOLATION",
                            details={
                                "candidate_id": child.candidate_id,
                                "family_id": fid,
                                "creation_method": (
                                    child.creation_method.value
                                    if hasattr(child.creation_method, "value")
                                    else str(child.creation_method)
                                ),
                                "coherence_reason": "outside_family_grammar",
                            },
                            reject_counts=descendant_reject_counts[fid],
                        )
                        return False
                    domain_reason = validate_tree_threshold_domains(child.entry_tree, grammar)
                    if domain_reason is None and child.exit_tree is not None:
                        domain_reason = validate_tree_threshold_domains(child.exit_tree, grammar)
                    if domain_reason is not None:
                        self._record_rejected_descendant(
                            greg=greg,
                            child=child,
                            kind=kind,
                            rejection_reason="INVALID_FEATURE_THRESHOLD_DOMAIN",
                            details={
                                "candidate_id": child.candidate_id,
                                "family_id": fid,
                                "creation_method": (
                                    child.creation_method.value
                                    if hasattr(child.creation_method, "value")
                                    else str(child.creation_method)
                                ),
                                "coherence_reason": domain_reason,
                            },
                            reject_counts=descendant_reject_counts[fid],
                        )
                        return False
                    coherence = validate_family_direction_coherence(child, spec)
                    if not coherence.coherent:
                        self._record_rejected_descendant(
                            greg=greg,
                            child=child,
                            kind=kind,
                            rejection_reason=coherence.rejection_reason
                            or FAMILY_DIRECTION_INCOHERENT,
                            details=dict(coherence.details),
                            reject_counts=descendant_reject_counts[fid],
                        )
                        return False
                    stamped = self._stamp_candidate_budget_metadata(
                        child,
                        family_id=fid,
                        generation_budget_consumed=campaign_generated + 1,
                        initial_or_descendant="descendant",
                    )
                    if stamped.candidate_id != child.candidate_id:
                        # Identity must remain stable; provenance-only stamp.
                        stamped = child
                    seen_ids.add(stamped.candidate_id)
                    next_queue.append(stamped)
                    cand_by_id[stamped.candidate_id] = stamped
                    generated_counts[fid] += 1
                    descendant_counts[fid] += 1
                    campaign_generated += 1
                    highest_generation[fid] = max(highest_generation[fid], next_gen)
                    if kind == "crossover":
                        greg.crossover_candidate_ids.append(stamped.candidate_id)
                        crossover_counts[fid] += 1
                    else:
                        greg.mutation_candidate_ids.append(stamped.candidate_id)
                        mutation_counts[fid] += 1
                    self._emit(
                        "CANDIDATE_GENERATED",
                        {
                            "candidate_id": stamped.candidate_id,
                            "family": stamped.strategy_family,
                            "family_id": fid,
                            "generation": next_gen,
                            "creation_method": kind.upper(),
                            "parent_ids": list(stamped.parent_ids),
                            "initial_or_descendant": "descendant",
                            "generation_budget_consumed": campaign_generated,
                        },
                    )
                    return True

                def _try_mutate(parent: StrategyCandidate, salt: int) -> bool:
                    for attempt in range(32):
                        if len(next_queue) >= breed_budget:
                            return False
                        mut_seed = (breed_seed_base + salt + attempt * 97) % (2**31 - 1)
                        try:
                            mutant = mutator.mutate(
                                parent,
                                seed=mut_seed,
                                strict=True,
                                generation=next_gen,
                            )
                        except MutationRejected:
                            continue
                        except Exception:  # noqa: BLE001
                            continue
                        if _accept_descendant(
                            mutant, kind="mutation", family_ref=parent.strategy_family
                        ):
                            return True
                    return False

                # Guarantee a mutation slot first when budget allows, then crossover,
                # then fill remaining slots with further mutations.
                if parents:
                    _try_mutate(parents[0], salt=11)

                if len(parents) >= 2 and len(next_queue) < breed_budget:
                    pa, pb = parents[0], parents[1]
                    for attempt in range(24):
                        if greg.crossover_candidate_ids:
                            break
                        if len(next_queue) >= breed_budget:
                            break
                        try:
                            children = crossover.crossover(
                                pa,
                                pb,
                                seed=(breed_seed_base + attempt * 31) % (2**31 - 1),
                                strict=True,
                                require_same_family=True,
                                generation=next_gen,
                            )
                        except CrossoverError:
                            continue
                        for child in children:
                            _accept_descendant(
                                child, kind="crossover", family_ref=pa.strategy_family
                            )

                for i, parent in enumerate(parents):
                    if len(next_queue) >= breed_budget:
                        break
                    _try_mutate(parent, salt=17 * (i + 1) + 100)

                gen_queues[next_gen] = next_queue
                greg.stop_reason = None
                generation_records.append(greg)
                if not next_queue:
                    family_stop = "no_valid_descendants"
                    break

                self._emit(
                    "FAMILY_GENERATION_COMPLETED",
                    {
                        "family_id": fid,
                        "generation": gen,
                        "full_wfo": full_wfo_counts[fid],
                        "stagnation_count": stagnation_count,
                    },
                )

            if family_stop and (
                not generation_records
                or generation_records[-1].family_id != fid
                or generation_records[-1].stop_reason is None
            ):
                pass  # stop_reason already recorded on last greg when possible

            # Resolve family stop reason from last generation record when unset.
            if family_stop is None:
                fam_gregs = [g for g in generation_records if g.family_id == fid]
                if fam_gregs and fam_gregs[-1].stop_reason:
                    family_stop = fam_gregs[-1].stop_reason
                else:
                    family_stop = "evolution_generations_completed"
            family_stop_reasons[fid] = family_stop
            # Return unused evolutionary slots for adaptive reallocation.
            unused_evo = max(0, int(family_gen_caps[fid]) - int(generated_counts[fid]))
            if cfg.adaptive_reallocation and unused_evo > 0:
                evo_reallocation_pool += unused_evo
                family_gen_caps[fid] = int(generated_counts[fid])

        # Phase 3B.1: Score Qualified → real Stress.
        stress_accounting = self._run_stress_phase(
            families=families,
            records_by_family=records_by_family,
            cand_by_id=cand_by_id,
            t0=t0,
        )
        # Phase 3B.2: STRESS_PASSED → real Parameter Robustness.
        robustness_accounting = self._run_robustness_phase(
            families=families,
            records_by_family=records_by_family,
            cand_by_id=cand_by_id,
            t0=t0,
            stress_pipeline_complete=True,
        )
        # Phase 3C: DSR/PBO → Behavioral Clustering → Research Shortlist.
        shortlist_phase = self._run_statistics_clustering_shortlist_phase(
            families=families,
            records_by_family=records_by_family,
            cand_by_id=cand_by_id,
            robustness_pipeline_complete=True,
        )
        stress_passed_ids = {
            s.candidate_id
            for s in self.candidate_stress_summaries
            if s.final_decision == STRESS_PASSED
        }
        robustness_passed_ids = {
            s.candidate_id
            for s in self.candidate_robustness_summaries
            if s.final_decision == ROBUSTNESS_PASSED
        }
        statistically_passed_ids = {
            s.candidate_id
            for s in shortlist_phase.statistics_summaries
            if s.final_decision == STATISTICALLY_PASSED
        }
        shortlist_ids = {e.candidate_id for e in shortlist_phase.research_shortlist}
        pipeline = PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_RESEARCH_SHORTLIST

        family_stats = []
        for spec in families:
            fid = spec.family_id
            st = self._stats_from_records(
                spec=spec,
                records=records_by_family[fid],
                generated=generated_counts[fid],
                allocation_generated=int(family_gen_caps[fid]),
                allocation_wfo=wfo_alloc[fid],
                full_wfo=full_wfo_counts[fid],
            )
            for reason, count in descendant_reject_counts.get(fid, {}).items():
                st.rejection_reasons[reason] = st.rejection_reasons.get(reason, 0) + int(count)
            fam_cids = {r.candidate_id for r in records_by_family[fid]}
            st.stress_passed = sum(1 for cid in stress_passed_ids if cid in fam_cids)
            st.robustness_passed = sum(1 for cid in robustness_passed_ids if cid in fam_cids)
            st.statistically_passed = sum(
                1 for cid in statistically_passed_ids if cid in fam_cids
            )
            st.research_shortlisted = sum(1 for cid in shortlist_ids if cid in fam_cids)
            st.generation_0_generated = int(generation_0_counts[fid])
            st.descendants_generated = int(descendant_counts[fid])
            st.mutation_children = int(mutation_counts[fid])
            st.crossover_children = int(crossover_counts[fid])
            st.highest_generation_reached = int(highest_generation[fid])
            st.structural_parents_found = int(structural_parents_found[fid])
            st.family_stop_reason = family_stop_reasons.get(fid)
            family_stats.append(st)

        initial_candidates_generated = sum(generation_0_counts.values())
        evolutionary_candidates_generated = sum(descendant_counts.values())
        unused_evolutionary_budget = max(
            0,
            int(evolutionary_candidate_budget) - int(evolutionary_candidates_generated),
        )

        # Budget invariant assertions (soft: encode into stop / payload).
        assert campaign_generated <= cfg.total_candidate_budget + 0  # noqa: S101
        assert campaign_full_wfo <= cfg.max_full_wfo
        assert campaign_evaluated <= max_evaluated
        if not resuming:
            assert initial_candidates_generated == len(family_ids) * initial_per
            assert campaign_generated == (
                initial_candidates_generated + evolutionary_candidates_generated
            )
            for fid in family_ids:
                assert full_wfo_counts[fid] <= wfo_alloc[fid]
                assert generation_0_counts[fid] == int(initial_alloc[fid])
                assert (
                    generated_counts[fid]
                    == generation_0_counts[fid] + descendant_counts[fid]
                )
                assert generated_counts[fid] <= cfg.total_candidate_budget
        else:
            for fid in family_ids:
                assert full_wfo_counts.get(fid, 0) <= wfo_alloc.get(fid, cfg.max_full_wfo)
                assert generated_counts.get(fid, 0) <= cfg.total_candidate_budget
        rankings: list[dict[str, Any]] = []
        for st in family_stats:
            for cid in st.best_candidate_ids:
                rankings.append(
                    {
                        "candidate_id": cid,
                        "fitness": st.best_fitness,
                        "ranking_source": "validation_oos",
                        "family": st.family_id,
                    }
                )
        empty_reasons = {
            k: list(v) for k, v in EMPTY_COLLECTIONS_REASONS_AFTER_SHORTLIST.items()
        }
        if shortlist_phase.research_shortlist:
            empty_reasons["research_shortlist"] = []
            empty_reasons["vault_candidates"] = [VAULT_NOT_RUN]
            empty_reasons["paper_candidates"] = [PAPER_NOT_RUN]
        else:
            empty_reasons["research_shortlist"] = [NOT_RESEARCH_SHORTLISTED]
        if not shortlist_phase.clusters:
            empty_reasons["clusters"] = [CLUSTERING_NOT_RUN]
        else:
            empty_reasons["clusters"] = []
        shortlist_payload = [e.as_dict() for e in shortlist_phase.research_shortlist]
        discovery = DiscoveryRunResult(
            discovery_run_id=self.discovery_run_id,
            budget_id="budget_multi_family",
            stop_reason=stop_reason,
            generations=completed_generation_count,
            evaluated=sum(s.evaluated for s in family_stats),
            registered_trials=len(self.registry.all_trials()),
            rankings=rankings,
            finalists=[],
            promoted=[],
            portfolio_pool={"members": [], "size": 0},
            clusters=list(shortlist_phase.clusters),
            reproducible_fingerprint="",
            pipeline_level=pipeline,
            post_wfo_pipeline_complete=False,
            score_qualified_meaning=SCORE_QUALIFIED_MEANING,
            empty_collections_reasons=empty_reasons,
            research_shortlist=shortlist_payload,
            vault_candidates=[],
            paper_candidates=[],
        )
        allocation_payload = {
            "generated": {fid: int(family_gen_caps[fid]) for fid in family_ids},
            "generated_cap_alloc": {
                fid: int(initial_alloc[fid]) + int(evolutionary_alloc[fid])
                for fid in family_ids
            },
            "initial_alloc": dict(initial_alloc),
            "evolutionary_alloc": dict(evolutionary_alloc),
            "full_wfo_initial": wfo_alloc,
            "full_wfo_early": wfo_alloc,
            "full_wfo_final": wfo_alloc,
            "adaptive_reallocation": bool(cfg.adaptive_reallocation),
            "allocation_detail": {},
            "min_candidates_per_family": cfg.min_candidates_per_family,
            "initial_candidates_per_family": initial_per,
            "total_candidate_budget": cfg.total_candidate_budget,
            "max_full_wfo": cfg.max_full_wfo,
            "family_local_evolution": True,
            "allow_cross_family_crossover": False,
            "pipeline_level": pipeline,
            "post_wfo_pipeline_complete": False,
            "stress_pipeline_complete": True,
            "robustness_pipeline_complete": True,
            "statistics_pipeline_complete": True,
            "clustering_pipeline_complete": True,
            "research_shortlist_pipeline_complete": True,
            "vault_pipeline_complete": False,
            "paper_pipeline_complete": False,
            "live_pipeline_complete": False,
            "completed_generations": completed_generation_count,
            "campaign_generated": campaign_generated,
            "campaign_evaluated": campaign_evaluated,
            "campaign_full_wfo": campaign_full_wfo,
            "initial_population_budget": int(initial_population_budget),
            "initial_candidates_generated": int(initial_candidates_generated),
            "evolutionary_candidate_budget": int(evolutionary_candidate_budget),
            "evolutionary_candidates_generated": int(evolutionary_candidates_generated),
            "unused_evolutionary_budget": int(unused_evolutionary_budget),
            "generated_total": int(campaign_generated),
            "generated_cap": int(cfg.total_candidate_budget),
            "generation_0_generated": int(initial_candidates_generated),
            "descendants_generated": int(evolutionary_candidates_generated),
            "mutation_children": int(sum(mutation_counts.values())),
            "crossover_children": int(sum(crossover_counts.values())),
            "highest_generation_reached": int(
                max(highest_generation.values()) if highest_generation else 0
            ),
            "per_family_generation_counts": {
                fid: {
                    "generation_0_generated": int(generation_0_counts[fid]),
                    "descendants_generated": int(descendant_counts[fid]),
                    "mutation_children": int(mutation_counts[fid]),
                    "crossover_children": int(crossover_counts[fid]),
                    "highest_generation_reached": int(highest_generation[fid]),
                    "structural_parents_found": int(structural_parents_found[fid]),
                    "generated_total": int(generated_counts[fid]),
                    "family_stop_reason": family_stop_reasons.get(fid),
                }
                for fid in family_ids
            },
            "family_stop_reasons": dict(family_stop_reasons),
            "stress_accounting": stress_accounting.as_dict(),
            "robustness_accounting": robustness_accounting.as_dict(),
            "statistics_accounting": shortlist_phase.statistics_accounting.as_dict(),
            "clustering_accounting": shortlist_phase.clustering_accounting.as_dict(),
            "search_program_id": self.search_program_id,
            "search_mode": self.search_mode,
            "compatibility_fingerprint": self.compatibility_fingerprint,
            "source_run_id": self.source_run_id,
            "resumed_from_run_id": self.resumed_from_run_id,
            "session_totals": {
                "generated": self._session_generated,
                "evaluated": self._session_evaluated,
                "full_wfo": self._session_full_wfo,
            },
            "cumulative_totals": {
                "generated": campaign_generated,
                "evaluated": campaign_evaluated,
                "full_wfo": campaign_full_wfo,
            },
            "evaluation_cache_hits": self._cache_hits,
            "evaluation_cache_misses": self._cache_misses,
            "pending_by_gate": (
                self._live_checkpoint.pending_by_gate()
                if self._live_checkpoint is not None
                else {}
            ),
        }
        # Final atomic checkpoint — crash after this loses nothing completed.
        if self._live_checkpoint is not None:
            from discovery.search_program import PipelinePhase
            from discovery.search_resume import mark_score_qualified_pending

            ckpt = self._live_checkpoint
            ckpt.pipeline_phase = (
                PipelinePhase.COMPLETE.value
                if stop_reason
                not in {"max_runtime_seconds", "max_full_wfo", "max_evaluated_candidates"}
                else PipelinePhase.STRESS.value
            )
            if stop_reason == "max_runtime_seconds":
                mark_score_qualified_pending(ckpt)
                from discovery.search_resume import choose_resume_phase

                ckpt.pipeline_phase = choose_resume_phase(ckpt).value
            ckpt.campaign_generated = campaign_generated
            ckpt.campaign_evaluated = campaign_evaluated
            ckpt.campaign_full_wfo = campaign_full_wfo
            ckpt.session_generated = self._session_generated
            ckpt.session_evaluated = self._session_evaluated
            ckpt.session_full_wfo = self._session_full_wfo
            ckpt.stop_reason = stop_reason
            ckpt.generation_records = [g.as_dict() for g in generation_records]
            ckpt.stress_summaries = [s.as_dict() for s in self.candidate_stress_summaries]
            ckpt.robustness_summaries = [
                s.as_dict() for s in self.candidate_robustness_summaries
            ]
            ckpt.statistics_summaries = [
                s.as_dict() for s in shortlist_phase.statistics_summaries
            ]
            ckpt.clusters = list(shortlist_phase.clusters)
            ckpt.research_shortlist = list(shortlist_payload)
            ckpt.population_stats = dict(shortlist_phase.population_stats or {})
            ckpt.family_specs = [f.as_dict() for f in families]
            for cid, cand in cand_by_id.items():
                ckpt.candidates[cid] = cand.as_dict()
            for fid, recs in records_by_family.items():
                for rec in recs:
                    ckpt.evaluation_records[rec.candidate_id] = rec.as_dict()
            self._persist_live_checkpoint()

        fingerprint = sha256_json(
            {
                "families": [f.canonical_hash() for f in families],
                "allocation": allocation_payload,
                "config": cfg.as_dict(),
                "pipeline_level": pipeline,
                "generation_records": [g.as_dict() for g in generation_records],
                "candidate_stress_summaries": [
                    s.as_dict() for s in self.candidate_stress_summaries
                ],
                "candidate_robustness_summaries": [
                    s.as_dict() for s in self.candidate_robustness_summaries
                ],
                "research_shortlist_phase": shortlist_phase.reproducible_fingerprint_payload,
            }
        )
        discovery.reproducible_fingerprint = fingerprint
        return MultiFamilyCampaignResult(
            campaign_id=self.discovery_run_id,
            families=families,
            family_stats=family_stats,
            budget_allocation=allocation_payload,
            discovery_results=[discovery],
            aggregated_stop_reason=stop_reason,
            reproducible_fingerprint=fingerprint,
            pipeline_level=pipeline,
            post_wfo_pipeline_complete=False,
            stress_pipeline_complete=True,
            robustness_pipeline_complete=True,
            statistics_pipeline_complete=True,
            clustering_pipeline_complete=True,
            research_shortlist_pipeline_complete=True,
            vault_pipeline_complete=False,
            paper_pipeline_complete=False,
            live_pipeline_complete=False,
            score_qualified_meaning=SCORE_QUALIFIED_MEANING,
            score_qualified_does_not_mean=SCORE_QUALIFIED_DOES_NOT_MEAN,
            stress_passed_does_not_mean=STRESS_PASSED_DOES_NOT_MEAN,
            robustness_passed_does_not_mean=ROBUSTNESS_PASSED_DOES_NOT_MEAN,
            research_shortlisted_does_not_mean=RESEARCH_SHORTLISTED_DOES_NOT_MEAN,
            post_wfo_blocked_reasons=list(POST_SHORTLIST_BLOCKED_REASONS),
            empty_collections_reasons=empty_reasons,
            research_shortlist=shortlist_payload,
            vault_candidates=[],
            paper_candidates=[],
            generation_records=generation_records,
            candidate_status_history=list(self.candidate_status_history),
            candidate_stress_summaries=list(self.candidate_stress_summaries),
            stress_accounting=stress_accounting,
            candidate_robustness_summaries=list(self.candidate_robustness_summaries),
            robustness_accounting=robustness_accounting,
            candidate_statistics_summaries=list(shortlist_phase.statistics_summaries),
            statistics_accounting=shortlist_phase.statistics_accounting,
            clusters=list(shortlist_phase.clusters),
            behavioral_signatures=list(shortlist_phase.signatures),
            clustering_accounting=shortlist_phase.clustering_accounting,
            shortlist_rejects=list(shortlist_phase.shortlist_rejects),
            population_stats=dict(shortlist_phase.population_stats),
        )

    def run(self) -> MultiFamilyCampaignResult:
        cfg = self.config
        # Cross-family crossover remains explicitly unsupported.
        if cfg.allow_cross_family_crossover:
            raise RuntimeError(CROSS_FAMILY_CROSSOVER_UNSUPPORTED)

        families = self.family_generator.generate(
            count=cfg.requested_family_count,
            family_ids=cfg.family_ids,
            provenance={"campaign_id": self.discovery_run_id},
        )
        assert_diverse_family_grammars(families)
        for spec in families:
            gen = CandidateGenerator.from_family_spec(spec)
            if gen.strategy_family == "dsl_generated":
                raise RuntimeError(
                    f"FAMILY_LABEL_COLLAPSE: {spec.family_id!r} resolved to dsl_generated"
                )

        family_ids = [f.family_id for f in families]
        if cfg.family_local_evolution:
            evo_split = evolution_budget_split(
                family_ids=family_ids,
                total_budget=cfg.total_candidate_budget,
                initial_per_family=cfg.effective_initial_candidates_per_family(),
            )
            gen_alloc = dict(evo_split["generated_cap_alloc"])
            initial_alloc = dict(evo_split["initial_alloc"])
            evolutionary_alloc = dict(evo_split["evolutionary_alloc"])
            initial_population_budget = int(evo_split["initial_population_budget"])
            evolutionary_candidate_budget = int(evo_split["evolutionary_candidate_budget"])
        else:
            gen_alloc = equal_initial_allocation(
                family_ids=family_ids,
                total_budget=cfg.total_candidate_budget,
                min_per_family=cfg.min_candidates_per_family,
            )
            initial_alloc = {}
            evolutionary_alloc = {}
            initial_population_budget = 0
            evolutionary_candidate_budget = 0
        wfo_floor = 1 if cfg.max_full_wfo >= len(family_ids) else 0
        initial_wfo = equal_wfo_allocation(
            family_ids=family_ids,
            max_full_wfo=cfg.max_full_wfo,
            min_per_family=wfo_floor,
        )

        if cfg.family_local_evolution:
            # Exact initial population + reserved evolutionary slots.
            # Adaptive reallocation may move unused evolutionary remainder only
            # to families with active structural parents.
            return self._run_family_local_evolution(
                families=families,
                gen_alloc=gen_alloc,
                wfo_alloc=initial_wfo,
                initial_alloc=initial_alloc,
                evolutionary_alloc=evolutionary_alloc,
                initial_population_budget=initial_population_budget,
                evolutionary_candidate_budget=evolutionary_candidate_budget,
            )

        self._emit(
            "MULTI_FAMILY_STARTED",
            {"families": family_ids, "gen_alloc": gen_alloc, "wfo_alloc": initial_wfo},
        )

        # --- Generate family-constrained candidates (no WFO yet) ---
        pool: dict[str, list[Any]] = {fid: [] for fid in family_ids}
        for spec in families:
            fid = spec.family_id
            generator = CandidateGenerator.from_family_spec(spec)
            target = int(gen_alloc[fid])
            attempts = 0
            seen: set[str] = set()
            while len(pool[fid]) < target and attempts < target * 8:
                attempts += 1
                seed_i = stable_seed(cfg.seed, fid, attempts, salt=997)
                try:
                    cand = generator.generate(seed=seed_i)
                except Exception:  # noqa: BLE001 — keep generating under budget
                    continue
                if cand.strategy_family == "dsl_generated":
                    raise RuntimeError(
                        f"FAMILY_LABEL_COLLAPSE: generated dsl_generated under {fid!r}"
                    )
                if cand.candidate_id in seen:
                    continue
                seen.add(cand.candidate_id)
                pool[fid].append(cand)
                self._emit(
                    "CANDIDATE_GENERATED",
                    {
                        "candidate_id": cand.candidate_id,
                        "family": cand.strategy_family,
                        "family_id": fid,
                    },
                )
            if len(pool[fid]) < cfg.min_candidates_per_family:
                raise RuntimeError(
                    f"FAMILY_GENERATION_SHORTFALL: {fid} produced {len(pool[fid])} "
                    f"< min_candidates_per_family={cfg.min_candidates_per_family}"
                )
            # Persist generated-only visibility in the registry (not yet Full-WFO'd).
            ev_reg = self._make_evaluator(spec=spec, wfo_cap=1, gen_cap=len(pool[fid]))
            for cand in pool[fid]:
                ev_reg._register(  # noqa: SLF001
                    cand,
                    rejection_reason=None,
                    ranking_score=None,
                    net_metrics={"generated_only": True, "awaiting_full_wfo": True},
                    extra_snapshot={
                        "family_provenance": dict(cand.family_provenance or {}),
                        "multi_family_phase": "generated",
                    },
                )

        # --- Early WFO: >=2 candidates/family when budget permits ---
        early_floor = wfo_floor
        if cfg.adaptive_reallocation and cfg.max_full_wfo >= len(family_ids) * 2:
            early_floor = 2
        elif cfg.max_full_wfo >= len(family_ids):
            early_floor = max(wfo_floor, 1)
        early_wfo = (
            {fid: early_floor for fid in family_ids}
            if cfg.adaptive_reallocation and cfg.max_full_wfo > len(family_ids) * early_floor
            else dict(initial_wfo)
        )
        # Cap early total to max_full_wfo.
        early_total = sum(early_wfo.values())
        if early_total > cfg.max_full_wfo:
            early_wfo = equal_wfo_allocation(
                family_ids=family_ids,
                max_full_wfo=cfg.max_full_wfo,
                min_per_family=wfo_floor,
            )

        records_by_family: dict[str, list[Any]] = {fid: [] for fid in family_ids}
        evaluated_ids: dict[str, set[str]] = {fid: set() for fid in family_ids}
        full_wfo_counts: dict[str, int] = {fid: 0 for fid in family_ids}
        early_scores: dict[str, float] = {fid: float("-inf") for fid in family_ids}
        early_candidate_ids: dict[str, list[str]] = {fid: [] for fid in family_ids}

        def _evaluate_quota(fid: str, spec: FamilySpec, quota: int) -> None:
            if quota <= 0:
                return
            remaining = [c for c in pool[fid] if c.candidate_id not in evaluated_ids[fid]]
            take = remaining[:quota]
            if not take:
                return
            ev = self._make_evaluator(
                spec=spec, wfo_cap=quota, gen_cap=max(len(pool[fid]), 1)
            )
            self._bind_evaluator_features(ev)
            for cand in take:
                # Authoritative counter delta: the evaluator only advances
                # counters.full_wfo when a real WFO evaluation completed and
                # its own returned artifacts prove it (signal_source ==
                # candidate_dsl_trees, is_full_event_wfo == True,
                # completed_fold_count > 0). This is the single source of
                # truth — outcomes like PRECHECK_FAILED, INVALID_DSL_TYPE,
                # DUPLICATE_SKIPPED, FEATURE_UNAVAILABLE, or an evaluation
                # exception never move this counter.
                rec = self._evaluate_candidate(
                    ev=ev, cand=cand, full_wfo_counts=full_wfo_counts, fid=fid
                )
                records_by_family[fid].append(rec)
                evaluated_ids[fid].add(cand.candidate_id)
                # FEATURE_UNAVAILABLE must not consume Full WFO budget or
                # count toward early-phase scoring/candidate tracking.
                if rec.outcome is EvalOutcome.FEATURE_UNAVAILABLE:
                    continue
                early_candidate_ids[fid].append(cand.candidate_id)
                if rec.fitness is not None and not rec.fitness.rejected:
                    early_scores[fid] = max(early_scores[fid], float(rec.fitness.fitness))
                elif rec.fitness is not None:
                    early_scores[fid] = max(early_scores[fid], float(rec.fitness.fitness) - 1e6)

        for spec in families:
            _evaluate_quota(spec.family_id, spec, int(early_wfo[spec.family_id]))
            self._emit(
                "FAMILY_PHASE_COMPLETED",
                {"family_id": spec.family_id, "phase": 1, "full_wfo": full_wfo_counts[spec.family_id]},
            )

        final_wfo = dict(initial_wfo)
        alloc_report: dict[str, dict[str, Any]] = {}
        if cfg.adaptive_reallocation and cfg.max_full_wfo > len(family_ids) * wfo_floor:
            final_wfo = adaptive_reallocate_wfo(
                family_ids=family_ids,
                max_full_wfo=cfg.max_full_wfo,
                early_scores=early_scores,
                min_quota=wfo_floor,
                early_counts=full_wfo_counts,
            )
            alloc_report = adaptive_allocation_report(
                family_ids=family_ids,
                initial_quota=early_wfo,
                final_quota=final_wfo,
                early_scores=early_scores,
                early_candidate_ids=early_candidate_ids,
                early_counts=full_wfo_counts,
            )
            for spec in families:
                fid = spec.family_id
                extra = int(final_wfo[fid]) - int(full_wfo_counts[fid])
                if extra > 0:
                    _evaluate_quota(fid, spec, extra)
                self._emit(
                    "FAMILY_PHASE_COMPLETED",
                    {"family_id": fid, "phase": 2, "full_wfo": full_wfo_counts[fid]},
                )

        family_stats = []
        for spec in families:
            st = self._stats_from_records(
                spec=spec,
                records=records_by_family[spec.family_id],
                generated=len(pool[spec.family_id]),
                allocation_generated=gen_alloc[spec.family_id],
                allocation_wfo=final_wfo[spec.family_id],
                full_wfo=full_wfo_counts[spec.family_id],
            )
            st.allocation_detail = dict(alloc_report.get(spec.family_id) or {})
            family_stats.append(st)

        rankings: list[dict[str, Any]] = []
        for st in family_stats:
            for cid in st.best_candidate_ids:
                rankings.append(
                    {
                        "candidate_id": cid,
                        "fitness": st.best_fitness,
                        "ranking_source": "validation_oos",
                        "family": st.family_id,
                    }
                )
        empty_reasons = {k: list(v) for k, v in EMPTY_COLLECTIONS_REASONS.items()}
        discovery = DiscoveryRunResult(
            discovery_run_id=self.discovery_run_id,
            budget_id="budget_multi_family",
            stop_reason="multi_family_wfo_screening_completed",
            generations=1,
            evaluated=sum(s.evaluated for s in family_stats),
            registered_trials=len(self.registry.all_trials()),
            rankings=rankings,
            # Post-WFO pipeline not implemented — keep empty with explicit reasons.
            finalists=[],
            promoted=[],
            portfolio_pool={"members": [], "size": 0},
            clusters=[],
            reproducible_fingerprint="",
            pipeline_level=PIPELINE_LEVEL_MULTI_FAMILY_WFO_SCREENING,
            post_wfo_pipeline_complete=False,
            score_qualified_meaning=SCORE_QUALIFIED_MEANING,
            empty_collections_reasons=empty_reasons,
            research_shortlist=[],
            vault_candidates=[],
            paper_candidates=[],
        )
        allocation_payload = {
            "generated": gen_alloc,
            "full_wfo_initial": initial_wfo,
            "full_wfo_early": early_wfo,
            "full_wfo_final": final_wfo,
            "adaptive_reallocation": cfg.adaptive_reallocation,
            "allocation_detail": alloc_report,
            "min_candidates_per_family": cfg.min_candidates_per_family,
            "total_candidate_budget": cfg.total_candidate_budget,
            "max_full_wfo": cfg.max_full_wfo,
            "family_local_evolution": False,
            "allow_cross_family_crossover": False,
            "pipeline_level": PIPELINE_LEVEL_MULTI_FAMILY_WFO_SCREENING,
            "post_wfo_pipeline_complete": False,
        }
        fingerprint = sha256_json(
            {
                "families": [f.canonical_hash() for f in families],
                "allocation": allocation_payload,
                "config": cfg.as_dict(),
                "pipeline_level": PIPELINE_LEVEL_MULTI_FAMILY_WFO_SCREENING,
            }
        )
        discovery.reproducible_fingerprint = fingerprint
        return MultiFamilyCampaignResult(
            campaign_id=self.discovery_run_id,
            families=families,
            family_stats=family_stats,
            budget_allocation=allocation_payload,
            discovery_results=[discovery],
            aggregated_stop_reason="multi_family_wfo_screening_completed",
            reproducible_fingerprint=fingerprint,
            pipeline_level=PIPELINE_LEVEL_MULTI_FAMILY_WFO_SCREENING,
            post_wfo_pipeline_complete=False,
            score_qualified_meaning=SCORE_QUALIFIED_MEANING,
            score_qualified_does_not_mean=SCORE_QUALIFIED_DOES_NOT_MEAN,
            post_wfo_blocked_reasons=list(POST_WFO_BLOCKED_REASONS),
            empty_collections_reasons=empty_reasons,
            research_shortlist=[],
            vault_candidates=[],
            paper_candidates=[],
        )


def family_campaign_from_config(raw: dict[str, Any] | None) -> FamilyCampaignConfig | None:
    if not raw or not raw.get("enabled"):
        return None
    return FamilyCampaignConfig(
        requested_family_count=int(raw.get("requested_family_count", raw.get("family_count", 6))),
        min_candidates_per_family=int(raw.get("min_candidates_per_family", 10)),
        initial_candidates_per_family=(
            int(raw["initial_candidates_per_family"])
            if raw.get("initial_candidates_per_family") is not None
            else None
        ),
        total_candidate_budget=int(raw.get("total_candidate_budget", 60)),
        max_full_wfo=int(raw.get("max_full_wfo", raw.get("max_full_wfo_evaluations", 18))),
        adaptive_reallocation=bool(raw.get("adaptive_reallocation", True)),
        seed=int(raw.get("seed", 42)),
        family_ids=list(raw["family_ids"]) if raw.get("family_ids") else None,
        max_evaluated_candidates=(
            int(raw["max_evaluated_candidates"])
            if raw.get("max_evaluated_candidates") is not None
            else None
        ),
        max_runtime_seconds=float(raw.get("max_runtime_seconds", 600)),
        min_oos_trades=int(raw.get("min_oos_trades", 1)),
        min_oos_trades_per_fold=int(raw.get("min_oos_trades_per_fold", 1)),
        max_oos_drawdown=float(raw.get("max_oos_drawdown", raw.get("max_drawdown_limit", 0.20))),
        population_size=int(raw.get("population_size", 2)),
        stagnation_generations=int(raw.get("stagnation_generations", 99)),
        family_local_evolution=bool(raw.get("family_local_evolution", False)),
        evolution_generations=int(raw.get("evolution_generations", 2)),
        allow_cross_family_crossover=bool(raw.get("allow_cross_family_crossover", False)),
        minimum_improvement=float(raw.get("minimum_improvement", 1e-4)),
        max_stress_evaluations=int(raw.get("max_stress_evaluations", 24)),
        max_stress_scenarios_per_candidate=int(
            raw.get("max_stress_scenarios_per_candidate", 10)
        ),
        min_stress_pass_rate=float(raw.get("min_stress_pass_rate", 0.5)),
        stress_scenarios=(
            tuple(str(s) for s in raw["stress_scenarios"])
            if raw.get("stress_scenarios") is not None
            else None
        ),
        fail_closed_unsupported_stress=bool(
            raw.get("fail_closed_unsupported_stress", True)
        ),
        max_robustness_candidates=int(raw.get("max_robustness_candidates", 2)),
        max_robustness_evaluations=int(raw.get("max_robustness_evaluations", 30)),
        max_parameters_per_candidate=int(raw.get("max_parameters_per_candidate", 2)),
        max_points_per_parameter=int(raw.get("max_points_per_parameter", 5)),
        allow_one_sided_neighborhood=bool(raw.get("allow_one_sided_neighborhood", False)),
        min_valid_neighborhood_points=int(raw.get("min_valid_neighborhood_points", 3)),
        min_dsr=float(raw.get("min_dsr", 0.95)),
        max_pbo=float(raw.get("max_pbo", 0.50)),
        pbo_n_splits=int(raw.get("pbo_n_splits", 4)),
        behavioral_similarity_threshold=float(
            raw.get("behavioral_similarity_threshold", 0.85)
        ),
        min_oos_observations_for_dsr=int(raw.get("min_oos_observations_for_dsr", 20)),
    )
