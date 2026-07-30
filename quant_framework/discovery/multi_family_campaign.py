"""Multi-family Alpha Miner campaign — allocate budget across FamilySpecs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable
from uuid import uuid4

from discovery.candidate import StrategyCandidate
from discovery.crossover import Crossover, CrossoverError
from discovery.evaluator import CandidateEvaluator, EvalOutcome, EvaluationRecord
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
from discovery.types import NodeKind
from registry.experiment_registry import ExperimentRegistry
from registry.hashing import sha256_json

STRUCTURAL_PARENT_ELIGIBLE = "STRUCTURAL_PARENT_ELIGIBLE"
SCORE_QUALIFIED = "SCORE_QUALIFIED"
NOT_PARENT_ELIGIBLE = "NOT_PARENT_ELIGIBLE"
FAMILY_DIRECTION_INCOHERENT = "FAMILY_DIRECTION_INCOHERENT"

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
    }
)
# Breakout context features inherit follow economics only inside breakout families.
_BREAKOUT_CONTEXT_FEATURES = frozenset(
    {
        "vol.range_compression_20",
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

# Honest capability declaration: MultiFamily currently screens via Full WFO only.
PIPELINE_LEVEL_MULTI_FAMILY_WFO_SCREENING = "MULTI_FAMILY_WFO_SCREENING"
PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_WFO_SCREENING = (
    "MULTI_FAMILY_EVOLUTIONARY_WFO_SCREENING"
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

POST_WFO_PIPELINE_NOT_RUN = "POST_WFO_PIPELINE_NOT_RUN"
STRESS_NOT_RUN = "STRESS_NOT_RUN"
ROBUSTNESS_NOT_RUN = "ROBUSTNESS_NOT_RUN"
CLUSTERING_NOT_RUN = "CLUSTERING_NOT_RUN"
NOT_RESEARCH_SHORTLISTED = "NOT_RESEARCH_SHORTLISTED"
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
    """Return ``'fade'``, ``'follow'``, or ``None`` if feature is not directional."""
    if feature_id in _FADE_FEATURES:
        # Gap and displacement features are fade economics regardless of family label.
        return "fade"
    if feature_id in _FOLLOW_FEATURES:
        return "follow"
    if feature_id in _BREAKOUT_CONTEXT_FEATURES and family_id == "breakout":
        return "follow"
    return None


def _expected_direction_from_comparison(
    *,
    mechanism: str,
    op_name: str,
    threshold: float | None,
) -> tuple[str | None, str]:
    """Map a signed comparison onto the economically required entry direction."""
    is_up = op_name in _UP_OPS
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
    evidence: list[dict[str, Any]] = []
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
        mechanism = _mechanism_for_feature(family_id, feature_id)
        if mechanism is None:
            continue
        if wrapped_abs:
            # ABS(signed_feature) alone cannot prove trade direction.
            abs_only_directional = True
            evidence.append(
                {
                    "condition_feature": feature_id,
                    "comparison_operator": node.name,
                    "threshold_value": _numeric_threshold(right),
                    "coherence_reason": "abs_signed_feature_ambiguous",
                }
            )
            continue
        thr = _numeric_threshold(right)
        expected, reason = _expected_direction_from_comparison(
            mechanism=mechanism,
            op_name=node.name,
            threshold=thr,
        )
        evidence.append(
            {
                "condition_feature": feature_id,
                "comparison_operator": node.name,
                "threshold_value": thr,
                "expected_direction": expected,
                "coherence_reason": reason,
            }
        )
        if expected is None:
            details = {**base_details, **evidence[-1]}
            details["coherence_reason"] = reason
            return DirectionCoherenceResult(
                coherent=False,
                rejection_reason=FAMILY_DIRECTION_INCOHERENT,
                details=details,
            )
        expected_votes.append(expected)

    if not expected_votes:
        reason = (
            "abs_signed_feature_without_directional_proof"
            if abs_only_directional
            else "no_provable_directional_condition"
        )
        details = {**base_details, "coherence_reason": reason}
        if evidence:
            details.update(evidence[0])
            details["coherence_reason"] = reason
        return DirectionCoherenceResult(
            coherent=False,
            rejection_reason=FAMILY_DIRECTION_INCOHERENT,
            details=details,
        )

    if len(set(expected_votes)) != 1:
        details = {
            **base_details,
            **(evidence[0] if evidence else {}),
            "expected_direction": sorted(set(expected_votes)),
            "coherence_reason": "conflicting_directional_conditions",
        }
        return DirectionCoherenceResult(
            coherent=False,
            rejection_reason=FAMILY_DIRECTION_INCOHERENT,
            details=details,
        )

    expected_direction = expected_votes[0]
    primary = evidence[0] if evidence else {}
    details = {
        **base_details,
        **primary,
        "expected_direction": expected_direction,
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
    CLUSTERING_NOT_RUN,
    NOT_RESEARCH_SHORTLISTED,
)

EMPTY_COLLECTIONS_REASONS: dict[str, list[str]] = {
    "finalists": [POST_WFO_PIPELINE_NOT_RUN, STRESS_NOT_RUN, ROBUSTNESS_NOT_RUN, CLUSTERING_NOT_RUN],
    "promoted": [POST_WFO_PIPELINE_NOT_RUN, STRESS_NOT_RUN, ROBUSTNESS_NOT_RUN],
    "clusters": [POST_WFO_PIPELINE_NOT_RUN, CLUSTERING_NOT_RUN],
    "research_shortlist": [POST_WFO_PIPELINE_NOT_RUN, NOT_RESEARCH_SHORTLISTED],
    "vault_candidates": [POST_WFO_PIPELINE_NOT_RUN, NOT_RESEARCH_SHORTLISTED],
    "paper_candidates": [POST_WFO_PIPELINE_NOT_RUN, NOT_RESEARCH_SHORTLISTED],
}


@dataclass
class FamilyCampaignConfig:
    requested_family_count: int = 6
    min_candidates_per_family: int = 10
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

    def as_dict(self) -> dict[str, Any]:
        return {
            "requested_family_count": self.requested_family_count,
            "min_candidates_per_family": self.min_candidates_per_family,
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
    best_fitness: float | None = None
    median_oos_expectancy: float | None = None
    median_pf: float | None = None
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    best_candidate_ids: list[str] = field(default_factory=list)
    constraints: dict[str, Any] = field(default_factory=dict)
    parent_status_counts: dict[str, int] = field(default_factory=dict)
    allocation_detail: dict[str, Any] = field(default_factory=dict)

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
            "best_fitness": self.best_fitness,
            "median_oos_expectancy": self.median_oos_expectancy,
            "median_pf": self.median_pf,
            "rejection_reasons": dict(self.rejection_reasons),
            "best_candidate_ids": list(self.best_candidate_ids),
            "constraints": dict(self.constraints),
            "parent_status_counts": dict(self.parent_status_counts),
            "allocation_detail": dict(self.allocation_detail),
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
    score_qualified_meaning: str = SCORE_QUALIFIED_MEANING
    score_qualified_does_not_mean: tuple[str, ...] = SCORE_QUALIFIED_DOES_NOT_MEAN
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

    def as_dict(self) -> dict[str, Any]:
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
            "score_qualified_meaning": self.score_qualified_meaning,
            "score_qualified_does_not_mean": list(self.score_qualified_does_not_mean),
            "post_wfo_blocked_reasons": list(self.post_wfo_blocked_reasons),
            "empty_collections_reasons": {
                k: list(v) for k, v in self.empty_collections_reasons.items()
            },
            "research_shortlist": list(self.research_shortlist),
            "vault_candidates": list(self.vault_candidates),
            "paper_candidates": list(self.paper_candidates),
            "generation_records": [g.as_dict() for g in self.generation_records],
            # Explicit empty post-WFO surfaces — never silent.
            "finalists": [],
            "promoted": [],
            "clusters": [],
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
    ) -> None:
        self.config = config
        self.registry = registry
        self.backend = backend
        self.system_version = system_version
        self.discovery_run_id = discovery_run_id or ("mfc_" + uuid4().hex[:16])
        self.progress_hook = progress_hook
        self.family_generator = StrategyFamilyGenerator(seed=config.seed)

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
                passed = sum(
                    1
                    for v in rec.stress_results.values()
                    if isinstance(v, dict) and v.get("passed") is True
                )
                if passed and passed == len(rec.stress_results):
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
        payload = {
            "rejection_reason": rejection_reason,
            "creation_kind": kind,
            **dict(details),
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
        prev_full_wfo = ev.counters.full_wfo
        rec = ev.evaluate(cand)
        if ev.counters.full_wfo > prev_full_wfo:
            full_wfo_counts[fid] = full_wfo_counts.get(fid, 0) + 1
        return rec

    def _generate_initial_population(
        self,
        *,
        spec: FamilySpec,
        target: int,
        seen: set[str],
    ) -> list[StrategyCandidate]:
        cfg = self.config
        generator = CandidateGenerator.from_family_spec(spec)
        pool: list[StrategyCandidate] = []
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
            if cand.candidate_id in seen:
                continue
            seen.add(cand.candidate_id)
            # Force generation index 0 for the initial queue.
            if cand.generation != 0:
                from discovery.candidate import build_candidate

                cand = build_candidate(
                    entry_tree=cand.entry_tree,
                    exit_tree=cand.exit_tree,
                    stop=cand.stop,
                    target=cand.target,
                    sizing=cand.sizing,
                    regime_gates=cand.regime_gates,
                    strategy_family=cand.strategy_family,
                    creation_method=cand.creation_method,
                    generation=0,
                    parent_ids=cand.parent_ids,
                    grammar_version=cand.grammar_version,
                    feature_set_version=cand.feature_set_version,
                    cost_model_version=cand.cost_model_version,
                    asset_universe=cand.asset_universe,
                    random_seed=cand.random_seed,
                    family_provenance=dict(cand.family_provenance or {}),
                )
                if cand.candidate_id in seen:
                    continue
                seen.add(cand.candidate_id)
            pool.append(cand)
            self._emit(
                "CANDIDATE_GENERATED",
                {
                    "candidate_id": cand.candidate_id,
                    "family": cand.strategy_family,
                    "family_id": spec.family_id,
                    "generation": 0,
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
    ) -> MultiFamilyCampaignResult:
        """Deterministic per-family generation queues with mutation/crossover."""
        cfg = self.config
        t0 = time.perf_counter()
        pipeline = PIPELINE_LEVEL_MULTI_FAMILY_EVOLUTIONARY_WFO_SCREENING
        family_ids = [f.family_id for f in families]
        records_by_family: dict[str, list[EvaluationRecord]] = {fid: [] for fid in family_ids}
        full_wfo_counts: dict[str, int] = {fid: 0 for fid in family_ids}
        generated_counts: dict[str, int] = {fid: 0 for fid in family_ids}
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
        stop_reason = "multi_family_evolutionary_wfo_screening_completed"
        completed_generation_count = 0

        self._emit(
            "MULTI_FAMILY_EVOLUTION_STARTED",
            {"families": family_ids, "gen_alloc": gen_alloc, "wfo_alloc": wfo_alloc},
        )

        for spec in families:
            fid = spec.family_id
            grammar = spec.to_grammar()
            mutator = Mutator(grammar=grammar)
            crossover = Crossover(grammar=grammar)
            family_gen_cap = int(gen_alloc[fid])
            family_wfo_cap = int(wfo_alloc[fid])
            initial_n = max(2, min(int(cfg.population_size), family_gen_cap))
            # Leave headroom for descendants when allocation allows.
            if family_gen_cap > initial_n + 1:
                initial_n = min(initial_n, max(2, family_gen_cap // 2))

            # --- Generation 0 queue (initial population only) ---
            gen_queues: dict[int, list[StrategyCandidate]] = {
                0: self._generate_initial_population(
                    spec=spec, target=initial_n, seen=seen_ids
                )
            }
            if len(gen_queues[0]) < 2:
                raise RuntimeError(
                    f"FAMILY_GENERATION_SHORTFALL: {fid} produced {len(gen_queues[0])} "
                    f"< 2 required for family-local evolution"
                )
            generated_counts[fid] = len(gen_queues[0])
            campaign_generated += len(gen_queues[0])
            for c in gen_queues[0]:
                cand_by_id[c.candidate_id] = c

            best_score_so_far = float("-inf")
            stagnation_count = 0
            family_stop: str | None = None
            population: list[StrategyCandidate] = list(gen_queues[0])
            rec_by_id: dict[str, EvaluationRecord] = {}

            max_gen_index = max(0, int(cfg.evolution_generations) - 1)
            for gen in range(0, max_gen_index + 1):
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
                take = queue[:eval_cap]
                greg.generated_count = len(queue)
                valid_eval_count = 0
                gen_scores: list[float] = []

                for cand in take:
                    if campaign_evaluated >= max_evaluated:
                        family_stop = "max_evaluated_candidates"
                        stop_reason = "max_evaluated_candidates"
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
                    greg.evaluated_candidate_ids.append(cand.candidate_id)
                    greg.evaluated_count += 1

                    if full_wfo_counts[fid] > prev_family_wfo:
                        delta = full_wfo_counts[fid] - prev_family_wfo
                        campaign_full_wfo += delta
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

                n_parents = min(max(2, int(cfg.population_size)), len(eligible_scored))
                parents, parent_reasons = self._select_parents_deterministic(
                    eligible=eligible_scored, n_parents=n_parents
                )
                greg.selected_parent_ids = [p.candidate_id for p in parents]
                greg.parent_selection_reasons = dict(parent_reasons)
                population = list(parents) if parents else list(population)

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
                    greg.stop_reason = family_stop or "budget_exhausted_before_breed"
                    generation_records.append(greg)
                    break

                breed_seed_base = int(stable_seed(cfg.seed, fid, next_gen, salt=4242) % (2**31 - 1))

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
                    seen_ids.add(child.candidate_id)
                    next_queue.append(child)
                    cand_by_id[child.candidate_id] = child
                    generated_counts[fid] += 1
                    campaign_generated += 1
                    if kind == "crossover":
                        greg.crossover_candidate_ids.append(child.candidate_id)
                    else:
                        greg.mutation_candidate_ids.append(child.candidate_id)
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

        family_stats = []
        for spec in families:
            st = self._stats_from_records(
                spec=spec,
                records=records_by_family[spec.family_id],
                generated=generated_counts[spec.family_id],
                allocation_generated=gen_alloc[spec.family_id],
                allocation_wfo=wfo_alloc[spec.family_id],
                full_wfo=full_wfo_counts[spec.family_id],
            )
            for reason, count in descendant_reject_counts.get(spec.family_id, {}).items():
                st.rejection_reasons[reason] = st.rejection_reasons.get(reason, 0) + int(count)
            family_stats.append(st)

        # Budget invariant assertions (soft: encode into stop / payload).
        assert campaign_generated <= cfg.total_candidate_budget + 0  # noqa: S101
        assert campaign_full_wfo <= cfg.max_full_wfo
        assert campaign_evaluated <= max_evaluated
        for fid in family_ids:
            assert generated_counts[fid] <= gen_alloc[fid]
            assert full_wfo_counts[fid] <= wfo_alloc[fid]

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
            stop_reason=stop_reason,
            generations=completed_generation_count,
            evaluated=sum(s.evaluated for s in family_stats),
            registered_trials=len(self.registry.all_trials()),
            rankings=rankings,
            finalists=[],
            promoted=[],
            portfolio_pool={"members": [], "size": 0},
            clusters=[],
            reproducible_fingerprint="",
            pipeline_level=pipeline,
            post_wfo_pipeline_complete=False,
            score_qualified_meaning=SCORE_QUALIFIED_MEANING,
            empty_collections_reasons=empty_reasons,
            research_shortlist=[],
            vault_candidates=[],
            paper_candidates=[],
        )
        allocation_payload = {
            "generated": gen_alloc,
            "full_wfo_initial": wfo_alloc,
            "full_wfo_early": wfo_alloc,
            "full_wfo_final": wfo_alloc,
            "adaptive_reallocation": False,
            "allocation_detail": {},
            "min_candidates_per_family": cfg.min_candidates_per_family,
            "total_candidate_budget": cfg.total_candidate_budget,
            "max_full_wfo": cfg.max_full_wfo,
            "family_local_evolution": True,
            "allow_cross_family_crossover": False,
            "pipeline_level": pipeline,
            "post_wfo_pipeline_complete": False,
            "completed_generations": completed_generation_count,
            "campaign_generated": campaign_generated,
            "campaign_evaluated": campaign_evaluated,
            "campaign_full_wfo": campaign_full_wfo,
        }
        fingerprint = sha256_json(
            {
                "families": [f.canonical_hash() for f in families],
                "allocation": allocation_payload,
                "config": cfg.as_dict(),
                "pipeline_level": pipeline,
                "generation_records": [g.as_dict() for g in generation_records],
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
            score_qualified_meaning=SCORE_QUALIFIED_MEANING,
            score_qualified_does_not_mean=SCORE_QUALIFIED_DOES_NOT_MEAN,
            post_wfo_blocked_reasons=list(POST_WFO_BLOCKED_REASONS),
            empty_collections_reasons=empty_reasons,
            research_shortlist=[],
            vault_candidates=[],
            paper_candidates=[],
            generation_records=generation_records,
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
        gen_alloc = equal_initial_allocation(
            family_ids=family_ids,
            total_budget=cfg.total_candidate_budget,
            min_per_family=cfg.min_candidates_per_family,
        )
        wfo_floor = 1 if cfg.max_full_wfo >= len(family_ids) else 0
        initial_wfo = equal_wfo_allocation(
            family_ids=family_ids,
            max_full_wfo=cfg.max_full_wfo,
            min_per_family=wfo_floor,
        )

        if cfg.family_local_evolution:
            # Evolution uses the same equal allocations; adaptive reallocation is
            # intentionally not applied inside the family-local generation loop.
            return self._run_family_local_evolution(
                families=families,
                gen_alloc=gen_alloc,
                wfo_alloc=initial_wfo,
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
    )
