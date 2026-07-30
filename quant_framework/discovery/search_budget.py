"""Frozen search budget — changing it creates a new discovery run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from registry.hashing import sha256_json

SEARCH_BUDGET_VERSION = "search_budget_v1"

# Exact per-bucket rejection reasons (never collapse to a generic string).
FAMILY_CAP = "FAMILY_CAP"
COMPLEXITY_TIER_CAP = "COMPLEXITY_TIER_CAP"
FEATURE_FAMILY_CAP = "FEATURE_FAMILY_CAP"
BUDGET_BUCKETS_EXHAUSTED = "budget_buckets_exhausted"
GENERATION_DIVERSITY_EXHAUSTED = "generation_diversity_exhausted"
DUPLICATE_CANDIDATE_ID = "DUPLICATE_CANDIDATE_ID"
_COMPLEXITY_TIERS = ("low", "mid", "high")


def _optional_int(cfg: dict[str, Any], key: str) -> int | None:
    if key not in cfg or cfg[key] is None:
        return None
    return int(cfg[key])


def resolve_bucket_caps(
    budget_cfg: dict[str, Any],
    *,
    max_generated_candidates: int,
) -> dict[str, Any]:
    """
    Resolve family/tier/feature-family caps from Search budget JSON.

    When a key is omitted, the effective limit equals ``max_generated_candidates``
    (never a hidden hardcode like 20).
    """
    requested = {
        "max_candidates_per_family": _optional_int(budget_cfg, "max_candidates_per_family"),
        "max_candidates_per_complexity_tier": _optional_int(
            budget_cfg, "max_candidates_per_complexity_tier"
        ),
        "max_candidates_per_feature_family": _optional_int(
            budget_cfg, "max_candidates_per_feature_family"
        ),
    }
    effective = {
        key: (val if val is not None else int(max_generated_candidates))
        for key, val in requested.items()
    }
    return {"requested": requested, "effective": effective}


def search_budget_from_config(budget_cfg: dict[str, Any] | None = None) -> SearchBudget:
    """Build a frozen :class:`SearchBudget` from control-plane Search budget JSON."""
    cfg = dict(budget_cfg or {})
    max_gen = int(cfg.get("max_generated_candidates", 12))
    pop_size = int(cfg.get("population_size", 4))
    stag_gens = cfg.get("stagnation_generations")
    if stag_gens is None:
        stag_gens = cfg.get("stagnation_limit", 8)
    min_before_stag = cfg.get("minimum_generations_before_stagnation")
    if min_before_stag is None:
        min_before_stag = max(1, pop_size)
    caps = resolve_bucket_caps(cfg, max_generated_candidates=max_gen)["effective"]
    return SearchBudget(
        max_generated_candidates=max_gen,
        max_evaluated_candidates=int(cfg.get("max_evaluated_candidates", 10)),
        max_full_wfo_evaluations=int(cfg.get("max_full_wfo_evaluations", 10)),
        max_stress_evaluations=int(cfg.get("max_stress_evaluations", 4)),
        population_size=pop_size,
        elite_count=int(cfg.get("elite_count", 1)),
        stagnation_limit=int(stag_gens),
        stagnation_generations=int(stag_gens),
        minimum_generations_before_stagnation=int(min_before_stag),
        max_runtime_seconds=float(cfg.get("max_runtime_seconds", 60)),
        max_candidates_per_family=int(caps["max_candidates_per_family"]),
        max_candidates_per_complexity_tier=int(caps["max_candidates_per_complexity_tier"]),
        max_candidates_per_feature_family=int(caps["max_candidates_per_feature_family"]),
        min_oos_trades=int(cfg.get("min_oos_trades", 8)),
        min_oos_trades_per_fold=int(cfg.get("min_oos_trades_per_fold", 1)),
        max_oos_drawdown=float(
            cfg.get("max_oos_drawdown", cfg.get("max_drawdown_limit", 0.20))
        ),
    )


@dataclass(frozen=True)
class SearchBudget:
    """
    Hard caps for an Alpha Miner discovery run.

    Freeze before execution. Mutating any field produces a new ``budget_id``
    and therefore a new discovery run identity.
    """

    max_generated_candidates: int = 64
    max_evaluated_candidates: int = 48
    max_runtime_seconds: float = 120.0
    max_cpu_seconds: float = 120.0
    max_memory_mb: float = 2048.0
    # Defaults equal max_generated_candidates when resolved from JSON via
    # search_budget_from_config / resolve_bucket_caps. Standalone dataclass
    # defaults mirror the historical SearchBudget generation default (64).
    max_candidates_per_family: int = 64
    max_candidates_per_complexity_tier: int = 64
    max_candidates_per_feature_family: int = 64
    max_full_wfo_evaluations: int = 32
    max_stress_evaluations: int = 8
    max_vault_submissions: int = 2
    # Closed OOS trade floors — zero-trade candidates cannot score-qualify.
    min_oos_trades: int = 8
    min_oos_trades_per_fold: int = 1
    # Absolute MaxDD ceiling (fraction); |worst fold MaxDD| must be <= this.
    max_oos_drawdown: float = 0.20
    stagnation_limit: int = 8
    # Explicit alias — generations without meaningful improvement before stop
    stagnation_generations: int | None = None
    # Do not apply stagnation until this many evolutionary generations completed
    # (default None → max(1, population_size) warm-up)
    minimum_generations_before_stagnation: int | None = None
    convergence_fitness_delta: float = 1e-4
    population_size: int = 8
    elite_count: int = 2
    version: str = SEARCH_BUDGET_VERSION

    def effective_stagnation_generations(self) -> int:
        if self.stagnation_generations is not None:
            return int(self.stagnation_generations)
        return int(self.stagnation_limit)

    def effective_minimum_generations_before_stagnation(self) -> int:
        if self.minimum_generations_before_stagnation is not None:
            return int(self.minimum_generations_before_stagnation)
        # Warm-up: at least the initial population generation requirement
        return max(1, int(self.population_size))

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def budget_id(self) -> str:
        return "budget_" + sha256_json(self.as_dict())[:24]

    def with_overrides(self, **kwargs: Any) -> SearchBudget:
        """Return a new frozen budget (new discovery run)."""
        payload = self.as_dict()
        payload.update(kwargs)
        return SearchBudget(**payload)


@dataclass
class BudgetCounters:
    """Mutable counters checked against a frozen :class:`SearchBudget`."""

    generated: int = 0
    generated_attempts: int = 0
    unique_generated: int = 0
    duplicate_attempts: int = 0
    evaluated: int = 0
    full_wfo: int = 0
    invalid: int = 0
    stress: int = 0
    vault_submissions: int = 0
    stagnant_generations: int = 0
    generations_completed: int = 0
    runtime_seconds: float = 0.0
    cpu_seconds: float = 0.0
    peak_memory_mb: float = 0.0
    # Consecutive FAMILY/COMPLEXITY/FEATURE cap rejections (reset on accept).
    bucket_cap_rejection_streak: int = 0
    # Consecutive duplicate candidate_id attempts (reset on unique accept).
    duplicate_rejection_streak: int = 0
    per_family: dict[str, int] | None = None
    per_complexity_tier: dict[str, int] | None = None
    per_feature_family: dict[str, int] | None = None
    _seen_candidate_ids: set[str] | None = None

    def __post_init__(self) -> None:
        if self.per_family is None:
            self.per_family = {}
        if self.per_complexity_tier is None:
            self.per_complexity_tier = {}
        if self.per_feature_family is None:
            self.per_feature_family = {}
        if self._seen_candidate_ids is None:
            self._seen_candidate_ids = set()

    def complexity_tier(self, score: float) -> str:
        if score < 5:
            return "low"
        if score < 12:
            return "mid"
        return "high"

    def feature_family(self, feature_ids: tuple[str, ...]) -> str:
        prefixes = sorted({fid.split(".", 1)[0] for fid in feature_ids if "." in fid})
        return "+".join(prefixes) if prefixes else "unknown"

    def max_generation_attempts(self, budget: SearchBudget) -> int:
        """Bounded attempt safety cap — prevents infinite duplicate loops."""
        cap = int(budget.max_generated_candidates)
        return max(cap * 3, cap + 32)

    def complexity_tiers_exhausted(self, budget: SearchBudget) -> bool:
        assert self.per_complexity_tier is not None
        return all(
            self.per_complexity_tier.get(tier, 0) >= budget.max_candidates_per_complexity_tier
            for tier in _COMPLEXITY_TIERS
        )

    def tracked_families_exhausted(self, budget: SearchBudget) -> bool:
        """True when every family seen so far is at its per-family cap."""
        assert self.per_family is not None
        if not self.per_family:
            return False
        return all(
            count >= budget.max_candidates_per_family for count in self.per_family.values()
        )

    def tracked_feature_families_exhausted(self, budget: SearchBudget) -> bool:
        assert self.per_feature_family is not None
        if not self.per_feature_family:
            return False
        return all(
            count >= budget.max_candidates_per_feature_family
            for count in self.per_feature_family.values()
        )

    def buckets_exhausted(self, budget: SearchBudget) -> bool:
        """Stop further generation when remaining bucket capacity is effectively zero."""
        streak_limit = max(16, int(budget.population_size) * 3)
        if self.bucket_cap_rejection_streak >= streak_limit:
            return True
        if self.complexity_tiers_exhausted(budget):
            return True
        # Cap rejects only — duplicates are tracked separately for diversity stop.
        cap_rejects = max(
            0, self.generated_attempts - self.generated - self.duplicate_attempts
        )
        if cap_rejects >= max(1, int(budget.max_generated_candidates)):
            return True
        return False

    def diversity_exhausted(self, budget: SearchBudget) -> bool:
        """True when replacement attempts cannot produce new unique candidate IDs."""
        if self.generated >= budget.max_generated_candidates:
            return False
        if self.generated_attempts >= self.max_generation_attempts(budget):
            return True
        dup_streak_limit = max(32, int(budget.population_size) * 8)
        if self.duplicate_rejection_streak >= dup_streak_limit:
            return True
        return False

    def stop_reason(self, budget: SearchBudget) -> str | None:
        # Unique candidate budget — never count duplicate attempts here.
        if self.generated >= budget.max_generated_candidates:
            return "max_generated_candidates"
        if self.evaluated >= budget.max_evaluated_candidates:
            return "max_evaluated_candidates"
        if self.full_wfo >= budget.max_full_wfo_evaluations:
            return "max_full_wfo_evaluations"
        if self.stress >= budget.max_stress_evaluations:
            return "max_stress_evaluations"
        if self.vault_submissions >= budget.max_vault_submissions:
            return "max_vault_submissions"
        if self.runtime_seconds >= budget.max_runtime_seconds:
            return "max_runtime_seconds"
        if self.cpu_seconds >= budget.max_cpu_seconds:
            return "max_cpu_seconds"
        if self.peak_memory_mb >= budget.max_memory_mb:
            return "max_memory_mb"
        if self.diversity_exhausted(budget):
            return GENERATION_DIVERSITY_EXHAUSTED
        if self.buckets_exhausted(budget):
            return BUDGET_BUCKETS_EXHAUSTED
        min_gens = budget.effective_minimum_generations_before_stagnation()
        stag_limit = budget.effective_stagnation_generations()
        if (
            self.generations_completed >= min_gens
            and self.stagnant_generations >= stag_limit
        ):
            return "stagnation_limit"
        return None

    def allows_family(self, budget: SearchBudget, family: str) -> bool:
        assert self.per_family is not None
        return self.per_family.get(family, 0) < budget.max_candidates_per_family

    def allows_complexity(self, budget: SearchBudget, score: float) -> bool:
        assert self.per_complexity_tier is not None
        tier = self.complexity_tier(score)
        return self.per_complexity_tier.get(tier, 0) < budget.max_candidates_per_complexity_tier

    def allows_feature_family(self, budget: SearchBudget, feature_ids: tuple[str, ...]) -> bool:
        assert self.per_feature_family is not None
        key = self.feature_family(feature_ids)
        return self.per_feature_family.get(key, 0) < budget.max_candidates_per_feature_family

    def cap_rejection_reason(
        self,
        budget: SearchBudget,
        *,
        family: str,
        complexity: float,
        feature_ids: tuple[str, ...],
    ) -> str | None:
        """Return exact cap reason if the candidate would be rejected; else None."""
        if not self.allows_family(budget, family):
            return FAMILY_CAP
        if not self.allows_complexity(budget, complexity):
            return COMPLEXITY_TIER_CAP
        if not self.allows_feature_family(budget, feature_ids):
            return FEATURE_FAMILY_CAP
        return None

    def record_generated(
        self,
        budget: SearchBudget,
        *,
        family: str,
        complexity: float,
        feature_ids: tuple[str, ...],
        candidate_id: str | None = None,
    ) -> str | None:
        """
        Increment generation counters if within per-bucket caps.

        ``max_generated_candidates`` counts **unique** candidate IDs only.
        Duplicate IDs increment ``duplicate_attempts`` / ``generated_attempts``
        but do not consume the unique-generation budget.

        Returns ``None`` on accept, otherwise an exact rejection reason:
        ``DUPLICATE_CANDIDATE_ID``, ``FAMILY_CAP``, ``COMPLEXITY_TIER_CAP``,
        or ``FEATURE_FAMILY_CAP``.
        """
        self.generated_attempts += 1
        assert self._seen_candidate_ids is not None
        if candidate_id and candidate_id in self._seen_candidate_ids:
            self.duplicate_attempts += 1
            self.duplicate_rejection_streak += 1
            return DUPLICATE_CANDIDATE_ID
        reason = self.cap_rejection_reason(
            budget,
            family=family,
            complexity=complexity,
            feature_ids=feature_ids,
        )
        if reason is not None:
            self.bucket_cap_rejection_streak += 1
            return reason
        assert self.per_family is not None
        assert self.per_complexity_tier is not None
        assert self.per_feature_family is not None
        self.bucket_cap_rejection_streak = 0
        self.duplicate_rejection_streak = 0
        self.generated += 1
        self.unique_generated += 1
        if candidate_id:
            self._seen_candidate_ids.add(candidate_id)
        self.per_family[family] = self.per_family.get(family, 0) + 1
        tier = self.complexity_tier(complexity)
        self.per_complexity_tier[tier] = self.per_complexity_tier.get(tier, 0) + 1
        ff = self.feature_family(feature_ids)
        self.per_feature_family[ff] = self.per_feature_family.get(ff, 0) + 1
        return None
