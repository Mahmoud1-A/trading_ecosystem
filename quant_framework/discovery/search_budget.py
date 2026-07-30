"""Frozen search budget — changing it creates a new discovery run."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from registry.hashing import sha256_json

SEARCH_BUDGET_VERSION = "search_budget_v1"


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
    max_candidates_per_family: int = 16
    max_candidates_per_complexity_tier: int = 16
    max_candidates_per_feature_family: int = 16
    max_full_wfo_evaluations: int = 32
    max_stress_evaluations: int = 8
    max_vault_submissions: int = 2
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
    per_family: dict[str, int] | None = None
    per_complexity_tier: dict[str, int] | None = None
    per_feature_family: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if self.per_family is None:
            self.per_family = {}
        if self.per_complexity_tier is None:
            self.per_complexity_tier = {}
        if self.per_feature_family is None:
            self.per_feature_family = {}

    def complexity_tier(self, score: float) -> str:
        if score < 5:
            return "low"
        if score < 12:
            return "mid"
        return "high"

    def feature_family(self, feature_ids: tuple[str, ...]) -> str:
        prefixes = sorted({fid.split(".", 1)[0] for fid in feature_ids if "." in fid})
        return "+".join(prefixes) if prefixes else "unknown"

    def stop_reason(self, budget: SearchBudget) -> str | None:
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

    def record_generated(
        self,
        budget: SearchBudget,
        *,
        family: str,
        complexity: float,
        feature_ids: tuple[str, ...],
    ) -> bool:
        """Increment generation counters if within per-bucket caps. Returns False if rejected."""
        self.generated_attempts += 1
        if not self.allows_family(budget, family):
            return False
        if not self.allows_complexity(budget, complexity):
            return False
        if not self.allows_feature_family(budget, feature_ids):
            return False
        assert self.per_family is not None
        assert self.per_complexity_tier is not None
        assert self.per_feature_family is not None
        self.generated += 1
        self.unique_generated += 1
        self.per_family[family] = self.per_family.get(family, 0) + 1
        tier = self.complexity_tier(complexity)
        self.per_complexity_tier[tier] = self.per_complexity_tier.get(tier, 0) + 1
        ff = self.feature_family(feature_ids)
        self.per_feature_family[ff] = self.per_feature_family.get(ff, 0) + 1
        return True
