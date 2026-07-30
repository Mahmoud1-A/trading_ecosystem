"""Import legacy strategies and force reevaluation from zero."""

from __future__ import annotations

from dataclasses import dataclass, field

from discovery.evaluator import CandidateEvaluator, EvalOutcome
from discovery.search_budget import BudgetCounters, SearchBudget
from legacy.candidate_adapter import adapt_legacy_record
from legacy.migration_report import MigrationReport
from legacy.records import LegacyStrategyRecord, make_legacy_book
from legacy.result_mapper import map_reevaluation
from registry.experiment_registry import ExperimentRegistry


@dataclass
class LegacyImporter:
    """
    Read-only bridge from the legacy archive into the new evaluation pipeline.

    Every import is reevaluated; no Book / Vault / fitness status is inherited.
    """

    registry: ExperimentRegistry
    budget: SearchBudget = field(default_factory=SearchBudget)
    feature_set_version: str = "feature_set_v1_phase6b"
    score_qualify_threshold: float = -1e9

    def import_and_reevaluate(
        self,
        records: list[LegacyStrategyRecord] | None = None,
        *,
        discovery_run_id: str = "legacy_migration",
    ) -> MigrationReport:
        records = records if records is not None else make_legacy_book()
        counters = BudgetCounters()
        # Generous budget so migration is not truncated mid-book in tests
        budget = self.budget.with_overrides(
            max_generated_candidates=max(self.budget.max_generated_candidates, len(records) + 8),
            max_evaluated_candidates=max(self.budget.max_evaluated_candidates, len(records) + 8),
            max_full_wfo_evaluations=max(self.budget.max_full_wfo_evaluations, len(records) + 8),
            max_candidates_per_family=max(self.budget.max_candidates_per_family, len(records) + 8),
        )
        evaluator = CandidateEvaluator(
            registry=self.registry,
            budget=budget,
            counters=counters,
            discovery_run_id=discovery_run_id,
        )

        report = MigrationReport(legacy_book_size_snapshot=len(records))
        seen_ids: set[str] = set()

        for i, record in enumerate(records):
            candidate = adapt_legacy_record(
                record,
                feature_set_version=self.feature_set_version,
                random_seed=1000 + i,
            )
            evaluation = evaluator.evaluate(candidate)
            mapped = map_reevaluation(record, candidate, evaluation)

            qualified = (
                evaluation.outcome is EvalOutcome.REGISTERED
                and evaluation.fitness is not None
                and evaluation.fitness.fitness >= self.score_qualify_threshold
            )
            unique = candidate.candidate_id not in seen_ids
            if unique:
                seen_ids.add(candidate.candidate_id)
            finalist = qualified and unique
            report.record_mapped(mapped, qualified=qualified, unique=unique, finalist=finalist)

        return report
