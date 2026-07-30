"""Evolutionary search controller — Alpha Miner orchestration (Phase 6D)."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

import numpy as np

from discovery.behavioral_dedup import (
    BehavioralDeduper,
    signature_from_record,
)
from discovery.candidate import StrategyCandidate
from discovery.crossover import Crossover
from discovery.evaluator import CandidateEvaluator, EvalOutcome, EvaluationRecord
from discovery.fitness import RobustFitness
from discovery.freezing import FrozenCandidate, freeze_candidate
from discovery.generator import CandidateGenerator
from discovery.mutation import Mutator
from discovery.parameter_robustness import ParameterRobustness
from discovery.portfolio_candidates import PortfolioCandidate, PortfolioCandidatePool
from discovery.promotion import PromotionDecision, PromotionGate, PromotionStatus
from discovery.search_budget import (
    BUDGET_BUCKETS_EXHAUSTED,
    DUPLICATE_CANDIDATE_ID,
    GENERATION_DIVERSITY_EXHAUSTED,
    BudgetCounters,
    SearchBudget,
)
from discovery.selection import DiverseSelector, ScoredCandidate
from discovery.stress import StressTester, attach_stress
from discovery.expression_tree import DSLValidationError
from discovery.typecheck import InvalidDslTypeError
from discovery.vault_gateway import MinerVaultError
from registry.experiment_registry import ExperimentRegistry
from registry.hashing import sha256_json


@dataclass
class DiscoveryRunResult:
    discovery_run_id: str
    budget_id: str
    stop_reason: str
    generations: int
    evaluated: int
    registered_trials: int
    rankings: list[dict[str, Any]]
    finalists: list[str]
    promoted: list[dict[str, Any]]
    portfolio_pool: dict[str, Any]
    clusters: list[dict[str, Any]]
    reproducible_fingerprint: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "discovery_run_id": self.discovery_run_id,
            "budget_id": self.budget_id,
            "stop_reason": self.stop_reason,
            "generations": self.generations,
            "evaluated": self.evaluated,
            "registered_trials": self.registered_trials,
            "rankings": list(self.rankings),
            "finalists": list(self.finalists),
            "promoted": list(self.promoted),
            "portfolio_pool": dict(self.portfolio_pool),
            "clusters": list(self.clusters),
            "reproducible_fingerprint": self.reproducible_fingerprint,
        }


@dataclass
class SearchController:
    """
    Autonomous Alpha Miner loop.

    Never reads Vault data. Ranking uses OOS fitness only. Search stops when the
    frozen budget is exhausted or stagnation criteria fire.
    """

    registry: ExperimentRegistry
    budget: SearchBudget
    seed: int = 42
    system_version: str = "0.6.3-phase6d"
    data_hash: str = "discovery_data_v1"
    code_hash: str = "discovery_code_v1"
    discovery_run_id: str = field(default_factory=lambda: "disc_" + uuid4().hex[:16])
    generator: CandidateGenerator = field(default_factory=CandidateGenerator)
    mutator: Mutator = field(default_factory=Mutator)
    crossover: Crossover = field(default_factory=Crossover)
    selector: DiverseSelector = field(default_factory=DiverseSelector)
    deduper: BehavioralDeduper = field(default_factory=BehavioralDeduper)
    promotion_gate: PromotionGate = field(default_factory=PromotionGate)
    counters: BudgetCounters = field(default_factory=BudgetCounters)
    _vault_touch_attempted: bool = False
    # Optional instrumentation — None preserves quantitative behavior (Phase 12 dashboard)
    progress_hook: Callable[[str, dict[str, Any]], None] | None = None
    # UI canary: skip seed template; generate exactly canary_generate_seed once.
    canary_generate_seed: int | None = None
    skip_seed_template: bool = False

    def __post_init__(self) -> None:
        trade_fitness = RobustFitness(
            min_total_oos_trades=int(self.budget.min_oos_trades),
            min_oos_trades_per_fold=int(self.budget.min_oos_trades_per_fold),
            max_oos_drawdown=float(self.budget.max_oos_drawdown),
        )
        self.evaluator = CandidateEvaluator(
            registry=self.registry,
            budget=self.budget,
            counters=self.counters,
            fitness_model=trade_fitness,
            system_version=self.system_version,
            data_hash=self.data_hash,
            code_hash=self.code_hash,
            discovery_run_id=self.discovery_run_id,
            grammar=self.generator.grammar,
            progress_hook=self.progress_hook,
        )
        self.stress_tester = StressTester(
            budget=self.budget,
            counters=self.counters,
            fitness_model=trade_fitness,
        )
        self.robustness = ParameterRobustness(fitness_model=trade_fitness)

    def _emit_progress(self, event_name: str, payload: dict[str, Any] | None = None) -> None:
        if self.progress_hook is not None:
            self.progress_hook(event_name, payload or {})

    def assert_no_vault_access(self) -> None:
        """Hard guard — miner must never touch Vault frames."""
        if self._vault_touch_attempted:
            raise MinerVaultError("Miner attempted Vault data access during search")

    def _safe_generate(self, seed: int, *, prefer_seed_template: bool = False) -> StrategyCandidate | None:
        """Generate a candidate; INVALID_DSL_TYPE is recorded and returns None."""
        try:
            if prefer_seed_template:
                return self.generator.seed_template_mean_reversion(seed=seed)
            return self.generator.generate(seed=seed)
        except (DSLValidationError, InvalidDslTypeError) as exc:
            self.evaluator.register_invalid_dsl(exc, seed=seed, operation="generate")
            return None

    def _safe_mutate(self, parent: StrategyCandidate, *, seed: int) -> StrategyCandidate | None:
        try:
            return self.mutator.mutate(parent, seed=seed)
        except (DSLValidationError, InvalidDslTypeError) as exc:
            self.evaluator.register_invalid_dsl(
                exc,
                seed=seed,
                operation="mutation",
                parent_ids=(parent.candidate_id,),
            )
            return None

    def _safe_crossover(
        self,
        parent_a: StrategyCandidate,
        parent_b: StrategyCandidate,
        *,
        seed: int,
    ) -> tuple[StrategyCandidate, StrategyCandidate] | None:
        from discovery.crossover import CrossoverError

        try:
            return self.crossover.crossover(parent_a, parent_b, seed=seed)
        except (DSLValidationError, InvalidDslTypeError) as exc:
            self.evaluator.register_invalid_dsl(
                exc,
                seed=seed,
                operation="crossover",
                parent_ids=(parent_a.candidate_id, parent_b.candidate_id),
            )
            return None
        except CrossoverError:
            return None

    def run(self) -> DiscoveryRunResult:
        t0 = time.perf_counter()
        rng = np.random.default_rng(self.seed)
        population: list[ScoredCandidate] = []
        cand_by_id: dict[str, StrategyCandidate] = {}
        rec_by_id: dict[str, EvaluationRecord] = {}
        best_fitness = float("-inf")
        generations = 0
        stop_reason = "completed"
        # Seed template has a fixed AST → fixed candidate_id. Attempt it once only;
        # thereafter always use random generation while the population is empty.
        seed_template_attempted = False

        self._emit_progress("GENERATION_STARTED", {"generation": 0, "phase": "seed"})

        # Seed population
        canary_used = False
        while len(population) < self.budget.population_size:
            self.counters.runtime_seconds = time.perf_counter() - t0
            reason = self.counters.stop_reason(self.budget)
            if reason:
                stop_reason = reason
                break
            if self.counters.invalid >= self.budget.max_generated_candidates:
                stop_reason = "max_generated_candidates"
                break
            if self.canary_generate_seed is not None and not canary_used:
                seed_i = int(self.canary_generate_seed)
                prefer_seed = False
                canary_used = True
            else:
                seed_i = int(rng.integers(0, 1_000_000))
                prefer_seed = (not seed_template_attempted) and (not self.skip_seed_template)
                if prefer_seed:
                    seed_template_attempted = True
            cand = self._safe_generate(seed_i, prefer_seed_template=prefer_seed)
            if cand is None:
                # Replacement attempt while budget remains
                continue
            cap_reason = self.counters.record_generated(
                self.budget,
                family=cand.strategy_family,
                complexity=cand.complexity_score,
                feature_ids=cand.feature_ids,
                candidate_id=cand.candidate_id,
            )
            if cap_reason is not None:
                if cap_reason == DUPLICATE_CANDIDATE_ID:
                    # Visibility in registry; do not consume unique budget.
                    # Fresh-seed replacement continues on the next loop iteration.
                    self.evaluator.evaluate(cand)
                else:
                    self.evaluator._register(  # noqa: SLF001 — explicit budget reject path
                        cand,
                        rejection_reason=cap_reason,
                        ranking_score=None,
                        net_metrics={},
                    )
                    self._emit_progress(
                        "CANDIDATE_REJECTED",
                        {"candidate_id": cand.candidate_id, "reason": cap_reason},
                    )
                if self.counters.diversity_exhausted(self.budget):
                    stop_reason = GENERATION_DIVERSITY_EXHAUSTED
                    break
                if self.counters.buckets_exhausted(self.budget):
                    stop_reason = BUDGET_BUCKETS_EXHAUSTED
                    break
                continue
            self._emit_progress(
                "CANDIDATE_GENERATED",
                {
                    "candidate_id": cand.candidate_id,
                    "family": cand.strategy_family,
                    "generation": cand.generation,
                    "generated": self.counters.generated,
                    "generated_attempts": self.counters.generated_attempts,
                    "unique_generated": self.counters.unique_generated,
                    "duplicate_attempts": self.counters.duplicate_attempts,
                },
            )
            rec = self.evaluator.evaluate(cand)
            cand_by_id[cand.candidate_id] = cand
            # Keep the first non-duplicate evaluation record for this ID.
            prev = rec_by_id.get(cand.candidate_id)
            if prev is None or (
                prev.outcome is EvalOutcome.DUPLICATE_SKIPPED
                and rec.outcome is not EvalOutcome.DUPLICATE_SKIPPED
            ):
                rec_by_id[cand.candidate_id] = rec
            if rec.fitness is not None and rec.outcome is EvalOutcome.REGISTERED:
                sig = signature_from_record(cand, rec)
                population.append(
                    ScoredCandidate(candidate=cand, fitness=rec.fitness.fitness, signature=sig)
                )
            # If unique candidate did not enter population, keep generating
            # random replacements (seed already attempted at most once).

        while True:
            self.counters.runtime_seconds = time.perf_counter() - t0
            reason = self.counters.stop_reason(self.budget)
            if reason:
                stop_reason = reason
                break

            generations += 1
            self.counters.generations_completed = generations
            self._emit_progress("GENERATION_STARTED", {"generation": generations})
            if not population:
                stop_reason = "empty_population"
                break

            # Behavioral clustering
            sigs = [p.signature for p in population if p.signature is not None]
            clusters = self.deduper.cluster(sigs)
            for p in population:
                if p.signature is None:
                    continue
                for cl in clusters:
                    if p.candidate.candidate_id in cl.member_ids:
                        rec_by_id[p.candidate.candidate_id].behavioral_cluster = cl.cluster_id
            self._emit_progress(
                "BEHAVIORAL_CLUSTER_UPDATED",
                {"generation": generations, "clusters": [c.as_dict() for c in clusters]},
            )

            elites = self.selector.select_elites(population, n=self.budget.elite_count)
            gen_best = max(p.fitness for p in population)
            # Stagnation measures generations without meaningful improvement only
            # after the seed population exists — never from probe-only warm-up noise
            # before minimum_generations_before_stagnation (enforced in stop_reason).
            if gen_best > best_fitness + self.budget.convergence_fitness_delta:
                best_fitness = gen_best
                self.counters.stagnant_generations = 0
            else:
                self.counters.stagnant_generations += 1

            # Breed next generation
            n_pairs = max(1, (self.budget.population_size - len(elites)) // 2)
            pairs = self.selector.select_parents(population, n_pairs=n_pairs, rng=rng)
            children: list[StrategyCandidate] = list(elites)

            for i, (pa, pb) in enumerate(pairs):
                if self.counters.stop_reason(self.budget):
                    break
                crossed = self._safe_crossover(
                    pa, pb, seed=self.seed + generations * 100 + i
                )
                child_iter = crossed if crossed is not None else ()
                for child in child_iter:
                    if self.counters.generated >= self.budget.max_generated_candidates:
                        break
                    cap_reason = self.counters.record_generated(
                        self.budget,
                        family=child.strategy_family,
                        complexity=child.complexity_score,
                        feature_ids=child.feature_ids,
                        candidate_id=child.candidate_id,
                    )
                    if cap_reason is not None:
                        if cap_reason == DUPLICATE_CANDIDATE_ID:
                            self.evaluator.evaluate(child)
                        else:
                            self.evaluator._register(  # noqa: SLF001
                                child,
                                rejection_reason=cap_reason,
                                ranking_score=None,
                                net_metrics={},
                            )
                            self._emit_progress(
                                "CANDIDATE_REJECTED",
                                {
                                    "candidate_id": child.candidate_id,
                                    "reason": cap_reason,
                                },
                            )
                        if self.counters.diversity_exhausted(self.budget):
                            stop_reason = GENERATION_DIVERSITY_EXHAUSTED
                            break
                        if self.counters.buckets_exhausted(self.budget):
                            stop_reason = BUDGET_BUCKETS_EXHAUSTED
                            break
                        continue
                    self._emit_progress(
                        "CANDIDATE_GENERATED",
                        {
                            "candidate_id": child.candidate_id,
                            "family": child.strategy_family,
                            "generation": generations,
                            "generated": self.counters.generated,
                            "generated_attempts": self.counters.generated_attempts,
                            "unique_generated": self.counters.unique_generated,
                            "duplicate_attempts": self.counters.duplicate_attempts,
                        },
                    )
                    rec = self.evaluator.evaluate(child)
                    cand_by_id[child.candidate_id] = child
                    prev = rec_by_id.get(child.candidate_id)
                    if prev is None or (
                        prev.outcome is EvalOutcome.DUPLICATE_SKIPPED
                        and rec.outcome is not EvalOutcome.DUPLICATE_SKIPPED
                    ):
                        rec_by_id[child.candidate_id] = rec
                    if rec.fitness is not None and rec.outcome is EvalOutcome.REGISTERED:
                        children.append(child)

                # Mutation branch — on INVALID_DSL_TYPE, generate a replacement
                if self.counters.stop_reason(self.budget):
                    break
                mutant = self._safe_mutate(
                    pa, seed=self.seed + generations * 200 + i
                )
                if mutant is None:
                    mutant = self._safe_generate(self.seed + generations * 300 + i)
                if mutant is None:
                    continue
                mutant_cap_reason = self.counters.record_generated(
                    self.budget,
                    family=mutant.strategy_family,
                    complexity=mutant.complexity_score,
                    feature_ids=mutant.feature_ids,
                    candidate_id=mutant.candidate_id,
                )
                if mutant_cap_reason is None:
                    self._emit_progress(
                        "CANDIDATE_GENERATED",
                        {
                            "candidate_id": mutant.candidate_id,
                            "family": mutant.strategy_family,
                            "generation": generations,
                            "generated": self.counters.generated,
                            "generated_attempts": self.counters.generated_attempts,
                            "unique_generated": self.counters.unique_generated,
                            "duplicate_attempts": self.counters.duplicate_attempts,
                        },
                    )
                    rec = self.evaluator.evaluate(mutant)
                    cand_by_id[mutant.candidate_id] = mutant
                    prev = rec_by_id.get(mutant.candidate_id)
                    if prev is None or (
                        prev.outcome is EvalOutcome.DUPLICATE_SKIPPED
                        and rec.outcome is not EvalOutcome.DUPLICATE_SKIPPED
                    ):
                        rec_by_id[mutant.candidate_id] = rec
                    if rec.fitness is not None and rec.outcome is EvalOutcome.REGISTERED:
                        children.append(mutant)
                elif mutant_cap_reason == DUPLICATE_CANDIDATE_ID:
                    # Duplicate → attempt a fresh random replacement for this slot.
                    self.evaluator.evaluate(mutant)
                    replacement = self._safe_generate(
                        self.seed + generations * 400 + i + self.counters.generated_attempts
                    )
                    if replacement is None:
                        if self.counters.diversity_exhausted(self.budget):
                            stop_reason = GENERATION_DIVERSITY_EXHAUSTED
                            break
                        continue
                    repl_reason = self.counters.record_generated(
                        self.budget,
                        family=replacement.strategy_family,
                        complexity=replacement.complexity_score,
                        feature_ids=replacement.feature_ids,
                        candidate_id=replacement.candidate_id,
                    )
                    if repl_reason is None:
                        self._emit_progress(
                            "CANDIDATE_GENERATED",
                            {
                                "candidate_id": replacement.candidate_id,
                                "family": replacement.strategy_family,
                                "generation": generations,
                                "generated": self.counters.generated,
                                "generated_attempts": self.counters.generated_attempts,
                                "unique_generated": self.counters.unique_generated,
                                "duplicate_attempts": self.counters.duplicate_attempts,
                            },
                        )
                        rec = self.evaluator.evaluate(replacement)
                        cand_by_id[replacement.candidate_id] = replacement
                        rec_by_id[replacement.candidate_id] = rec
                        if rec.fitness is not None and rec.outcome is EvalOutcome.REGISTERED:
                            children.append(replacement)
                    elif repl_reason == DUPLICATE_CANDIDATE_ID:
                        self.evaluator.evaluate(replacement)
                    else:
                        self.evaluator._register(  # noqa: SLF001
                            replacement,
                            rejection_reason=repl_reason,
                            ranking_score=None,
                            net_metrics={},
                        )
                    if self.counters.diversity_exhausted(self.budget):
                        stop_reason = GENERATION_DIVERSITY_EXHAUSTED
                        break
                else:
                    self.evaluator._register(  # noqa: SLF001
                        mutant,
                        rejection_reason=mutant_cap_reason,
                        ranking_score=None,
                        net_metrics={},
                    )
                    self._emit_progress(
                        "CANDIDATE_REJECTED",
                        {
                            "candidate_id": mutant.candidate_id,
                            "reason": mutant_cap_reason,
                        },
                    )
                    if self.counters.diversity_exhausted(self.budget):
                        stop_reason = GENERATION_DIVERSITY_EXHAUSTED
                        break
                    if self.counters.buckets_exhausted(self.budget):
                        stop_reason = BUDGET_BUCKETS_EXHAUSTED
                        break

            # Rebuild scored population from children
            new_pop: list[ScoredCandidate] = []
            for ch in children:
                rec = rec_by_id.get(ch.candidate_id)
                if rec is None or rec.fitness is None:
                    continue
                if rec.outcome is not EvalOutcome.REGISTERED:
                    continue
                sig = signature_from_record(ch, rec)
                new_pop.append(ScoredCandidate(candidate=ch, fitness=rec.fitness.fitness, signature=sig))
            # Deduplicate by candidate_id keeping best fitness
            uniq: dict[str, ScoredCandidate] = {}
            for s in new_pop:
                prev = uniq.get(s.candidate.candidate_id)
                if prev is None or s.fitness > prev.fitness:
                    uniq[s.candidate.candidate_id] = s
            population = sorted(uniq.values(), key=lambda s: s.fitness, reverse=True)[
                : self.budget.population_size
            ]

        # Finalists: top unique behavior clusters
        scored = [
            ScoredCandidate(
                candidate=cand_by_id[cid],
                fitness=rec.fitness.fitness if rec.fitness else float("-inf"),
                signature=signature_from_record(cand_by_id[cid], rec)
                if cid in cand_by_id
                else None,
            )
            for cid, rec in rec_by_id.items()
            if rec.outcome is EvalOutcome.REGISTERED and rec.fitness is not None and cid in cand_by_id
        ]
        sigs = [s.signature for s in scored if s.signature is not None]
        clusters = self.deduper.cluster(sigs)
        rep_ids = {c.representative_id for c in clusters}
        finalists = [s for s in scored if s.candidate.candidate_id in rep_ids]
        finalists = sorted(finalists, key=lambda s: s.fitness, reverse=True)
        self._emit_progress(
            "FINALIST_SELECTED",
            {"finalists": [f.candidate.candidate_id for f in finalists], "count": len(finalists)},
        )
        self._emit_progress(
            "BEHAVIORAL_CLUSTER_UPDATED",
            {"clusters": [c.as_dict() for c in clusters], "phase": "final"},
        )

        frozen_list: list[FrozenCandidate] = []
        promoted: list[PromotionDecision] = []
        pool = PortfolioCandidatePool()

        for fin in finalists[: max(1, self.budget.max_vault_submissions + 2)]:
            cand = fin.candidate
            rec = rec_by_id[cand.candidate_id]
            assert rec.fitness is not None
            # SCORE_QUALIFIED gate: never stress fitness-rejected candidates.
            from discovery.fitness import SCORE_QUALIFIED_REJECT_REASONS

            if (
                rec.outcome is not EvalOutcome.REGISTERED
                or rec.rejection_reason in SCORE_QUALIFIED_REJECT_REASONS
                or (rec.fitness is not None and rec.fitness.rejected)
            ):
                continue

            # Stress + robustness (persist on record)
            stress = self.stress_tester.run(
                cand,
                base_fitness=rec.fitness.fitness,
                scenarios=(
                    "base_costs",
                    "costs_2x",
                    "costs_4x",
                    "wider_spread",
                    "worse_slippage",
                    "delayed_execution",
                    "removed_best_day",
                    "parameter_perturbation",
                ),
            )
            attach_stress(rec, stress)
            rob = self.robustness.probe(cand)
            rec.robustness_results = {r.parameter: r.as_dict() for r in rob}
            instability = self.robustness.instability_score(rob)

            # Re-score with stress/robustness penalties for promotion threshold
            pass_rate = sum(1 for s in stress if s.passed) / len(stress) if stress else 1.0
            frozen = freeze_candidate(
                cand,
                oos_fitness=rec.fitness.fitness,
                ranking_source=rec.fitness.ranking_source,
                discovery_run_id=self.discovery_run_id,
                code_hash=self.code_hash,
                data_hash=self.data_hash,
            )
            frozen_list.append(frozen)
            decision = self.promotion_gate.decide(
                frozen,
                stress_results=stress,
                train_score=rec.train_metrics.get("sharpe"),
            )
            promoted.append(decision)

            cluster_id = rec.behavioral_cluster or "beh_0"
            pool.add(
                PortfolioCandidate(
                    candidate=cand,
                    frozen=frozen,
                    fitness=rec.fitness.fitness - 0.1 * instability,
                    cluster_id=cluster_id,
                    stress_pass_rate=pass_rate,
                    robustness_ok=all(r.accepted for r in rob) if rob else True,
                )
            )

            if decision.status is PromotionStatus.PROMOTED:
                self.counters.vault_submissions += 1  # reservation toward vault cap

        rankings = sorted(
            [
                {
                    "candidate_id": s.candidate.candidate_id,
                    "fitness": s.fitness,
                    "ranking_source": "validation_oos",
                    "complexity": s.candidate.complexity_score,
                }
                for s in scored
            ],
            key=lambda r: r["fitness"],
            reverse=True,
        )

        fingerprint = sha256_json(
            {
                "seed": self.seed,
                "budget_id": self.budget.budget_id,
                "rankings": rankings,
                "finalists": [f.candidate.candidate_id for f in finalists],
            }
        )

        self.assert_no_vault_access()

        return DiscoveryRunResult(
            discovery_run_id=self.discovery_run_id,
            budget_id=self.budget.budget_id,
            stop_reason=stop_reason,
            generations=generations,
            evaluated=self.counters.evaluated,
            registered_trials=len(self.registry.all_trials()),
            rankings=rankings,
            finalists=[f.candidate.candidate_id for f in finalists],
            promoted=[p.as_dict() for p in promoted],
            portfolio_pool=pool.as_dict(),
            clusters=[c.as_dict() for c in clusters],
            reproducible_fingerprint=fingerprint,
        )
