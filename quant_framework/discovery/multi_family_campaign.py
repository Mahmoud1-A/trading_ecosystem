"""Multi-family Alpha Miner campaign — allocate budget across FamilySpecs."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, Callable
from uuid import uuid4

from discovery.evaluator import CandidateEvaluator, EvalOutcome
from discovery.family_generator import StrategyFamilyGenerator
from discovery.family_spec import FamilySpec, assert_diverse_family_grammars
from discovery.fitness import RobustFitness
from discovery.generator import CandidateGenerator
from discovery.mutation import Mutator
from discovery.search_budget import BudgetCounters, SearchBudget
from discovery.search_controller import DiscoveryRunResult
from discovery.stable_hash import stable_seed
from registry.experiment_registry import ExperimentRegistry
from registry.hashing import sha256_json

STRUCTURAL_PARENT_ELIGIBLE = "STRUCTURAL_PARENT_ELIGIBLE"
SCORE_QUALIFIED = "SCORE_QUALIFIED"
NOT_PARENT_ELIGIBLE = "NOT_PARENT_ELIGIBLE"


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
class MultiFamilyCampaignResult:
    campaign_id: str
    families: list[FamilySpec]
    family_stats: list[FamilyStats]
    budget_allocation: dict[str, Any]
    discovery_results: list[DiscoveryRunResult]
    aggregated_stop_reason: str
    reproducible_fingerprint: str

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
            if rec.outcome is EvalOutcome.REGISTERED and rec.fitness is not None and not rec.fitness.rejected:
                score_qualified += 1
                status = SCORE_QUALIFIED
                fitnesses.append((float(rec.fitness.fitness), rec.candidate_id))
                comps = rec.fitness.components or {}
                if "pos_median_oos_expectancy" in comps:
                    expectancies.append(float(comps["pos_median_oos_expectancy"]))
                if "pos_oos_profit_factor" in comps:
                    pfs.append(float(comps["pos_oos_profit_factor"]))
            elif rec.fitness is not None and not rec.fitness.rejected:
                status = STRUCTURAL_PARENT_ELIGIBLE
            elif rec.outcome is EvalOutcome.REGISTERED:
                status = STRUCTURAL_PARENT_ELIGIBLE
            elif rec.rejection_reason in {
                "NEGATIVE_EXPECTANCY",
                "PF_BELOW_ONE",
                "INSUFFICIENT_OOS_TRADES",
                "MAX_DRAWDOWN_EXCEEDED",
            }:
                # Structural parent may still be eligible without Score Qualified.
                status = STRUCTURAL_PARENT_ELIGIBLE
            parent_status[status] = parent_status.get(status, 0) + 1
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

    def run(self) -> MultiFamilyCampaignResult:
        cfg = self.config
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
            avail = getattr(self.backend, "available_feature_ids", None)
            if avail is not None:
                ev.available_feature_ids = frozenset(avail)
                ev.dataset_capabilities = tuple(
                    getattr(self.backend, "dataset_capabilities", ()) or ()
                )
            for cand in take:
                # Authoritative counter delta: the evaluator only advances
                # counters.full_wfo when a real WFO evaluation completed and
                # its own returned artifacts prove it (signal_source ==
                # candidate_dsl_trees, is_full_event_wfo == True,
                # completed_fold_count > 0). This is the single source of
                # truth — outcomes like PRECHECK_FAILED, INVALID_DSL_TYPE,
                # DUPLICATE_SKIPPED, FEATURE_UNAVAILABLE, or an evaluation
                # exception never move this counter.
                prev_full_wfo = ev.counters.full_wfo
                rec = ev.evaluate(cand)
                records_by_family[fid].append(rec)
                evaluated_ids[fid].add(cand.candidate_id)
                if ev.counters.full_wfo > prev_full_wfo:
                    full_wfo_counts[fid] += 1
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

        # Optional family-local evolution (same FamilySpec grammar only).
        if cfg.family_local_evolution:
            for spec in families:
                fid = spec.family_id
                parents = [
                    c
                    for c in pool[fid]
                    if c.candidate_id in evaluated_ids[fid]
                ]
                # Prefer structural / score-qualified parents from records.
                parent_ids = {
                    rec.candidate_id
                    for rec in records_by_family[fid]
                    if rec.outcome
                    not in {EvalOutcome.FEATURE_UNAVAILABLE, EvalOutcome.INVALID_DSL_TYPE}
                }
                parents = [c for c in parents if c.candidate_id in parent_ids] or pool[fid][:2]
                mutator = Mutator(grammar=spec.to_grammar())
                for gen_i in range(int(cfg.evolution_generations)):
                    if len(pool[fid]) >= int(gen_alloc[fid]) + 4:
                        break
                    for pi, parent in enumerate(parents[: max(1, cfg.population_size)]):
                        child_seed = stable_seed(cfg.seed, fid, gen_i, pi, salt=333)
                        try:
                            child = mutator.mutate(parent, seed=child_seed)
                        except Exception:  # noqa: BLE001
                            continue
                        if child.strategy_family not in {fid, spec.family_id}:
                            continue
                        if child.candidate_id in {c.candidate_id for c in pool[fid]}:
                            continue
                        pool[fid].append(child)

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
        discovery = DiscoveryRunResult(
            discovery_run_id=self.discovery_run_id,
            budget_id="budget_multi_family",
            stop_reason="multi_family_completed",
            generations=1,
            evaluated=sum(s.evaluated for s in family_stats),
            registered_trials=len(self.registry.all_trials()),
            rankings=rankings,
            finalists=[],
            promoted=[],
            portfolio_pool={"members": [], "size": 0},
            clusters=[],
            reproducible_fingerprint="",
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
            "family_local_evolution": cfg.family_local_evolution,
        }
        fingerprint = sha256_json(
            {
                "families": [f.canonical_hash() for f in families],
                "allocation": allocation_payload,
                "config": cfg.as_dict(),
            }
        )
        discovery.reproducible_fingerprint = fingerprint
        return MultiFamilyCampaignResult(
            campaign_id=self.discovery_run_id,
            families=families,
            family_stats=family_stats,
            budget_allocation=allocation_payload,
            discovery_results=[discovery],
            aggregated_stop_reason="multi_family_completed",
            reproducible_fingerprint=fingerprint,
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
    )
