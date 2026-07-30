"""Real event-driven stress backends — no synthetic fold haircuts."""

from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import TYPE_CHECKING, Any, Callable

import pandas as pd

from config.cost_model import CostModel
from config.models import IntrabarAmbiguityPolicy
from config.walk_forward_config import WalkForwardConfig
from discovery.candidate import StrategyCandidate, build_candidate
from discovery.evaluator import BacktestBackend, SyntheticOOSBackend
from discovery.expression_tree import ExprNode, NodeKind, parameter_node
from discovery.fitness import FoldOOSMetrics
from discovery.stable_hash import stable_int_hash, stable_seed
from discovery.types import CreationMethod
from engine.event_execution import ExecutionConfig
from validation.event_driven_wfo import EventWFOContext

if TYPE_CHECKING:
    from discovery.event_wfo_backend import EventDrivenDiscoveryBackend

SYNTHETIC_STRESS_FORBIDDEN = "SYNTHETIC_STRESS_FORBIDDEN"
STRESS_BACKEND_KIND = "real_event_driven"
STRESS_BASELINE_TRADES_UNAVAILABLE = "STRESS_BASELINE_TRADES_UNAVAILABLE"
UNSUPPORTED_STRESS_SCENARIO = "UNSUPPORTED_STRESS_SCENARIO"
NOT_APPLICABLE_SINGLE_SYMBOL = "NOT_APPLICABLE_SINGLE_SYMBOL"
# Fraction (by count) of top realized net-PnL closed trades removed for the
# "removed_best_trades" scenario — distinct from "removed_best_day", which
# removes only the single highest realized net-PnL calendar day.
REMOVED_BEST_TRADES_FRACTION = 0.1
# Scenario names the real Stress path knows how to execute or explicitly
# mark not-applicable/unavailable. Anything else is UNSUPPORTED_STRESS_SCENARIO.
KNOWN_STRESS_SCENARIOS = frozenset(
    {
        "base_costs",
        "costs_2x",
        "costs_4x",
        "wider_spread",
        "worse_slippage",
        "delayed_execution",
        "conservative_intrabar",
        "reduced_participation",
        "alt_wfo_alignment",
        "alt_start_dates",
        "parameter_perturbation",
        "regime_exclusion",
        "removed_best_day",
        "removed_best_trades",
        "symbol_exclusion",
    }
)


def _copy_cost(cost: CostModel, **updates: Any) -> CostModel:
    data = cost.model_dump()
    data.update(updates)
    return CostModel(**data)


def _clone_event_backend(
    base: "EventDrivenDiscoveryBackend",
    *,
    cost_model: CostModel | None = None,
    exec_config: ExecutionConfig | None = None,
    wfo_config: WalkForwardConfig | None = None,
    bars: pd.DataFrame | None = None,
    backend_kind: str = STRESS_BACKEND_KIND,
) -> "EventDrivenDiscoveryBackend":
    from discovery.event_wfo_backend import EventDrivenDiscoveryBackend

    assert base._context is not None
    ctx = base._context
    new_ctx = EventWFOContext(
        asset=ctx.asset,
        cost_model=cost_model if cost_model is not None else ctx.cost_model,
        prop_profile=ctx.prop_profile,
        starting_equity=ctx.starting_equity,
        exec_config=exec_config if exec_config is not None else ctx.exec_config,
        bars_per_year=ctx.bars_per_year,
        freq=ctx.freq,
        run_id=ctx.run_id,
    )
    return EventDrivenDiscoveryBackend(
        bars=bars if bars is not None else base.bars,
        wfo_config=wfo_config if wfo_config is not None else base.wfo_config,
        backend_kind=backend_kind,
        is_full_event_wfo=True,
        require_real_bars=bool(base.require_real_bars),
        asset_class=base.asset_class,
        intraday_only=bool(base.intraday_only),
        artifact_dir=None,
        silver_resolution=dict(base.silver_resolution or {}),
        _context=new_ctx,
    )


def _replace_param(tree: ExprNode, name: str, value: float) -> ExprNode:
    kids = tuple(_replace_param(c, name, value) for c in tree.children)
    if tree.kind is NodeKind.PARAMETER and tree.name == name:
        return parameter_node(name, value)
    return ExprNode(tree.kind, tree.value_type, tree.name, kids, dict(tree.meta))


def _perturb_candidate(candidate: StrategyCandidate, scenario: str) -> StrategyCandidate:
    if not candidate.parameters:
        return candidate
    name = sorted(candidate.parameters)[0]
    base = float(candidate.parameters[name])
    delta = 0.05 if abs(base) < 1e-12 else 0.05 * abs(base)
    sign = 1.0 if (stable_int_hash(scenario + name) % 2 == 0) else -1.0
    val = base + sign * delta
    entry = _replace_param(candidate.entry_tree, name, val)
    exit_tree = (
        _replace_param(candidate.exit_tree, name, val) if candidate.exit_tree else None
    )
    return build_candidate(
        entry_tree=entry,
        exit_tree=exit_tree,
        stop=candidate.stop,
        target=candidate.target,
        sizing=candidate.sizing,
        regime_gates=candidate.regime_gates,
        strategy_family=candidate.strategy_family,
        creation_method=CreationMethod.MUTATION,
        generation=candidate.generation,
        parent_ids=(candidate.candidate_id,),
        grammar_version=candidate.grammar_version,
        feature_set_version=candidate.feature_set_version,
        cost_model_version=candidate.cost_model_version,
        asset_universe=candidate.asset_universe,
        random_seed=stable_seed(candidate.random_seed, name, scenario),
        family_provenance=dict(candidate.family_provenance or {}),
    )


def _drop_calendar_days(bars: pd.DataFrame, days: set[str]) -> pd.DataFrame:
    work = bars.copy()
    if "timestamp" in work.columns:
        ts = pd.to_datetime(work["timestamp"], utc=True)
    else:
        ts = pd.to_datetime(work.index, utc=True)
    mask = [ _day_key(t) not in days for t in ts ]
    out = work.loc[mask].reset_index(drop=True) if "timestamp" in work.columns else work.loc[mask]
    return out


@dataclass
class StressScenarioBackend:
    """Wraps a real EventDrivenDiscoveryBackend configured for one scenario."""

    scenario: str
    inner: Any
    cost_model_changes: dict[str, Any] = field(default_factory=dict)
    execution_changes: dict[str, Any] = field(default_factory=dict)
    candidate_transform: Callable[[StrategyCandidate], StrategyCandidate] | None = None
    backend_kind: str = STRESS_BACKEND_KIND
    is_full_event_wfo: bool = True

    def evaluate(self, candidate: StrategyCandidate) -> tuple[list[FoldOOSMetrics], dict[str, float]]:
        cand = self.candidate_transform(candidate) if self.candidate_transform else candidate
        return self.inner.evaluate(cand)


@dataclass
class StressScenarioStatus:
    """A Stress scenario that was NOT genuinely rerun against the real backend.

    Used for ``NOT_APPLICABLE_SINGLE_SYMBOL``, ``UNSUPPORTED_STRESS_SCENARIO``,
    and ``STRESS_BASELINE_TRADES_UNAVAILABLE``. Never calls a real backend and
    never fabricates fold metrics — :class:`StressTester` must detect
    ``stress_status`` and record it as excluded from pass/fail/executed and
    the pass-rate denominator, instead of silently rerunning the baseline.
    """

    scenario: str
    stress_status: str
    backend_kind: str = "stress_status_no_rerun"
    is_full_event_wfo: bool = False

    def evaluate(self, candidate: StrategyCandidate) -> tuple[list[FoldOOSMetrics], dict[str, float]]:
        return [], {"signal_source": "not_executed", "stress_status": self.stress_status}


def _day_key(ts: Any) -> str:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return str(t.floor("D"))


def _extract_closed_trades(baseline_artifacts: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Real baseline closed-trade records from the last real backend run.

    Returns an empty list (never a fabricated fallback) when unavailable so
    callers can honestly report ``STRESS_BASELINE_TRADES_UNAVAILABLE``.
    """
    if not baseline_artifacts:
        return []
    trades = baseline_artifacts.get("closed_trades")
    return list(trades) if trades else []


def _aggregate_daily_net_pnl(trades: list[dict[str, Any]]) -> dict[str, float]:
    """Realized net PnL aggregated by the calendar day each trade closed on."""
    daily: dict[str, float] = {}
    for t in trades:
        ts = t.get("exit_time")
        if not ts:
            continue
        day = _day_key(ts)
        daily[day] = daily.get(day, 0.0) + float(t.get("net_pnl", 0.0) or 0.0)
    return daily


def build_stress_backend_for_scenario(
    base: Any,
    scenario: str,
    *,
    baseline_artifacts: dict[str, Any] | None = None,
) -> StressScenarioBackend | StressScenarioStatus:
    """Build a real stress backend that mutates execution context, not metrics."""
    assert base._context is not None
    cost = base._context.cost_model
    exec_cfg = base._context.exec_config or ExecutionConfig()
    wfo = base.wfo_config
    bars = base.bars
    cost_changes: dict[str, Any] = {}
    exec_changes: dict[str, Any] = {}
    transform: Callable[[StrategyCandidate], StrategyCandidate] | None = None

    if scenario == "base_costs":
        pass
    elif scenario == "costs_2x":
        cost = _copy_cost(
            cost,
            commission_per_contract=float(cost.commission_per_contract) * 2.0,
            fixed_spread_ticks=float(cost.fixed_spread_ticks) * 2.0,
            slippage_ticks_mean=float(cost.slippage_ticks_mean) * 2.0,
            version=f"{cost.version}_stress_2x",
        )
        cost_changes = {"multiplier": 2.0}
    elif scenario == "costs_4x":
        cost = _copy_cost(
            cost,
            commission_per_contract=float(cost.commission_per_contract) * 4.0,
            fixed_spread_ticks=float(cost.fixed_spread_ticks) * 4.0,
            slippage_ticks_mean=float(cost.slippage_ticks_mean) * 4.0,
            version=f"{cost.version}_stress_4x",
        )
        cost_changes = {"multiplier": 4.0}
    elif scenario == "wider_spread":
        cost = _copy_cost(
            cost,
            fixed_spread_ticks=float(cost.fixed_spread_ticks) * 2.5,
            version=f"{cost.version}_wider_spread",
        )
        cost_changes = {"fixed_spread_ticks_mult": 2.5}
    elif scenario == "worse_slippage":
        cost = _copy_cost(
            cost,
            slippage_ticks_mean=float(cost.slippage_ticks_mean) * 3.0,
            slippage_ticks_std=float(cost.slippage_ticks_std) * 2.0,
            version=f"{cost.version}_worse_slippage",
        )
        cost_changes = {"slippage_mult": 3.0}
    elif scenario == "delayed_execution":
        # One bar of latency (freq-dependent; use 1 minute as minimum research delay).
        delay = timedelta(minutes=1) if "1" in str(base._context.freq) else timedelta(minutes=5)
        exec_cfg = replace(exec_cfg, latency=exec_cfg.latency + delay)
        exec_changes = {"latency_added_seconds": delay.total_seconds()}
    elif scenario == "conservative_intrabar":
        exec_cfg = replace(
            exec_cfg, intrabar_policy=IntrabarAmbiguityPolicy.CONSERVATIVE_WORST_CASE
        )
        exec_changes = {"intrabar_policy": "CONSERVATIVE_WORST_CASE"}
    elif scenario == "reduced_participation":
        new_cap = max(0.01, float(cost.participation_rate_cap) * 0.35)
        cost = _copy_cost(
            cost,
            participation_rate_cap=new_cap,
            version=f"{cost.version}_reduced_participation",
        )
        cost_changes = {"participation_rate_cap": new_cap}
    elif scenario == "alt_wfo_alignment" and wfo is not None:
        wfo = WalkForwardConfig(
            train_window_days=wfo.train_window_days,
            validation_window_days=wfo.validation_window_days,
            step_forward_days=max(1, int(wfo.step_forward_days) + 1),
            purge_gap_bars=int(wfo.purge_gap_bars) + 1,
            embargo_gap_bars=int(wfo.embargo_gap_bars) + 1,
            bars_per_day=wfo.bars_per_day,
            max_folds=wfo.max_folds,
            param_grid=dict(wfo.param_grid or {}),
            optimize_metric=wfo.optimize_metric,
        )
        exec_changes = {"step_forward_days": wfo.step_forward_days, "purge_gap_bars": wfo.purge_gap_bars}
    elif scenario == "alt_start_dates" and bars is not None:
        shift = min(max(5, len(bars) // 20), max(0, len(bars) - 50))
        bars = bars.iloc[shift:].reset_index(drop=True) if "timestamp" in bars.columns else bars.iloc[shift:]
        exec_changes = {"bars_dropped_from_start": shift}
    elif scenario == "parameter_perturbation":
        transform = lambda c, s=scenario: _perturb_candidate(c, s)  # noqa: E731
        exec_changes = {"parameter_perturbation": True}
    elif scenario == "removed_best_day":
        if bars is None:
            return StressScenarioStatus(scenario=scenario, stress_status=STRESS_BASELINE_TRADES_UNAVAILABLE)
        trades = _extract_closed_trades(baseline_artifacts)
        daily_pnl = _aggregate_daily_net_pnl(trades)
        if not daily_pnl:
            # Honest failure — never infer the "best day" from fold metrics,
            # OOS range boundaries, or an arbitrary middle date.
            return StressScenarioStatus(scenario=scenario, stress_status=STRESS_BASELINE_TRADES_UNAVAILABLE)
        best_day, best_day_pnl = max(daily_pnl.items(), key=lambda kv: kv[1])
        removed = [t for t in trades if t.get("exit_time") and _day_key(t["exit_time"]) == best_day]
        bars = _drop_calendar_days(bars, {best_day})
        exec_changes = {
            "removed_day": best_day,
            "baseline_day_net_pnl": best_day_pnl,
            "removed_trade_ids": [t.get("trade_id") for t in removed],
            "removed_trade_timestamps": [t.get("exit_time") for t in removed],
            "rerun_methodology": (
                "aggregated_realized_net_pnl_by_exit_day_from_baseline_closed_trades; "
                "removed_calendar_day_with_highest_net_pnl; reran_real_event_driven_backend"
            ),
        }
    elif scenario == "removed_best_trades":
        if bars is None:
            return StressScenarioStatus(scenario=scenario, stress_status=STRESS_BASELINE_TRADES_UNAVAILABLE)
        trades = _extract_closed_trades(baseline_artifacts)
        if not trades:
            return StressScenarioStatus(scenario=scenario, stress_status=STRESS_BASELINE_TRADES_UNAVAILABLE)
        ranked = sorted(trades, key=lambda t: float(t.get("net_pnl", 0.0) or 0.0), reverse=True)
        n_remove = max(1, int(round(len(ranked) * REMOVED_BEST_TRADES_FRACTION)))
        removed = ranked[:n_remove]
        removed_days = {_day_key(t["exit_time"]) for t in removed if t.get("exit_time")}
        if removed_days:
            bars = _drop_calendar_days(bars, removed_days)
        exec_changes = {
            "removed_trade_ids": [t.get("trade_id") for t in removed],
            "removed_trade_timestamps": [t.get("exit_time") for t in removed],
            "removed_fraction": REMOVED_BEST_TRADES_FRACTION,
            "baseline_removed_net_pnl": float(
                sum(float(t.get("net_pnl", 0.0) or 0.0) for t in removed)
            ),
            "rerun_methodology": (
                "ranked_baseline_closed_trades_by_realized_net_pnl_desc; "
                f"removed_top_{REMOVED_BEST_TRADES_FRACTION:.0%}_by_trade_count; "
                "dropped_calendar_days_containing_those_trade_exit_timestamps; "
                "reran_real_event_driven_backend; NOT_an_alias_for_removed_best_day"
            ),
        }
    elif scenario == "symbol_exclusion":
        # This codebase only ever runs single-symbol campaigns (asset_universe
        # is always a single instrument) — there is no multi-symbol universe
        # to defensibly exclude a member from.
        return StressScenarioStatus(scenario=scenario, stress_status=NOT_APPLICABLE_SINGLE_SYMBOL)
    elif scenario == "regime_exclusion" and bars is not None:
        # Drop bars in the strongest absolute trend tercile using causal features.
        from features.regime_features import compute_regime_features

        work = bars.copy()
        if "timestamp" in work.columns:
            work = work.set_index(pd.to_datetime(work["timestamp"], utc=True))
        try:
            reg = compute_regime_features(work)
            thr = float(reg["regime.trend_state"].abs().quantile(0.67))
            keep = reg["regime.trend_state"].abs() <= thr
            work = work.loc[keep]
            if "timestamp" in bars.columns:
                bars = work.reset_index().rename(columns={"index": "timestamp"})
                if "timestamp" not in bars.columns and work.index.name:
                    bars = work.reset_index()
            else:
                bars = work
            exec_changes = {"regime_exclusion_abs_trend_cap": thr}
        except Exception as exc:  # noqa: BLE001
            exec_changes = {"regime_exclusion_error": str(exc)}
    elif scenario not in KNOWN_STRESS_SCENARIOS:
        # Unknown scenario name — never silently rerun the unchanged baseline.
        return StressScenarioStatus(scenario=scenario, stress_status=UNSUPPORTED_STRESS_SCENARIO)

    inner = _clone_event_backend(
        base,
        cost_model=cost,
        exec_config=exec_cfg,
        wfo_config=wfo,
        bars=bars,
    )
    return StressScenarioBackend(
        scenario=scenario,
        inner=inner,
        cost_model_changes=cost_changes,
        execution_changes=exec_changes,
        candidate_transform=transform,
    )


def make_stress_backend_factory(
    base_backend: BacktestBackend,
    *,
    research_eligible: bool,
    synthetic_stress_forbidden: bool | None = None,
) -> Callable[[str], BacktestBackend]:
    """
    Factory for StressTester.

    Research-eligible runs must use EventDrivenDiscoveryBackend and never
    SyntheticOOSBackend.
    """
    forbid = (
        bool(synthetic_stress_forbidden)
        if synthetic_stress_forbidden is not None
        else bool(research_eligible)
    )

    from discovery.event_wfo_backend import EventDrivenDiscoveryBackend

    if isinstance(base_backend, SyntheticOOSBackend):
        if forbid or research_eligible:
            def _boom(scenario: str) -> BacktestBackend:
                raise RuntimeError(
                    f"{SYNTHETIC_STRESS_FORBIDDEN}: scenario={scenario} "
                    "research-eligible path cannot use SyntheticOOSBackend"
                )

            return _boom

        def _synthetic(scenario: str) -> BacktestBackend:
            salt = stable_int_hash(scenario, bits=32) % 10_000
            return SyntheticOOSBackend(seed_salt=salt)

        return _synthetic

    if not isinstance(base_backend, EventDrivenDiscoveryBackend):
        if forbid or research_eligible:
            def _boom2(scenario: str) -> BacktestBackend:
                raise RuntimeError(
                    f"{SYNTHETIC_STRESS_FORBIDDEN}: scenario={scenario} "
                    f"unsupported backend {type(base_backend).__name__}"
                )

            return _boom2

        def _fallback(scenario: str) -> BacktestBackend:
            return SyntheticOOSBackend(seed_salt=stable_int_hash(scenario, bits=32) % 10_000)

        return _fallback

    baseline_cache: dict[str, Any] = {"arts": None}

    def _factory(scenario: str) -> BacktestBackend:
        if forbid and isinstance(base_backend, SyntheticOOSBackend):
            raise RuntimeError(SYNTHETIC_STRESS_FORBIDDEN)
        arts = baseline_cache["arts"]
        built = build_stress_backend_for_scenario(
            base_backend, scenario, baseline_artifacts=arts
        )
        return built

    # Attach helper for StressTester to seed baseline artifacts after base_costs.
    _factory.base_event_backend = base_backend  # type: ignore[attr-defined]
    _factory.baseline_cache = baseline_cache  # type: ignore[attr-defined]
    _factory.research_eligible = research_eligible  # type: ignore[attr-defined]
    _factory.synthetic_stress_forbidden = forbid  # type: ignore[attr-defined]
    return _factory


def assert_not_synthetic_stress(backend: BacktestBackend, *, research_eligible: bool) -> None:
    if research_eligible and isinstance(backend, SyntheticOOSBackend):
        raise RuntimeError(SYNTHETIC_STRESS_FORBIDDEN)
    if research_eligible and getattr(backend, "backend_kind", "") == "synthetic_oos_probe":
        raise RuntimeError(SYNTHETIC_STRESS_FORBIDDEN)
