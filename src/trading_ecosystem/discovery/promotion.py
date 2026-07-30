"""Immutable Validation Vault evaluation (promotion-only)."""

from __future__ import annotations

from typing import Any

from trading_ecosystem.backtest.engine import BacktestEngine
from trading_ecosystem.common.contracts import Bar
from trading_ecosystem.discovery.multiple_testing import (
    deflated_sharpe_ratio,
    equity_to_returns,
    probability_of_backtest_overfitting,
    sharpe_ratio,
)
from trading_ecosystem.discovery.trial_ledger import load_trial_scores, register_vault_exposure
from trading_ecosystem.discovery.walk_forward import Fold, filter_panel
from trading_ecosystem.execution.costs import load_execution_costs
from trading_ecosystem.risk.master import MasterRiskManager
from trading_ecosystem.strategies.registry import build_strategy


def vault_folds(folds: list[Fold]) -> list[Fold]:
    return [f for f in folds if f.kind == "vault"]


def evaluate_vault_member(
    member: dict[str, Any],
    panel: dict[str, list[Bar]],
    vault_fold: Fold,
    *,
    starting_equity: float,
    universe_id: str | None = None,
    burn: bool = True,
) -> dict[str, Any]:
    c = member.get("candidate") or member
    if not c.get("class_name") or not c.get("strategy_id"):
        return {"ok": False, "reason": "missing_candidate"}
    strat = build_strategy(
        c["class_name"],
        c["strategy_id"],
        list(c.get("symbols") or []),
        dict(c.get("params") or {}),
    )
    sub = filter_panel(panel, vault_fold.start, vault_fold.end)
    costs = load_execution_costs(universe_id=universe_id)
    result = BacktestEngine(
        strategies=[strat],
        risk_manager=MasterRiskManager(),
        starting_equity=starting_equity,
        costs=costs,
    ).run(sub)
    rets = equity_to_returns(result.equity_curve)
    sr = sharpe_ratio(rets)
    trials = load_trial_scores(universe_id)
    dsr = deflated_sharpe_ratio(sr, n_trials=max(len(trials), 1), n_returns=max(len(rets), 2))
    pbo = probability_of_backtest_overfitting(trials) if trials else {"pbo": None}
    metrics = result.metrics.as_dict()
    out = {
        "ok": True,
        "strategy_id": c["strategy_id"],
        "class_name": c["class_name"],
        "vault_fold": {"name": vault_fold.name, "start": str(vault_fold.start), "end": str(vault_fold.end)},
        "metrics": metrics,
        "sharpe": sr,
        "dsr": dsr,
        "pbo": pbo,
    }
    if burn:
        out["vault_exposure"] = register_vault_exposure(
            strategy_id=str(c["strategy_id"]),
            class_name=str(c["class_name"]),
            params=dict(c.get("params") or {}),
            vault_metrics=metrics,
            universe_id=universe_id,
        )
    return out


def evaluate_vault_book(
    members: list[dict[str, Any]],
    panel: dict[str, list[Bar]],
    vault_fold: Fold,
    *,
    starting_equity: float,
    universe_id: str | None = None,
    burn: bool = True,
) -> dict[str, Any]:
    from trading_ecosystem.discovery.aggregate_book import run_combined_book

    sub = filter_panel(panel, vault_fold.start, vault_fold.end)
    metrics = run_combined_book(members, sub, starting_equity=starting_equity, universe_id=universe_id)
    member_reports = [
        evaluate_vault_member(
            m,
            panel,
            vault_fold,
            starting_equity=starting_equity / max(len(members), 1),
            universe_id=universe_id,
            burn=burn,
        )
        for m in members
    ]
    return {
        "ok": bool(metrics),
        "book_metrics": metrics,
        "members": member_reports,
        "vault_fold": {"name": vault_fold.name, "start": str(vault_fold.start), "end": str(vault_fold.end)},
        "note": "Vault metrics must not feed discovery fitness or invent weights.",
    }
