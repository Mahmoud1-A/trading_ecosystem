"""Stability gate + portfolio readiness report for real-market promotion."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

from trading_ecosystem.backtest.engine import BacktestEngine
from trading_ecosystem.common.config import CONFIG_DIR, load_yaml
from trading_ecosystem.common.contracts import Bar
from trading_ecosystem.discovery.universe import get_active_universe, processed_artifact
from trading_ecosystem.execution.costs import load_execution_costs
from trading_ecosystem.risk.master import MasterRiskManager
from trading_ecosystem.strategies.registry import build_strategy


def load_stability_cfg() -> dict[str, Any]:
    return load_yaml(CONFIG_DIR / "stability_gate.yaml")


def readiness_path(universe_id: str | None = None) -> Any:
    return processed_artifact("portfolio_readiness", universe_id=universe_id)


def load_readiness(universe_id: str | None = None) -> dict[str, Any]:
    path = readiness_path(universe_id)
    if not path.exists():
        return {
            "ready": False,
            "level": "not_ready",
            "universe": universe_id or get_active_universe(),
            "checks": [],
            "message": "No readiness report yet — wait for discovery generations",
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_readiness(payload: dict[str, Any], universe_id: str | None = None) -> Any:
    path = readiness_path(universe_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    out["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    return path


def _benchmark_symbol(universe_id: str, cfg: dict[str, Any]) -> str:
    benches = cfg.get("benchmarks") or {}
    return str(benches.get(universe_id) or benches.get("etf") or "SPY")


def classify_years(
    benchmark_bars: list[Bar],
    *,
    bull_year_return: float = 0.10,
    bear_year_return: float = -0.10,
) -> dict[int, dict[str, Any]]:
    """Label each calendar year from benchmark buy-hold return."""
    by_year: dict[int, list[Bar]] = defaultdict(list)
    for b in benchmark_bars:
        by_year[b.ts.year].append(b)
    out: dict[int, dict[str, Any]] = {}
    for year, bars in sorted(by_year.items()):
        if len(bars) < 20:
            continue
        bars = sorted(bars, key=lambda x: x.ts)
        start = float(bars[0].close)
        end = float(bars[-1].close)
        if start <= 0:
            continue
        ret = end / start - 1.0
        if ret >= bull_year_return:
            label = "bull"
        elif ret <= bear_year_return:
            label = "bear"
        else:
            label = "sideways"
        out[year] = {"label": label, "benchmark_return": ret, "bars": len(bars)}
    return out


def yearly_returns_from_equity(equity_curve: list[tuple]) -> dict[int, float]:
    if not equity_curve:
        return {}
    by_year: dict[int, list[tuple]] = defaultdict(list)
    for ts, eq in equity_curve:
        by_year[ts.year].append((ts, float(eq)))
    out: dict[int, float] = {}
    for year, pts in by_year.items():
        pts = sorted(pts, key=lambda x: x[0])
        if len(pts) < 5:
            continue
        start_eq = pts[0][1]
        end_eq = pts[-1][1]
        if start_eq <= 0:
            continue
        out[year] = end_eq / start_eq - 1.0
    return out


def _run_book_with_curve(
    members: list[dict[str, Any]],
    panel: dict[str, list[Bar]],
    *,
    starting_equity: float,
    universe_id: str,
    mode: str | None = None,
) -> tuple[dict[str, Any], list[tuple]]:
    mode_l = (mode or "").strip().lower()
    if mode_l in {"sector_champions", "champions", "per_coin", "sparse_heroes", "heroes"}:
        from trading_ecosystem.discovery.aggregate_book import run_equal_weight_champions

        return run_equal_weight_champions(
            members,
            panel,
            starting_equity=starting_equity,
            universe_id=universe_id,
        )

    strategies = []
    for e in members:
        c = e.get("candidate") or {}
        if not c.get("class_name") or not c.get("strategy_id"):
            continue
        strategies.append(
            build_strategy(
                c["class_name"],
                c["strategy_id"],
                list(c.get("symbols") or []),
                dict(c.get("params") or {}),
            )
        )
    if not strategies:
        return {}, []
    needed = sorted({s for st in strategies for s in st.symbols})
    sub = {s: panel.get(s, []) for s in needed}
    costs = load_execution_costs(universe_id=universe_id)
    result = BacktestEngine(
        strategies=strategies,
        risk_manager=MasterRiskManager(),
        starting_equity=starting_equity,
        costs=costs,
    ).run(sub)
    metrics = result.metrics.as_dict()
    metrics["members"] = len(strategies)
    metrics["mode"] = "aggregate_book"
    metrics["risk_rejects"] = result.risk_rejects
    metrics["cost_profile"] = costs.profile
    return metrics, list(result.equity_curve)


def _check(
    name: str,
    ok: bool,
    detail: str,
    *,
    hard: bool = True,
) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "hard": hard, "detail": detail}


def _member_fingerprint(members: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for m in members:
        c = m.get("candidate") or {}
        syms = ",".join(c.get("symbols") or [])
        parts.append(f"{c.get('strategy_id')}|{c.get('class_name')}|{syms}")
    return "\n".join(sorted(parts))


def evaluate_portfolio_readiness(
    members: list[dict[str, Any]],
    panel: dict[str, list[Bar]],
    *,
    universe_id: str | None = None,
    starting_equity: float = 100_000.0,
    portfolio_metrics: dict[str, Any] | None = None,
    prop_rules: dict[str, Any] | None = None,
    gate_cfg: dict[str, Any] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """
    Stability gate for promoting an aggregate book to paper/live.
    Re-simulates the combined book under the universe cost profile.
    Skips full re-sim when member fingerprint matches the last report (unless force=True).
    """
    cfg = gate_cfg or load_stability_cfg()
    prop = prop_rules or load_yaml(CONFIG_DIR / "prop_rules.yaml")
    uid = (universe_id or get_active_universe()).strip().lower()
    checks: list[dict[str, Any]] = []
    fingerprint = _member_fingerprint(members)

    from trading_ecosystem.discovery.universe import load_universe

    uni = load_universe(uid)
    book_mode = str(
        (portfolio_metrics or {}).get("mode")
        or uni.get("portfolio_mode")
        or "aggregate_book"
    ).strip().lower()

    if not members:
        payload = {
            "ready": False,
            "level": "not_ready",
            "universe": uid,
            "checks": [_check("has_members", False, "empty book", hard=True)],
            "message": "Empty aggregate book",
            "metrics": {},
            "regimes": {},
            "promotion": {"allow_freeze_paper": False, "allow_live": False},
            "fingerprint": "",
        }
        save_readiness(payload, uid)
        return payload

    if not force:
        prev = load_readiness(uid)
        if prev.get("fingerprint") == fingerprint and prev.get("checks"):
            # Same book composition — reuse expensive regime/sim work.
            out = dict(prev)
            out["skipped_resim"] = True
            out["fingerprint"] = fingerprint
            if portfolio_metrics:
                metrics = dict(out.get("metrics") or {})
                for k in (
                    "cagr",
                    "max_drawdown",
                    "num_trades",
                    "profit_factor",
                    "win_rate",
                    "book_score",
                    "members",
                ):
                    if portfolio_metrics.get(k) is not None:
                        metrics[k] = portfolio_metrics[k]
                out["metrics"] = metrics
            save_readiness(out, uid)
            return out

    metrics, equity_curve = _run_book_with_curve(
        members,
        panel,
        starting_equity=starting_equity,
        universe_id=uid,
        mode=book_mode,
    )
    # Prefer freshly simulated metrics; fall back to provided.
    if not metrics and portfolio_metrics:
        metrics = dict(portfolio_metrics)

    cagr = float(metrics.get("cagr") or 0.0)
    max_dd = float(metrics.get("max_drawdown") or 1.0)
    trades = int(metrics.get("num_trades") or 0)
    pf = float(metrics.get("profit_factor") or 0.0)
    win_rate = float(metrics.get("win_rate") or 0.0)
    book_score = float(
        (portfolio_metrics or {}).get("book_score")
        if portfolio_metrics and portfolio_metrics.get("book_score") is not None
        else metrics.get("book_score") or metrics.get("mar") or -1e9
    )
    n_members = len(members)
    prop_dd = float(prop.get("max_total_drawdown_pct", 10.0)) / 100.0
    max_dd_gate = min(float(cfg.get("max_book_dd", prop_dd)), prop_dd)

    checks.append(_check("min_members", n_members >= int(cfg.get("min_members", 5)), f"{n_members}"))
    checks.append(_check("min_book_trades", trades >= int(cfg.get("min_book_trades", 40)), f"{trades}"))
    checks.append(_check("min_book_cagr", cagr >= float(cfg.get("min_book_cagr", 0.0)), f"{cagr:.4f}"))
    checks.append(_check("max_book_dd", max_dd <= max_dd_gate + 1e-12, f"{max_dd:.4f}<={max_dd_gate:.4f}"))
    checks.append(_check("min_profit_factor", pf >= float(cfg.get("min_profit_factor", 1.05)), f"{pf:.3f}"))
    checks.append(_check("min_win_rate", win_rate >= float(cfg.get("min_win_rate", 0.20)), f"{win_rate:.3f}"))
    checks.append(_check("min_book_score", book_score >= float(cfg.get("min_book_score", 0.0)), f"{book_score:.4f}"))

    passed_members = 0
    weak_members = 0
    min_member_cagr = float(cfg.get("min_member_oos_cagr", -0.05))
    for m in members:
        fit = m.get("fitness") or {}
        if fit.get("passed"):
            passed_members += 1
        oos = fit.get("oos_cagr")
        if oos is not None and float(oos) < min_member_cagr:
            weak_members += 1
    passed_ratio = passed_members / max(1, n_members)
    checks.append(
        _check(
            "passed_member_ratio",
            passed_ratio >= float(cfg.get("min_passed_member_ratio", 0.60)),
            f"{passed_ratio:.2f} ({passed_members}/{n_members})",
        )
    )
    checks.append(_check("no_very_weak_members", weak_members == 0, f"weak={weak_members}", hard=False))

    # Regime robustness
    bench_sym = _benchmark_symbol(uid, cfg)
    bench_bars = list(panel.get(bench_sym) or [])
    year_labels = classify_years(
        bench_bars,
        bull_year_return=float(cfg.get("bull_year_return", 0.10)),
        bear_year_return=float(cfg.get("bear_year_return", -0.10)),
    )
    book_years = yearly_returns_from_equity(equity_curve)
    bull_years = [y for y, info in year_labels.items() if info["label"] == "bull" and y in book_years]
    bear_years = [y for y, info in year_labels.items() if info["label"] == "bear" and y in book_years]

    require_regimes = bool(cfg.get("require_bull_and_bear_years", True))
    checks.append(
        _check(
            "has_bull_years",
            (not require_regimes) or len(bull_years) >= int(cfg.get("min_bull_years", 1)),
            f"bull_years={bull_years}",
        )
    )
    checks.append(
        _check(
            "has_bear_years",
            (not require_regimes) or len(bear_years) >= int(cfg.get("min_bear_years", 1)),
            f"bear_years={bear_years}",
        )
    )

    min_bull_book = float(cfg.get("min_book_return_in_bull_year", 0.0))
    min_bear_book = float(cfg.get("min_book_return_in_bear_year", -0.15))
    bull_ok = all(book_years[y] >= min_bull_book for y in bull_years) if bull_years else (not require_regimes)
    bear_ok = all(book_years[y] >= min_bear_book for y in bear_years) if bear_years else (not require_regimes)
    checks.append(
        _check(
            "book_ok_in_bull_years",
            bull_ok,
            {str(y): round(book_years[y], 4) for y in bull_years},
        )
    )
    checks.append(
        _check(
            "book_ok_in_bear_years",
            bear_ok,
            {str(y): round(book_years[y], 4) for y in bear_years},
        )
    )

    hard_checks = [c for c in checks if c.get("hard", True)]
    hard_pass = all(c["ok"] for c in hard_checks)
    soft_pass = all(c["ok"] for c in checks if not c.get("hard", True))

    # Robust score: not "CAGR > X" alone
    concentration = 0.0
    classes = [(m.get("candidate") or {}).get("class_name") for m in members]
    if classes:
        from collections import Counter

        top = Counter(classes).most_common(1)[0][1]
        concentration = top / max(1, len(classes))
    robust_score = (
        cagr * 100.0
        - max_dd * 80.0
        - concentration * 15.0
        + min(pf, 5.0) * 2.0
        + (0.0 if hard_pass else -25.0)
    )
    checks.append(
        _check(
            "robust_score_floor",
            robust_score >= float(cfg.get("min_robust_score", -5.0)),
            f"{robust_score:.3f}",
            hard=False,
        )
    )
    soft_pass = all(c["ok"] for c in checks if not c.get("hard", True))

    # Live tier (stricter)
    live = cfg.get("live") or {}
    live_checks = [
        _check("live_min_cagr", cagr >= float(live.get("min_book_cagr", 0.08)), f"{cagr:.4f}"),
        _check("live_min_pf", pf >= float(live.get("min_profit_factor", 1.20)), f"{pf:.3f}"),
        _check("live_min_trades", trades >= int(live.get("min_book_trades", 80)), f"{trades}"),
        _check("live_min_members", n_members >= int(live.get("min_members", 8)), f"{n_members}"),
        _check(
            "live_bear_floor",
            all(book_years[y] >= float(live.get("min_book_return_in_bear_year", -0.10)) for y in bear_years)
            if bear_years
            else False,
            {str(y): round(book_years[y], 4) for y in bear_years},
        ),
    ]
    live_pass = hard_pass and all(c["ok"] for c in live_checks)

    # Candidate = basic structure present even if returns weak
    candidate = n_members >= max(3, int(cfg.get("min_members", 5)) // 2) and trades >= 10

    if live_pass:
        level = "ready_for_live"
        message = "All paper + live stability gates passed"
    elif hard_pass:
        level = "ready_for_paper"
        message = "Stability gate passed — eligible to freeze for paper trading"
    elif candidate:
        level = "candidate"
        message = "Book exists but failed hard stability gates — keep discovering"
    else:
        level = "not_ready"
        message = "Insufficient book quality for promotion"

    vault_report = None
    if hard_pass:
        # Open immutable vault ONLY at promotion — burns strategy versions.
        try:
            from trading_ecosystem.discovery.promotion import evaluate_vault_book
            from trading_ecosystem.discovery.walk_forward import build_folds

            folds = build_folds(panel, vault_ratio=0.20, seal_vault=True, use_yearly_rolls=False)
            vault_fold = next((f for f in folds if f.kind == "vault"), None)
            if vault_fold is not None:
                vault_report = evaluate_vault_book(
                    members,
                    panel,
                    vault_fold,
                    starting_equity=starting_equity,
                    universe_id=uid,
                    burn=True,
                )
                vm = (vault_report or {}).get("book_metrics") or {}
                checks.append(
                    _check(
                        "vault_non_negative_cagr",
                        float(vm.get("cagr") or 0.0) >= 0.0,
                        f"vault_cagr={vm.get('cagr')}",
                        hard=False,
                    )
                )
        except Exception as exc:  # noqa: BLE001
            vault_report = {"ok": False, "error": str(exc)}

    payload = {
        "ready": hard_pass,
        "level": level,
        "universe": uid,
        "benchmark": bench_sym,
        "message": message,
        "checks": checks,
        "live_checks": live_checks,
        "soft_pass": soft_pass,
        "fingerprint": fingerprint,
        "skipped_resim": False,
        "metrics": {
            "cagr": cagr,
            "max_drawdown": max_dd,
            "num_trades": trades,
            "profit_factor": pf,
            "win_rate": win_rate,
            "book_score": book_score,
            "robust_score": robust_score,
            "family_concentration": concentration,
            "members": n_members,
            "cost_profile": metrics.get("cost_profile"),
        },
        "regimes": {
            "year_labels": {str(y): v for y, v in year_labels.items()},
            "book_year_returns": {str(y): round(v, 6) for y, v in book_years.items()},
            "bull_years_used": bull_years,
            "bear_years_used": bear_years,
        },
        "vault": vault_report,
        "promotion": {
            "allow_freeze_paper": hard_pass,
            "allow_live": live_pass,
            "vault_opened": bool(vault_report and vault_report.get("ok")),
        },
    }
    save_readiness(payload, uid)
    return payload
