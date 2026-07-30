from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def finalize_report(
    evaluated: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    portfolio_metrics: dict[str, Any] | None,
    target_cagr: float,
    prop_rules: dict[str, Any],
    top_n: int = 20,
) -> dict[str, Any]:
    ranked = sorted(
        evaluated,
        key=lambda e: (
            1 if e.get("fitness", {}).get("passed") else 0,
            e.get("fitness", {}).get("score", -1e9),
        ),
        reverse=True,
    )
    passed = [e for e in ranked if e.get("fitness", {}).get("passed")]
    hit = [e for e in passed if e.get("fitness", {}).get("hit_target_cagr")]
    best_any = max(ranked, key=lambda e: e.get("fitness", {}).get("oos_cagr", -1e9), default=None)
    best_passed = passed[0] if passed else None

    return {
        "summary": {
            "candidates_evaluated": len(ranked),
            "passed_prop_and_stability": len(passed),
            "hit_target_cagr": len(hit),
            "target_cagr": target_cagr,
            "prop_max_daily_dd_pct": prop_rules.get("max_daily_drawdown_pct"),
            "prop_max_total_dd_pct": prop_rules.get("max_total_drawdown_pct"),
            "best_oos_cagr_any": None if best_any is None else best_any["fitness"]["oos_cagr"],
            "best_passed_strategy_id": None
            if best_passed is None
            else best_passed["candidate"]["strategy_id"],
            "best_passed_oos_cagr": None if best_passed is None else best_passed["fitness"]["oos_cagr"],
            "target_reached": len(hit) > 0,
            "gap_to_target": None
            if best_passed is None
            else max(0.0, target_cagr - float(best_passed["fitness"]["oos_cagr"])),
            "note": (
                "Under HARD prop DD caps, 30% CAGR may be unreachable; "
                "best stable survivors are kept without relaxing risk."
            ),
        },
        "top_candidates": ranked[:top_n],
        "selected_portfolio": selected,
        "portfolio_metrics": portfolio_metrics or {},
    }


def write_report(report: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return p


def candidates_to_strategies_yaml(selected: list[dict[str, Any]]) -> dict[str, Any]:
    strategies = []
    for item in selected:
        c = item["candidate"]
        strategies.append(
            {
                "id": c["strategy_id"],
                "class": c["class_name"],
                "enabled": True,
                "symbols": list(c["symbols"]),
                "params": dict(c["params"]),
            }
        )
    return {"strategies": strategies}


def write_strategies_yaml(payload: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)
    return p
