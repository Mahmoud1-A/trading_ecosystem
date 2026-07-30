#!/usr/bin/env python
"""CLI: decisive open-execution stress on live GapFade signals."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

logging.disable(logging.WARNING)

from trading_ecosystem.audit.open_execution_stress import (  # noqa: E402
    OpenCosts,
    run_open_execution_stress,
    write_report,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--start", default="2024-01-01")
    p.add_argument("--end", default=None)
    p.add_argument("--max-signals", type=int, default=250)
    p.add_argument("--spread-bps", type=float, default=2.0)
    p.add_argument("--slip-bps", type=float, default=5.0)
    args = p.parse_args()
    costs = OpenCosts(spread_bps=args.spread_bps, slippage_bps=args.slip_bps)
    print(f"running open-execution stress start={args.start} max_signals={args.max_signals}", flush=True)
    payload = run_open_execution_stress(
        start=args.start,
        end=args.end,
        max_signals=args.max_signals,
        costs=costs,
    )
    jp, mp = write_report(payload)
    v = payload.get("verdict") or {}
    print("verdict", v.get("label"), flush=True)
    for k, row in (payload.get("scenarios") or {}).items():
        print(
            f"{k}: cagr={row['cagr']:.2%} filled={row['n_filled']} missed={row['n_missed']}",
            flush=True,
        )
    print("json", jp, flush=True)
    print("md", mp, flush=True)


if __name__ == "__main__":
    main()
