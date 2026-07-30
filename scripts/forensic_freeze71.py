#!/usr/bin/env python
"""CLI: forensic re-test of strategies.paper.live.yaml (~71% freeze claim)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from trading_ecosystem.audit.forensic_book import run_forensic, write_forensic_report  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Forensic freeze-71 audit")
    p.add_argument("--start", default="2018-01-01")
    p.add_argument("--end", default=None)
    args = p.parse_args()
    report = run_forensic(start=args.start, end=args.end)
    json_path, md_path = write_forensic_report(report)
    print(f"trust={report.get('trust')}")
    print(f"json={json_path}")
    print(f"md={md_path}")
    base = (report.get("scenarios") or {}).get("base_costs") or {}
    delayed = (report.get("scenarios") or {}).get("next_open_delayed_entry") or {}
    print(f"base_cagr={base.get('cagr')} next_open_cagr={delayed.get('cagr')}")


if __name__ == "__main__":
    main()
