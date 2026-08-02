"""Idempotently wire direct multi-year BID/ASK datasets into Alpha Miner."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
JOBS = ROOT / "quant_framework" / "control_plane" / "jobs.py"


OLD = '''        try:
            canary_max_rows = canary_cfg.get("max_rows")
            max_rows = int(canary_max_rows) if canary_max_rows is not None else None
            resolved = resolve_bid_ask_bars(entry, timeframe=timeframe, max_rows=max_rows)
        except (SilverResolutionError, ValueError) as exc:
'''

NEW = '''        try:
            canary_max_rows = canary_cfg.get("max_rows")
            max_rows = int(canary_max_rows) if canary_max_rows is not None else None
            from control_plane.direct_bid_ask_dataset import (
                DirectBidAskResolutionError,
                direct_bid_ask_path,
                resolve_direct_bid_ask_bars,
            )

            if direct_bid_ask_path(entry) is not None:
                resolved = resolve_direct_bid_ask_bars(
                    entry,
                    timeframe=timeframe,
                    max_rows=max_rows,
                )
            else:
                resolved = resolve_bid_ask_bars(
                    entry,
                    timeframe=timeframe,
                    max_rows=max_rows,
                )
        except (SilverResolutionError, DirectBidAskResolutionError, ValueError) as exc:
'''


def main() -> None:
    text = JOBS.read_text(encoding="utf-8")
    if NEW in text:
        print("Direct multi-year dataset support already wired.")
        return
    count = text.count(OLD)
    if count != 1:
        raise RuntimeError(f"expected one Alpha Miner Silver-resolution block, found {count}")
    JOBS.write_text(text.replace(OLD, NEW, 1), encoding="utf-8")
    print("Direct multi-year dataset support wired.")


if __name__ == "__main__":
    main()
