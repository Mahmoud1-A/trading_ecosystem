"""One-shot cleanup: split crypto contamination out of ETF discovery artifacts."""
from __future__ import annotations

import json
from pathlib import Path

root = Path("data/processed")


def primary(row: dict) -> str:
    c = row.get("candidate") or {}
    syms = c.get("symbols") or []
    return str(syms[0]) if syms else str(row.get("sector") or "")


def is_crypto(sym: str) -> bool:
    return "-USD" in (sym or "")


def split_rows(rows: list) -> tuple[list, list]:
    etf, cry = [], []
    for r in rows or []:
        (cry if is_crypto(primary(r)) else etf).append(r)
    return etf, cry


def main() -> None:
    book_path = root / "discovery_portfolio_book.json"
    book = json.loads(book_path.read_text(encoding="utf-8"))
    members = list(book.get("members") or [])
    etf_m = [m for m in members if not is_crypto(primary(m))]
    crypto_m = [m for m in members if is_crypto(primary(m))]
    print("book members", len(members), "etf", len(etf_m), "crypto", len(crypto_m))

    clean_book = dict(book)
    clean_book["members"] = etf_m
    clean_book["universe"] = "etf"
    met = dict(clean_book.get("metrics") or {})
    met["members"] = len(etf_m)
    met["note"] = "cleaned_crypto_contamination"
    clean_book["metrics"] = met
    book_path.write_text(json.dumps(clean_book, indent=2, default=str), encoding="utf-8")

    crypto_book_path = root / "discovery_portfolio_book_crypto.json"
    if crypto_m and not crypto_book_path.exists():
        crypto_book_path.write_text(
            json.dumps(
                {
                    "universe": "crypto",
                    "members": crypto_m,
                    "metrics": {"mode": "aggregate_book", "members": len(crypto_m)},
                    "score": None,
                    "best_ever": None,
                    "objective": "aggregate_book_mar_cagr_prop",
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print("seeded crypto book", len(crypto_m))

    top_path = root / "discovery_top50.json"
    top = json.loads(top_path.read_text(encoding="utf-8"))
    e_ent, c_ent = split_rows(top.get("entries") or [])
    e_pool, c_pool = split_rows(top.get("pool") or [])
    print("top50 entries", len(top.get("entries") or []), "-> etf", len(e_ent), "crypto", len(c_ent))
    top["entries"] = e_ent
    top["pool"] = e_pool
    top["universe"] = "etf"
    top_path.write_text(json.dumps(top, indent=2, default=str), encoding="utf-8")

    crypto_top = root / "discovery_top50_crypto.json"
    if (c_ent or c_pool) and not crypto_top.exists():
        crypto_top.write_text(
            json.dumps(
                {
                    "updated_at": top.get("updated_at"),
                    "size": top.get("size", 50),
                    "universe": "crypto",
                    "entries": c_ent[:50],
                    "pool": c_pool[:400],
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print("seeded crypto top50")

    b3_path = root / "discovery_best3_by_sector.json"
    b3 = json.loads(b3_path.read_text(encoding="utf-8"))
    sectors = dict(b3.get("sectors") or {})
    etf_sec = {k: v for k, v in sectors.items() if not is_crypto(k)}
    cry_sec = {k: v for k, v in sectors.items() if is_crypto(k)}
    print("sectors", len(sectors), "etf", len(etf_sec), "crypto", len(cry_sec))
    b3["sectors"] = etf_sec
    b3["universe"] = "etf"
    b3_path.write_text(json.dumps(b3, indent=2, default=str), encoding="utf-8")

    crypto_b3 = root / "discovery_best3_by_sector_crypto.json"
    if cry_sec and not crypto_b3.exists():
        crypto_b3.write_text(
            json.dumps(
                {
                    "updated_at": b3.get("updated_at"),
                    "best_per_sector": b3.get("best_per_sector", 3),
                    "universe": "crypto",
                    "sectors": cry_sec,
                },
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        print("seeded crypto best3")

    print("done")


if __name__ == "__main__":
    main()
