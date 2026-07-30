from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Candidate:
    strategy_id: str
    class_name: str
    symbols: list[str]
    params: dict[str, Any]
    source: str = "families"  # families | anomaly
    meta: dict[str, Any] = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.class_name}:{','.join(self.symbols)}:{sorted(self.params.items())}"


def _product_grid(grids: dict[str, list[Any]]) -> list[dict[str, Any]]:
    if not grids:
        return [{}]
    keys = list(grids.keys())
    values = [grids[k] for k in keys]
    return [dict(zip(keys, combo, strict=True)) for combo in itertools.product(*values)]


def expand_family_candidates(
    discovery_cfg: dict[str, Any],
    *,
    max_per_family: int | None = None,
    seed: int = 42,
) -> list[Candidate]:
    symbols = list(discovery_cfg.get("symbols") or [])
    families = discovery_cfg.get("families") or {}
    limit = max_per_family
    if limit is None:
        limit = int(discovery_cfg.get("max_candidates_per_family", 80))
    rng = random.Random(seed)
    out: list[Candidate] = []

    for class_name, fam in families.items():
        if not fam.get("enabled", True):
            continue
        grids = dict(fam.get("grids") or {})
        combos = _product_grid(grids)
        pairs: list[tuple[str, dict[str, Any]]] = []
        for sym in symbols:
            for params in combos:
                pairs.append((sym, dict(params)))
        if len(pairs) > limit:
            pairs = rng.sample(pairs, limit)
        for i, (sym, params) in enumerate(pairs):
            sid = f"{class_name.lower()}_{sym.lower()}_{i:03d}"
            out.append(
                Candidate(
                    strategy_id=sid,
                    class_name=class_name,
                    symbols=[sym],
                    params=params,
                    source="families",
                )
            )
    return out


def candidate_from_dict(raw: dict[str, Any]) -> Candidate:
    return Candidate(
        strategy_id=str(raw["strategy_id"]),
        class_name=str(raw["class_name"]),
        symbols=list(raw["symbols"]),
        params=dict(raw.get("params") or {}),
        source=str(raw.get("source", "families")),
        meta=dict(raw.get("meta") or {}),
    )
