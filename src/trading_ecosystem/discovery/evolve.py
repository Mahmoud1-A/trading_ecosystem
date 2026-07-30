from __future__ import annotations

import random
from typing import Any

from trading_ecosystem.discovery.search_space import Candidate, expand_family_candidates
from trading_ecosystem.discovery.anomaly import scan_anomalies
from trading_ecosystem.common.contracts import Bar


# Soft clamps keep mutated params in economically plausible ranges (not live copies).
_PARAM_CLAMP: dict[str, dict[str, tuple[float, float]]] = {
    "GapFade": {
        "gap_pct": (0.0005, 0.012),
        "hold_bars": (1, 15),
        "atr_period": (5, 40),
        "atr_stop_mult": (0.25, 4.0),
        "risk_fraction": (0.0025, 0.015),
    },
}


def _jitter_number(rng: random.Random, value: float | int, scale: float = 0.25) -> float | int:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        delta = max(1, int(round(abs(value) * scale))) or 1
        out = value + rng.randint(-delta, delta)
        return max(1, out)
    span = abs(value) * scale if value != 0 else scale
    out = value + rng.uniform(-span, span)
    return float(round(out, 6))


def _clamp_params(class_name: str, params: dict[str, Any]) -> dict[str, Any]:
    bounds = _PARAM_CLAMP.get(class_name) or {}
    out = dict(params)
    for key, (lo, hi) in bounds.items():
        if key not in out:
            continue
        val = out[key]
        if isinstance(val, bool):
            continue
        if isinstance(val, int) and not isinstance(val, bool):
            out[key] = int(max(lo, min(hi, val)))
        elif isinstance(val, (int, float)):
            out[key] = float(max(lo, min(hi, float(val))))
    return out


def mutate_candidate(
    parent: Candidate,
    *,
    symbols: list[str],
    rng: random.Random,
    generation: int,
    seq: int,
) -> Candidate:
    params = dict(parent.params)
    keys = list(params.keys())
    if keys:
        for key in rng.sample(keys, k=max(1, len(keys) // 2)):
            params[key] = _jitter_number(rng, params[key])
    params = _clamp_params(parent.class_name, params)
    syms = list(parent.symbols)
    if symbols and rng.random() < 0.35:
        syms = [rng.choice(symbols)]
    return Candidate(
        strategy_id=f"mut_g{generation}_{seq:04d}",
        class_name=parent.class_name,
        symbols=syms,
        params=params,
        source="mutate",
        meta={"parent": parent.strategy_id, "generation": generation},
    )


def crossover_candidates(
    a: Candidate,
    b: Candidate,
    *,
    symbols: list[str],
    rng: random.Random,
    generation: int,
    seq: int,
) -> Candidate:
    """Mate two elites: class from A or B, params blended, symbol from either."""
    class_name = a.class_name if rng.random() < 0.5 else b.class_name
    # Prefer parent that matches chosen class for param backbone
    primary = a if a.class_name == class_name else b
    secondary = b if primary is a else a
    params = dict(primary.params)
    for key, val in secondary.params.items():
        if key in params and rng.random() < 0.5:
            # numeric blend when both numeric
            if isinstance(params[key], (int, float)) and isinstance(val, (int, float)):
                if isinstance(params[key], int) and isinstance(val, int):
                    params[key] = int(round((params[key] + val) / 2))
                else:
                    params[key] = float((float(params[key]) + float(val)) / 2)
            else:
                params[key] = val
        elif key not in params and rng.random() < 0.3:
            params[key] = val
    params = _clamp_params(class_name, params)
    sym = rng.choice([primary.symbols[0], secondary.symbols[0]]) if primary.symbols and secondary.symbols else (
        primary.symbols[0] if primary.symbols else rng.choice(symbols)
    )
    if symbols and rng.random() < 0.2:
        sym = rng.choice(symbols)
    return Candidate(
        strategy_id=f"xover_g{generation}_{seq:04d}",
        class_name=class_name,
        symbols=[sym],
        params=params,
        source="crossover",
        meta={
            "parents": [a.strategy_id, b.strategy_id],
            "generation": generation,
        },
    )


def _invent_family_weights(discovery_cfg: dict[str, Any]) -> dict[str, float]:
    """
    Adaptive budget controller:
    - floor exploration per family
    - boost families with recent OOS improvement
    - penalize duplicate-heavy families (from trial ledger when available)
    Falls back to static invent_family_weights / equal weights.
    """
    cont = discovery_cfg.get("continuous") or {}
    families = discovery_cfg.get("families") or {}
    enabled = [n for n, f in families.items() if f.get("enabled", True)]
    if not enabled:
        return {}

    static = cont.get("invent_family_weights") or discovery_cfg.get("invent_family_weights") or {}
    min_floor = float(cont.get("family_min_explore_weight", 0.08))
    weights: dict[str, float] = {}
    for name in enabled:
        weights[name] = float(static.get(name, 1.0 / len(enabled)))

    # Adaptive tilt from recent trial ledger
    try:
        from collections import defaultdict

        from trading_ecosystem.discovery.trial_ledger import trial_ledger_path
        from trading_ecosystem.discovery.universe import get_active_universe

        path = trial_ledger_path(str(discovery_cfg.get("active_universe") or get_active_universe()))
        if path.exists():
            lines = path.read_text(encoding="utf-8").splitlines()[-400:]
            by_cls: dict[str, list[float]] = defaultdict(list)
            for line in lines:
                try:
                    row = __import__("json").loads(line)
                except Exception:  # noqa: BLE001
                    continue
                cls = row.get("class_name")
                oos = row.get("oos_cagr")
                if cls in weights and oos is not None:
                    by_cls[cls].append(float(oos))
            if by_cls:
                means = {c: (sum(v) / len(v)) for c, v in by_cls.items() if v}
                if means:
                    best = max(means.values())
                    for c, m in means.items():
                        # Boost relative to best recent mean; penalize near-zero clones
                        boost = 1.0 + max(0.0, m) / max(best, 1e-6)
                        dup_pen = 1.0 / (1.0 + max(0, len(by_cls[c]) - 20) * 0.02)
                        weights[c] = weights.get(c, min_floor) * boost * dup_pen
    except Exception:  # noqa: BLE001
        pass

    # Enforce floors + renormalize
    for name in enabled:
        weights[name] = max(min_floor, float(weights.get(name, min_floor)))
    total = sum(weights.values()) or 1.0
    return {k: v / total for k, v in weights.items()}


def _sample_invent_from_families(
    discovery_cfg: dict[str, Any],
    *,
    rng: random.Random,
    generation: int,
    n: int,
) -> list[Candidate]:
    """Draw invent candidates with optional family prior weights."""
    if n <= 0:
        return []
    weights = _invent_family_weights(discovery_cfg)
    if not weights:
        seed = rng.randint(0, 10_000_000)
        pool = expand_family_candidates(
            discovery_cfg,
            max_per_family=max(2, n),
            seed=seed,
        )
        rng.shuffle(pool)
        out: list[Candidate] = []
        for i, c in enumerate(pool[:n]):
            out.append(
                Candidate(
                    strategy_id=f"invent_g{generation}_{i:04d}",
                    class_name=c.class_name,
                    symbols=list(c.symbols),
                    params=_clamp_params(c.class_name, dict(c.params)),
                    source="invent",
                    meta={"generation": generation, "seed": seed},
                )
            )
        return out

    # Build a modest pool per enabled family, then weighted-pick classes.
    seed = rng.randint(0, 10_000_000)
    per_family = max(4, n)
    by_class: dict[str, list[Candidate]] = {name: [] for name in weights}
    # Sample each family separately so rare-weighted families still have combos.
    for class_name in list(weights):
        fam_cfg = dict(discovery_cfg)
        fam_cfg["families"] = {class_name: (discovery_cfg.get("families") or {})[class_name]}
        pool = expand_family_candidates(
            fam_cfg,
            max_per_family=per_family,
            seed=seed + (sum(ord(ch) for ch in class_name) % 10_000),
        )
        by_class[class_name] = pool

    names = list(weights.keys())
    wts = [weights[n_] for n_ in names]
    out: list[Candidate] = []
    for i in range(n):
        # Fallback if a family pool emptied.
        alive = [nm for nm in names if by_class.get(nm)]
        if not alive:
            break
        alive_w = [weights[nm] for nm in alive]
        pick = rng.choices(alive, weights=alive_w, k=1)[0]
        c = by_class[pick].pop(rng.randrange(len(by_class[pick])))
        out.append(
            Candidate(
                strategy_id=f"invent_g{generation}_{i:04d}",
                class_name=c.class_name,
                symbols=list(c.symbols),
                params=_clamp_params(c.class_name, dict(c.params)),
                source="invent",
                meta={
                    "generation": generation,
                    "seed": seed,
                    "family_prior": pick,
                    "family_weight": weights.get(pick),
                },
            )
        )
    return out


def invent_candidates(
    discovery_cfg: dict[str, Any],
    panel: dict[str, list[Bar]],
    *,
    rng: random.Random,
    generation: int,
    n_random: int = 8,
    n_anomaly: int = 6,
) -> list[Candidate]:
    """Invent fresh candidates via weighted family draws + anomaly rescan."""
    out: list[Candidate] = []
    out.extend(
        _sample_invent_from_families(
            discovery_cfg,
            rng=rng,
            generation=generation,
            n=n_random,
        )
    )

    anom = scan_anomalies(panel, top_k=max(n_anomaly * 2, 10))
    rng.shuffle(anom)
    for i, c in enumerate(anom[:n_anomaly]):
        # slight param jitter so invent is not identical every generation
        params = _clamp_params(c.class_name, dict(c.params))
        if params:
            k = rng.choice(list(params.keys()))
            params[k] = _jitter_number(rng, params[k], scale=0.15)
            params = _clamp_params(c.class_name, params)
        out.append(
            Candidate(
                strategy_id=f"anom_g{generation}_{i:04d}",
                class_name=c.class_name,
                symbols=list(c.symbols),
                params=params,
                source="invent_anomaly",
                meta={"generation": generation, "pattern": c.meta.get("pattern")},
            )
        )
    return out


def breed_generation(
    elites: list[Candidate],
    discovery_cfg: dict[str, Any],
    panel: dict[str, list[Bar]],
    *,
    generation: int,
    batch_size: int,
    rng: random.Random,
) -> list[Candidate]:
    """Build next batch: mutate + crossover elites + invent new."""
    symbols = list(discovery_cfg.get("symbols") or [])
    children: list[Candidate] = []
    seq = 0

    if elites:
        n_mut = max(1, batch_size // 3)
        n_xover = max(1, batch_size // 3)
        for _ in range(n_mut):
            parent = rng.choice(elites)
            children.append(
                mutate_candidate(parent, symbols=symbols, rng=rng, generation=generation, seq=seq)
            )
            seq += 1
        if len(elites) >= 2:
            for _ in range(n_xover):
                a, b = rng.sample(elites, 2)
                children.append(
                    crossover_candidates(
                        a, b, symbols=symbols, rng=rng, generation=generation, seq=seq
                    )
                )
                seq += 1
        else:
            parent = elites[0]
            for _ in range(n_xover):
                children.append(
                    mutate_candidate(parent, symbols=symbols, rng=rng, generation=generation, seq=seq)
                )
                seq += 1

    remaining = max(0, batch_size - len(children))
    if remaining:
        children.extend(
            invent_candidates(
                discovery_cfg,
                panel,
                rng=rng,
                generation=generation,
                n_random=max(1, remaining // 2),
                n_anomaly=max(1, remaining - remaining // 2),
            )[:remaining]
        )

    # Deduplicate by key
    seen: set[str] = set()
    unique: list[Candidate] = []
    for c in children:
        k = c.key()
        if k in seen:
            continue
        seen.add(k)
        unique.append(c)
    return unique[:batch_size]
