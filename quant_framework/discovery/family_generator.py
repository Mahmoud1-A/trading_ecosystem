"""Generate deterministic, diverse FamilySpec instances."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from discovery.family_catalog import DEFAULT_FAMILY_ORDER, FAMILY_BLUEPRINTS
from discovery.family_spec import (
    DuplicateFamilyError,
    FamilySpec,
    assert_diverse_family_grammars,
    dedupe_family_specs,
)


def _jitter_ranges(
    ranges: dict[str, tuple[float, float]],
    rng: np.random.Generator,
) -> dict[str, tuple[float, float]]:
    out: dict[str, tuple[float, float]] = {}
    for key, (lo, hi) in ranges.items():
        span = float(hi) - float(lo)
        # Small deterministic jitter so seeds differ without collapsing grammars.
        shift = float(rng.uniform(-0.05, 0.05)) * (span if span != 0 else 1.0)
        width = float(rng.uniform(0.9, 1.1))
        mid = 0.5 * (float(lo) + float(hi)) + shift
        half = 0.5 * span * width
        a, b = mid - half, mid + half
        out[key] = (min(a, b), max(a, b))
    return out


def materialize_family_spec(
    family_id: str,
    *,
    seed: int,
    provenance: dict[str, Any] | None = None,
) -> FamilySpec:
    if family_id not in FAMILY_BLUEPRINTS:
        raise KeyError(f"unknown family blueprint {family_id!r}")
    bp = FAMILY_BLUEPRINTS[family_id]
    rng = np.random.default_rng(int(seed) ^ (hash(family_id) & 0xFFFFFFFF))
    limits = dict(bp["complexity_limits"])
    # Light per-seed complexity jitter that stays family-distinct.
    limits["max_nodes"] = int(max(10, limits.get("max_nodes", 18) + int(rng.integers(-1, 2))))
    ranges = _jitter_ranges(dict(bp["parameter_ranges"]), rng)
    return FamilySpec(
        family_id=str(bp["family_id"]),
        hypothesis=str(bp["hypothesis"]),
        allowed_features=tuple(bp["allowed_features"]),
        allowed_operators=tuple(bp["allowed_operators"]),
        entry_patterns=tuple(bp["entry_patterns"]),
        exit_patterns=tuple(bp["exit_patterns"]),
        regime_constraints=tuple(bp["regime_constraints"]),
        parameter_ranges=ranges,
        complexity_limits=limits,
        random_seed=int(seed),
        provenance={
            "source": "family_catalog",
            "blueprint_id": family_id,
            "generator": "StrategyFamilyGenerator",
            **(provenance or {}),
        },
    )


class StrategyFamilyGenerator:
    """Produce a deterministic set of diverse FamilySpecs."""

    def __init__(self, *, seed: int = 42) -> None:
        self.seed = int(seed)

    def available_family_ids(self) -> tuple[str, ...]:
        return DEFAULT_FAMILY_ORDER

    def generate(
        self,
        *,
        count: int,
        family_ids: Sequence[str] | None = None,
        provenance: dict[str, Any] | None = None,
    ) -> list[FamilySpec]:
        if count < 1:
            raise ValueError("requested family count must be >= 1")
        order = list(family_ids) if family_ids else list(DEFAULT_FAMILY_ORDER)
        if count > len(order):
            raise ValueError(
                f"requested {count} families but only {len(order)} blueprints available"
            )
        selected = order[: int(count)]
        families: list[FamilySpec] = []
        for i, fid in enumerate(selected):
            fam_seed = int(self.seed) + 17 * i + (hash(fid) & 0xFFFF)
            families.append(
                materialize_family_spec(
                    fid,
                    seed=fam_seed,
                    provenance={"campaign_seed": self.seed, "ordinal": i, **(provenance or {})},
                )
            )
        families = dedupe_family_specs(families)
        assert_diverse_family_grammars(families)
        return families

    def generate_with_duplicate_trap(self, family_id: str, *, seed: int) -> list[FamilySpec]:
        """Test helper: attempt to register the same effective spec twice."""
        a = materialize_family_spec(family_id, seed=seed)
        b = materialize_family_spec(family_id, seed=seed)
        try:
            return dedupe_family_specs([a, b])
        except DuplicateFamilyError:
            raise
