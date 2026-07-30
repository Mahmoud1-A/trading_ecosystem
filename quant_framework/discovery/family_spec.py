"""Hypothesis-driven strategy FamilySpec — constrains DSL search spaces."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

from discovery.grammar import (
    DEFAULT_FEATURE_LEAVES,
    GRAMMAR_VERSION,
    FeatureLeaf,
    Grammar,
    GrammarLimits,
)
from discovery.operators import OPERATOR_REGISTRY, OperatorId
from registry.hashing import sha256_json


@dataclass(frozen=True)
class FamilySpec:
    """Immutable specification of a strategy family DSL search space."""

    family_id: str
    hypothesis: str
    allowed_features: tuple[str, ...]
    allowed_operators: tuple[str, ...]
    entry_patterns: tuple[str, ...]
    exit_patterns: tuple[str, ...]
    regime_constraints: tuple[str, ...]
    parameter_ranges: Mapping[str, tuple[float, float]]
    complexity_limits: Mapping[str, int]
    random_seed: int
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def canonical_dict(self) -> dict[str, Any]:
        """Stable payload for duplicate detection (excludes provenance noise)."""
        return {
            "family_id": self.family_id,
            "hypothesis": self.hypothesis,
            "allowed_features": list(self.allowed_features),
            "allowed_operators": list(self.allowed_operators),
            "entry_patterns": list(self.entry_patterns),
            "exit_patterns": list(self.exit_patterns),
            "regime_constraints": list(self.regime_constraints),
            "parameter_ranges": {
                k: [float(v[0]), float(v[1])] for k, v in sorted(self.parameter_ranges.items())
            },
            "complexity_limits": {k: int(v) for k, v in sorted(self.complexity_limits.items())},
            "random_seed": int(self.random_seed),
        }

    def canonical_hash(self) -> str:
        return "famhash_" + sha256_json(self.canonical_dict())[:24]

    def effective_grammar_fingerprint(self) -> str:
        """Fingerprint of the *effective* DSL grammar (not just the label)."""
        payload = {
            "allowed_features": list(self.allowed_features),
            "allowed_operators": list(self.allowed_operators),
            "entry_patterns": list(self.entry_patterns),
            "exit_patterns": list(self.exit_patterns),
            "regime_constraints": list(self.regime_constraints),
            "complexity_limits": {k: int(v) for k, v in sorted(self.complexity_limits.items())},
            "parameter_range_keys": sorted(self.parameter_ranges.keys()),
        }
        return "gram_" + sha256_json(payload)[:24]

    def as_dict(self) -> dict[str, Any]:
        return {
            **self.canonical_dict(),
            "canonical_hash": self.canonical_hash(),
            "effective_grammar_fingerprint": self.effective_grammar_fingerprint(),
            "provenance": dict(self.provenance),
        }

    def to_grammar(self) -> Grammar:
        leaf_by_id = {leaf.feature_id: leaf for leaf in DEFAULT_FEATURE_LEAVES}
        leaves: list[FeatureLeaf] = []
        feature_ids = list(self.allowed_features)
        # Inject features required by executable regime constraints.
        for constraint in self.regime_constraints:
            if constraint in {"require_trend_regime", "prefer_range_regime"}:
                feature_ids.append("regime.trend_state")
            elif constraint == "prefer_vol_expansion":
                feature_ids.append("regime.volatility_state")
            elif constraint == "intraday_session_only":
                feature_ids.append("temp.minutes_since_open")
            elif constraint == "prefer_liquid_session":
                feature_ids.append("liq.volume_pct_20")
        # ATR always available for family stops/targets.
        feature_ids.append("vol.atr_14")
        seen: set[str] = set()
        for fid in feature_ids:
            if fid in seen:
                continue
            seen.add(fid)
            if fid not in leaf_by_id:
                raise KeyError(f"FamilySpec {self.family_id!r} references unknown feature {fid!r}")
            leaves.append(leaf_by_id[fid])
        if not leaves:
            raise ValueError(f"FamilySpec {self.family_id!r} has empty allowed_features")

        op_ids: set[OperatorId] = set()
        for name in self.allowed_operators:
            try:
                oid = OperatorId(name)
            except ValueError as exc:
                raise KeyError(
                    f"FamilySpec {self.family_id!r} references unknown operator {name!r}"
                ) from exc
            if oid not in OPERATOR_REGISTRY:
                raise KeyError(f"FamilySpec {self.family_id!r}: operator {name!r} not in registry")
            op_ids.add(oid)
        # Always permit structural wrappers used by generator/repair/regime gates.
        for required in (
            OperatorId.ENTRY_LONG,
            OperatorId.ENTRY_SHORT,
            OperatorId.EXIT_SIGNAL,
            OperatorId.AND,
            OperatorId.OR,
            OperatorId.NOT,
            OperatorId.ATR_STOP,
            OperatorId.ATR_TARGET,
            OperatorId.REGIME_GATE,
            OperatorId.ABS,
            OperatorId.GREATER_THAN,
            OperatorId.LESS_THAN,
            OperatorId.GREATER_EQUAL,
            OperatorId.LESS_EQUAL,
        ):
            if required in OPERATOR_REGISTRY:
                op_ids.add(required)

        lim = self.complexity_limits
        limits = GrammarLimits(
            max_tree_depth=int(lim.get("max_tree_depth", 5)),
            max_nodes=int(lim.get("max_nodes", 25)),
            max_distinct_features=int(lim.get("max_distinct_features", 5)),
            max_free_parameters=int(lim.get("max_free_parameters", 8)),
            max_entry_conditions=int(lim.get("max_entry_conditions", 4)),
            max_exit_conditions=int(lim.get("max_exit_conditions", 4)),
            max_regime_gates=int(lim.get("max_regime_gates", 2)),
            max_rolling_lookback=int(lim.get("max_rolling_lookback", 100)),
        )
        return Grammar(
            limits=limits,
            feature_leaves=tuple(leaves),
            version=f"{GRAMMAR_VERSION}::{self.family_id}",
            allowed_operators=frozenset(op_ids),
            family_id=self.family_id,
        )


class DuplicateFamilyError(ValueError):
    """Raised when a FamilySpec collides with an existing canonical hash."""


class HomogeneousFamilyGrammarError(RuntimeError):
    """Raised when all families collapse to the same effective DSL grammar."""


def assert_diverse_family_grammars(families: list[FamilySpec]) -> None:
    fps = {f.effective_grammar_fingerprint() for f in families}
    if len(families) >= 2 and len(fps) == 1:
        raise HomogeneousFamilyGrammarError(
            "FAMILY_GRAMMAR_COLLAPSE: all generated families resolve to the same "
            f"effective DSL grammar fingerprint {next(iter(fps))}"
        )


def dedupe_family_specs(families: list[FamilySpec]) -> list[FamilySpec]:
    """Keep first occurrence; raise on canonical-hash collision with different id."""
    seen: dict[str, FamilySpec] = {}
    out: list[FamilySpec] = []
    for spec in families:
        h = spec.canonical_hash()
        prior = seen.get(h)
        if prior is not None:
            raise DuplicateFamilyError(
                f"duplicate FamilySpec hash {h}: {prior.family_id!r} vs {spec.family_id!r}"
            )
        seen[h] = spec
        out.append(spec)
    return out
