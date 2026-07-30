"""Strategy Family Generator helpers for New Run UI + control-plane validation."""

from __future__ import annotations

from typing import Any

from discovery.family_catalog import DEFAULT_FAMILY_ORDER, FAMILY_BLUEPRINTS
from discovery.family_generator import StrategyFamilyGenerator
from discovery.family_spec import HomogeneousFamilyGrammarError, assert_diverse_family_grammars

MULTI_FAMILY_STRATEGY_FAMILY = "multi_family_generated"
FORBIDDEN_MULTI_FAMILY_LABELS = frozenset({"mean_reversion_vwap_bb", "dsl_generated", ""})


def list_family_blueprints(*, seed: int = 42) -> list[dict[str, Any]]:
    """Return blueprint metadata + deterministic grammar fingerprints for the UI."""
    gen = StrategyFamilyGenerator(seed=int(seed))
    out: list[dict[str, Any]] = []
    for fid in DEFAULT_FAMILY_ORDER:
        spec = gen.generate(count=1, family_ids=[fid])[0]
        bp = FAMILY_BLUEPRINTS[fid]
        out.append(
            {
                "family_id": fid,
                "hypothesis": str(bp["hypothesis"]),
                "allowed_features": list(bp["allowed_features"]),
                "allowed_operators": list(bp["allowed_operators"]),
                "entry_patterns": list(bp["entry_patterns"]),
                "exit_patterns": list(bp["exit_patterns"]),
                "regime_constraints": list(bp["regime_constraints"]),
                "effective_grammar_fingerprint": spec.effective_grammar_fingerprint(),
                "canonical_hash": spec.canonical_hash(),
            }
        )
    return out


def preview_multi_family_specs(
    *,
    seed: int,
    family_count: int,
    family_ids: list[str] | None,
) -> dict[str, Any]:
    """Build FamilySpec preview for Frozen config; fail if <2 distinct grammars."""
    count = int(family_count)
    if count < 2:
        raise ValueError(
            "MULTI_FAMILY_TOO_FEW: requested_family_count must be >= 2 "
            "(need at least 2 distinct grammar fingerprints)"
        )
    ids = list(family_ids) if family_ids else None
    if ids is not None and len(set(ids)) < 2:
        raise ValueError(
            "MULTI_FAMILY_TOO_FEW: select at least 2 distinct family blueprints "
            "(need at least 2 distinct grammar fingerprints)"
        )
    gen = StrategyFamilyGenerator(seed=int(seed))
    try:
        families = gen.generate(count=count, family_ids=ids)
    except HomogeneousFamilyGrammarError as exc:
        raise ValueError(str(exc)) from exc
    except ValueError as exc:
        raise ValueError(f"MULTI_FAMILY_INVALID: {exc}") from exc
    assert_diverse_family_grammars(families)
    fps = {f.effective_grammar_fingerprint() for f in families}
    if len(fps) < 2:
        raise ValueError(
            "MULTI_FAMILY_TOO_FEW: fewer than 2 distinct grammar fingerprints"
        )
    return {
        "strategy_family": MULTI_FAMILY_STRATEGY_FAMILY,
        "family_count": len(families),
        "distinct_grammar_fingerprints": len(fps),
        "families": [f.as_dict() for f in families],
        "grammar_fingerprints": sorted(fps),
    }


def validate_multi_family_launch(
    *,
    strategy_family: str,
    multi_family: dict[str, Any] | None,
    random_seed: int,
) -> dict[str, Any] | None:
    """
    Validate Multi-Family New Run payload.

    Returns preview dict when enabled; None when disabled.
    """
    mf = dict(multi_family or {})
    if not mf.get("enabled"):
        return None
    if strategy_family in FORBIDDEN_MULTI_FAMILY_LABELS:
        raise ValueError(
            "multi_family requires strategy_family='multi_family_generated' "
            "(must not remain mean_reversion_vwap_bb or dsl_generated)"
        )
    if strategy_family != MULTI_FAMILY_STRATEGY_FAMILY:
        raise ValueError(
            f"multi_family requires strategy_family={MULTI_FAMILY_STRATEGY_FAMILY!r}; "
            f"got {strategy_family!r}"
        )
    count = int(mf.get("requested_family_count", mf.get("family_count", 0)) or 0)
    ids = list(mf["family_ids"]) if mf.get("family_ids") else None
    seed = int(mf.get("seed", random_seed))
    return preview_multi_family_specs(
        seed=seed,
        family_count=count,
        family_ids=ids,
    )
