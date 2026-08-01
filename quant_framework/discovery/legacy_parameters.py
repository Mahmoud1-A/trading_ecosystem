"""Normalize legacy candidate parameter payloads for bootstrap / rehydration."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from discovery.expression_tree import ExprNode

LEGACY_PARAMETER_VALUE_AMBIGUOUS = "LEGACY_PARAMETER_VALUE_AMBIGUOUS"


class LegacyParameterError(ValueError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        candidate_id: str | None = None,
        parameter: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.candidate_id = candidate_id
        self.parameter = parameter
        self.details = dict(details or {})


def finite_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        out = float(value)
        return out if math.isfinite(out) else None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            out = float(text)
        except ValueError:
            return None
        return out if math.isfinite(out) else None
    return None


def value_shape(value: Any) -> dict[str, Any]:
    if isinstance(value, list):
        return {"type": "list", "length": len(value)}
    if isinstance(value, tuple):
        return {"type": "tuple", "length": len(value)}
    if isinstance(value, dict):
        return {"type": "dict", "keys": sorted(str(k) for k in value.keys())[:32]}
    return {"type": type(value).__name__}


def parameter_defaults_from_trees(
    *trees: ExprNode | None,
) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    for tree in trees:
        if tree is None:
            continue
        for node in tree.walk():
            if node.kind.value != "PARAMETER":
                continue
            raw = node.meta.get("default", node.meta.get("value"))
            scalar = finite_float(raw)
            if scalar is None:
                continue
            found.setdefault(str(node.name), []).append(scalar)
    return found


def normalize_legacy_candidate_parameters(
    candidate_tree: ExprNode | Sequence[ExprNode | None] | None,
    raw_parameters: Mapping[str, Any] | None,
    *,
    candidate_id: str = "",
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Normalize legacy trial parameters to scalar floats.

    Authoritative scalars live in DSL PARAMETER nodes. List/tuple/dict legacy
    metadata is never passed to ``float()`` directly; list-valued fields are
    recovered from the matching PARAMETER default when unambiguous.
    """
    if candidate_tree is None:
        trees: tuple[ExprNode | None, ...] = ()
    elif isinstance(candidate_tree, ExprNode):
        trees = (candidate_tree,)
    else:
        trees = tuple(candidate_tree)

    defaults = parameter_defaults_from_trees(*trees)
    recovered: list[dict[str, Any]] = []
    out: dict[str, float] = {}

    for key, raw in dict(raw_parameters or {}).items():
        name = str(key)
        direct = finite_float(raw)
        if direct is not None:
            out[name] = direct
            continue

        if isinstance(raw, (list, tuple, dict)):
            recovered.append(
                {
                    "candidate_id": candidate_id,
                    "parameter": name,
                    "raw_type": type(raw).__name__,
                    "raw_shape": value_shape(raw),
                }
            )
            choices = defaults.get(name) or []
            unique = sorted({round(v, 12) for v in choices})
            if len(unique) == 1:
                out[name] = float(choices[0])
                recovered[-1]["recovered_from_dsl"] = out[name]
                continue
            raise LegacyParameterError(
                LEGACY_PARAMETER_VALUE_AMBIGUOUS,
                (
                    f"cannot recover unambiguous scalar for parameter {name!r} "
                    f"(raw_type={type(raw).__name__}, shape={value_shape(raw)}, "
                    f"dsl_defaults={choices!r})"
                ),
                candidate_id=candidate_id or None,
                parameter=name,
                details={
                    "raw_type": type(raw).__name__,
                    "raw_shape": value_shape(raw),
                    "raw_value": (
                        raw
                        if not isinstance(raw, (list, tuple)) or len(raw) <= 16
                        else value_shape(raw)
                    ),
                    "dsl_defaults": choices,
                },
            )

        raise LegacyParameterError(
            LEGACY_PARAMETER_VALUE_AMBIGUOUS,
            f"unsupported legacy parameter value for {name!r}: type={type(raw).__name__}",
            candidate_id=candidate_id or None,
            parameter=name,
            details={"raw_type": type(raw).__name__, "raw_shape": value_shape(raw)},
        )

    for name, values in defaults.items():
        if name in out:
            continue
        unique = sorted({round(v, 12) for v in values})
        if len(unique) == 1:
            out[name] = float(values[0])

    return out, recovered


def sanitize_fitness_components(raw: Mapping[str, Any] | None) -> dict[str, float]:
    """Keep only finite scalar fitness components; expand fold trade-count lists."""
    components: dict[str, float] = {}
    for key, value in dict(raw or {}).items():
        name = str(key)
        direct = finite_float(value)
        if direct is not None:
            components[name] = direct
            continue
        if name == "fold_trade_counts" and isinstance(value, (list, tuple)):
            for i, item in enumerate(value):
                count = finite_float(item)
                if count is None:
                    continue
                components[f"fold_{i}_oos_trades"] = float(int(count))
            continue
    return components
