"""Causal series evaluation for Strategy DSL expression trees.

Compiles FEATURE / PARAMETER / OPERATOR trees into aligned pandas Series.
Strategy wrappers (ENTRY_*, EXIT_*, ATR_*, REGIME_GATE, …) evaluate to their
payload series (boolean intent or distance) for the signal adapter.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from discovery.expression_tree import ExprNode
from discovery.operators import OperatorId
from discovery.types import NodeKind


def _scalar_window(series: pd.Series, *, default: int = 1) -> int:
    """Extract a positive integer lookback from a constant/parameter series."""
    if series.empty:
        return default
    vals = series.replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return default
    w = int(round(float(vals.iloc[0])))
    return max(1, min(w, 500))


def _as_bool(series: pd.Series) -> pd.Series:
    return series.fillna(0.0).astype(float).ne(0.0)


def eval_series(
    node: ExprNode,
    features: pd.DataFrame,
    *,
    bindings: dict[str, float] | None = None,
) -> pd.Series:
    """Evaluate ``node`` over ``features`` index (causal; no future peek)."""
    bindings = bindings or {}
    index = features.index

    if node.kind is NodeKind.CONSTANT:
        return pd.Series(float(node.meta["value"]), index=index, dtype=float)

    if node.kind is NodeKind.PARAMETER:
        val = float(bindings.get(node.name, node.meta.get("default", 0.0)))
        return pd.Series(val, index=index, dtype=float)

    if node.kind is NodeKind.FEATURE:
        if node.name in features.columns:
            return features[node.name].astype(float)
        return pd.Series(np.nan, index=index, dtype=float)

    op = OperatorId(node.name)
    kids = [eval_series(c, features, bindings=bindings) for c in node.children]

    if op is OperatorId.ADD:
        return kids[0] + kids[1]
    if op is OperatorId.SUBTRACT:
        return kids[0] - kids[1]
    if op is OperatorId.MULTIPLY_BOUNDED:
        return (kids[0] * kids[1]).clip(-1e6, 1e6)
    if op is OperatorId.DIVIDE_PROTECTED:
        numer = kids[0].astype(float)
        denom = kids[1].astype(float)
        safe = denom.where(denom.abs() >= 1e-8, np.nan)
        return (numer / safe).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if op is OperatorId.ABS:
        return kids[0].abs()
    if op is OperatorId.NEGATE:
        return -kids[0]
    if op is OperatorId.MIN:
        return pd.concat([kids[0], kids[1]], axis=1).min(axis=1)
    if op is OperatorId.MAX:
        return pd.concat([kids[0], kids[1]], axis=1).max(axis=1)
    if op is OperatorId.CLIP:
        return kids[0].clip(kids[1], kids[2])

    if op is OperatorId.LAG:
        n = _scalar_window(kids[1], default=1)
        return kids[0].shift(n)
    if op is OperatorId.DIFFERENCE:
        return kids[0].diff()
    if op is OperatorId.PERCENT_CHANGE:
        return kids[0].pct_change()
    if op is OperatorId.ROLLING_MEAN:
        w = _scalar_window(kids[1])
        return kids[0].rolling(w, min_periods=w).mean()
    if op is OperatorId.ROLLING_MEDIAN:
        w = _scalar_window(kids[1])
        return kids[0].rolling(w, min_periods=w).median()
    if op is OperatorId.ROLLING_STD:
        w = _scalar_window(kids[1])
        return kids[0].rolling(w, min_periods=w).std(ddof=0)
    if op is OperatorId.ROLLING_MIN:
        w = _scalar_window(kids[1])
        return kids[0].rolling(w, min_periods=w).min()
    if op is OperatorId.ROLLING_MAX:
        w = _scalar_window(kids[1])
        return kids[0].rolling(w, min_periods=w).max()
    if op is OperatorId.ROLLING_RANK:
        w = _scalar_window(kids[1])
        return kids[0].rolling(w, min_periods=w).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )
    if op is OperatorId.ROLLING_ZSCORE:
        w = _scalar_window(kids[1])
        mu = kids[0].rolling(w, min_periods=w).mean()
        sd = kids[0].rolling(w, min_periods=w).std(ddof=0).replace(0.0, np.nan)
        return (kids[0] - mu) / sd
    if op is OperatorId.EMA:
        w = _scalar_window(kids[1])
        return kids[0].ewm(span=w, adjust=False, min_periods=w).mean()

    if op is OperatorId.GREATER_THAN:
        return (kids[0] > kids[1]).astype(float)
    if op is OperatorId.LESS_THAN:
        return (kids[0] < kids[1]).astype(float)
    if op is OperatorId.GREATER_EQUAL:
        return (kids[0] >= kids[1]).astype(float)
    if op is OperatorId.LESS_EQUAL:
        return (kids[0] <= kids[1]).astype(float)
    if op is OperatorId.BETWEEN:
        return ((kids[0] >= kids[1]) & (kids[0] <= kids[2])).astype(float)
    if op is OperatorId.CROSS_ABOVE:
        return ((kids[0] > kids[1]) & (kids[0].shift(1) <= kids[1].shift(1))).astype(float)
    if op is OperatorId.CROSS_BELOW:
        return ((kids[0] < kids[1]) & (kids[0].shift(1) >= kids[1].shift(1))).astype(float)
    if op is OperatorId.AND:
        return (_as_bool(kids[0]) & _as_bool(kids[1])).astype(float)
    if op is OperatorId.OR:
        return (_as_bool(kids[0]) | _as_bool(kids[1])).astype(float)
    if op is OperatorId.NOT:
        return (~_as_bool(kids[0])).astype(float)

    # Strategy wrappers — return payload series for the signal adapter
    if op in {
        OperatorId.ENTRY_LONG,
        OperatorId.ENTRY_SHORT,
        OperatorId.EXIT_SIGNAL,
        OperatorId.SESSION_EXIT,
    }:
        return kids[0]
    if op is OperatorId.TIME_EXIT:
        return kids[0]
    if op in {OperatorId.ATR_STOP, OperatorId.ATR_TARGET, OperatorId.TRAILING_STOP}:
        return kids[0] * kids[1]
    if op is OperatorId.REGIME_GATE:
        return (_as_bool(kids[0]) & _as_bool(kids[1])).astype(float)
    if op is OperatorId.LIQUIDITY_GATE:
        return (_as_bool(kids[0]) & kids[1].fillna(0.0).gt(0.0)).astype(float)
    if op is OperatorId.VOLATILITY_GATE:
        return (_as_bool(kids[0]) & kids[1].fillna(0.0).gt(0.0)).astype(float)

    if kids:
        return kids[0]
    return pd.Series(0.0, index=index, dtype=float)


def unwrap_boolean_intent(node: ExprNode | None, features: pd.DataFrame, bindings: dict[str, float]) -> pd.Series:
    """Evaluate a tree to a boolean Series (ENTRY/EXIT/GATE roots unwrap to cond)."""
    index = features.index
    if node is None:
        return pd.Series(False, index=index)
    series = eval_series(node, features, bindings=bindings)
    return _as_bool(series)


def unwrap_distance(node: ExprNode | None, features: pd.DataFrame, bindings: dict[str, float]) -> pd.Series | None:
    """Evaluate ATR_STOP / ATR_TARGET style trees to a non-negative distance Series."""
    if node is None:
        return None
    series = eval_series(node, features, bindings=bindings).astype(float)
    return series.abs()


def time_exit_bars(node: ExprNode | None, bindings: dict[str, float]) -> float | None:
    """Extract TIME_EXIT max-hold bars if present on an exit/stop tree."""
    if node is None:
        return None
    if node.kind is NodeKind.OPERATOR and node.name == OperatorId.TIME_EXIT.value:
        child = node.children[0]
        if child.kind is NodeKind.CONSTANT:
            return float(child.meta["value"])
        if child.kind is NodeKind.PARAMETER:
            return float(bindings.get(child.name, child.meta.get("default", 0.0)))
    for c in node.children:
        found = time_exit_bars(c, bindings)
        if found is not None:
            return found
    return None


__all__ = [
    "eval_series",
    "time_exit_bars",
    "unwrap_boolean_intent",
    "unwrap_distance",
]
