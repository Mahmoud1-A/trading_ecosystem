from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from trading_ecosystem.common.contracts import Bar
from trading_ecosystem.discovery.search_space import Candidate


@dataclass
class PatternHit:
    symbol: str
    kind: str
    score: float
    params: dict[str, Any]
    class_name: str
    detail: dict[str, Any]


def _closes(bars: list[Bar]) -> np.ndarray:
    return np.array([b.close for b in bars], dtype=float)


def _scan_gaps(symbol: str, bars: list[Bar]) -> list[PatternHit]:
    if len(bars) < 40:
        return []
    gaps = []
    next_rets = []
    for i in range(1, len(bars)):
        prev = bars[i - 1].close
        if prev <= 0:
            continue
        gap = (bars[i].open - prev) / prev
        # next-day close vs open after gap
        ret = (bars[i].close - bars[i].open) / bars[i].open if bars[i].open else 0.0
        gaps.append(gap)
        next_rets.append(ret)
    gaps_a = np.array(gaps)
    rets_a = np.array(next_rets)
    hits: list[PatternHit] = []
    # Micro-gap thresholds match liquid daily ETFs; keep a couple larger ones.
    for thr in (0.001, 0.0015, 0.002, 0.003, 0.005, 0.008, 0.01):
        mask = gaps_a <= -thr
        min_n = 8 if thr >= 0.005 else 12
        if mask.sum() < min_n:
            continue
        mean_ret = float(rets_a[mask].mean())
        # Positive mean ret after down-gap => fade edge
        if mean_ret <= 0:
            continue
        hits.append(
            PatternHit(
                symbol=symbol,
                kind="gap_fade",
                score=mean_ret * float(mask.sum()) ** 0.5,
                class_name="GapFade",
                params={
                    "gap_pct": thr,
                    "hold_bars": 5,
                    "atr_period": 20,
                    "atr_stop_mult": 1.5,
                    "risk_fraction": 0.0075,
                },
                detail={"n": int(mask.sum()), "mean_next_ret": mean_ret},
            )
        )
    return hits


def _scan_zscore(symbol: str, bars: list[Bar]) -> list[PatternHit]:
    closes = _closes(bars)
    if len(closes) < 80:
        return []
    hits: list[PatternHit] = []
    for lookback in (10, 20, 40):
        if len(closes) <= lookback + 5:
            continue
        reversion = []
        for i in range(lookback, len(closes) - 1):
            window = closes[i - lookback : i]
            mu = window.mean()
            sd = window.std()
            if sd <= 1e-12:
                continue
            z = (closes[i] - mu) / sd
            if z <= -2.0:
                fwd = (closes[i + 1] - closes[i]) / closes[i]
                reversion.append(fwd)
        if len(reversion) < 8:
            continue
        mean_fwd = float(np.mean(reversion))
        if mean_fwd <= 0:
            continue
        hits.append(
            PatternHit(
                symbol=symbol,
                kind="zscore_revert",
                score=mean_fwd * len(reversion) ** 0.5,
                class_name="ZScoreRevert",
                params={
                    "lookback": lookback,
                    "entry_z": 2.0,
                    "exit_z": 0.5,
                    "risk_fraction": 0.0075,
                    "atr_period": 14,
                    "atr_stop_mult": 2.5,
                },
                detail={"n": len(reversion), "mean_fwd": mean_fwd},
            )
        )
    return hits


def _scan_breakout_followthrough(symbol: str, bars: list[Bar]) -> list[PatternHit]:
    if len(bars) < 80:
        return []
    hits: list[PatternHit] = []
    for n in (10, 20, 40):
        follow = []
        for i in range(n + 1, len(bars) - 1):
            prior_high = max(b.high for b in bars[i - n : i])
            if bars[i].close > prior_high:
                fwd = (bars[i + 1].close - bars[i].close) / bars[i].close
                follow.append(fwd)
        if len(follow) < 8:
            continue
        mean_fwd = float(np.mean(follow))
        if mean_fwd <= 0:
            continue
        hits.append(
            PatternHit(
                symbol=symbol,
                kind="breakout_follow",
                score=mean_fwd * len(follow) ** 0.5,
                class_name="BreakoutATR",
                params={
                    "lookback": n,
                    "atr_period": 14,
                    "atr_stop_mult": 2.5,
                    "risk_fraction": 0.0075,
                },
                detail={"n": len(follow), "mean_fwd": mean_fwd},
            )
        )
    return hits


def scan_anomalies(
    bars_by_symbol: dict[str, list[Bar]],
    *,
    top_k: int = 40,
) -> list[Candidate]:
    hits: list[PatternHit] = []
    for symbol, bars in bars_by_symbol.items():
        if len(bars) < 40:
            continue
        hits.extend(_scan_gaps(symbol, bars))
        hits.extend(_scan_zscore(symbol, bars))
        hits.extend(_scan_breakout_followthrough(symbol, bars))

    hits.sort(key=lambda h: h.score, reverse=True)
    selected = hits[:top_k]
    candidates: list[Candidate] = []
    for i, h in enumerate(selected):
        candidates.append(
            Candidate(
                strategy_id=f"anom_{h.kind}_{h.symbol.lower()}_{i:03d}",
                class_name=h.class_name,
                symbols=[h.symbol],
                params=dict(h.params),
                source="anomaly",
                meta={"pattern": h.kind, "score": h.score, **h.detail},
            )
        )
    return candidates
