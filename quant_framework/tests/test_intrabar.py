"""Phase 2 acceptance tests — intrabar stop/target ambiguity policies."""

from __future__ import annotations

import pytest

from config.models import IntrabarAmbiguityPolicy, Side
from engine.intrabar_policy import IntrabarOutcome, resolve_intrabar


class TestIntrabarAmbiguity:
    def test_default_conservative_prefers_stop_when_both_touched(self) -> None:
        # Long: low <= stop and high >= target in same bar
        res = resolve_intrabar(
            position_side=Side.BUY,
            open_=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            stop=95.0,
            target=108.0,
            policy=IntrabarAmbiguityPolicy.CONSERVATIVE_WORST_CASE,
        )
        assert res.outcome == IntrabarOutcome.STOP
        assert res.fill_price == pytest.approx(95.0)
        assert "conservative" in res.notes

    def test_optimistic_prefers_target_when_both_touched(self) -> None:
        res = resolve_intrabar(
            position_side=Side.BUY,
            open_=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            stop=95.0,
            target=108.0,
            policy=IntrabarAmbiguityPolicy.OPTIMISTIC,
        )
        assert res.outcome == IntrabarOutcome.TARGET
        assert res.fill_price == pytest.approx(108.0)

    def test_randomized_is_deterministic_with_seed(self) -> None:
        kwargs = dict(
            position_side=Side.BUY,
            open_=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            stop=95.0,
            target=108.0,
            policy=IntrabarAmbiguityPolicy.RANDOMIZED_WITH_SEED,
        )
        a = resolve_intrabar(**kwargs, seed=7)  # type: ignore[arg-type]
        b = resolve_intrabar(**kwargs, seed=7)  # type: ignore[arg-type]
        c = resolve_intrabar(**kwargs, seed=99)  # type: ignore[arg-type]
        assert a.outcome == b.outcome
        assert a.fill_price == b.fill_price
        # Different seed may differ (not required, but usually does); at least both valid
        assert c.outcome in {IntrabarOutcome.STOP, IntrabarOutcome.TARGET}

    def test_lower_timeframe_replay_stop_first(self) -> None:
        # Path dips to stop before reaching target
        res = resolve_intrabar(
            position_side=Side.BUY,
            open_=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            stop=95.0,
            target=108.0,
            policy=IntrabarAmbiguityPolicy.LOWER_TIMEFRAME_REPLAY,
            lower_tf_path=[100.0, 97.0, 95.0, 100.0, 109.0],
        )
        assert res.outcome == IntrabarOutcome.STOP

    def test_lower_timeframe_replay_target_first(self) -> None:
        res = resolve_intrabar(
            position_side=Side.BUY,
            open_=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            stop=95.0,
            target=108.0,
            policy=IntrabarAmbiguityPolicy.LOWER_TIMEFRAME_REPLAY,
            lower_tf_path=[100.0, 105.0, 108.0, 94.0],
        )
        assert res.outcome == IntrabarOutcome.TARGET
        assert res.fill_price == pytest.approx(108.0)

    def test_ltf_missing_falls_back_to_conservative(self) -> None:
        res = resolve_intrabar(
            position_side=Side.BUY,
            open_=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            stop=95.0,
            target=108.0,
            policy=IntrabarAmbiguityPolicy.LOWER_TIMEFRAME_REPLAY,
            lower_tf_path=None,
        )
        assert res.outcome == IntrabarOutcome.STOP
        assert "fallback_conservative" in res.notes

    def test_gap_through_stop_on_open(self) -> None:
        res = resolve_intrabar(
            position_side=Side.BUY,
            open_=90.0,  # gapped through stop 95
            high=92.0,
            low=88.0,
            close=91.0,
            stop=95.0,
            target=108.0,
            policy=IntrabarAmbiguityPolicy.CONSERVATIVE_WORST_CASE,
        )
        assert res.outcome == IntrabarOutcome.STOP
        assert res.gap_through is True
        assert res.fill_price == pytest.approx(90.0)

    def test_stop_only(self) -> None:
        res = resolve_intrabar(
            position_side=Side.BUY,
            open_=100.0,
            high=101.0,
            low=94.0,
            close=96.0,
            stop=95.0,
            target=108.0,
        )
        assert res.outcome == IntrabarOutcome.STOP
        assert res.fill_price == pytest.approx(95.0)

    def test_target_only(self) -> None:
        res = resolve_intrabar(
            position_side=Side.BUY,
            open_=100.0,
            high=109.0,
            low=99.0,
            close=108.5,
            stop=95.0,
            target=108.0,
        )
        assert res.outcome == IntrabarOutcome.TARGET

    def test_short_conservative_both_touched(self) -> None:
        res = resolve_intrabar(
            position_side=Side.SELL,
            open_=100.0,
            high=110.0,
            low=90.0,
            close=95.0,
            stop=105.0,  # short stop above
            target=92.0,  # short target below
            policy=IntrabarAmbiguityPolicy.CONSERVATIVE_WORST_CASE,
        )
        assert res.outcome == IntrabarOutcome.STOP
        assert res.fill_price == pytest.approx(105.0)
