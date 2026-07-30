"""
Phase 6B acceptance tests — Causal Feature Store.

Proves future invariance, causal session VWAP, DST timing, cross-asset as-of
joins, staleness, train-only stateful fits, determinism, capability gating,
tick-volume labeling discipline, warm-up enforcement, and no Vault access.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from data.calendars import ExchangeCalendar
from data.events.enums import DataCapability, VolumeType
from data.synchronization import asof_join, measure_staleness, reject_stale, StaleDataError
from features import (
    FeatureApprovalStatus,
    FeatureGenerator,
    FeatureGeneratorConfig,
    TrainOnlyScaler,
    assert_no_vault_columns,
    assert_tick_volume_not_mislabeled,
    assert_warmup_enforced,
    build_default_catalog,
    causal_session_vwap,
    future_invariance_test,
    miner_feature_ids,
)
from features.liquidity import causal_session_vwap as session_vwap_fn


CHI = ZoneInfo("America/Chicago")
NY = ZoneInfo("America/New_York")


def _bars(n: int = 80, *, start: str = "2024-01-02 09:00", tz=CHI, seed: int = 0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="5min", tz=tz)
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    return pd.DataFrame(
        {
            "open": close + rng.normal(0, 0.05, n),
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": rng.integers(500, 2000, n).astype(float),
        },
        index=idx,
    )


class TestCatalogAndMinerEligibility:
    def test_miner_only_gets_approved_causal_with_capabilities(self) -> None:
        cat = build_default_catalog()
        ohlcv_only = miner_feature_ids(cat, available_capabilities={DataCapability.OHLCV_BARS})
        assert "micro.quoted_spread" not in ohlcv_only
        assert "micro.volume_delta_ohlcv_forbidden" not in ohlcv_only
        assert "price.rolling_z_20" in ohlcv_only
        with_quotes = miner_feature_ids(
            cat,
            available_capabilities={DataCapability.OHLCV_BARS, DataCapability.BID_ASK_QUOTES},
        )
        assert "micro.quoted_spread" in with_quotes
        forbidden = cat.get("micro.volume_delta_ohlcv_forbidden")
        assert forbidden.approval_status is FeatureApprovalStatus.DISABLED
        assert forbidden.causal is False

    def test_unsupported_data_disables_dependent_features(self) -> None:
        gen = FeatureGenerator(
            config=FeatureGeneratorConfig(available_capabilities=(DataCapability.OHLCV_BARS,))
        )
        frame = gen.generate(_bars(60))
        assert "micro.quoted_spread" in frame.disabled_features
        assert "micro.quoted_spread" not in frame.feature_ids
        assert "micro.volume_delta_ohlcv_forbidden" in frame.disabled_features


class TestFutureInvariance:
    def test_future_data_cannot_modify_past_features(self) -> None:
        bars = _bars(100, seed=3)
        gen = FeatureGenerator(
            config=FeatureGeneratorConfig(
                available_capabilities=(DataCapability.OHLCV_BARS,),
                calendar="CME",
            )
        )
        result = future_invariance_test(
            bars,
            generate_fn=gen.generate_values,
            cutoff_index=40,
        )
        assert result.passed, result.as_dict()
        assert result.compared_rows == 41
        assert result.failing_columns == ()


class TestCausalSessionVWAP:
    def test_session_vwap_is_causal(self) -> None:
        bars = _bars(20, seed=1)
        vwap = causal_session_vwap(bars)
        # Manually: first bar VWAP equals typical price
        typical0 = (bars["high"].iloc[0] + bars["low"].iloc[0] + bars["close"].iloc[0]) / 3.0
        assert vwap.iloc[0] == pytest.approx(typical0)
        # Mutating a later bar does not change earlier VWAP
        mutated = bars.copy()
        mutated.iloc[-1, mutated.columns.get_loc("close")] = 999.0
        vwap2 = session_vwap_fn(mutated)
        assert vwap.iloc[:-1].to_numpy() == pytest.approx(vwap2.iloc[:-1].to_numpy())


class TestDSTSessionTiming:
    def test_dst_session_timing_uses_exchange_local_clock(self) -> None:
        # US DST spring forward 2024-03-10 (Sun); first CDT weekday session 2024-03-11
        cal = ExchangeCalendar(
            name="CME_DST",
            timezone="America/Chicago",
            session_open=__import__("datetime").time(8, 30),
            session_close=__import__("datetime").time(15, 0),
        )
        # 09:30 Chicago on Mon after DST = 14:30 UTC (CDT = UTC-5)
        idx = pd.DatetimeIndex([pd.Timestamp("2024-03-11 14:30", tz="UTC")])
        bars = pd.DataFrame(
            {"open": [1], "high": [1], "low": [1], "close": [1], "volume": [1.0]},
            index=idx,
        )
        gen = FeatureGenerator(
            config=FeatureGeneratorConfig(
                calendar=cal,
                available_capabilities=(DataCapability.OHLCV_BARS,),
            )
        )
        frame = gen.generate(bars)
        # 09:30 - 08:30 = 60 minutes since open in Chicago local time
        assert frame.values["temp.minutes_since_open"].iloc[0] == pytest.approx(60.0)


class TestCrossAssetAsOf:
    def test_cross_asset_joins_never_select_future_events(self) -> None:
        primary = _bars(10, seed=0)
        secondary = _bars(10, start="2024-01-02 08:55", seed=1)
        # Explicit availability: secondary bar available at its own timestamp
        left = pd.DataFrame(
            {
                "decision_timestamp": primary.index,
                "pid": range(len(primary)),
            }
        )
        right = pd.DataFrame(
            {
                "availability_timestamp": secondary.index,
                "s_close": secondary["close"].to_numpy(),
            }
        )
        joined = asof_join(left, right)
        for i, dec in enumerate(left["decision_timestamp"]):
            avail = joined["availability_timestamp"].iloc[i]
            if pd.notna(avail):
                assert pd.Timestamp(avail) <= pd.Timestamp(dec)


class TestStalenessAndWarmup:
    def test_stale_values_are_rejected(self) -> None:
        src = pd.Timestamp("2024-01-02 09:00", tz="UTC")
        dec = pd.Timestamp("2024-01-02 12:00", tz="UTC")
        result = measure_staleness(
            source_timestamp=src, decision_timestamp=dec, max_age=pd.Timedelta("30min")
        )
        assert result.is_stale
        with pytest.raises(StaleDataError):
            reject_stale(result)

    def test_warmup_periods_are_enforced(self) -> None:
        cat = build_default_catalog()
        gen = FeatureGenerator()
        frame = gen.generate(_bars(40))
        contract = cat.get("price.rolling_z_20")
        assert_warmup_enforced(frame.values, contract)


class TestStatefulTrainOnly:
    def test_stateful_transforms_fit_only_on_training_data(self) -> None:
        bars = _bars(60)
        gen = FeatureGenerator()
        values = gen.generate(bars).values
        train = values.iloc[:40]
        test = values.iloc[40:]
        scaler = TrainOnlyScaler()
        with pytest.raises(RuntimeError, match="before fit"):
            scaler.transform(test)
        scaler.fit(train)
        scaled_test = scaler.transform(test)
        # Mean/std come from train only — transforming train yields ~0 mean
        scaled_train = scaler.transform(train)
        numeric = scaled_train.select_dtypes(include=[np.number])
        assert abs(float(numeric.mean().mean())) < 0.05


class TestDeterminismAndVault:
    def test_feature_generation_is_deterministic(self) -> None:
        bars = _bars(50, seed=9)
        gen = FeatureGenerator(code_hash="code")
        a = gen.generate(bars)
        b = gen.generate(bars)
        pd.testing.assert_frame_equal(a.values, b.values)
        assert a.feature_set_hash == b.feature_set_hash
        assert list(a.feature_ids) == list(b.feature_ids)

    def test_no_feature_accesses_vault_data(self) -> None:
        gen = FeatureGenerator()
        frame = gen.generate(_bars(30))
        assert_no_vault_columns(frame.values)
        with pytest.raises(ValueError, match="Vault"):
            assert_no_vault_columns(pd.DataFrame({"vault_return": [1.0]}))

    def test_tick_volume_not_mislabeled(self) -> None:
        assert_tick_volume_not_mislabeled(VolumeType.TICK_ACTIVITY_PROXY.value)
        assert_tick_volume_not_mislabeled(VolumeType.EXCHANGE_EXECUTED_VOLUME.value)
        with pytest.raises(ValueError):
            assert_tick_volume_not_mislabeled("EXCHANGE_VOLUME_FAKE")
