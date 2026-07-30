"""Phase 5 acceptance tests — reproducibility and experiment registry."""

from __future__ import annotations

from datetime import timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from config import default_es_futures, default_futures_cost_model, default_prop_profile
from config.cost_model import CostModel
from engine.event_execution import EventExecutionEngine, ExecutionConfig
from engine.events import BarEvent, InformationTiming, SignalEvent
from engine.ids import DeterministicIdFactory
from engine.portfolio import Portfolio
from registry import ExperimentRegistry, hash_directory_py, sha256_json
from risk import PropRiskFSM


TZ = ZoneInfo("America/Chicago")


def _zero_cost() -> CostModel:
    return CostModel(
        version="repro_cost_v1",
        commission_per_contract=0.0,
        minimum_commission=0.0,
        fixed_spread_ticks=0.0,
        dynamic_spread_enabled=False,
        slippage_ticks_mean=0.0,
        slippage_ticks_std=0.0,
        volatility_dependent_slippage=False,
        time_of_day_slippage=False,
        liquidity_dependent_slippage=False,
        overnight_swap_enabled=False,
        participation_rate_cap=1.0,
    )


def _bars() -> list[BarEvent]:
    out: list[BarEvent] = []
    for i, o in enumerate([100.0, 100.5, 101.0, 100.0, 99.5]):
        start = pd.Timestamp("2024-01-02 08:30", tz=TZ) + pd.Timedelta(minutes=5 * i)
        out.append(
            BarEvent(
                timestamp=start,
                open=o,
                high=o + 1,
                low=o - 1,
                close=o,
                volume=10_000,
                symbol="ES",
                contract="ESH24",
                bar_end=start + pd.Timedelta(minutes=5),
            )
        )
    return out


def _run_once(seed: int = 7, *, run_id: str = "repro", fold_id: int = 0) -> dict:
    """
    Run the engine with an explicit deterministic identity.

    Nothing is stripped from the comparison: order_id, fill_id, trade_id and
    signal_id must all be reproducible from (run_id, candidate_id, fold_id).
    """
    bars = _bars()
    ids = DeterministicIdFactory(run_id=run_id, candidate_id="c1", fold_id=fold_id)

    def signal_fn(bar: BarEvent, portfolio: Portfolio, i: int) -> SignalEvent | None:
        if i not in {0, 2}:
            return None
        side = "BUY" if i == 0 else "FLAT"
        return SignalEvent(
            timing=InformationTiming(
                source_timestamp=bar.timestamp,
                availability_timestamp=bar.bar_end,
                decision_timestamp=bar.bar_end,
            ),
            symbol="ES",
            side=side,
            quantity=1.0,
            signal_id=f"sig_{run_id}_{fold_id}_{i}",
        )

    engine = EventExecutionEngine(
        default_es_futures(),
        _zero_cost(),
        starting_equity=100_000.0,
        exec_config=ExecutionConfig(latency=timedelta(0), random_seed=seed),
        risk_manager=PropRiskFSM(default_prop_profile(), starting_equity=100_000.0),
        id_factory=ids,
    )
    result = engine.run(bars, signal_fn)
    return {
        "orders": [o.snapshot() for o in result.orders],
        "fills": [f.as_dict() for f in result.fills],
        "trades": [t.as_dict() for t in result.portfolio.trades],
        "equity": [(str(t), e) for t, e in result.portfolio.equity_curve],
        "risk_events": list(result.risk_events),
    }


class TestReproducibility:
    def test_identical_seed_produces_identical_results(self) -> None:
        a = _run_once(seed=11)
        b = _run_once(seed=11)
        assert sha256_json(a["orders"]) == sha256_json(b["orders"])
        assert sha256_json(a["fills"]) == sha256_json(b["fills"])
        assert sha256_json(a["trades"]) == sha256_json(b["trades"])
        assert sha256_json(a["equity"]) == sha256_json(b["equity"])
        assert sha256_json(a["risk_events"]) == sha256_json(b["risk_events"])
        # Status history / prices / quantities must match
        assert a["orders"][0]["status"] == b["orders"][0]["status"]
        assert a["fills"][0]["price"] == b["fills"][0]["price"]
        assert a["fills"][0]["quantity"] == b["fills"][0]["quantity"]

    def test_order_and_fill_ids_are_deterministic_not_random(self) -> None:
        a = _run_once(seed=11)
        b = _run_once(seed=11)
        assert [o["order_id"] for o in a["orders"]] == [o["order_id"] for o in b["orders"]]
        assert [f["fill_id"] for f in a["fills"]] == [f["fill_id"] for f in b["fills"]]
        assert [t["trade_id"] for t in a["trades"]] == [t["trade_id"] for t in b["trades"]]
        # Every ID is present and carries the deterministic prefix
        assert a["orders"] and a["fills"]
        assert all(o["order_id"].startswith("ord_") for o in a["orders"])
        assert all(f["fill_id"].startswith("fil_") for f in a["fills"])
        # No uuid4 leaked in (uuid4 hex has dashes and is 36 chars)
        assert all("-" not in o["order_id"] for o in a["orders"])
        assert all("-" not in f["fill_id"] for f in a["fills"])

    def test_different_run_identity_yields_different_ids_same_economics(self) -> None:
        a = _run_once(seed=11, run_id="repro", fold_id=0)
        b = _run_once(seed=11, run_id="other_run", fold_id=3)
        assert [o["order_id"] for o in a["orders"]] != [o["order_id"] for o in b["orders"]]
        assert [f["fill_id"] for f in a["fills"]] != [f["fill_id"] for f in b["fills"]]
        # Identity changes must not change prices, quantities or equity
        assert [f["price"] for f in a["fills"]] == [f["price"] for f in b["fills"]]
        assert [f["quantity"] for f in a["fills"]] == [f["quantity"] for f in b["fills"]]
        assert sha256_json(a["equity"]) == sha256_json(b["equity"])

    def test_id_factory_defaults_when_not_supplied(self) -> None:
        engine = EventExecutionEngine(
            default_es_futures(),
            _zero_cost(),
            starting_equity=100_000.0,
        )
        assert isinstance(engine.id_factory, DeterministicIdFactory)
        assert engine.id_factory.run_id == "default"


class TestExperimentRegistry:
    def test_persists_rejected_trials(self, tmp_path) -> None:
        reg = ExperimentRegistry(tmp_path / "registry")
        reg.create_trial(
            candidate_id="c1",
            lineage_id="l1",
            strategy_family="mean_reversion_vwap_bb",
            parameters={"lookback": 20},
            config_snapshot={"v": 1},
            system_version="0.5.0-phase5",
            data_hash="d",
            random_seed=42,
            cost_model_version="futures_cost_v1",
            code_hash="code",
            ranking_score=0.1,
            rejection_reason="failed_stability",
        )
        reg.create_trial(
            candidate_id="c2",
            lineage_id="l2",
            strategy_family="mean_reversion_vwap_bb",
            parameters={"lookback": 15},
            config_snapshot={"v": 1},
            system_version="0.5.0-phase5",
            data_hash="d",
            random_seed=42,
            cost_model_version="futures_cost_v1",
            code_hash="code",
            ranking_score=1.2,
            rejection_reason=None,
        )
        assert len(reg.all_trials()) == 2
        assert len(reg.rejected_trials()) == 1
        assert len(reg.accepted_trials()) == 1
        # DSR/PBO must see full history including rejects
        hist = reg.trial_history_for_dsr()
        assert len(hist) == 2
        assert 0.1 in hist and 1.2 in hist

    def test_code_hash_stable(self, tmp_path) -> None:
        # Hashing empty dir is stable
        (tmp_path / "a.py").write_text("x=1\n", encoding="utf-8")
        h1 = hash_directory_py(tmp_path)
        h2 = hash_directory_py(tmp_path)
        assert h1 == h2
