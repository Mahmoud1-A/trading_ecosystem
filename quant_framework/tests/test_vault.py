"""Phase 5 acceptance tests — immutable validation vault."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from validation import CandidateLineage, ValidationVault, VaultAccessError


TZ = ZoneInfo("America/Chicago")


def _lineage(**overrides: object) -> CandidateLineage:
    params = {"lookback": 20, "z_entry": 2.0}
    params.update({k: v for k, v in overrides.items() if k in {"lookback", "z_entry"}})
    return CandidateLineage.create(
        strategy_family="mean_reversion_vwap_bb",
        parameters=params,
        config_snapshot={"system_version": "0.5.0-phase5"},
        code_hash="abc",
        data_hash="def",
        cost_model_version="futures_cost_v1",
    )


def _vault(tmp_path) -> ValidationVault:
    idx = pd.date_range("2024-06-01 08:30", periods=50, freq="5min", tz=TZ)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "volume": 1000.0,
        }
    )
    return ValidationVault(
        vault_version="vault_v1",
        data=df,
        data_hash="vaulthash",
        store_path=tmp_path / "vault",
    )


class TestValidationVault:
    def test_blocks_optimization_access(self, tmp_path) -> None:
        vault = _vault(tmp_path)
        with pytest.raises(VaultAccessError, match="optimization"):
            vault.evaluate_frozen_candidate(
                _lineage(),
                evaluate_fn=lambda df: {"n": len(df)},
                for_optimization=True,
            )

    def test_blocks_ranking_access(self, tmp_path) -> None:
        vault = _vault(tmp_path)
        with pytest.raises(VaultAccessError, match="ranking"):
            vault.evaluate_frozen_candidate(
                _lineage(),
                evaluate_fn=lambda df: {"n": len(df)},
                for_ranking=True,
            )

    def test_blocks_discovery_dashboard_access(self, tmp_path) -> None:
        vault = _vault(tmp_path)
        with pytest.raises(VaultAccessError, match="discovery"):
            vault.evaluate_frozen_candidate(
                _lineage(),
                evaluate_fn=lambda df: {"n": len(df)},
                for_discovery_dashboard=True,
            )

    def test_one_shot_access_per_lineage(self, tmp_path) -> None:
        vault = _vault(tmp_path)
        lineage = _lineage()
        first = vault.evaluate_frozen_candidate(lineage, evaluate_fn=lambda df: {"n": len(df)})
        assert first["access_count"] == 1
        with pytest.raises(VaultAccessError, match="already accessed"):
            vault.evaluate_frozen_candidate(lineage, evaluate_fn=lambda df: {"n": len(df)})

    def test_modified_candidate_new_lineage_can_access(self, tmp_path) -> None:
        vault = _vault(tmp_path)
        lineage = _lineage(lookback=20)
        vault.evaluate_frozen_candidate(lineage, evaluate_fn=lambda df: {"ok": True})
        modified = lineage.with_parameter_change({"lookback": 25, "z_entry": 2.0})
        assert modified.lineage_id != lineage.lineage_id
        second = vault.evaluate_frozen_candidate(modified, evaluate_fn=lambda df: {"ok": True})
        assert second["access_count"] == 1

    def test_persists_access_metadata(self, tmp_path) -> None:
        vault = _vault(tmp_path)
        lineage = _lineage()
        result = vault.evaluate_frozen_candidate(lineage, evaluate_fn=lambda df: {"n": len(df)})
        assert result["vault_version"] == "vault_v1"
        assert result["lineage_id"] == lineage.lineage_id
        assert result["candidate_id"] == lineage.candidate_id
        assert "evaluation_timestamp" in result
        # Reload vault and confirm second access still blocked
        vault2 = ValidationVault(
            vault_version="vault_v1",
            data=vault.data,
            data_hash=vault.data_hash,
            store_path=tmp_path / "vault",
        )
        with pytest.raises(VaultAccessError):
            vault2.evaluate_frozen_candidate(lineage, evaluate_fn=lambda df: {"n": 1})
