"""Paper promotion gates — Vault pass required; legacy Sim cannot promote."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from brokers.capabilities import BrokerCapabilities
from brokers.legacy_alpaca_adapter import LegacyAlpacaAdapter
from portfolio.versioning import PortfolioPipelineStage, PortfolioVersion
from runtime.kill_switch import KillSwitch


class PaperPromotionError(ValueError):
    pass


@dataclass
class PaperPromotionGate:
    """
    A portfolio enters Paper only when all gates pass.

    Legacy Sim performance is explicitly rejected as a promotion input.
    """

    require_vault_passed: bool = True
    require_frozen: bool = True
    require_kill_switch_tested: bool = True

    def evaluate(
        self,
        portfolio: PortfolioVersion,
        *,
        capabilities: BrokerCapabilities,
        unresolved_recon_errors: int = 0,
        kill_switch_tested: bool = False,
        legacy_sim_sharpe: float | None = None,
        feature_set_frozen: bool = True,
        cost_model_frozen: bool = True,
        risk_profile_frozen: bool = True,
    ) -> dict[str, Any]:
        reasons: list[str] = []

        if legacy_sim_sharpe is not None:
            reasons.append("legacy_sim_performance_cannot_promote")

        if self.require_frozen and not portfolio.frozen:
            reasons.append("portfolio_not_frozen")
        if self.require_vault_passed and portfolio.stage is not PortfolioPipelineStage.VAULT_PASSED:
            reasons.append("vault_not_passed")
        if not feature_set_frozen:
            reasons.append("feature_set_not_frozen")
        if not cost_model_frozen:
            reasons.append("cost_model_not_frozen")
        if not risk_profile_frozen:
            reasons.append("risk_profile_not_frozen")
        if not capabilities.paper_support:
            reasons.append("broker_lacks_paper_support")
        if capabilities.live_support:
            # Research framework: refuse brokers that advertise live in this path
            reasons.append("live_capable_broker_not_allowed_for_paper_gate")
        if unresolved_recon_errors > 0:
            reasons.append("unresolved_reconciliation_errors")
        if self.require_kill_switch_tested and not kill_switch_tested:
            reasons.append("kill_switch_not_tested")

        return {
            "accepted": len(reasons) == 0,
            "reasons": reasons,
            "portfolio_id": portfolio.portfolio_id,
            "stage": portfolio.stage.value,
        }

    def promote_or_raise(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        result = self.evaluate(*args, **kwargs)
        if not result["accepted"]:
            raise PaperPromotionError(",".join(result["reasons"]))
        return result


def reject_legacy_sim_promotion(adapter: LegacyAlpacaAdapter) -> None:
    perf = adapter.legacy_sim_performance()
    if not perf.get("promotable", False):
        raise PaperPromotionError(perf["reason"])
