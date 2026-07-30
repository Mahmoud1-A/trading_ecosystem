"""Maker/taker fees + slippage model for realistic fills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trading_ecosystem.common.config import CONFIG_DIR, load_yaml
from trading_ecosystem.common.contracts import OrderSide


@dataclass(frozen=True)
class ExecutionCosts:
    slippage_bps: float = 5.0
    maker_fee_bps: float = 0.0
    taker_fee_bps: float = 0.0
    commission_per_share: float = 0.0
    default_liquidity: str = "taker"
    profile: str = "etf"

    def liquidity(self, requested: str | None = None) -> str:
        liq = (requested or self.default_liquidity or "taker").strip().lower()
        return "maker" if liq == "maker" else "taker"

    def fee_bps(self, liquidity: str | None = None) -> float:
        return self.maker_fee_bps if self.liquidity(liquidity) == "maker" else self.taker_fee_bps

    def fill_price(self, mid: float, side: OrderSide, liquidity: str | None = None) -> float:
        """Adverse slippage vs mid. Maker still pays half-spread proxy via slippage_bps."""
        if mid <= 0:
            return mid
        slip = mid * (self.slippage_bps / 10_000.0)
        # Maker assumes slightly better fill (half slippage).
        if self.liquidity(liquidity) == "maker":
            slip *= 0.5
        return mid + slip if side == OrderSide.BUY else mid - slip

    def commission(self, qty: float, price: float, liquidity: str | None = None) -> float:
        notional = abs(float(qty) * float(price))
        fee = notional * (self.fee_bps(liquidity) / 10_000.0)
        fee += self.commission_per_share * abs(float(qty))
        return float(fee)

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "slippage_bps": self.slippage_bps,
            "maker_fee_bps": self.maker_fee_bps,
            "taker_fee_bps": self.taker_fee_bps,
            "commission_per_share": self.commission_per_share,
            "default_liquidity": self.default_liquidity,
        }


def _resolve_profile(universe_id: str | None, profile: str | None) -> str:
    if profile and str(profile).strip():
        return str(profile).strip().lower()
    uid = (universe_id or "etf").strip().lower()
    uni_path = CONFIG_DIR / "universes" / f"{uid}.yaml"
    if uni_path.exists():
        try:
            raw = load_yaml(uni_path)
            return str(raw.get("cost_profile") or uid).strip().lower()
        except Exception:  # noqa: BLE001
            return uid
    return uid


def load_execution_costs(
    *,
    universe_id: str | None = None,
    profile: str | None = None,
) -> ExecutionCosts:
    cfg = load_yaml(CONFIG_DIR / "execution_costs.yaml")
    profiles = cfg.get("profiles") or {}
    prof = _resolve_profile(universe_id, profile)
    raw = dict(profiles.get(prof) or profiles.get("etf") or {})
    return ExecutionCosts(
        slippage_bps=float(raw.get("slippage_bps", 5.0)),
        maker_fee_bps=float(raw.get("maker_fee_bps", 0.0)),
        taker_fee_bps=float(raw.get("taker_fee_bps", 0.0)),
        commission_per_share=float(raw.get("commission_per_share", 0.0)),
        default_liquidity=str(raw.get("default_liquidity") or "taker"),
        profile=prof or "etf",
    )
