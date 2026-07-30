from __future__ import annotations

from trading_ecosystem.common.contracts import (
    Bar,
    OrderIntent,
    OrderSide,
    OrderType,
    Position,
    Signal,
    SignalAction,
)


def long_only_intents(
    strategy_id: str,
    signal: Signal,
    bar: Bar,
    position: Position | None,
    params: dict,
) -> list[OrderIntent]:
    """Shared long-only market entry/exit sizing used by simple strategy families."""
    risk_fraction = float(params.get("risk_fraction", 0.0075))
    price_ref = str(params.get("price_ref") or "close").lower()
    ref_px = float(bar.open) if price_ref == "open" else float(bar.close)

    if signal.action == SignalAction.ENTER_LONG:
        stop = signal.stop_price
        if stop is None or stop >= ref_px:
            return []
        risk_per_share = ref_px - stop
        if risk_per_share <= 0:
            return []
        equity_hint = float(params.get("equity_hint", 100_000.0))
        qty = (equity_hint * risk_fraction) / risk_per_share
        qty = max(1.0, float(int(qty)))
        return [
            OrderIntent(
                strategy_id=strategy_id,
                symbol=bar.symbol,
                side=OrderSide.BUY,
                qty=qty,
                order_type=OrderType.MARKET,
                stop_price=stop,
                ts=bar.ts,
                meta={"signal": signal.action.value, "price_ref": price_ref},
            )
        ]

    if signal.action == SignalAction.EXIT_LONG and position and position.qty > 0:
        return [
            OrderIntent(
                strategy_id=strategy_id,
                symbol=bar.symbol,
                side=OrderSide.SELL,
                qty=abs(position.qty),
                order_type=OrderType.MARKET,
                ts=bar.ts,
                reduce_only=True,
                meta={"signal": signal.action.value},
            )
        ]

    if signal.action == SignalAction.HOLD and signal.stop_price and position:
        position.stop_price = signal.stop_price
    return []
