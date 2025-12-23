"""Crypto fractional position sizing.

Allows fractional quantities (e.g., BTC min lot) with step sizes.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING, Literal

from liq.risk.types import TargetPosition

if TYPE_CHECKING:
    from liq.core import PortfolioState
    from liq.signals import Signal

    from liq.risk.config import MarketState, RiskConfig


class CryptoFractionalSizer:
    """Allocate fraction of equity to crypto positions with fractional lots.

    Designed for cryptocurrency trading where fractional quantities are common.

    Args:
        fraction: Fraction of equity to allocate per position (0, 1].
        min_qty: Minimum quantity for an order. Orders below this are skipped.
        step_qty: Step size for quantity rounding. If None, uses 4 decimal places.

    Example:
        >>> sizer = CryptoFractionalSizer(fraction=0.02, min_qty=Decimal("0.0001"))
        >>> targets = sizer.size_positions(signals, portfolio, market, config)

    Raises:
        ValueError: If fraction, min_qty, or step_qty have invalid values.
    """

    def __init__(
        self,
        fraction: float = 0.02,
        min_qty: Decimal = Decimal("0.0001"),
        step_qty: Decimal | None = Decimal("0.0001"),
    ) -> None:
        # Validate fraction
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"fraction must be in range (0, 1], got {fraction}")

        # Validate min_qty
        if min_qty <= 0:
            raise ValueError(f"min_qty must be positive, got {min_qty}")

        # Validate step_qty
        if step_qty is not None and step_qty <= 0:
            raise ValueError(f"step_qty must be positive if provided, got {step_qty}")

        self._fraction = fraction
        self._min_qty = min_qty
        self._step_qty = step_qty

    @property
    def fraction(self) -> float:
        return self._fraction

    def size_positions(
        self,
        signals: list[Signal],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[TargetPosition]:
        """Size positions for crypto with fractional lots.

        Args:
            signals: Trading signals to size.
            portfolio_state: Current portfolio snapshot.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            List of TargetPosition objects with target quantities.
        """
        targets: list[TargetPosition] = []
        equity = portfolio_state.equity

        for signal in signals:
            if signal.direction == "flat":
                continue
            bar = market_state.current_bars.get(signal.symbol)
            if bar is None:
                continue
            price = bar.close
            if price <= 0:
                continue

            allocation = equity * Decimal(str(self._fraction))
            raw_quantity = allocation / price

            if self._step_qty:
                steps = (raw_quantity // self._step_qty) * self._step_qty
                quantity = steps
            else:
                quantity = raw_quantity.quantize(Decimal("0.0001"), rounding=ROUND_DOWN)

            if quantity <= 0 or quantity < self._min_qty:
                continue

            # Get current position quantity
            position = portfolio_state.positions.get(signal.symbol)
            current_quantity = position.quantity if position else Decimal("0")

            # Determine target quantity and direction
            direction: Literal["long", "short", "flat"]
            if signal.direction == "long":
                target_quantity = quantity
                direction = "long"
            else:
                target_quantity = -quantity
                direction = "short"

            target = TargetPosition(
                symbol=signal.symbol,
                target_quantity=target_quantity,
                current_quantity=current_quantity,
                direction=direction,
                signal_strength=signal.strength,
            )
            targets.append(target)

        return targets
