"""Fixed fractional position sizing.

FixedFractionalSizer allocates a fixed percentage of equity to each position,
providing simple and predictable position sizing.

Formula:
    quantity = (equity * fraction) / price
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide, OrderType

if TYPE_CHECKING:
    from liq.core import PortfolioState
    from liq.signals import Signal

    from liq.risk.config import MarketState, RiskConfig


class FixedFractionalSizer:
    """Allocate fixed percentage of equity to each position.

    Simple sizing strategy that allocates a constant fraction of
    portfolio equity to each new position.

    Attributes:
        fraction: Fraction of equity to allocate per position (0, 1].

    Example:
        >>> sizer = FixedFractionalSizer(fraction=0.02)  # 2% per position
        >>> orders = sizer.size_positions(signals, portfolio, market, config)
    """

    def __init__(self, fraction: float = 0.02) -> None:
        """Initialize FixedFractionalSizer.

        Args:
            fraction: Fraction of equity to allocate per position.
                     Must be in range (0, 1]. Default is 0.02 (2%).

        Raises:
            ValueError: If fraction is not in valid range.
        """
        if fraction <= 0 or fraction > 1:
            raise ValueError(f"fraction must be in range (0, 1], got {fraction}")
        self._fraction = fraction

    @property
    def fraction(self) -> float:
        """Get the allocation fraction."""
        return self._fraction

    def size_positions(
        self,
        signals: list[Signal],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[OrderRequest]:
        """Size positions using fixed fraction of equity.

        Args:
            signals: Trading signals to size.
            portfolio_state: Current portfolio snapshot.
            market_state: Current market conditions.
            risk_config: Risk parameters (not used, kept for protocol).

        Returns:
            List of sized OrderRequest objects.
        """
        orders: list[OrderRequest] = []
        equity = portfolio_state.equity

        for signal in signals:
            # Skip flat signals
            if signal.direction == "flat":
                continue

            # Get bar data
            bar = market_state.current_bars.get(signal.symbol)
            if bar is None:
                continue

            # Use close price for sizing
            price = bar.close

            if price <= 0:
                continue

            # Calculate quantity using fixed fractional formula
            # qty = (equity * fraction) / price
            allocation = equity * Decimal(str(self._fraction))
            raw_quantity = allocation / price

            # Round down to whole shares
            quantity = raw_quantity.to_integral_value(rounding=ROUND_DOWN)

            # Skip if quantity < 1
            if quantity < 1:
                continue

            # Determine order side
            side = OrderSide.BUY if signal.direction == "long" else OrderSide.SELL

            # Create order
            order = OrderRequest(
                symbol=signal.symbol,
                side=side,
                order_type=OrderType.MARKET,
                quantity=quantity,
                timestamp=signal.normalized_timestamp(),
                confidence=signal.strength,
            )
            orders.append(order)

        return orders
