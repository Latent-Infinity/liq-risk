"""Volatility-based position sizing.

VolatilitySizer scales position size inversely with volatility,
ensuring consistent risk per trade regardless of market conditions.

Formula:
    quantity = (equity * risk_per_trade) / (price * atr_multiple * atr)

Higher volatility → smaller position.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide, OrderType

if TYPE_CHECKING:
    from liq.core import PortfolioState
    from liq.signals import Signal

    from liq.risk.config import MarketState, RiskConfig


class VolatilitySizer:
    """Scale position inversely with volatility.

    Uses ATR (Average True Range) or range-based volatility to determine
    position size such that each trade risks approximately the same
    dollar amount.

    Attributes:
        risk_per_trade: Override for config.risk_per_trade if provided.
        atr_multiple: Stop-loss distance in ATR multiples.
        use_midrange_price: If True, use (high+low)/2 instead of close.

    Example:
        >>> sizer = VolatilitySizer(atr_multiple=2.0)
        >>> orders = sizer.size_positions(signals, portfolio, market, config)
    """

    def __init__(
        self,
        risk_per_trade: float | None = None,
        atr_multiple: float = 2.0,
        use_midrange_price: bool = True,
    ) -> None:
        """Initialize VolatilitySizer.

        Args:
            risk_per_trade: Fraction of equity to risk per trade.
                           If None, uses config.risk_per_trade.
            atr_multiple: Stop-loss distance in ATR multiples.
            use_midrange_price: Use midrange price instead of close.
        """
        self.risk_per_trade = risk_per_trade
        self.atr_multiple = atr_multiple
        self.use_midrange_price = use_midrange_price

    def size_positions(
        self,
        signals: list[Signal],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[OrderRequest]:
        """Size positions based on volatility-adjusted risk.

        Args:
            signals: Trading signals to size.
            portfolio_state: Current portfolio snapshot.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            List of sized OrderRequest objects.
        """
        orders: list[OrderRequest] = []
        equity = portfolio_state.equity
        risk_pct = (
            self.risk_per_trade
            if self.risk_per_trade is not None
            else risk_config.risk_per_trade
        )

        for signal in signals:
            # Skip flat signals
            if signal.direction == "flat":
                continue

            # Get bar data
            bar = market_state.current_bars.get(signal.symbol)
            if bar is None:
                continue

            # Get volatility
            volatility = market_state.volatility.get(signal.symbol)
            if volatility is None or volatility <= 0:
                continue

            # Calculate price to use
            if self.use_midrange_price:
                price = bar.midrange
            else:
                price = bar.close

            # Calculate quantity using volatility sizing formula
            # qty = (equity * risk_per_trade) / (price * atr_multiple * atr)
            risk_amount = equity * Decimal(str(risk_pct))
            divisor = price * Decimal(str(self.atr_multiple)) * volatility

            if divisor <= 0:
                continue

            raw_quantity = risk_amount / divisor

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
