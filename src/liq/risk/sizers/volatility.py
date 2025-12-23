"""Volatility-based position sizing.

VolatilitySizer scales position size inversely with volatility,
ensuring consistent risk per trade regardless of market conditions.

Formula:
    quantity = (equity * risk_per_trade) / (price * atr_multiple * atr)

Higher volatility → smaller position.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING, Literal

from liq.risk.types import TargetPosition

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
        >>> targets = sizer.size_positions(signals, portfolio, market, config)
    """

    def __init__(
        self,
        risk_per_trade: float | None = None,
        atr_multiple: float = 2.0,
        use_midrange_price: bool = True,
        min_quantity: Decimal = Decimal("0.0001"),
        quantize_step: Decimal | None = Decimal("0.0001"),
    ) -> None:
        """Initialize VolatilitySizer.

        Args:
            risk_per_trade: Fraction of equity to risk per trade.
                           If None, uses config.risk_per_trade.
            atr_multiple: Stop-loss distance in ATR multiples.
            use_midrange_price: Use midrange price instead of close.
            min_quantity: Minimum tradable quantity (default supports crypto fractional sizing).
            quantize_step: Optional lot size to quantize quantities (None = no quantization).
        """
        self.risk_per_trade = risk_per_trade
        self.atr_multiple = atr_multiple
        self.use_midrange_price = use_midrange_price
        self.min_quantity = min_quantity
        self.quantize_step = quantize_step

    def size_positions(
        self,
        signals: list[Signal],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[TargetPosition]:
        """Size positions based on volatility-adjusted risk.

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
            if self.quantize_step:
                steps = (raw_quantity / self.quantize_step).to_integral_value(rounding=ROUND_DOWN)
                quantity = steps * self.quantize_step
            else:
                quantity = raw_quantity

            # Skip if below minimum tradable size
            if quantity < self.min_quantity:
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

            # Calculate stop price based on volatility
            stop_distance = volatility * Decimal(str(self.atr_multiple))
            if direction == "long":
                stop_price = price - stop_distance
            else:
                stop_price = price + stop_distance

            # Create target position
            target = TargetPosition(
                symbol=signal.symbol,
                target_quantity=target_quantity,
                current_quantity=current_quantity,
                direction=direction,
                signal_strength=signal.strength,
                stop_price=stop_price,
            )
            targets.append(target)

        return targets
