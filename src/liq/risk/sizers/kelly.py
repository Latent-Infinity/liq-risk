"""Kelly criterion position sizer.

KellySizer: Sizes positions using Kelly criterion for optimal growth.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING, Literal

from liq.risk.types import TargetPosition

if TYPE_CHECKING:
    from liq.core import PortfolioState
    from liq.signals import Signal

    from liq.risk.config import MarketState, RiskConfig


class KellySizer:
    """Position sizer using Kelly criterion.

    The Kelly criterion determines the optimal bet size to maximize
    long-term geometric growth. We use signal strength as a proxy
    for win probability.

    Formula:
        Full Kelly: f* = 2p - 1 (for symmetric returns where b=1)
        Where p = win probability (signal strength)

    We apply fractional Kelly (from config) for safety:
        Actual fraction = f* × kelly_fraction

    Example:
        >>> sizer = KellySizer()
        >>> targets = sizer.size_positions(signals, portfolio, market, config)

    With strength=0.75 and quarter Kelly (0.25):
        - Full Kelly: f* = 2 * 0.75 - 1 = 0.5 (50%)
        - Quarter Kelly: 0.5 * 0.25 = 0.125 (12.5%)
        - Position = equity × 0.125
    """

    def size_positions(
        self,
        signals: list[Signal],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[TargetPosition]:
        """Size positions using Kelly criterion.

        Args:
            signals: Trading signals to size.
            portfolio_state: Current portfolio snapshot.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            List of TargetPosition objects with target quantities.
        """
        if not signals:
            return []

        equity = portfolio_state.equity
        kelly_fraction = Decimal(str(risk_config.kelly_fraction))

        targets: list[TargetPosition] = []

        for signal in signals:
            # Skip flat signals
            if signal.direction == "flat":
                continue

            # Get bar data
            bar = market_state.current_bars.get(signal.symbol)
            if bar is None:
                continue

            # Calculate Kelly fraction
            # p = signal strength (win probability proxy)
            # Full Kelly for symmetric returns: f* = 2p - 1
            p = Decimal(str(signal.strength))
            full_kelly = 2 * p - 1

            # If no edge (f* <= 0), skip this signal
            if full_kelly <= 0:
                continue

            # Apply fractional Kelly for safety
            position_fraction = full_kelly * kelly_fraction

            # Calculate position value and quantity
            position_value = equity * position_fraction
            price = bar.close
            quantity = (position_value / price).to_integral_value(rounding=ROUND_DOWN)

            if quantity < 1:
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
