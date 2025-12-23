"""Equal weight position sizer.

EqualWeightSizer: Allocates equal dollar weight to each signal.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING, Literal

from liq.risk.types import TargetPosition

if TYPE_CHECKING:
    from liq.core import PortfolioState
    from liq.signals import Signal

    from liq.risk.config import MarketState, RiskConfig


class EqualWeightSizer:
    """Position sizer that allocates equal weight to each signal.

    Divides equity equally among all signals (or max_positions if fewer).
    Uses close price for sizing calculations.

    Example:
        >>> sizer = EqualWeightSizer()
        >>> targets = sizer.size_positions(signals, portfolio, market, config)

    With 3 signals and $100,000 equity:
        - Each signal gets $33,333 allocation
        - Shares = allocation / price (rounded down)
    """

    def size_positions(
        self,
        signals: list[Signal],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[TargetPosition]:
        """Size positions using equal weight allocation.

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

        # Filter out flat signals
        active_signals = [s for s in signals if s.direction != "flat"]
        if not active_signals:
            return []

        # Sort by strength (descending) for max_positions limit
        active_signals = sorted(active_signals, key=lambda s: s.strength, reverse=True)

        # Limit to max_positions
        max_positions = risk_config.max_positions
        if len(active_signals) > max_positions:
            active_signals = active_signals[:max_positions]

        # Calculate allocation per signal
        equity = portfolio_state.equity
        n_signals = len(active_signals)
        allocation_per_signal = equity / Decimal(str(n_signals))

        targets: list[TargetPosition] = []

        for signal in active_signals:
            bar = market_state.current_bars.get(signal.symbol)
            if bar is None:
                continue

            price = bar.close

            # Calculate quantity
            quantity = (allocation_per_signal / price).to_integral_value(
                rounding=ROUND_DOWN
            )

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
