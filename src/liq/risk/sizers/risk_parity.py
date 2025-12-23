"""Risk parity position sizer.

RiskParitySizer: Equal risk contribution from each position.
Sizes positions so each contributes equal volatility to portfolio.

Formula:
    weight_i = (1/vol_i) / Σ(1/vol_j)

Where vol_i is the volatility (ATR) of asset i.
Higher volatility assets get smaller positions.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING, Literal

from liq.risk.types import TargetPosition

if TYPE_CHECKING:
    from liq.core import PortfolioState
    from liq.signals import Signal

    from liq.risk.config import MarketState, RiskConfig


class RiskParitySizer:
    """Equal risk contribution position sizer.

    Sizes positions inversely proportional to their volatility,
    so each position contributes equal risk to the portfolio.

    Formula:
        weight_i = (1/vol_i) / Σ(1/vol_j)
        allocation_i = equity * risk_per_trade * weight_i
        quantity_i = allocation_i / price_i

    Example:
        >>> sizer = RiskParitySizer()
        >>> targets = sizer.size_positions(signals, portfolio, market, config)

    Notes:
        - Higher volatility assets receive smaller positions
        - Equal volatility means equal position sizes
        - Zero or missing volatility signals are skipped
        - Quantities are rounded down to whole shares
    """

    def size_positions(
        self,
        signals: list[Signal],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[TargetPosition]:
        """Size positions using risk parity allocation.

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

        # Filter valid signals and collect their volatilities
        valid_signals: list[tuple[Signal, Decimal, Decimal]] = []  # (signal, vol, price)

        for signal in signals:
            # Skip flat signals
            if signal.direction == "flat":
                continue

            # Get bar data
            bar = market_state.current_bars.get(signal.symbol)
            if bar is None:
                continue

            # Get volatility
            vol = market_state.volatility.get(signal.symbol)
            if vol is None or vol <= 0:
                continue

            # Use midrange price
            price = (bar.high + bar.low) / 2
            valid_signals.append((signal, vol, price))

        if not valid_signals:
            return []

        # Calculate inverse volatility weights
        # weight_i = (1/vol_i) / Σ(1/vol_j)
        inverse_vols = [Decimal("1") / vol for _, vol, _ in valid_signals]
        total_inverse_vol = sum(inverse_vols)

        if total_inverse_vol <= 0:
            return []

        weights = [iv / total_inverse_vol for iv in inverse_vols]

        # Calculate total allocation
        equity = portfolio_state.equity
        risk_per_trade = Decimal(str(risk_config.risk_per_trade))
        total_allocation = equity * risk_per_trade

        targets: list[TargetPosition] = []

        for i, (signal, _vol, price) in enumerate(valid_signals):
            # Allocation for this asset
            allocation = total_allocation * weights[i]

            # Calculate quantity
            quantity = (allocation / price).to_integral_value(rounding=ROUND_DOWN)

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
