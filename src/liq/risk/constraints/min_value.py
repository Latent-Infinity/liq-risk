"""Minimum position value constraint.

MinPositionValueConstraint: Filters out orders below minimum notional value.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide

if TYPE_CHECKING:
    from liq.core import PortfolioState

    from liq.risk.config import MarketState, RiskConfig


class MinPositionValueConstraint:
    """Filter orders below minimum notional value.

    Removes orders whose value (quantity * price) is below
    the configured min_position_value threshold.

    Example:
        >>> constraint = MinPositionValueConstraint()
        >>> orders = constraint.apply(orders, portfolio, market, config)
    """

    def apply(
        self,
        orders: list[OrderRequest],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[OrderRequest]:
        """Apply minimum value constraint.

        Args:
            orders: Orders to constrain.
            portfolio_state: Current portfolio.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            Filtered order list with orders below minimum removed.
        """
        result: list[OrderRequest] = []
        min_value = risk_config.min_position_value

        for order in orders:
            # Sell orders always pass (reducing position)
            if order.side == OrderSide.SELL:
                result.append(order)
                continue

            # Get bar data for price
            bar = market_state.current_bars.get(order.symbol)
            if bar is None:
                continue

            price = bar.close
            order_value = order.quantity * price

            if order_value >= min_value:
                result.append(order)

        return result
