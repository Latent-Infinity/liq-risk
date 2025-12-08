"""Leverage constraints.

GrossLeverageConstraint: Limits total gross exposure / equity ratio.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide

if TYPE_CHECKING:
    from liq.core import PortfolioState

    from liq.risk.config import MarketState, RiskConfig


class GrossLeverageConstraint:
    """Limit total gross exposure as multiple of equity.

    Scales down orders proportionally if they would result in
    gross exposure exceeding the configured max_gross_leverage.

    Gross exposure = sum of absolute position values.

    Example:
        >>> constraint = GrossLeverageConstraint()
        >>> orders = constraint.apply(orders, portfolio, market, config)
    """

    def apply(
        self,
        orders: list[OrderRequest],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[OrderRequest]:
        """Apply gross leverage constraint.

        Args:
            orders: Orders to constrain.
            portfolio_state: Current portfolio.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            Constrained order list with scaled quantities.
        """
        equity = portfolio_state.equity
        max_exposure = equity * Decimal(str(risk_config.max_gross_leverage))

        # Calculate current gross exposure
        current_exposure = Decimal("0")
        for position in portfolio_state.positions.values():
            current_exposure += abs(position.market_value)

        # Separate buy and sell orders
        buy_orders: list[OrderRequest] = []
        sell_orders: list[OrderRequest] = []

        for order in orders:
            if order.side == OrderSide.SELL:
                sell_orders.append(order)
            else:
                buy_orders.append(order)

        # Sell orders always pass (reduce exposure)
        result: list[OrderRequest] = list(sell_orders)

        if not buy_orders:
            return result

        # Calculate total new exposure from buy orders
        total_new_exposure = Decimal("0")
        order_values: list[tuple[OrderRequest, Decimal]] = []

        for order in buy_orders:
            bar = market_state.current_bars.get(order.symbol)
            if bar is None:
                continue

            price = bar.close
            order_value = order.quantity * price
            total_new_exposure += order_value
            order_values.append((order, order_value))

        if not order_values:
            return result

        # Calculate remaining capacity
        remaining_capacity = max_exposure - current_exposure

        if remaining_capacity <= 0:
            # Already at or over limit
            return result

        if total_new_exposure <= remaining_capacity:
            # All orders fit
            result.extend(order for order, _ in order_values)
            return result

        # Scale down proportionally
        scale_factor = remaining_capacity / total_new_exposure

        for order, order_value in order_values:
            bar = market_state.current_bars.get(order.symbol)
            if bar is None:
                continue

            price = bar.close
            scaled_value = order_value * scale_factor
            scaled_quantity = (scaled_value / price).to_integral_value(
                rounding=ROUND_DOWN
            )

            if scaled_quantity >= 1:
                new_order = OrderRequest(
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=scaled_quantity,
                    limit_price=order.limit_price,
                    stop_price=order.stop_price,
                    time_in_force=order.time_in_force,
                    timestamp=order.timestamp,
                    strategy_id=order.strategy_id,
                    confidence=order.confidence,
                    tags=order.tags,
                    metadata=order.metadata,
                )
                result.append(new_order)

        return result
