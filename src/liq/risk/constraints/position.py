"""Position-related constraints.

MaxPositionConstraint: Limits individual position size as % of equity.
MaxPositionsConstraint: Limits total number of concurrent positions.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide

if TYPE_CHECKING:
    from liq.core import PortfolioState

    from liq.risk.config import MarketState, RiskConfig


class MaxPositionConstraint:
    """Limit individual position size as percentage of equity.

    Reduces order quantity if it would result in a position
    exceeding the configured max_position_pct limit.

    Example:
        >>> constraint = MaxPositionConstraint()
        >>> orders = constraint.apply(orders, portfolio, market, config)
    """

    def apply(
        self,
        orders: list[OrderRequest],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[OrderRequest]:
        """Apply max position constraint.

        Args:
            orders: Orders to constrain.
            portfolio_state: Current portfolio.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            Constrained order list with reduced quantities.
        """
        result: list[OrderRequest] = []
        equity = portfolio_state.equity
        max_position_value = equity * Decimal(str(risk_config.max_position_pct))

        for order in orders:
            # Sell orders are not limited
            if order.side == OrderSide.SELL:
                result.append(order)
                continue

            # Get bar data for price
            bar = market_state.current_bars.get(order.symbol)
            if bar is None:
                continue

            price = bar.close

            # Get existing position value
            existing_position = portfolio_state.positions.get(order.symbol)
            existing_value = Decimal("0")
            if existing_position is not None:
                existing_value = abs(existing_position.market_value)

            # Calculate remaining room
            remaining_room = max_position_value - existing_value

            if remaining_room <= 0:
                # Already at or over limit
                continue

            # Calculate order value
            order_value = order.quantity * price

            if order_value <= remaining_room:
                # Order fits within limit
                result.append(order)
            else:
                # Scale down order
                max_quantity = (remaining_room / price).to_integral_value(
                    rounding=ROUND_DOWN
                )
                if max_quantity >= 1:
                    new_order = OrderRequest(
                        client_order_id=order.client_order_id,
                        symbol=order.symbol,
                        side=order.side,
                        order_type=order.order_type,
                        quantity=max_quantity,
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


class MaxPositionsConstraint:
    """Limit total number of concurrent positions.

    Filters orders that would create new positions beyond the
    configured max_positions limit. Orders are prioritized by
    confidence (signal strength).

    Example:
        >>> constraint = MaxPositionsConstraint()
        >>> orders = constraint.apply(orders, portfolio, market, config)
    """

    def apply(
        self,
        orders: list[OrderRequest],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[OrderRequest]:
        """Apply max positions constraint.

        Args:
            orders: Orders to constrain.
            portfolio_state: Current portfolio.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            Filtered order list respecting position count limit.
        """
        existing_symbols = set(portfolio_state.positions.keys())
        current_count = len(existing_symbols)
        max_positions = risk_config.max_positions

        # Separate orders into categories
        sell_orders: list[OrderRequest] = []
        existing_position_orders: list[OrderRequest] = []
        new_position_orders: list[OrderRequest] = []

        for order in orders:
            if order.side == OrderSide.SELL:
                # Sell orders always pass
                sell_orders.append(order)
            elif order.symbol in existing_symbols:
                # Adding to existing position
                existing_position_orders.append(order)
            else:
                # New position
                new_position_orders.append(order)

        # Calculate room for new positions
        room_for_new = max_positions - current_count

        if room_for_new <= 0:
            # No room for new positions
            return sell_orders + existing_position_orders

        # Sort new position orders by confidence (descending)
        new_position_orders.sort(
            key=lambda o: o.confidence if o.confidence is not None else 0.0,
            reverse=True,
        )

        # Take top N new positions
        accepted_new = new_position_orders[:room_for_new]

        return sell_orders + existing_position_orders + accepted_new
