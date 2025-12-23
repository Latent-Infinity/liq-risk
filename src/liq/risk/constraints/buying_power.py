"""Buying power constraint.

BuyingPowerConstraint: Limits buy orders to available cash.
"""

from __future__ import annotations

import logging
from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide

from liq.risk.types import ConstraintResult, RejectedOrder

if TYPE_CHECKING:
    from liq.core import PortfolioState

    from liq.risk.config import MarketState, RiskConfig

logger = logging.getLogger(__name__)


class BuyingPowerConstraint:
    """Limit buy orders to available cash.

    Ensures that the total value of buy orders does not exceed
    available cash, accounting for commission and slippage.

    Note:
        Sell orders pass freely (they generate cash, not consume it).
        Buy orders covering shorts still require cash for purchase.

    Example:
        >>> constraint = BuyingPowerConstraint()
        >>> result = constraint.apply(orders, portfolio, market, config)
        >>> for r in result.rejected:
        ...     print(f"Rejected {r.order.symbol}: {r.reason}")
    """

    @property
    def name(self) -> str:
        """Human-readable constraint name for logging and audit."""
        return "BuyingPowerConstraint"

    def classify_risk(
        self,
        order: OrderRequest,
        portfolio_state: PortfolioState,
    ) -> bool:
        """Classify if this order is risk-increasing.

        Args:
            order: The order to classify.
            portfolio_state: Current portfolio for context.

        Returns:
            True if risk-increasing, False if risk-reducing.
        """
        position = portfolio_state.positions.get(order.symbol)
        current_qty = position.quantity if position else Decimal("0")

        if order.side == OrderSide.BUY:
            return current_qty >= 0
        else:
            return current_qty <= 0

    def apply(
        self,
        orders: list[OrderRequest],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> ConstraintResult:
        """Apply buying power constraint.

        Args:
            orders: Orders to constrain.
            portfolio_state: Current portfolio.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            ConstraintResult with passed orders, rejected orders, and warnings.
        """
        rejected: list[RejectedOrder] = []
        warnings: list[str] = []

        cash = portfolio_state.cash

        # Calculate cost multiplier from fees
        commission_pct = Decimal(str(risk_config.default_commission_pct))
        slippage_pct = Decimal(str(risk_config.default_slippage_pct))
        cost_multiplier = Decimal("1") + commission_pct + slippage_pct

        # Separate buy and sell orders
        sell_orders: list[OrderRequest] = []
        buy_orders: list[tuple[OrderRequest, Decimal]] = []  # (order, cost)

        for order in orders:
            if order.side == OrderSide.SELL:
                # Sell orders pass freely
                sell_orders.append(order)
            else:
                # Buy orders need cash check
                bar = market_state.current_bars.get(order.symbol)
                if bar is None:
                    rejected.append(
                        RejectedOrder(
                            order=order,
                            constraint_name=self.name,
                            reason=f"No bar data for {order.symbol}",
                        )
                    )
                    continue

                price = bar.close
                order_cost = order.quantity * price * cost_multiplier
                buy_orders.append((order, order_cost))

        # Start with sell orders (always pass)
        result: list[OrderRequest] = list(sell_orders)

        if not buy_orders:
            return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)

        # Calculate total buy cost
        total_buy_cost = sum(cost for _, cost in buy_orders)

        if total_buy_cost <= cash:
            # All buys fit within cash
            result.extend(order for order, _ in buy_orders)
            return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)

        if cash <= 0:
            # No cash available, no buys allowed
            for order, _ in buy_orders:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason="No cash available for buy orders",
                    )
                )
            return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)

        # Scale down proportionally
        scale_factor = cash / total_buy_cost

        for order, order_cost in buy_orders:
            bar = market_state.current_bars.get(order.symbol)
            if bar is None:
                continue

            price = bar.close
            # Calculate scaled cost and back-calculate quantity
            scaled_cost = order_cost * scale_factor
            # Quantity = cost / (price * cost_multiplier)
            scaled_quantity = (scaled_cost / (price * cost_multiplier)).to_integral_value(
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
                if scaled_quantity < order.quantity:
                    rejected.append(
                        RejectedOrder(
                            order=order,
                            constraint_name=self.name,
                            reason=f"Scaled from {order.quantity} to {scaled_quantity} "
                            f"(insufficient cash ${cash:.2f})",
                            original_quantity=order.quantity,
                        )
                    )
            else:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Scaled quantity < 1 (insufficient cash ${cash:.2f})",
                    )
                )

        return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)
