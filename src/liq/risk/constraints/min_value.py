"""Minimum position value constraint.

MinPositionValueConstraint: Filters out orders below minimum notional value.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide

from liq.risk.types import ConstraintResult, RejectedOrder

if TYPE_CHECKING:
    from liq.core import PortfolioState

    from liq.risk.config import MarketState, RiskConfig


class MinPositionValueConstraint:
    """Filter orders below minimum notional value.

    Removes orders whose value (quantity * price) is below
    the configured min_position_value threshold.

    Example:
        >>> constraint = MinPositionValueConstraint()
        >>> result = constraint.apply(orders, portfolio, market, config)
        >>> for r in result.rejected:
        ...     print(f"Rejected {r.order.symbol}: {r.reason}")
    """

    @property
    def name(self) -> str:
        """Human-readable constraint name for logging and audit."""
        return "MinPositionValueConstraint"

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
        """Apply minimum value constraint.

        Args:
            orders: Orders to constrain.
            portfolio_state: Current portfolio.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            ConstraintResult with passed orders, rejected orders, and warnings.
        """
        result: list[OrderRequest] = []
        rejected: list[RejectedOrder] = []
        warnings: list[str] = []

        min_value = risk_config.min_position_value

        for order in orders:
            # Sell orders always pass (reducing position)
            if order.side == OrderSide.SELL:
                result.append(order)
                continue

            # Get bar data for price
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
            order_value = order.quantity * price

            if order_value >= min_value:
                result.append(order)
            else:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Order value ${order_value:.2f} below minimum ${min_value:.2f}",
                    )
                )

        return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)
