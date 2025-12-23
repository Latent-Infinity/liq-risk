"""Short selling constraint.

ShortSellingConstraint: Filters or trims sell orders that would create short positions
when shorting is disabled.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide

from liq.risk.types import ConstraintResult, RejectedOrder

if TYPE_CHECKING:
    from liq.core import PortfolioState

    from liq.risk.config import MarketState, RiskConfig


class ShortSellingConstraint:
    """Filter sell orders that would create short positions when shorts disabled.

    When allow_shorts=False in RiskConfig:
    - Buy orders always pass
    - Sell orders are allowed only to close existing long positions
    - Sell quantity is trimmed to avoid going short

    When allow_shorts=True (default):
    - All orders pass through unchanged

    Example:
        >>> constraint = ShortSellingConstraint()
        >>> result = constraint.apply(orders, portfolio, market, config)
        >>> for r in result.rejected:
        ...     print(f"Rejected {r.order.symbol}: {r.reason}")
    """

    @property
    def name(self) -> str:
        """Human-readable constraint name for logging and audit."""
        return "ShortSellingConstraint"

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
        """Apply short selling constraint.

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

        # If shorts are allowed, pass all orders through unchanged
        if risk_config.allow_shorts:
            return ConstraintResult(orders=list(orders), rejected=rejected, warnings=warnings)

        result: list[OrderRequest] = []

        for order in orders:
            # Buy orders always pass
            if order.side == OrderSide.BUY:
                result.append(order)
                continue

            # For sell orders, check if it would go short
            position = portfolio_state.positions.get(order.symbol)
            current_qty = position.quantity if position else Decimal("0")

            # If current position is zero or already short, block the sell
            if current_qty <= 0:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason="Short selling not allowed (allow_shorts=False)",
                    )
                )
                continue

            # If sell quantity exceeds position, trim to position size
            if order.quantity > current_qty:
                # Create new order with trimmed quantity
                new_order = OrderRequest(
                    client_order_id=order.client_order_id,
                    symbol=order.symbol,
                    side=order.side,
                    order_type=order.order_type,
                    quantity=current_qty,  # Trim to position size
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
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Trimmed from {order.quantity} to {current_qty} "
                        f"to avoid short position (allow_shorts=False)",
                        original_quantity=order.quantity,
                    )
                )
            else:
                # Sell is within position size - pass unchanged
                result.append(order)

        return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)
