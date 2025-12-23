"""Net leverage constraint.

NetLeverageConstraint: Limits net exposure (longs - shorts) to equity multiple.
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


class NetLeverageConstraint:
    """Limit net exposure to equity multiple.

    Net exposure = sum of signed position values (longs positive, shorts negative).
    This constraint limits abs(net_exposure) to max_net_leverage * equity.

    Unlike gross leverage which limits total absolute exposure,
    net leverage allows balanced long/short portfolios to have
    more positions as long as they're offsetting.

    Example:
        >>> constraint = NetLeverageConstraint()
        >>> result = constraint.apply(orders, portfolio, market, config)
        >>> for r in result.rejected:
        ...     print(f"Rejected {r.order.symbol}: {r.reason}")
    """

    @property
    def name(self) -> str:
        """Human-readable constraint name for logging and audit."""
        return "NetLeverageConstraint"

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
        """Apply net leverage constraint.

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

        equity = portfolio_state.equity
        max_net_leverage = Decimal(str(risk_config.max_net_leverage))
        max_net_exposure = equity * max_net_leverage

        # Calculate current net exposure from positions
        # Use position.market_value which uses current_price or average_price
        current_net_exposure = Decimal("0")
        for position in portfolio_state.positions.values():
            # market_value is signed: positive for long, negative for short
            current_net_exposure += position.market_value

        # Categorize orders into reducing vs increasing net exposure
        reducing_orders: list[OrderRequest] = []
        increasing_orders: list[tuple[OrderRequest, Decimal]] = []  # (order, delta)

        for order in orders:
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

            # Calculate net exposure delta for this order
            if order.side == OrderSide.BUY:
                order_delta = order.quantity * price
            else:
                order_delta = -order.quantity * price

            # Determine if this order increases or decreases absolute net exposure
            new_net_exposure = current_net_exposure + order_delta
            if abs(new_net_exposure) < abs(current_net_exposure):
                # This order reduces absolute net exposure - always pass
                reducing_orders.append(order)
            else:
                # This order increases absolute net exposure - may need constraining
                increasing_orders.append((order, order_delta))

        # Start with reducing orders (always pass)
        result: list[OrderRequest] = list(reducing_orders)

        if not increasing_orders:
            return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)

        # Calculate proposed total delta from increasing orders
        proposed_delta = sum(delta for _, delta in increasing_orders)
        proposed_net_exposure = current_net_exposure + proposed_delta

        if abs(proposed_net_exposure) <= max_net_exposure:
            # All orders fit within limit
            result.extend(order for order, _ in increasing_orders)
            return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)

        # Calculate available room in the direction we're going
        available: Decimal
        if proposed_delta > 0:
            # Trying to go more net long (or less net short)
            available = max_net_exposure - current_net_exposure
        else:
            # Trying to go more net short (or less net long)
            available = max_net_exposure + current_net_exposure

        if available <= 0:
            # No room for more exposure in this direction
            for order, _ in increasing_orders:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Net leverage at max ({risk_config.max_net_leverage}x), "
                        f"no capacity in this direction",
                    )
                )
            return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)

        # Scale down proportionally
        total_delta: Decimal = abs(proposed_delta)  # type: ignore[assignment]
        scale_factor = available / total_delta

        for order, delta in increasing_orders:
            bar = market_state.current_bars.get(order.symbol)
            if bar is None:
                continue

            price = bar.close
            scaled_delta = abs(delta) * scale_factor
            scaled_quantity = (scaled_delta / price).to_integral_value(rounding=ROUND_DOWN)

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
                            f"(net leverage limit {risk_config.max_net_leverage}x)",
                            original_quantity=order.quantity,
                        )
                    )
            else:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Scaled quantity < 1 (net leverage limit {risk_config.max_net_leverage}x)",
                    )
                )

        return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)
