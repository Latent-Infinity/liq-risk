"""Leverage constraints.

GrossLeverageConstraint: Limits total gross exposure / equity ratio.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide

from liq.risk.types import ConstraintResult, RejectedOrder

if TYPE_CHECKING:
    from liq.core import PortfolioState

    from liq.risk.config import MarketState, RiskConfig


class GrossLeverageConstraint:
    """Limit total gross exposure as multiple of equity.

    Scales down orders proportionally if they would result in
    gross exposure exceeding the configured max_gross_leverage.

    Gross exposure = sum of absolute position values.

    Note:
        This constraint correctly handles short positions:
        - SELL orders that close/reduce long positions pass freely
        - SELL orders that initiate/increase short positions are constrained

    Example:
        >>> constraint = GrossLeverageConstraint()
        >>> result = constraint.apply(orders, portfolio, market, config)
        >>> for r in result.rejected:
        ...     print(f"Rejected {r.order.symbol}: {r.reason}")
    """

    @property
    def name(self) -> str:
        """Human-readable constraint name for logging and audit."""
        return "GrossLeverageConstraint"

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
            # Buying when short = reducing risk
            return current_qty >= 0
        else:
            # Selling when long = reducing risk
            return current_qty <= 0

    def apply(
        self,
        orders: list[OrderRequest],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> ConstraintResult:
        """Apply gross leverage constraint.

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
        max_exposure = equity * Decimal(str(risk_config.max_gross_leverage))

        # Calculate current gross exposure
        current_exposure = Decimal("0")
        for position in portfolio_state.positions.values():
            current_exposure += abs(position.market_value)

        # Categorize orders based on whether they increase or reduce exposure
        exposure_reducing_orders: list[OrderRequest] = []
        exposure_increasing_orders: list[tuple[OrderRequest, Decimal]] = []

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
            maybe_position = portfolio_state.positions.get(order.symbol)
            current_qty = maybe_position.quantity if maybe_position else Decimal("0")

            # Determine if this order increases or reduces gross exposure
            if order.side == OrderSide.BUY:
                if current_qty < 0:
                    # Covering a short - how much reduces vs increases exposure
                    cover_qty = min(order.quantity, abs(current_qty))
                    new_long_qty = order.quantity - cover_qty

                    # Cover portion always passes (reduces exposure)
                    if cover_qty > 0:
                        if new_long_qty > 0:
                            # Split: cover passes, new long constrained
                            cover_order = OrderRequest(
                                client_order_id=order.client_order_id,
                                symbol=order.symbol,
                                side=order.side,
                                order_type=order.order_type,
                                quantity=cover_qty,
                                limit_price=order.limit_price,
                                stop_price=order.stop_price,
                                time_in_force=order.time_in_force,
                                timestamp=order.timestamp,
                                strategy_id=order.strategy_id,
                                confidence=order.confidence,
                                tags=order.tags,
                                metadata=order.metadata,
                            )
                            exposure_reducing_orders.append(cover_order)

                            # Create order for the new long portion
                            new_long_order = OrderRequest(
                                client_order_id=order.client_order_id,
                                symbol=order.symbol,
                                side=order.side,
                                order_type=order.order_type,
                                quantity=new_long_qty,
                                limit_price=order.limit_price,
                                stop_price=order.stop_price,
                                time_in_force=order.time_in_force,
                                timestamp=order.timestamp,
                                strategy_id=order.strategy_id,
                                confidence=order.confidence,
                                tags=order.tags,
                                metadata=order.metadata,
                            )
                            exposure_increasing_orders.append(
                                (new_long_order, new_long_qty * price)
                            )
                        else:
                            # All cover - passes freely
                            exposure_reducing_orders.append(order)
                    else:
                        # No short to cover, all increases exposure
                        exposure_increasing_orders.append(
                            (order, order.quantity * price)
                        )
                else:
                    # No short position - buy increases exposure
                    exposure_increasing_orders.append((order, order.quantity * price))
            else:  # SELL
                if current_qty > 0:
                    # Closing a long - how much closes vs goes short
                    close_qty = min(order.quantity, current_qty)
                    new_short_qty = order.quantity - close_qty

                    # Close portion always passes (reduces exposure)
                    if close_qty > 0:
                        if new_short_qty > 0:
                            # Split: close passes, new short constrained
                            close_order = OrderRequest(
                                client_order_id=order.client_order_id,
                                symbol=order.symbol,
                                side=order.side,
                                order_type=order.order_type,
                                quantity=close_qty,
                                limit_price=order.limit_price,
                                stop_price=order.stop_price,
                                time_in_force=order.time_in_force,
                                timestamp=order.timestamp,
                                strategy_id=order.strategy_id,
                                confidence=order.confidence,
                                tags=order.tags,
                                metadata=order.metadata,
                            )
                            exposure_reducing_orders.append(close_order)

                            # Create order for the new short portion
                            new_short_order = OrderRequest(
                                client_order_id=order.client_order_id,
                                symbol=order.symbol,
                                side=order.side,
                                order_type=order.order_type,
                                quantity=new_short_qty,
                                limit_price=order.limit_price,
                                stop_price=order.stop_price,
                                time_in_force=order.time_in_force,
                                timestamp=order.timestamp,
                                strategy_id=order.strategy_id,
                                confidence=order.confidence,
                                tags=order.tags,
                                metadata=order.metadata,
                            )
                            exposure_increasing_orders.append(
                                (new_short_order, new_short_qty * price)
                            )
                        else:
                            # All close - passes freely
                            exposure_reducing_orders.append(order)
                    else:
                        # No long to close, all goes short (increases exposure)
                        exposure_increasing_orders.append(
                            (order, order.quantity * price)
                        )
                else:
                    # No long position - sell increases short exposure
                    exposure_increasing_orders.append((order, order.quantity * price))

        # Start with exposure-reducing orders (always pass)
        result: list[OrderRequest] = list(exposure_reducing_orders)

        if not exposure_increasing_orders:
            return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)

        # Calculate total new exposure
        total_new_exposure = sum(value for _, value in exposure_increasing_orders)

        # Calculate remaining capacity
        remaining_capacity = max_exposure - current_exposure

        if remaining_capacity <= 0:
            # Already at or over limit - no exposure-increasing orders allowed
            for order, _ in exposure_increasing_orders:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Gross leverage at max ({risk_config.max_gross_leverage}x), "
                        f"no capacity for new exposure",
                    )
                )
            return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)

        if total_new_exposure <= remaining_capacity:
            # All orders fit
            result.extend(order for order, _ in exposure_increasing_orders)
            return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)

        # Scale down proportionally
        scale_factor = remaining_capacity / total_new_exposure

        for order, order_value in exposure_increasing_orders:
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
                if scaled_quantity < order.quantity:
                    rejected.append(
                        RejectedOrder(
                            order=order,
                            constraint_name=self.name,
                            reason=f"Scaled from {order.quantity} to {scaled_quantity} "
                            f"(gross leverage limit {risk_config.max_gross_leverage}x)",
                            original_quantity=order.quantity,
                        )
                    )
            else:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Scaled quantity < 1 (gross leverage limit {risk_config.max_gross_leverage}x)",
                    )
                )

        return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)
