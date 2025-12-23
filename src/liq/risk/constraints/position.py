"""Position-related constraints.

MaxPositionConstraint: Limits individual position size as % of equity.
MaxPositionsConstraint: Limits total number of concurrent positions.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide

from liq.risk.types import ConstraintResult, RejectedOrder

if TYPE_CHECKING:
    from liq.core import PortfolioState

    from liq.risk.config import MarketState, RiskConfig


class MaxPositionConstraint:
    """Limit individual position size as percentage of equity.

    Reduces order quantity if it would result in a position
    exceeding the configured max_position_pct limit.

    Note:
        This constraint correctly handles short positions:
        - SELL orders that close/reduce long positions pass freely
        - SELL orders that initiate/increase short positions are constrained
        - The absolute position value is constrained (both long and short)

    Example:
        >>> constraint = MaxPositionConstraint()
        >>> result = constraint.apply(orders, portfolio, market, config)
        >>> for r in result.rejected:
        ...     print(f"Rejected {r.order.symbol}: {r.reason}")
    """

    @property
    def name(self) -> str:
        """Human-readable constraint name for logging and audit."""
        return "MaxPositionConstraint"

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
            # Buying when short = reducing risk, buying when flat/long = increasing
            return current_qty >= 0
        else:  # SELL
            # Selling when long = reducing risk, selling when flat/short = increasing
            return current_qty <= 0

    def apply(
        self,
        orders: list[OrderRequest],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> ConstraintResult:
        """Apply max position constraint.

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

        equity = portfolio_state.equity
        max_position_value = equity * Decimal(str(risk_config.max_position_pct))

        for order in orders:
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

            # Get existing position
            existing_position = portfolio_state.positions.get(order.symbol)
            current_qty = existing_position.quantity if existing_position else Decimal("0")

            if order.side == OrderSide.BUY:
                if current_qty < 0:
                    # Buying to cover a short
                    cover_qty = min(order.quantity, abs(current_qty))
                    new_long_qty = order.quantity - cover_qty

                    if new_long_qty <= 0:
                        # All cover - passes freely (reduces position)
                        result.append(order)
                        continue

                    # Split: cover portion passes, new long constrained
                    # Calculate room for new long position
                    remaining_room = max_position_value  # New position, starts at 0
                    constrained_qty = min(
                        new_long_qty,
                        (remaining_room / price).to_integral_value(rounding=ROUND_DOWN)
                    )

                    total_qty = cover_qty + constrained_qty
                    if total_qty >= 1:
                        new_order = OrderRequest(
                            client_order_id=order.client_order_id,
                            symbol=order.symbol,
                            side=order.side,
                            order_type=order.order_type,
                            quantity=total_qty,
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
                        # Track partial reduction
                        if total_qty < order.quantity:
                            rejected.append(
                                RejectedOrder(
                                    order=order,
                                    constraint_name=self.name,
                                    reason=f"Reduced from {order.quantity} to {total_qty} "
                                    f"(max position {risk_config.max_position_pct:.1%} of equity)",
                                    original_quantity=order.quantity,
                                )
                            )
                    else:
                        rejected.append(
                            RejectedOrder(
                                order=order,
                                constraint_name=self.name,
                                reason=f"Position would exceed {risk_config.max_position_pct:.1%} of equity",
                            )
                        )
                else:
                    # No short position - normal buy constrained
                    existing_value = abs(current_qty * price) if current_qty > 0 else Decimal("0")
                    remaining_room = max_position_value - existing_value

                    if remaining_room <= 0:
                        rejected.append(
                            RejectedOrder(
                                order=order,
                                constraint_name=self.name,
                                reason=f"Position already at max ({risk_config.max_position_pct:.1%} of equity)",
                            )
                        )
                        continue

                    order_value = order.quantity * price
                    if order_value <= remaining_room:
                        result.append(order)
                    else:
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
                            rejected.append(
                                RejectedOrder(
                                    order=order,
                                    constraint_name=self.name,
                                    reason=f"Reduced from {order.quantity} to {max_quantity} "
                                    f"(max position {risk_config.max_position_pct:.1%} of equity)",
                                    original_quantity=order.quantity,
                                )
                            )
                        else:
                            rejected.append(
                                RejectedOrder(
                                    order=order,
                                    constraint_name=self.name,
                                    reason=f"Position would exceed {risk_config.max_position_pct:.1%} of equity",
                                )
                            )
            else:  # SELL
                if current_qty > 0:
                    # Selling to close a long
                    close_qty = min(order.quantity, current_qty)
                    new_short_qty = order.quantity - close_qty

                    if new_short_qty <= 0:
                        # All close - passes freely (reduces position)
                        result.append(order)
                        continue

                    # Split: close portion passes, new short constrained
                    # Calculate room for new short position
                    remaining_room = max_position_value  # New position, starts at 0
                    constrained_qty = min(
                        new_short_qty,
                        (remaining_room / price).to_integral_value(rounding=ROUND_DOWN)
                    )

                    total_qty = close_qty + constrained_qty
                    if total_qty >= 1:
                        new_order = OrderRequest(
                            client_order_id=order.client_order_id,
                            symbol=order.symbol,
                            side=order.side,
                            order_type=order.order_type,
                            quantity=total_qty,
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
                        # Track partial reduction
                        if total_qty < order.quantity:
                            rejected.append(
                                RejectedOrder(
                                    order=order,
                                    constraint_name=self.name,
                                    reason=f"Reduced from {order.quantity} to {total_qty} "
                                    f"(max position {risk_config.max_position_pct:.1%} of equity)",
                                    original_quantity=order.quantity,
                                )
                            )
                    else:
                        rejected.append(
                            RejectedOrder(
                                order=order,
                                constraint_name=self.name,
                                reason=f"Position would exceed {risk_config.max_position_pct:.1%} of equity",
                            )
                        )
                else:
                    # No long position - sell initiates/increases short (constrained)
                    existing_value = abs(current_qty * price) if current_qty < 0 else Decimal("0")
                    remaining_room = max_position_value - existing_value

                    if remaining_room <= 0:
                        rejected.append(
                            RejectedOrder(
                                order=order,
                                constraint_name=self.name,
                                reason=f"Position already at max ({risk_config.max_position_pct:.1%} of equity)",
                            )
                        )
                        continue

                    order_value = order.quantity * price
                    if order_value <= remaining_room:
                        result.append(order)
                    else:
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
                            rejected.append(
                                RejectedOrder(
                                    order=order,
                                    constraint_name=self.name,
                                    reason=f"Reduced from {order.quantity} to {max_quantity} "
                                    f"(max position {risk_config.max_position_pct:.1%} of equity)",
                                    original_quantity=order.quantity,
                                )
                            )
                        else:
                            rejected.append(
                                RejectedOrder(
                                    order=order,
                                    constraint_name=self.name,
                                    reason=f"Position would exceed {risk_config.max_position_pct:.1%} of equity",
                                )
                            )

        return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)


class MaxPositionsConstraint:
    """Limit total number of concurrent positions.

    Filters orders that would create new positions beyond the
    configured max_positions limit. Orders are prioritized by
    confidence (signal strength).

    Note:
        This constraint correctly handles short positions:
        - SELL orders that close/reduce long positions pass freely
        - SELL orders that initiate new short positions count as new positions
        - Both long and short positions count toward the position limit

    Example:
        >>> constraint = MaxPositionsConstraint()
        >>> result = constraint.apply(orders, portfolio, market, config)
        >>> for r in result.rejected:
        ...     print(f"Rejected {r.order.symbol}: {r.reason}")
    """

    @property
    def name(self) -> str:
        """Human-readable constraint name for logging and audit."""
        return "MaxPositionsConstraint"

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
        else:  # SELL
            # Selling when long = reducing risk
            return current_qty <= 0

    def apply(
        self,
        orders: list[OrderRequest],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> ConstraintResult:
        """Apply max positions constraint.

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

        existing_symbols = set(portfolio_state.positions.keys())
        current_count = len(existing_symbols)
        max_positions = risk_config.max_positions

        # Separate orders into categories based on whether they create new positions
        reducing_orders: list[OrderRequest] = []  # Close/reduce existing positions
        existing_position_orders: list[OrderRequest] = []  # Add to existing positions
        new_position_orders: list[OrderRequest] = []  # Create new positions

        for order in orders:
            position = portfolio_state.positions.get(order.symbol)
            current_qty = position.quantity if position else Decimal("0")

            if order.side == OrderSide.BUY:
                if current_qty < 0:
                    # Buying to cover a short - reducing position
                    reducing_orders.append(order)
                elif order.symbol in existing_symbols:
                    # Adding to existing long
                    existing_position_orders.append(order)
                else:
                    # New long position
                    new_position_orders.append(order)
            else:  # SELL
                if current_qty > 0:
                    # Selling to close a long - reducing position
                    reducing_orders.append(order)
                elif order.symbol in existing_symbols:
                    # Adding to existing short
                    existing_position_orders.append(order)
                else:
                    # New short position
                    new_position_orders.append(order)

        # Calculate room for new positions
        room_for_new = max_positions - current_count

        if room_for_new <= 0:
            # No room for new positions - reject all new position orders
            for order in new_position_orders:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Max positions ({max_positions}) reached, "
                        f"currently holding {current_count} positions",
                    )
                )
            return ConstraintResult(
                orders=reducing_orders + existing_position_orders,
                rejected=rejected,
                warnings=warnings,
            )

        # Sort new position orders by confidence (descending)
        new_position_orders.sort(
            key=lambda o: o.confidence if o.confidence is not None else 0.0,
            reverse=True,
        )

        # Take top N new positions
        accepted_new = new_position_orders[:room_for_new]
        rejected_new = new_position_orders[room_for_new:]

        # Track rejected orders
        for order in rejected_new:
            rejected.append(
                RejectedOrder(
                    order=order,
                    constraint_name=self.name,
                    reason=f"Exceeds max positions limit ({max_positions}), "
                    f"lower confidence than accepted orders",
                )
            )

        return ConstraintResult(
            orders=reducing_orders + existing_position_orders + accepted_new,
            rejected=rejected,
            warnings=warnings,
        )
