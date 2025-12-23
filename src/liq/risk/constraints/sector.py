"""Sector exposure constraint.

SectorExposureConstraint: Limits exposure to any single sector.
"""

from __future__ import annotations

from decimal import ROUND_DOWN, Decimal
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide

from liq.risk.types import ConstraintResult, RejectedOrder

if TYPE_CHECKING:
    from liq.core import PortfolioState

    from liq.risk.config import MarketState, RiskConfig


class SectorExposureConstraint:
    """Limit exposure to any single sector.

    Scales down orders to ensure no sector exceeds max_sector_pct
    of total equity. Tracks cumulative exposure from existing
    positions and pending orders.

    Example:
        >>> constraint = SectorExposureConstraint()
        >>> result = constraint.apply(orders, portfolio, market, config)
        >>> for r in result.rejected:
        ...     print(f"Rejected {r.order.symbol}: {r.reason}")
    """

    @property
    def name(self) -> str:
        """Human-readable constraint name for logging and audit."""
        return "SectorExposureConstraint"

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
        """Apply sector exposure constraint.

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

        if not orders:
            return ConstraintResult(orders=[], rejected=rejected, warnings=warnings)

        # If no sector map, pass all orders through
        sector_map = market_state.sector_map
        if sector_map is None:
            return ConstraintResult(orders=list(orders), rejected=rejected, warnings=warnings)

        equity = portfolio_state.equity
        max_sector_exposure = equity * Decimal(str(risk_config.max_sector_pct))

        # Calculate current sector exposure from existing positions
        sector_exposure: dict[str, Decimal] = {}
        for symbol, position in portfolio_state.positions.items():
            sector = sector_map.get(symbol)
            if sector is None:
                continue

            # Use current market price for position value
            bar = market_state.current_bars.get(symbol)
            if bar is not None:
                position_value = abs(position.quantity) * bar.close
            else:
                position_value = position.market_value

            if sector not in sector_exposure:
                sector_exposure[sector] = Decimal("0")
            sector_exposure[sector] += position_value

        result: list[OrderRequest] = []

        for order in orders:
            # Sell orders always pass (reduce exposure)
            if order.side == OrderSide.SELL:
                result.append(order)
                continue

            # Get bar data for pricing
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

            # Get sector for this symbol
            sector = sector_map.get(order.symbol)
            if sector is None:
                # Unknown sector - pass through
                result.append(order)
                continue

            price = bar.close
            order_value = order.quantity * price

            # Get current sector exposure
            current_exposure = sector_exposure.get(sector, Decimal("0"))

            # Calculate remaining capacity
            remaining_capacity = max_sector_exposure - current_exposure

            if remaining_capacity <= 0:
                # No room in this sector
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Sector '{sector}' at max exposure ({risk_config.max_sector_pct:.0%} of equity)",
                    )
                )
                continue

            if order_value <= remaining_capacity:
                # Order fits within limit
                result.append(order)
                # Update tracking
                sector_exposure[sector] = current_exposure + order_value
            else:
                # Scale down to fit
                scaled_value = remaining_capacity
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
                    # Update tracking with actual value
                    sector_exposure[sector] = current_exposure + (
                        scaled_quantity * price
                    )
                    rejected.append(
                        RejectedOrder(
                            order=order,
                            constraint_name=self.name,
                            reason=f"Scaled from {order.quantity} to {scaled_quantity} "
                            f"(sector '{sector}' limit {risk_config.max_sector_pct:.0%})",
                            original_quantity=order.quantity,
                        )
                    )
                else:
                    rejected.append(
                        RejectedOrder(
                            order=order,
                            constraint_name=self.name,
                            reason=f"Sector '{sector}' at max exposure, scaled quantity < 1",
                        )
                    )

        return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)
