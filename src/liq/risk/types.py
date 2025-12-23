"""Core type definitions for liq-risk.

Provides execution-agnostic types for position sizing and constraint output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import ROUND_CEILING, ROUND_DOWN, ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, Literal, Union

if TYPE_CHECKING:
    from liq.core import OrderRequest, OrderType


@dataclass(frozen=True)
class RoundingPolicy:
    """Provider-specific quantity rounding rules.

    Passed as input to avoid provider-coupling in liq-risk.
    Different providers (Alpaca, Binance, etc.) have different
    minimum order sizes and quantity increments.

    Attributes:
        lot_size: Minimum tradeable unit (e.g., 1 for stocks, 0.001 for BTC).
        step_size: Quantity increment (usually same as lot_size).
        min_notional: Minimum order value in quote currency.
        max_precision: Maximum decimal places for quantity.

    Example:
        >>> policy = RoundingPolicy(lot_size=Decimal("0.001"))
        >>> policy.round_quantity(Decimal("1.23456789"))
        Decimal('1.234')
    """

    lot_size: Decimal = Decimal("1")
    step_size: Decimal = Decimal("1")
    min_notional: Decimal = Decimal("1")
    max_precision: int = 8

    def round_quantity(
        self, qty: Decimal, direction: str = "down"
    ) -> Decimal:
        """Round quantity to valid lot size.

        Args:
            qty: Raw quantity to round.
            direction: Rounding direction - "down", "up", or "nearest".

        Returns:
            Rounded quantity that is a multiple of lot_size.

        Example:
            >>> policy = RoundingPolicy(lot_size=Decimal("10"))
            >>> policy.round_quantity(Decimal("157"))
            Decimal('150')
        """
        if qty == 0:
            return Decimal("0")

        if self.lot_size == 0:
            return qty

        # Calculate how many lots
        lots = qty / self.lot_size

        if direction == "down":
            rounded_lots = lots.to_integral_value(rounding=ROUND_DOWN)
        elif direction == "up":
            rounded_lots = lots.to_integral_value(rounding=ROUND_CEILING)
        elif direction == "nearest":
            rounded_lots = lots.to_integral_value(rounding=ROUND_HALF_UP)
        else:
            rounded_lots = lots.to_integral_value(rounding=ROUND_DOWN)

        return rounded_lots * self.lot_size


@dataclass(frozen=True)
class TargetPosition:
    """Execution-agnostic position target from risk engine.

    This is the primary output of liq-risk sizers. It expresses
    "where we want to be" without knowing execution mechanics
    like order types or time-in-force.

    Use `to_order_request()` adapter to convert for execution.

    Attributes:
        symbol: Trading symbol (e.g., "AAPL", "BTC_USDT").
        target_quantity: Absolute target quantity. Positive for long,
            negative for short, zero for flat.
        current_quantity: Current position quantity for reference.
        direction: Position direction - "long", "short", or "flat".
        urgency: Execution urgency - "normal", "urgent", "immediate".
        stop_price: Suggested stop-loss price.
        take_profit_price: Suggested take-profit price.
        signal_strength: Original signal confidence (0.0 to 1.0).
        risk_tags: Audit trail tags (strategy, sector, etc.).

    Properties:
        delta_quantity: Quantity change needed (target - current).
        is_risk_increasing: Whether this increases position risk.

    Example:
        >>> tp = TargetPosition(
        ...     symbol="AAPL",
        ...     target_quantity=Decimal("150"),
        ...     current_quantity=Decimal("50"),
        ...     direction="long",
        ... )
        >>> tp.delta_quantity
        Decimal('100')
        >>> tp.is_risk_increasing
        True
    """

    symbol: str
    target_quantity: Decimal
    current_quantity: Decimal
    direction: Literal["long", "short", "flat"]
    urgency: Literal["normal", "urgent", "immediate"] = "normal"
    stop_price: Decimal | None = None
    take_profit_price: Decimal | None = None
    signal_strength: float = 1.0
    risk_tags: dict[str, str] = field(default_factory=dict)

    @property
    def delta_quantity(self) -> Decimal:
        """Quantity change needed: target - current."""
        return self.target_quantity - self.current_quantity

    @property
    def is_risk_increasing(self) -> bool:
        """Does this increase position risk?

        Risk increases when moving away from zero (flat) or
        adding to an existing directional position.

        Returns:
            True if risk-increasing, False if risk-reducing.
        """
        current_abs = abs(self.current_quantity)
        target_abs = abs(self.target_quantity)
        return target_abs > current_abs

    def to_order_request(
        self,
        timestamp: datetime,
        order_type: "OrderType | None" = None,
        rounding: RoundingPolicy | None = None,
    ) -> "OrderRequest | None":
        """Convert to OrderRequest for execution.

        Args:
            timestamp: Order timestamp (must be timezone-aware).
            order_type: Order type (default: MARKET).
            rounding: Optional rounding policy for quantity.

        Returns:
            OrderRequest if action needed, None if delta is zero
            or rounds to zero.

        Example:
            >>> tp = TargetPosition(...)
            >>> order = tp.to_order_request(timestamp=now)
        """
        from liq.core import OrderRequest, OrderSide, OrderType

        delta = self.delta_quantity

        if delta == 0:
            return None

        # Determine side based on delta direction
        if delta > 0:
            side = OrderSide.BUY
            quantity = delta
        else:
            side = OrderSide.SELL
            quantity = abs(delta)

        # Apply rounding if provided
        if rounding is not None:
            quantity = rounding.round_quantity(quantity)
            if quantity == 0:
                return None

        # Default to MARKET order
        if order_type is None:
            order_type = OrderType.MARKET

        return OrderRequest(
            symbol=self.symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            timestamp=timestamp,
            confidence=self.signal_strength,
        )


# Type alias for order or target position
OrderOrTarget = Union["OrderRequest", TargetPosition]


@dataclass(frozen=True)
class RejectedOrder:
    """An order that was rejected or modified by a constraint.

    Provides audit trail for constraint decisions, enabling
    debugging and analysis of why orders were filtered.

    Attributes:
        order: The rejected or modified order/target.
        constraint_name: Name of the constraint that rejected it.
        reason: Human-readable explanation of rejection.
        original_quantity: Original quantity if modified (not fully rejected).

    Example:
        >>> rejected = RejectedOrder(
        ...     order=order,
        ...     constraint_name="MaxPositionConstraint",
        ...     reason="Position would exceed 5% of equity",
        ... )
    """

    order: OrderOrTarget
    constraint_name: str
    reason: str
    original_quantity: Decimal | None = None


@dataclass
class ConstraintResult:
    """Structured result from constraint application.

    Constraints MUST return this, not just a filtered list.
    This enables proper audit trails and debugging.

    Attributes:
        orders: Orders that passed the constraint (may be modified).
        rejected: Orders that were rejected with reasons.
        warnings: Non-fatal issues to log (e.g., approaching limits).

    Example:
        >>> result = constraint.apply(orders, portfolio, market, config)
        >>> for rejection in result.rejected:
        ...     logger.info(f"Rejected {rejection.order.symbol}: {rejection.reason}")
    """

    orders: list["OrderRequest"]
    rejected: list[RejectedOrder]
    warnings: list[str] = field(default_factory=list)
