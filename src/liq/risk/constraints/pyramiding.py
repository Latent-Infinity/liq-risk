"""Pyramiding constraint.

PyramidingConstraint: Limits position scaling (adding to existing positions).
Prevents over-concentration by limiting how many times you can add to a position.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide

from liq.risk.types import ConstraintResult, RejectedOrder

if TYPE_CHECKING:
    from datetime import datetime

    from liq.core import PortfolioState

    from liq.risk.config import MarketState, RiskConfig

_QTY_EPS = Decimal("1e-9")


@dataclass
class PyramidingState:
    """Track pyramiding state for a symbol.

    Attributes:
        add_count: Number of times position has been added to.
        initial_quantity: Initial position size when first entered.
        total_added: Total quantity added since initial entry.
    """

    add_count: int = 0
    initial_quantity: Decimal = Decimal("0")
    total_added: Decimal = Decimal("0")


class PyramidingConstraint:
    """Limit position scaling (pyramiding) to prevent over-concentration.

    This constraint prevents adding to existing positions beyond configured limits.
    It tracks:
    - Number of adds (max_pyramid_adds)
    - Maximum add size as percentage of initial position (max_add_pct)

    When a position is fully closed, the pyramiding state resets.

    Args:
        max_pyramid_adds: Maximum number of times to add to a position.
            Default is 3 (initial entry + 3 adds = 4 total entries).
        max_add_pct: Maximum size of each add as fraction of initial position.
            Default is 0.5 (each add can be at most 50% of initial size).
        pyramiding_state: Optional pre-existing state for testing/recovery.

    Example:
        >>> constraint = PyramidingConstraint(max_pyramid_adds=2, max_add_pct=0.5)
        >>> result = constraint.apply(orders, portfolio, market, config)
        >>> for r in result.rejected:
        ...     print(f"Rejected {r.order.symbol}: {r.reason}")

    Notes:
        - Initial position entry always passes (add_count=0)
        - Risk-reducing orders (closing positions) always pass
        - State is per-symbol and resets when position is fully closed
    """

    def __init__(
        self,
        max_pyramid_adds: int = 3,
        max_add_pct: float = 0.5,
        pyramiding_state: dict[str, PyramidingState] | None = None,
    ) -> None:
        if max_pyramid_adds < 0:
            raise ValueError(f"max_pyramid_adds must be >= 0, got {max_pyramid_adds}")
        if max_add_pct <= 0 or max_add_pct > 1:
            raise ValueError(f"max_add_pct must be in (0, 1], got {max_add_pct}")

        self._max_pyramid_adds = max_pyramid_adds
        self._max_add_pct = Decimal(str(max_add_pct))
        self._state: dict[str, PyramidingState] = pyramiding_state or {}
        # Signed net quantity per symbol, maintained by record_trade so the
        # constraint can infer entry/add/close from a raw fill stream.
        self._net_qty: dict[str, Decimal] = {}

    @property
    def name(self) -> str:
        """Human-readable constraint name for logging and audit."""
        return "PyramidingConstraint"

    @property
    def max_pyramid_adds(self) -> int:
        """Maximum number of adds allowed."""
        return self._max_pyramid_adds

    @property
    def max_add_pct(self) -> Decimal:
        """Maximum add size as fraction of initial position."""
        return self._max_add_pct

    def get_state(self, symbol: str) -> PyramidingState:
        """Get pyramiding state for a symbol."""
        if symbol not in self._state:
            self._state[symbol] = PyramidingState()
        return self._state[symbol]

    def reset_state(self, symbol: str) -> None:
        """Reset pyramiding state for a symbol (called when position is closed)."""
        if symbol in self._state:
            del self._state[symbol]

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
            True if risk-increasing (adding to position), False if risk-reducing.
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
        """Apply pyramiding constraint.

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
        result: list[OrderRequest] = []

        # Running projected position per symbol across the batch, seeded from
        # the (single, possibly stale) portfolio snapshot. The rolling runner
        # calls apply() once per fold with the whole batch and a start-of-fold
        # snapshot, so entry/add classification MUST track intra-batch state,
        # not re-read the static snapshot for every order.
        projected: dict[str, Decimal] = {}
        # Adds already committed within THIS batch, on top of the cross-fold
        # state.add_count maintained by record_trade.
        in_batch_adds: dict[str, int] = {}
        # Initial entry size observed within this batch (add-size base when the
        # cross-fold state has no recorded initial_quantity yet).
        in_batch_initial: dict[str, Decimal] = {}

        for order in orders:
            symbol = order.symbol
            if symbol not in projected:
                maybe_position = portfolio_state.positions.get(symbol)
                projected[symbol] = (
                    maybe_position.quantity if maybe_position else Decimal("0")
                )
            current_qty = projected[symbol]
            signed = (
                order.quantity
                if order.side == OrderSide.BUY
                else -order.quantity
            )

            # Check if this is a risk-reducing order
            is_risk_reducing = self._is_risk_reducing(order, current_qty)

            if is_risk_reducing:
                # Risk-reducing orders always pass
                result.append(order)
                new_qty = current_qty + signed
                # Check if this fully closes (or flips) the position
                if self._would_close_position(order, current_qty) or (
                    (current_qty > 0) != (new_qty > 0) and new_qty != 0
                ):
                    self.reset_state(symbol)
                    in_batch_adds[symbol] = 0
                    in_batch_initial.pop(symbol, None)
                    projected[symbol] = new_qty if new_qty != 0 else Decimal("0")
                else:
                    projected[symbol] = new_qty
                continue

            # This is a risk-increasing order (adding to position)
            state = self.get_state(symbol)

            # Check if this is an initial entry (no existing position)
            if current_qty == 0:
                # Initial entry - record it and pass
                result.append(order)
                in_batch_adds[symbol] = 0
                in_batch_initial[symbol] = order.quantity
                projected[symbol] = signed
                continue

            # This is an add to existing position. The add budget is shared
            # between cross-fold state (record_trade) and this batch.
            effective_adds = state.add_count + in_batch_adds.get(symbol, 0)
            if effective_adds >= self._max_pyramid_adds:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Pyramiding limit reached: {effective_adds} adds "
                        f"(max {self._max_pyramid_adds})",
                    )
                )
                continue

            # Check max add size. Base on recorded initial quantity, else the
            # in-batch initial entry size, else current projected exposure.
            if state.initial_quantity > 0:
                base_qty = state.initial_quantity
            elif symbol in in_batch_initial:
                base_qty = in_batch_initial[symbol]
            else:
                base_qty = abs(current_qty)
            max_add_qty = base_qty * self._max_add_pct

            if order.quantity > max_add_qty:
                # Scale down to max allowed
                if max_add_qty >= 1:
                    scaled_qty = max_add_qty.to_integral_value()
                    new_order = OrderRequest(
                        client_order_id=order.client_order_id,
                        symbol=order.symbol,
                        side=order.side,
                        order_type=order.order_type,
                        quantity=scaled_qty,
                        limit_price=order.limit_price,
                        stop_price=order.stop_price,
                        time_in_force=order.time_in_force,
                        timestamp=order.timestamp,
                        policy_id=order.policy_id,
                        confidence=order.confidence,
                        tags=order.tags,
                        metadata=order.metadata,
                    )
                    result.append(new_order)
                    rejected.append(
                        RejectedOrder(
                            order=order,
                            constraint_name=self.name,
                            reason=f"Scaled from {order.quantity} to {scaled_qty} "
                            f"(max add {self._max_add_pct:.0%} of initial {base_qty})",
                            original_quantity=order.quantity,
                        )
                    )
                    # A (scaled) add was admitted -> consume add budget and
                    # advance the projected position by the scaled quantity.
                    in_batch_adds[symbol] = in_batch_adds.get(symbol, 0) + 1
                    projected[symbol] = current_qty + (
                        scaled_qty if order.side == OrderSide.BUY else -scaled_qty
                    )
                else:
                    rejected.append(
                        RejectedOrder(
                            order=order,
                            constraint_name=self.name,
                            reason=f"Add size {order.quantity} exceeds max {max_add_qty:.2f} "
                            f"({self._max_add_pct:.0%} of initial {base_qty})",
                        )
                    )
            else:
                # Add within limits - pass; consume add budget and advance the
                # projected position so the next batch order sees this add.
                result.append(order)
                in_batch_adds[symbol] = in_batch_adds.get(symbol, 0) + 1
                projected[symbol] = current_qty + signed

        return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)

    def _is_risk_reducing(self, order: OrderRequest, current_qty: Decimal) -> bool:
        """Check if order reduces position risk."""
        if order.side == OrderSide.BUY:
            # Buy reduces risk if we're short
            return current_qty < 0
        else:
            # Sell reduces risk if we're long
            return current_qty > 0

    def _would_close_position(self, order: OrderRequest, current_qty: Decimal) -> bool:
        """Check if order would fully close the position."""
        if current_qty == 0:
            return False

        if order.side == OrderSide.BUY:
            # Buying to close short
            return current_qty < 0 and order.quantity >= abs(current_qty)
        else:
            # Selling to close long
            return current_qty > 0 and order.quantity >= current_qty

    def record_trade(
        self,
        symbol: str,
        timestamp: datetime,
        side: OrderSide,
        quantity: Decimal,
    ) -> None:
        """Update pyramiding state from a confirmed fill.

        This is the runner's duck-typed fill-feedback contract (the same one
        `FrequencyCapConstraint.record_trade` implements). The rolling runner
        calls this for every fill after simulation. Unlike `record_fill`, no
        `is_add` flag is provided, so entry/add/close is inferred from running
        signed net quantity per symbol:

        - net leaves 0           -> initial entry (reset add budget)
        - |net| increases (same sign) -> add (add_count += 1)
        - net returns to ~0      -> position closed (reset state)
        - sign flip              -> close + immediate re-entry

        Args:
            symbol: Filled symbol.
            timestamp: Fill timestamp (unused; matches the runner contract).
            side: BUY adds positive signed quantity, SELL negative.
            quantity: Filled quantity (> 0).
        """
        del timestamp  # part of the runner contract; not needed here
        delta = quantity if side == OrderSide.BUY else -quantity
        prev = self._net_qty.get(symbol, Decimal("0"))
        new = prev + delta

        if abs(new) <= _QTY_EPS:
            # Flat -> position closed.
            self._net_qty[symbol] = Decimal("0")
            self.reset_state(symbol)
            return

        if prev == 0 or (prev > 0) != (new > 0):
            # Opening from flat, or sign flip (close + re-enter).
            self._net_qty[symbol] = new
            state = self.get_state(symbol)
            state.initial_quantity = abs(new)
            state.add_count = 0
            state.total_added = Decimal("0")
            return

        if abs(new) > abs(prev) + _QTY_EPS:
            # Same-direction increase -> add.
            state = self.get_state(symbol)
            state.add_count += 1
            state.total_added += quantity

        # else: partial reduction (still in position) -> no add-count change.
        self._net_qty[symbol] = new

    def record_fill(
        self,
        symbol: str,
        filled_qty: Decimal,
        is_add: bool,
    ) -> None:
        """Record a fill for state tracking.

        Call this after an order is filled to update pyramiding state.

        Args:
            symbol: The symbol that was filled.
            filled_qty: The quantity that was filled.
            is_add: Whether this was an add (True) or initial entry (False).
        """
        state = self.get_state(symbol)

        if not is_add:
            # Initial entry
            state.initial_quantity = filled_qty
            state.add_count = 0
            state.total_added = Decimal("0")
        else:
            # Add to position
            state.add_count += 1
            state.total_added += filled_qty
