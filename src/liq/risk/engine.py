"""Risk Engine - Core orchestrator for signal-to-order pipeline.

The RiskEngine transforms trading signals into sized, constrained orders
while applying risk management rules.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from liq.core import OrderRequest, OrderSide
from pydantic import BaseModel, ConfigDict, Field

from liq.risk.types import TargetPosition

if TYPE_CHECKING:
    from liq.core import PortfolioState
    from liq.signals import Signal

    from liq.risk.config import MarketState, RiskConfig
    from liq.risk.protocols import StructuredConstraint, TargetPositionSizer

logger = logging.getLogger(__name__)


class RiskEngineResult(BaseModel):
    """Result of processing signals through the risk engine.

    Attributes:
        orders: Sized and constrained orders ready for execution.
        rejected_signals: Signals that were rejected by constraints.
        constraint_violations: Map of constraint name to violation details.
        stop_losses: Map of symbol to stop-loss price.
        take_profits: Map of symbol to take-profit price.
        halted: Whether trading is halted (e.g., drawdown limit).
        halt_reason: Reason for halt if halted.

    Example:
        >>> result = engine.process_signals(signals, portfolio, market, config)
        >>> for order in result.orders:
        ...     execute(order)
        >>> if result.halted:
        ...     log_warning(result.halt_reason)
    """

    model_config = ConfigDict(frozen=True)

    orders: list[OrderRequest] = Field(
        default_factory=list,
        description="Sized and constrained orders",
    )
    rejected_signals: list[Any] = Field(
        default_factory=list,
        description="Signals rejected by constraints",
    )
    constraint_violations: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map of constraint name to violation details",
    )
    stop_losses: dict[str, Decimal] = Field(
        default_factory=dict,
        description="Map of symbol to stop-loss price",
    )
    take_profits: dict[str, Decimal] = Field(
        default_factory=dict,
        description="Map of symbol to take-profit price",
    )
    halted: bool = Field(
        default=False,
        description="Whether trading is halted",
    )
    halt_reason: str | None = Field(
        default=None,
        description="Reason for halt if halted",
    )


class RiskEngine:
    """Core orchestrator for signal-to-order pipeline.

    Transforms trading signals into sized orders while applying
    risk constraints. Supports customizable sizers and constraints.

    The default behavior:
    1. Check for drawdown halt
    2. Size positions using the configured sizer
    3. Apply constraint chain in order
    4. Calculate stop-losses for resulting orders

    Example:
        >>> engine = RiskEngine()
        >>> result = engine.process_signals(signals, portfolio, market, config)
        >>> for order in result.orders:
        ...     broker.submit(order)

    With custom components:
        >>> sizer = KellySizer()
        >>> constraints = [MaxPositionConstraint(), SectorConstraint()]
        >>> engine = RiskEngine(sizer=sizer, constraints=constraints)
    """

    def __init__(
        self,
        sizer: TargetPositionSizer | None = None,
        constraints: list[StructuredConstraint] | None = None,
    ) -> None:
        """Initialize the risk engine.

        Args:
            sizer: Position sizing algorithm. Defaults to VolatilitySizer.
            constraints: Constraint chain to apply. Defaults to standard chain.
        """
        self._sizer = sizer
        self._constraints = constraints

    def _get_sizer(self) -> TargetPositionSizer:
        """Get the position sizer, defaulting to VolatilitySizer."""
        if self._sizer is not None:
            return self._sizer

        from liq.risk.sizers import VolatilitySizer

        return VolatilitySizer()

    def _get_constraints(self) -> list[StructuredConstraint]:
        """Get the constraint chain, defaulting to standard chain."""
        if self._constraints is not None:
            return self._constraints

        return self._default_constraints()

    def _default_constraints(self) -> list[StructuredConstraint]:
        """Return the default constraint chain.

        Order matters:
        1. ShortSelling - Filter short sells if disabled (early exit)
        2. MinPositionValue - Filter tiny orders first
        3. MaxPosition - Limit individual position sizes
        4. MaxPositions - Limit total position count
        5. BuyingPower - Limit buys to available cash
        6. GrossLeverage - Limit total exposure
        7. NetLeverage - Limit net exposure
        """
        from liq.risk.constraints import (
            BuyingPowerConstraint,
            GrossLeverageConstraint,
            MaxPositionConstraint,
            MaxPositionsConstraint,
            MinPositionValueConstraint,
            NetLeverageConstraint,
            ShortSellingConstraint,
        )

        return [
            ShortSellingConstraint(),
            MinPositionValueConstraint(),
            MaxPositionConstraint(),
            MaxPositionsConstraint(),
            BuyingPowerConstraint(),
            GrossLeverageConstraint(),
            NetLeverageConstraint(),
        ]

    def process_signals(
        self,
        signals: list[Signal],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
        high_water_mark: Decimal | None = None,
        day_start_equity: Decimal | None = None,
    ) -> RiskEngineResult:
        """Process signals through the risk pipeline.

        Args:
            signals: Trading signals to process.
            portfolio_state: Current portfolio snapshot.
            market_state: Current market conditions.
            risk_config: Risk parameters.
            high_water_mark: Peak equity for drawdown calculation.
            day_start_equity: Equity at start of day for daily loss calculation.

        Returns:
            RiskEngineResult with orders, rejections, and stop-losses.
        """
        # Check for equity floor breach first
        halted, halt_reason = self._check_equity_floor(portfolio_state)

        # Check for drawdown halt if not already halted
        if not halted:
            halted, halt_reason = self._check_drawdown_halt(
                portfolio_state, risk_config, high_water_mark
            )

        # Check for daily loss halt if not already halted
        if not halted:
            halted, halt_reason = self._check_daily_loss_halt(
                portfolio_state, risk_config, day_start_equity
            )

        if not signals:
            return RiskEngineResult(
                orders=[],
                rejected_signals=[],
                constraint_violations={},
                stop_losses={},
                take_profits={},
                halted=halted,
                halt_reason=halt_reason,
            )

        # Size positions - sizers now return TargetPosition
        sizer = self._get_sizer()
        sizer_output = sizer.size_positions(signals, portfolio_state, market_state, risk_config)

        # Convert TargetPosition to OrderRequest
        # Use market state timestamp if available, otherwise UTC now
        timestamp = getattr(market_state, "timestamp", None)
        if timestamp is None:
            timestamp = datetime.now(UTC)

        orders: list[OrderRequest] = []
        for item in sizer_output:
            if isinstance(item, TargetPosition):
                # Convert TargetPosition to OrderRequest
                order = item.to_order_request(timestamp=timestamp)
                if order is not None:
                    orders.append(order)
            else:
                # Legacy: sizer returned OrderRequest directly
                orders.append(item)

        # If halted, only allow risk-reducing orders (sells for longs, buys for shorts)
        if halted:
            orders = [o for o in orders if o.side == OrderSide.SELL]

        # Apply constraint chain
        constraints = self._get_constraints()
        constraint_violations: dict[str, list[str]] = {}

        for constraint in constraints:
            constraint_result = constraint.apply(orders, portfolio_state, market_state, risk_config)

            # StructuredConstraint returns ConstraintResult
            orders = constraint_result.orders
            # Track violations from rejected orders
            if constraint_result.rejected:
                constraint_name = constraint.name
                if constraint_name not in constraint_violations:
                    constraint_violations[constraint_name] = []
                for rejected in constraint_result.rejected:
                    constraint_violations[constraint_name].append(
                        f"{rejected.order.symbol}: {rejected.reason}"
                    )

        # Identify rejected signals
        final_symbols = {o.symbol for o in orders}
        rejected_signals = [s for s in signals if s.symbol not in final_symbols]

        # Calculate stop-losses and take-profits
        stop_losses = self._calculate_stop_losses(orders, market_state, risk_config)
        take_profits = self._calculate_take_profits(orders, market_state, risk_config)

        return RiskEngineResult(
            orders=orders,
            rejected_signals=rejected_signals,
            constraint_violations=constraint_violations,
            stop_losses=stop_losses,
            take_profits=take_profits,
            halted=halted,
            halt_reason=halt_reason,
        )

    def _check_drawdown_halt(
        self,
        portfolio_state: PortfolioState,
        risk_config: RiskConfig,
        high_water_mark: Decimal | None = None,
    ) -> tuple[bool, str | None]:
        """Check if trading should be halted due to drawdown.

        Args:
            portfolio_state: Current portfolio.
            risk_config: Risk parameters.
            high_water_mark: Peak equity for drawdown calculation.

        Returns:
            Tuple of (halted, reason).
        """
        if high_water_mark is None or high_water_mark <= 0:
            return False, None

        current_equity = portfolio_state.equity
        drawdown = (high_water_mark - current_equity) / high_water_mark

        if drawdown >= Decimal(str(risk_config.max_drawdown_halt)):
            logger.warning(
                "HALT: Drawdown of %.1f%% exceeds limit of %.1f%% (hwm=%s, equity=%s)",
                float(drawdown * 100),
                risk_config.max_drawdown_halt * 100,
                high_water_mark,
                current_equity,
            )
            return True, f"Drawdown of {drawdown:.1%} exceeds limit of {risk_config.max_drawdown_halt:.1%}"

        return False, None

    def _check_equity_floor(
        self,
        portfolio_state: PortfolioState,
    ) -> tuple[bool, str | None]:
        """Check if equity has fallen to or below zero.

        This is a kill-switch that halts all new buys if equity
        is exhausted.

        Args:
            portfolio_state: Current portfolio.

        Returns:
            Tuple of (halted, reason).
        """
        if portfolio_state.equity <= 0:
            logger.warning(
                "HALT: Equity floor breached - equity is %s",
                portfolio_state.equity,
            )
            return True, f"Equity floor breached: equity is {portfolio_state.equity}"

        return False, None

    def _check_daily_loss_halt(
        self,
        portfolio_state: PortfolioState,
        risk_config: RiskConfig,
        day_start_equity: Decimal | None = None,
    ) -> tuple[bool, str | None]:
        """Check if trading should be halted due to daily loss.

        Args:
            portfolio_state: Current portfolio.
            risk_config: Risk parameters.
            day_start_equity: Equity at the start of the trading day.

        Returns:
            Tuple of (halted, reason).
        """
        # If daily loss halt not configured, skip
        if risk_config.max_daily_loss_halt is None:
            return False, None

        # Need day start equity to calculate daily loss
        if day_start_equity is None or day_start_equity <= 0:
            return False, None

        current_equity = portfolio_state.equity
        daily_loss = (day_start_equity - current_equity) / day_start_equity

        if daily_loss >= Decimal(str(risk_config.max_daily_loss_halt)):
            logger.warning(
                "HALT: Daily loss of %.1f%% exceeds limit of %.1f%%",
                float(daily_loss * 100),
                risk_config.max_daily_loss_halt * 100,
            )
            return True, f"Daily loss of {daily_loss:.1%} exceeds limit of {risk_config.max_daily_loss_halt:.1%}"

        return False, None

    def _calculate_stop_losses(
        self,
        orders: list[OrderRequest],
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> dict[str, Decimal]:
        """Calculate stop-loss prices for orders.

        Uses ATR-based stops:
        - Long: entry - (ATR * multiplier)
        - Short: entry + (ATR * multiplier)

        Args:
            orders: Orders to calculate stops for.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            Map of symbol to stop-loss price.
        """
        stop_losses: dict[str, Decimal] = {}
        atr_mult = Decimal(str(risk_config.stop_loss_atr_mult))

        for order in orders:
            bar = market_state.current_bars.get(order.symbol)
            if bar is None:
                continue

            atr = market_state.volatility.get(order.symbol)
            if atr is None:
                continue

            # Use midrange as entry price estimate
            midrange = (bar.high + bar.low) / 2

            if order.side == OrderSide.BUY:
                # Long stop below entry
                stop_losses[order.symbol] = midrange - (atr * atr_mult)
            else:
                # Short stop above entry
                stop_losses[order.symbol] = midrange + (atr * atr_mult)

        return stop_losses

    def _calculate_take_profits(
        self,
        orders: list[OrderRequest],
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> dict[str, Decimal]:
        """Calculate take-profit prices for orders.

        Uses ATR-based targets:
        - Long: entry + (ATR * multiplier)
        - Short: entry - (ATR * multiplier)

        Args:
            orders: Orders to calculate targets for.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            Map of symbol to take-profit price. Empty if not configured.
        """
        # If take-profit not configured, return empty
        if risk_config.take_profit_atr_mult is None:
            return {}

        take_profits: dict[str, Decimal] = {}
        atr_mult = Decimal(str(risk_config.take_profit_atr_mult))

        for order in orders:
            bar = market_state.current_bars.get(order.symbol)
            if bar is None:
                continue

            atr = market_state.volatility.get(order.symbol)
            if atr is None:
                continue

            # Use midrange as entry price estimate
            midrange = (bar.high + bar.low) / 2

            if order.side == OrderSide.BUY:
                # Long take-profit above entry
                take_profits[order.symbol] = midrange + (atr * atr_mult)
            else:
                # Short take-profit below entry
                take_profits[order.symbol] = midrange - (atr * atr_mult)

        return take_profits

    def calculate_stop_loss(
        self,
        symbol: str,
        side: OrderSide,
        entry_price: Decimal,
        atr: Decimal,
        atr_multiplier: float = 2.0,
    ) -> Decimal:
        """Calculate stop-loss price for a position.

        Args:
            symbol: Trading symbol.
            side: Order side (BUY for long, SELL for short).
            entry_price: Entry price.
            atr: Average True Range.
            atr_multiplier: Multiplier for ATR distance.

        Returns:
            Stop-loss price.

        Example:
            >>> engine = RiskEngine()
            >>> stop = engine.calculate_stop_loss(
            ...     "AAPL", OrderSide.BUY, Decimal("100"), Decimal("2"), 2.0
            ... )
            >>> stop
            Decimal('96')
        """
        mult = Decimal(str(atr_multiplier))
        stop_distance = atr * mult

        if side == OrderSide.BUY:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
