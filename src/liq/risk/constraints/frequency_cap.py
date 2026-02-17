"""Frequency cap constraint.

FrequencyCapConstraint: Limits trade frequency to prevent over-trading.
Configurable as N trades per Y timeframe (minute, hour, day, week, etc.).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide

from liq.risk.types import ConstraintResult, RejectedOrder

if TYPE_CHECKING:
    from liq.core import PortfolioState

    from liq.risk.config import MarketState, RiskConfig


class Timeframe(Enum):
    """Supported timeframes for frequency caps.

    Values represent the number of seconds in each timeframe.
    """

    SECOND = 1
    MINUTE = 60
    HOUR = 3600
    DAY = 86400
    WEEK = 604800
    MONTH = 2592000  # 30 days approximation

    @classmethod
    def from_string(cls, s: str) -> Timeframe:
        """Parse timeframe from string.

        Args:
            s: Timeframe string (case-insensitive).

        Returns:
            Matching Timeframe enum.

        Raises:
            ValueError: If string doesn't match a timeframe.

        Example:
            >>> Timeframe.from_string("hour")
            Timeframe.HOUR
            >>> Timeframe.from_string("1h")
            Timeframe.HOUR
        """
        s = s.lower().strip()

        # Direct matches
        mapping = {
            "second": cls.SECOND,
            "sec": cls.SECOND,
            "s": cls.SECOND,
            "1s": cls.SECOND,
            "minute": cls.MINUTE,
            "min": cls.MINUTE,
            "m": cls.MINUTE,
            "1m": cls.MINUTE,
            "hour": cls.HOUR,
            "hr": cls.HOUR,
            "h": cls.HOUR,
            "1h": cls.HOUR,
            "day": cls.DAY,
            "d": cls.DAY,
            "1d": cls.DAY,
            "week": cls.WEEK,
            "wk": cls.WEEK,
            "w": cls.WEEK,
            "1w": cls.WEEK,
            "month": cls.MONTH,
            "mo": cls.MONTH,
            "1mo": cls.MONTH,
        }

        if s in mapping:
            return mapping[s]

        raise ValueError(
            f"Unknown timeframe: {s}. "
            f"Valid options: second, minute, hour, day, week, month"
        )

    def to_timedelta(self) -> timedelta:
        """Convert timeframe to timedelta."""
        return timedelta(seconds=self.value)


@dataclass
class FrequencyCapConfig:
    """Configuration for a frequency cap rule.

    Attributes:
        max_trades: Maximum number of trades allowed in the window.
        timeframe: The timeframe for the window.
        per_symbol: If True, limit is per symbol. If False, limit is global.

    Example:
        >>> # Max 5 trades per hour per symbol
        >>> cap = FrequencyCapConfig(max_trades=5, timeframe=Timeframe.HOUR, per_symbol=True)
        >>> # Max 20 trades per day globally
        >>> cap = FrequencyCapConfig(max_trades=20, timeframe=Timeframe.DAY, per_symbol=False)
    """

    max_trades: int
    timeframe: Timeframe
    per_symbol: bool = True


@dataclass
class TradeRecord:
    """Record of a trade for frequency tracking.

    Attributes:
        symbol: Symbol traded.
        timestamp: When the trade occurred.
        side: Order side (BUY or SELL).
        quantity: Quantity traded.
    """

    symbol: str
    timestamp: datetime
    side: OrderSide
    quantity: Decimal


class FrequencyCapConstraint:
    """Limit trade frequency to prevent over-trading.

    Supports flexible configuration: N trades per Y timeframe where Y can be
    any supported timeframe (second, minute, hour, day, week, month).

    Multiple caps can be applied simultaneously (e.g., 3 per minute AND 10 per hour).
    Per-symbol and global caps can coexist.

    Args:
        caps: List of frequency cap configurations to apply.
            If None, uses default of 10 trades per minute per symbol.
        trade_history: Optional pre-existing trade history for testing/recovery.

    Example:
        >>> # Single cap: max 5 trades per hour per symbol
        >>> constraint = FrequencyCapConstraint([
        ...     FrequencyCapConfig(max_trades=5, timeframe=Timeframe.HOUR)
        ... ])

        >>> # Multiple caps
        >>> constraint = FrequencyCapConstraint([
        ...     FrequencyCapConfig(max_trades=3, timeframe=Timeframe.MINUTE, per_symbol=True),
        ...     FrequencyCapConfig(max_trades=20, timeframe=Timeframe.HOUR, per_symbol=True),
        ...     FrequencyCapConfig(max_trades=100, timeframe=Timeframe.DAY, per_symbol=False),
        ... ])

        >>> result = constraint.apply(orders, portfolio, market, config)

    Notes:
        - Risk-reducing orders (closing positions) are NOT exempt from frequency caps
        - Trade history should be updated via record_trade() after fill confirmation
        - History is automatically pruned to the longest cap timeframe
    """

    def __init__(
        self,
        caps: list[FrequencyCapConfig] | None = None,
        trade_history: list[TradeRecord] | None = None,
    ) -> None:
        if caps is None:
            # Default: 10 trades per minute per symbol
            caps = [
                FrequencyCapConfig(
                    max_trades=10, timeframe=Timeframe.MINUTE, per_symbol=True
                )
            ]

        # Validate caps
        for cap in caps:
            if cap.max_trades < 1:
                raise ValueError(f"max_trades must be >= 1, got {cap.max_trades}")

        self._caps = caps
        self._trade_history: deque[TradeRecord] = deque(trade_history or [])

        # Calculate max history retention (longest cap window + buffer)
        self._max_history_duration = max(cap.timeframe.to_timedelta() for cap in caps)

    @property
    def name(self) -> str:
        """Human-readable constraint name for logging and audit."""
        return "FrequencyCapConstraint"

    @property
    def caps(self) -> list[FrequencyCapConfig]:
        """Active frequency caps."""
        return self._caps

    def classify_risk(
        self,
        order: OrderRequest,
        portfolio_state: PortfolioState,
    ) -> bool:
        """Classify if this order is risk-increasing.

        Note: Frequency caps apply to ALL orders, not just risk-increasing ones.

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
        """Apply frequency cap constraint.

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

        # Get current time from market state or orders
        now = market_state.timestamp

        # Prune old history
        self._prune_history(now)

        # Track how many orders we're accepting in this batch
        # (for proper accounting within a single apply() call)
        batch_trades_by_symbol: dict[str, int] = {}
        batch_trades_global = 0

        for order in orders:
            # Check all caps
            violation = self._check_caps(order, now, batch_trades_by_symbol, batch_trades_global)

            if violation:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=violation,
                    )
                )
            else:
                result.append(order)
                # Track this order for remaining orders in batch
                batch_trades_by_symbol[order.symbol] = batch_trades_by_symbol.get(order.symbol, 0) + 1
                batch_trades_global += 1

        return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)

    def _check_caps(
        self,
        order: OrderRequest,
        now: datetime,
        batch_by_symbol: dict[str, int],
        batch_global: int,
    ) -> str | None:
        """Check if order would violate any frequency cap.

        Args:
            order: Order to check.
            now: Current timestamp.
            batch_by_symbol: Orders accepted in current batch by symbol.
            batch_global: Total orders accepted in current batch.

        Returns:
            Violation message if violated, None if OK.
        """
        for cap in self._caps:
            window_start = now - cap.timeframe.to_timedelta()

            if cap.per_symbol:
                # Count trades for this symbol in window
                history_count = sum(
                    1
                    for t in self._trade_history
                    if t.symbol == order.symbol and t.timestamp >= window_start
                )
                batch_count = batch_by_symbol.get(order.symbol, 0)
                total_count = history_count + batch_count

                if total_count >= cap.max_trades:
                    return (
                        f"Frequency cap exceeded for {order.symbol}: "
                        f"{total_count} trades in {cap.timeframe.name.lower()} "
                        f"(max {cap.max_trades})"
                    )
            else:
                # Count all trades in window
                history_count = sum(
                    1 for t in self._trade_history if t.timestamp >= window_start
                )
                total_count = history_count + batch_global

                if total_count >= cap.max_trades:
                    return (
                        f"Global frequency cap exceeded: "
                        f"{total_count} trades in {cap.timeframe.name.lower()} "
                        f"(max {cap.max_trades})"
                    )

        return None

    def _prune_history(self, now: datetime) -> None:
        """Remove trade records older than the max history duration."""
        cutoff = now - self._max_history_duration - timedelta(minutes=1)  # Small buffer
        while self._trade_history and self._trade_history[0].timestamp < cutoff:
            self._trade_history.popleft()

    def record_trade(
        self,
        symbol: str,
        timestamp: datetime,
        side: OrderSide,
        quantity: Decimal,
    ) -> None:
        """Record a completed trade for frequency tracking.

        Call this after an order is filled to update trade history.

        Args:
            symbol: Symbol traded.
            timestamp: When the trade occurred.
            side: Order side.
            quantity: Quantity filled.
        """
        record = TradeRecord(
            symbol=symbol,
            timestamp=timestamp,
            side=side,
            quantity=quantity,
        )
        self._trade_history.append(record)

    def get_trade_count(
        self,
        symbol: str | None = None,
        since: datetime | None = None,
    ) -> int:
        """Get count of trades in history.

        Args:
            symbol: If provided, count only trades for this symbol.
            since: If provided, count only trades after this time.

        Returns:
            Number of matching trades.
        """
        count = 0
        for record in self._trade_history:
            if symbol is not None and record.symbol != symbol:
                continue
            if since is not None and record.timestamp < since:
                continue
            count += 1
        return count

    def clear_history(self) -> None:
        """Clear all trade history."""
        self._trade_history.clear()


def create_frequency_cap(
    max_trades: int,
    per: str | Timeframe,
    per_symbol: bool = True,
) -> FrequencyCapConfig:
    """Convenience function to create a frequency cap config.

    Args:
        max_trades: Maximum trades allowed in window.
        per: Timeframe string (e.g., "hour", "day", "1m") or Timeframe enum.
        per_symbol: If True, limit is per symbol.

    Returns:
        FrequencyCapConfig instance.

    Example:
        >>> cap = create_frequency_cap(5, "hour")  # 5 per hour per symbol
        >>> cap = create_frequency_cap(100, "day", per_symbol=False)  # 100/day global
    """
    if isinstance(per, str):
        timeframe = Timeframe.from_string(per)
    else:
        timeframe = per

    return FrequencyCapConfig(
        max_trades=max_trades,
        timeframe=timeframe,
        per_symbol=per_symbol,
    )
