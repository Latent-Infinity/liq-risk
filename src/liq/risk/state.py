"""Layered market state types for liq-risk.

Provides structured inputs for sizing and constraint decisions,
organized into logical layers.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from liq.core import OrderRequest
    from liq.core.bar import Bar

from liq.core import OrderSide

from liq.risk.enums import PriceReference


@dataclass(frozen=True)
class PriceState:
    """Current price data - minimum required input.

    This is the core price data every sizer needs. Contains
    the most recent bar for each symbol being traded.

    Attributes:
        current_bars: Map of symbol to most recent Bar.
        timestamp: State snapshot time (UTC, timezone-aware).

    Example:
        >>> state = PriceState(
        ...     current_bars={"AAPL": bar_aapl},
        ...     timestamp=now,
        ... )
        >>> price = state.get_price("AAPL", PriceReference.MIDRANGE)
    """

    current_bars: dict[str, Bar]
    timestamp: datetime

    def get_price(self, symbol: str, ref: PriceReference) -> Decimal | None:
        """Get price for symbol using specified reference.

        Args:
            symbol: Trading symbol.
            ref: Price reference method (MIDRANGE, CLOSE, VWAP).

        Returns:
            Price as Decimal, or None if symbol not found.
        """
        bar = self.current_bars.get(symbol)
        if bar is None:
            return None

        if ref == PriceReference.MIDRANGE:
            return (bar.high + bar.low) / 2
        elif ref == PriceReference.CLOSE:
            return bar.close
        elif ref == PriceReference.VWAP:
            # VWAP requires additional data; fall back to close
            return bar.close
        else:
            return bar.close


@dataclass(frozen=True)
class RiskFactors:
    """Risk factor data - required for volatility-based sizing.

    Contains computed risk metrics like volatility, correlations,
    and market regime. Typically computed by liq-features or
    provided externally.

    Attributes:
        volatility: ATR or range-based volatility per symbol (float).
        correlations: Pairwise correlation matrix (optional, polars DataFrame).
        regime: Market regime label (optional, e.g., "high_volatility").

    Note:
        Volatility is stored as float for computation efficiency.
        Use Decimal only at currency boundaries.

    Example:
        >>> factors = RiskFactors(
        ...     volatility={"AAPL": 2.5, "GOOGL": 3.2},
        ...     regime="normal",
        ... )
    """

    volatility: dict[str, float]
    correlations: Any | None = None  # polars.DataFrame when available
    regime: str | None = None


@dataclass(frozen=True)
class AssetMetadata:
    """Static asset information for constraint checking.

    Contains classification and reference data that doesn't
    change during a trading session.

    Attributes:
        sector_map: Symbol to sector mapping (e.g., "AAPL" -> "Technology").
        group_map: Symbol to multi-dimensional grouping. Supports arbitrary
            groupings like asset_class, exchange, country, theme.
        liquidity: Average daily volume per symbol (float).
        borrow_rates: Per-symbol annualized borrow rates for shorts (float).

    Example:
        >>> metadata = AssetMetadata(
        ...     sector_map={"AAPL": "Technology", "JPM": "Financials"},
        ...     group_map={
        ...         "AAPL": {"country": "US", "size": "large"},
        ...     },
        ... )
    """

    sector_map: dict[str, str] | None = None
    group_map: dict[str, dict[str, str]] | None = None
    liquidity: dict[str, float] | None = None
    borrow_rates: dict[str, float] | None = None


@dataclass
class ExecutionState:
    """Execution context for netting calculations.

    Captures what's already in flight (open orders) so we can
    calculate proper deltas and reserved capital.

    Attributes:
        open_orders: Pending orders not yet filled.
        reserved_capital: Total cash reserved for open buy orders.

    Properties:
        reserved_by_symbol: Capital reserved per symbol from open orders.

    Example:
        >>> state = ExecutionState(
        ...     open_orders=[pending_order],
        ...     reserved_capital=Decimal("15000"),
        ... )
        >>> reserved = state.reserved_by_symbol
    """

    open_orders: list[OrderRequest]
    reserved_capital: Decimal

    @property
    def reserved_by_symbol(self) -> dict[str, Decimal]:
        """Capital reserved per symbol from open orders.

        Only counts buy orders since sells don't consume capital.

        Returns:
            Map of symbol to reserved capital amount.
        """
        reserved: dict[str, Decimal] = {}

        for order in self.open_orders:
            # Only buy orders reserve capital
            if order.side != OrderSide.BUY:
                continue

            # Calculate order value (quantity * limit_price or estimate)
            price = order.limit_price if order.limit_price else Decimal("0")
            if price == 0:
                # For market orders, we'd need current price
                # This is a simplification; real implementation would
                # use current market price
                continue

            order_value = order.quantity * price

            if order.symbol in reserved:
                reserved[order.symbol] += order_value
            else:
                reserved[order.symbol] = order_value

        return reserved
