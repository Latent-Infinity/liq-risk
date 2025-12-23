"""Enumerations for liq-risk.

Defines behavior modes for sizing, halts, and price references.
"""

from enum import Enum


class HaltMode(Enum):
    """Trading halt behavior modes.

    Determines which orders are blocked when trading is halted
    due to drawdown, daily loss, or equity floor breach.

    Attributes:
        HALT_BUYS_ONLY: Block new long entries only. Sells and
            short covers are allowed. Default behavior.
        HALT_ALL_RISK_INCREASING: Block any order that increases
            position risk (absolute size increase). Allows
            risk-reducing orders like closing positions.
        HALT_ALL_TRADES: Emergency mode - block all orders.
            Use only for system-wide issues.
    """

    HALT_BUYS_ONLY = "halt_buys_only"
    HALT_ALL_RISK_INCREASING = "halt_risk_inc"
    HALT_ALL_TRADES = "halt_all"


class SizingMode(Enum):
    """How to handle existing positions when sizing.

    Determines whether sizing is additive to existing positions
    or calculates absolute targets.

    Attributes:
        INCREMENTAL: Add to existing position. New signal generates
            additional quantity on top of current holdings.
            target = current + signal_sized_quantity
        REBALANCE: Target absolute position. Signal generates
            target quantity; order is the delta from current.
            order_qty = target - current - reserved
        REPLACE: Close existing position and open new. Used for
            direction changes or strategy rotation.
    """

    INCREMENTAL = "incremental"
    REBALANCE = "rebalance"
    REPLACE = "replace"


class PriceReference(Enum):
    """Which price to use for sizing calculations.

    Different price references have different characteristics
    for sizing accuracy and slippage.

    Attributes:
        MIDRANGE: (high + low) / 2. Default. More stable than
            close, represents typical trading range.
        CLOSE: Last traded price. Most recent but can be
            at extremes of the range.
        VWAP: Volume-weighted average price. Best execution
            estimate but requires volume data.
    """

    MIDRANGE = "midrange"
    CLOSE = "close"
    VWAP = "vwap"
