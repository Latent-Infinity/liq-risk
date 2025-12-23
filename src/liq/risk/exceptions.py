"""Exception types for liq-risk.

Defines a hierarchy of exceptions for risk and capital violations.
"""

from __future__ import annotations


class RiskError(Exception):
    """Base exception for all risk-related errors."""

    pass


class InsufficientBuyingPowerError(RiskError):
    """Raised when order value exceeds available cash.

    Attributes:
        order_value: The value of the attempted order.
        available_cash: The cash available for trading.
        symbol: The symbol being traded.
    """

    def __init__(
        self,
        message: str,
        *,
        order_value: float | None = None,
        available_cash: float | None = None,
        symbol: str | None = None,
    ) -> None:
        super().__init__(message)
        self.order_value = order_value
        self.available_cash = available_cash
        self.symbol = symbol


class LeverageExceededError(RiskError):
    """Raised when leverage limits are exceeded.

    Attributes:
        current_leverage: The current leverage ratio.
        max_leverage: The maximum allowed leverage.
        leverage_type: Type of leverage (gross, net).
    """

    def __init__(
        self,
        message: str,
        *,
        current_leverage: float | None = None,
        max_leverage: float | None = None,
        leverage_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.current_leverage = current_leverage
        self.max_leverage = max_leverage
        self.leverage_type = leverage_type


class EquityFloorBreachedError(RiskError):
    """Raised when equity falls to or below zero.

    Attributes:
        current_equity: The current equity value.
    """

    def __init__(
        self,
        message: str,
        *,
        current_equity: float | None = None,
    ) -> None:
        super().__init__(message)
        self.current_equity = current_equity


class TradingHaltedError(RiskError):
    """Raised when trading is halted due to risk limits.

    Attributes:
        halt_reason: The reason trading was halted.
        halt_type: Type of halt (drawdown, daily_loss, equity_floor).
    """

    def __init__(
        self,
        message: str,
        *,
        halt_reason: str | None = None,
        halt_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.halt_reason = halt_reason
        self.halt_type = halt_type
