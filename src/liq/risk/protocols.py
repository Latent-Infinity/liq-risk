"""Protocol definitions for liq-risk.

These protocols define the interfaces for position sizing algorithms
and risk constraints. They enable pluggable, composable risk management.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from liq.core import OrderRequest, PortfolioState
    from liq.signals import Signal

    from liq.risk.config import MarketState, RiskConfig


@runtime_checkable
class PositionSizer(Protocol):
    """Protocol for position sizing algorithms.

    Implementations transform signals into sized orders.
    Pure function: no side effects, no state mutation.

    Example:
        >>> class VolatilitySizer:
        ...     def size_positions(
        ...         self,
        ...         signals: list[Signal],
        ...         portfolio_state: PortfolioState,
        ...         market_state: MarketState,
        ...         risk_config: RiskConfig,
        ...     ) -> list[OrderRequest]:
        ...         # Size based on volatility
        ...         ...
    """

    def size_positions(
        self,
        signals: list[Signal],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[OrderRequest]:
        """Size positions for given signals.

        Args:
            signals: Trading signals to size.
            portfolio_state: Current portfolio snapshot.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            List of sized OrderRequest objects.
        """
        ...


@runtime_checkable
class Constraint(Protocol):
    """Protocol for risk constraints.

    Constraints are applied in sequence, each potentially
    modifying or filtering the order list.

    Example:
        >>> class MaxPositionConstraint:
        ...     def apply(
        ...         self,
        ...         orders: list[OrderRequest],
        ...         portfolio_state: PortfolioState,
        ...         market_state: MarketState,
        ...         risk_config: RiskConfig,
        ...     ) -> list[OrderRequest]:
        ...         # Filter or modify orders
        ...         ...
    """

    def apply(
        self,
        orders: list[OrderRequest],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[OrderRequest]:
        """Apply constraint to orders.

        May reduce quantities, remove orders, or leave unchanged.

        Args:
            orders: Orders to constrain.
            portfolio_state: Current portfolio.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            Constrained order list.
        """
        ...
