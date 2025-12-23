"""Protocol definitions for liq-risk.

These protocols define the interfaces for position sizing algorithms
and risk constraints. They enable pluggable, composable risk management.

Version History:
    v0.1: Original protocols (OrderRequest-based)
    v0.2: Added TargetPosition-based sizers and ConstraintResult-based constraints
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from liq.core import OrderRequest, PortfolioState
    from liq.signals import Signal

    from liq.risk.config import MarketState, RiskConfig
    from liq.risk.types import ConstraintResult, TargetPosition


@runtime_checkable
class PositionSizer(Protocol):
    """Protocol for position sizing algorithms.

    Implementations transform signals into sized orders.
    Pure function: no side effects, no state mutation.

    Note:
        This is the legacy protocol returning OrderRequest.
        New implementations should use TargetPositionSizer instead.

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
class TargetPositionSizer(Protocol):
    """Protocol for position sizing algorithms returning TargetPosition.

    This is the preferred protocol for new implementations. It returns
    execution-agnostic TargetPosition objects that can be converted
    to OrderRequests using adapters.

    Example:
        >>> class VolatilitySizer:
        ...     def size_positions(
        ...         self,
        ...         signals: list[Signal],
        ...         portfolio_state: PortfolioState,
        ...         market_state: MarketState,
        ...         risk_config: RiskConfig,
        ...     ) -> list[TargetPosition]:
        ...         # Size based on volatility, returning targets
        ...         ...
    """

    def size_positions(
        self,
        signals: list[Signal],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> list[TargetPosition]:
        """Size positions for given signals.

        Args:
            signals: Trading signals to size.
            portfolio_state: Current portfolio snapshot.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            List of TargetPosition objects with target quantities.
            Each TargetPosition knows its current vs target state.
        """
        ...


@runtime_checkable
class Constraint(Protocol):
    """Protocol for risk constraints.

    Constraints are applied in sequence, each potentially
    modifying or filtering the order list.

    Note:
        This is the legacy protocol returning list[OrderRequest].
        New implementations should use StructuredConstraint instead.

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


@runtime_checkable
class StructuredConstraint(Protocol):
    """Protocol for risk constraints with structured output.

    This is the preferred protocol for new implementations. It returns
    a ConstraintResult that includes rejected orders with reasons,
    enabling proper audit trails.

    Example:
        >>> class MaxPositionConstraint:
        ...     @property
        ...     def name(self) -> str:
        ...         return "MaxPositionConstraint"
        ...
        ...     def apply(
        ...         self,
        ...         orders: list[OrderRequest],
        ...         portfolio_state: PortfolioState,
        ...         market_state: MarketState,
        ...         risk_config: RiskConfig,
        ...     ) -> ConstraintResult:
        ...         # Filter or modify orders, tracking rejections
        ...         ...
        ...
        ...     def classify_risk(
        ...         self,
        ...         order: OrderRequest,
        ...         portfolio_state: PortfolioState,
        ...     ) -> bool:
        ...         # Return True if risk-increasing
        ...         ...
    """

    @property
    def name(self) -> str:
        """Human-readable constraint name for logging and audit."""
        ...

    def apply(
        self,
        orders: list[OrderRequest],
        portfolio_state: PortfolioState,
        market_state: MarketState,
        risk_config: RiskConfig,
    ) -> ConstraintResult:
        """Apply constraint to orders.

        May reduce quantities, remove orders, or leave unchanged.
        MUST track what was changed and why in the result.

        Args:
            orders: Orders to constrain.
            portfolio_state: Current portfolio.
            market_state: Current market conditions.
            risk_config: Risk parameters.

        Returns:
            ConstraintResult with:
            - orders: The surviving/modified orders
            - rejected: List of RejectedOrder with reasons
            - warnings: Non-fatal issues to log
        """
        ...

    def classify_risk(
        self,
        order: OrderRequest,
        portfolio_state: PortfolioState,
    ) -> bool:
        """Classify if this order is risk-increasing.

        Used by halts to allow risk-reducing orders through.

        Args:
            order: The order to classify.
            portfolio_state: Current portfolio for context.

        Returns:
            True if risk-increasing, False if risk-reducing.
        """
        ...
