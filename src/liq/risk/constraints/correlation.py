"""Correlation constraint.

CorrelationConstraint: Limits exposure to highly correlated assets.
"""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide

from liq.risk.types import ConstraintResult, RejectedOrder

if TYPE_CHECKING:
    from liq.core import PortfolioState

    from liq.risk.config import MarketState, RiskConfig


class CorrelationConstraint:
    """Limit exposure to highly correlated assets.

    Filters out buy orders for assets that are highly correlated
    with existing positions or pending orders. This promotes
    portfolio diversification.

    The constraint checks:
    1. Correlation with existing portfolio positions
    2. Correlation with already-accepted orders in the batch

    Sell orders always pass (reduce exposure).
    Missing correlation data is treated as allowed.

    Example:
        >>> constraint = CorrelationConstraint()
        >>> result = constraint.apply(orders, portfolio, market, config)
        >>> for r in result.rejected:
        ...     print(f"Rejected {r.order.symbol}: {r.reason}")
    """

    @property
    def name(self) -> str:
        """Human-readable constraint name for logging and audit."""
        return "CorrelationConstraint"

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
        """Apply correlation constraint.

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

        # If no max_correlation config, pass all orders through
        max_correlation = risk_config.max_correlation
        if max_correlation is None:
            return ConstraintResult(orders=list(orders), rejected=rejected, warnings=warnings)

        # If no correlation data, pass all orders through
        correlations = market_state.correlations
        if correlations is None:
            return ConstraintResult(orders=list(orders), rejected=rejected, warnings=warnings)

        # Get existing position symbols
        existing_symbols: set[str] = set(portfolio_state.positions.keys())

        # Track accepted symbols for this batch
        accepted_symbols: set[str] = set()

        result: list[OrderRequest] = []

        for order in orders:
            # Sell orders always pass (reduce exposure)
            if order.side == OrderSide.SELL:
                result.append(order)
                continue

            # Check correlation with existing positions and accepted orders
            all_check_symbols = existing_symbols | accepted_symbols

            correlated_with = self._find_highly_correlated(
                order.symbol, all_check_symbols, correlations, max_correlation
            )

            if correlated_with is not None:
                # Skip this order - too correlated
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Highly correlated with {correlated_with} "
                        f"(max correlation {max_correlation:.2f})",
                    )
                )
                continue

            # Accept the order
            result.append(order)
            accepted_symbols.add(order.symbol)

        return ConstraintResult(orders=result, rejected=rejected, warnings=warnings)

    def _find_highly_correlated(
        self,
        symbol: str,
        check_symbols: set[str],
        correlations: dict[tuple[str, str], float],
        max_correlation: float,
    ) -> str | None:
        """Find a symbol highly correlated with this one.

        Args:
            symbol: Symbol to check.
            check_symbols: Symbols to check correlation against.
            correlations: Correlation matrix as dict.
            max_correlation: Maximum allowed correlation.

        Returns:
            Symbol that is highly correlated, or None if none found.
        """
        for check_symbol in check_symbols:
            if check_symbol == symbol:
                continue

            # Try both orderings of the pair
            corr = correlations.get((symbol, check_symbol))
            if corr is None:
                corr = correlations.get((check_symbol, symbol))

            if corr is None:
                # No correlation data for this pair - allow
                continue

            # Check if absolute correlation exceeds threshold
            # Negative correlations are allowed (hedging)
            if corr > max_correlation:
                return check_symbol

        return None
