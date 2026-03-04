"""Bounded update constraints for promotion gate (Stage 2).

OutputSpaceBoundConstraint: limits per-asset weight delta, total turnover, and trade count.
RiskSpaceBoundConstraint: limits portfolio sigma and CVaR delta using EWMARiskModel.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
from liq.core import OrderRequest, OrderSide
from numpy.typing import NDArray

from liq.risk.types import ConstraintResult, RejectedOrder
from liq.risk.var_model import EWMARiskModel, RiskModelOutput

if TYPE_CHECKING:
    from liq.core import PortfolioState

    from liq.risk.config import MarketState, RiskConfig


@dataclass(frozen=True)
class OutputSpaceBoundConfig:
    """Configuration for output-space bounded update checks.

    Attributes:
        delta_w_max: Maximum allowed absolute weight change per asset.
        delta_turnover_max: Maximum total portfolio turnover (sum |Δw_i| / 2).
        delta_trades_max: Maximum number of trades allowed.
    """

    delta_w_max: float
    delta_turnover_max: float
    delta_trades_max: int


@dataclass(frozen=True)
class RiskSpaceBoundConfig:
    """Configuration for risk-space bounded update checks.

    Attributes:
        delta_sigma_max: Maximum change in portfolio standard deviation.
        delta_cvar_max: Maximum change in portfolio CVaR (Expected Shortfall).
        delta_mdd_pred_max: Maximum change in predicted max drawdown (optional).
    """

    delta_sigma_max: float
    delta_cvar_max: float
    delta_mdd_pred_max: float | None = None


class OutputSpaceBoundConstraint:
    """Limit per-asset weight delta, total turnover, and trade count.

    Implements the StructuredConstraint protocol from liq-risk.
    Orders are evaluated against output-space bounds:
    1. Per-asset weight delta: |proposed_w - current_w| <= delta_w_max
    2. Total turnover: sum(|Δw_i|) / 2 <= delta_turnover_max
    3. Trade count: n_trades <= delta_trades_max
    """

    def __init__(self, config: OutputSpaceBoundConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        return "OutputSpaceBoundConstraint"

    def classify_risk(
        self,
        order: OrderRequest,
        portfolio_state: PortfolioState,
    ) -> bool:
        """Return True if order increases position risk."""
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
        """Apply output-space bounded update checks."""
        if not orders:
            return ConstraintResult(orders=[], rejected=[])

        equity = portfolio_state.equity
        if equity <= 0:
            return ConstraintResult(
                orders=[],
                rejected=[
                    RejectedOrder(
                        order=o,
                        constraint_name=self.name,
                        reason="Zero or negative equity",
                    )
                    for o in orders
                ],
            )

        rejected: list[RejectedOrder] = []
        passed: list[OrderRequest] = []

        # 1. Check per-asset weight deltas
        weight_ok_orders: list[OrderRequest] = []
        for order in orders:
            bar = market_state.current_bars.get(order.symbol)
            if bar is None:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"No bar data for {order.symbol}",
                    )
                )
                continue

            price = bar.close
            order_value = order.quantity * price

            # Proposed weight change
            if order.side == OrderSide.BUY:
                delta_w = float(order_value / equity)
            else:
                delta_w = float(order_value / equity)

            if abs(delta_w) > self._config.delta_w_max:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Weight delta {abs(delta_w):.4f} exceeds "
                        f"max {self._config.delta_w_max:.4f}",
                    )
                )
            else:
                weight_ok_orders.append(order)

        if not weight_ok_orders:
            return ConstraintResult(orders=[], rejected=rejected)

        # 2. Check total turnover: sum(|Δw_i|) / 2
        total_abs_delta = Decimal("0")
        for order in weight_ok_orders:
            bar = market_state.current_bars[order.symbol]
            order_value = order.quantity * bar.close
            total_abs_delta += order_value

        turnover = float(total_abs_delta / equity) / 2.0

        if turnover > self._config.delta_turnover_max:
            for order in weight_ok_orders:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Total turnover {turnover:.4f} exceeds "
                        f"max {self._config.delta_turnover_max:.4f}",
                    )
                )
            return ConstraintResult(orders=[], rejected=rejected)

        # 3. Check trade count
        # Trade-count overflow: keep first N, reject rest individually.
        # The constraint is partial-permissive by design — it preserves
        # which orders would pass if the batch were smaller. The upstream
        # BoundedUpdateGate treats any non-empty rejected list as a hard
        # fail, so overall gating remains all-or-nothing.
        if len(weight_ok_orders) > self._config.delta_trades_max:
            passed = weight_ok_orders[: self._config.delta_trades_max]
            for order in weight_ok_orders[self._config.delta_trades_max :]:
                rejected.append(
                    RejectedOrder(
                        order=order,
                        constraint_name=self.name,
                        reason=f"Trade count {len(weight_ok_orders)} exceeds "
                        f"max {self._config.delta_trades_max}",
                    )
                )
        else:
            passed = weight_ok_orders

        return ConstraintResult(orders=passed, rejected=rejected)


class RiskSpaceBoundConstraint:
    """Limit portfolio risk metric deltas (sigma, CVaR, predicted MDD).

    Implements the StructuredConstraint protocol from liq-risk.
    Evaluates whether proposed orders would shift portfolio risk metrics
    beyond acceptable bounds. Since risk is portfolio-level, this is a
    batch accept/reject decision (all orders pass or all rejected).
    """

    def __init__(
        self,
        config: RiskSpaceBoundConfig,
        risk_model: EWMARiskModel,
        returns_history: NDArray[np.float64],
        current_weights: NDArray[np.float64],
        symbols: list[str] | None = None,
    ) -> None:
        self._config = config
        self._risk_model = risk_model
        self._returns_history = np.asarray(returns_history, dtype=np.float64)
        self._current_weights = np.asarray(current_weights, dtype=np.float64)
        self._symbols = symbols
        self._current_risk: RiskModelOutput = risk_model.compute(
            self._returns_history, self._current_weights
        )

    @property
    def name(self) -> str:
        return "RiskSpaceBoundConstraint"

    def classify_risk(
        self,
        order: OrderRequest,
        portfolio_state: PortfolioState,
    ) -> bool:
        """Return True if order increases position risk."""
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
        """Apply risk-space bounded update checks.

        Computes proposed weights from orders + portfolio, then compares
        risk metrics against current risk. Batch reject if any threshold breached.
        """
        if not orders:
            return ConstraintResult(orders=[], rejected=[])

        equity = portfolio_state.equity
        if equity <= 0:
            return ConstraintResult(
                orders=[],
                rejected=[
                    RejectedOrder(
                        order=o,
                        constraint_name=self.name,
                        reason="Zero or negative equity",
                    )
                    for o in orders
                ],
            )

        # Build proposed weights from portfolio + orders
        proposed_weights = self._compute_proposed_weights(orders, portfolio_state, market_state)

        if proposed_weights is None:
            return ConstraintResult(
                orders=[],
                rejected=[
                    RejectedOrder(
                        order=o,
                        constraint_name=self.name,
                        reason="Cannot compute proposed weights (missing bar data)",
                    )
                    for o in orders
                ],
            )

        # Compute proposed risk
        proposed_risk = self._risk_model.compute(self._returns_history, proposed_weights)

        # Check sigma delta
        violations: list[str] = []
        sigma_delta = abs(proposed_risk.sigma - self._current_risk.sigma)
        if sigma_delta > self._config.delta_sigma_max:
            violations.append(
                f"Sigma delta {sigma_delta:.6f} exceeds max {self._config.delta_sigma_max:.6f}"
            )

        # Check CVaR delta
        cvar_delta = abs(proposed_risk.cvar - self._current_risk.cvar)
        if cvar_delta > self._config.delta_cvar_max:
            violations.append(
                f"CVaR delta {cvar_delta:.6f} exceeds max {self._config.delta_cvar_max:.6f}"
            )

        # Check MDD pred delta (optional)
        if self._config.delta_mdd_pred_max is not None:
            mdd_delta = abs(proposed_risk.mdd_pred - self._current_risk.mdd_pred)
            if mdd_delta > self._config.delta_mdd_pred_max:
                violations.append(
                    f"MDD pred delta {mdd_delta:.6f} exceeds max "
                    f"{self._config.delta_mdd_pred_max:.6f}"
                )

        if violations:
            reason = "; ".join(violations)
            return ConstraintResult(
                orders=[],
                rejected=[
                    RejectedOrder(
                        order=o,
                        constraint_name=self.name,
                        reason=reason,
                    )
                    for o in orders
                ],
            )

        return ConstraintResult(orders=list(orders), rejected=[])

    def _compute_proposed_weights(
        self,
        orders: list[OrderRequest],
        portfolio_state: PortfolioState,
        market_state: MarketState,
    ) -> NDArray[np.float64] | None:
        """Compute proposed portfolio weights after applying orders.

        When ``self._symbols`` is set, uses explicit symbol→column-index mapping
        so that risk is always computed against the correct correlation columns.
        Returns None if bar data is missing for any traded symbol or if a symbol
        cannot be mapped to the returns matrix.
        """
        equity = portfolio_state.equity
        n_assets = self._returns_history.shape[1]

        if self._symbols is not None:
            # Explicit symbol mapping mode
            symbol_to_idx: dict[str, int] = {sym: idx for idx, sym in enumerate(self._symbols)}

            proposed = np.zeros(n_assets)

            # Map current positions
            for sym, position in portfolio_state.positions.items():
                idx = symbol_to_idx.get(sym)
                if idx is None:
                    # Position in a symbol not in the returns matrix — reject
                    return None
                proposed[idx] = float(position.market_value / equity)

            # Apply order effects
            for order in orders:
                bar = market_state.current_bars.get(order.symbol)
                if bar is None:
                    return None

                idx = symbol_to_idx.get(order.symbol)
                if idx is None:
                    # Order for a symbol not in the returns matrix — reject
                    return None

                order_value = float(order.quantity * bar.close)
                weight_delta = order_value / float(equity)

                if order.side == OrderSide.BUY:
                    proposed[idx] += weight_delta
                else:
                    proposed[idx] -= weight_delta

            return proposed

        # Fallback: no explicit symbol mapping
        # Build symbol list from portfolio + orders
        symbols = list(portfolio_state.positions.keys())
        for order in orders:
            if order.symbol not in symbols:
                symbols.append(order.symbol)

        if len(symbols) != n_assets:
            # Symbol count mismatch with no mapping — cannot safely compute risk
            return None

        # Direct mapping: symbols align with weight vector positions
        proposed = np.zeros(n_assets)
        for i, sym in enumerate(symbols):
            position = portfolio_state.positions.get(sym)
            if position:
                proposed[i] = float(position.market_value / equity)

        # Apply order effects
        for order in orders:
            bar = market_state.current_bars.get(order.symbol)
            if bar is None:
                return None

            order_value = float(order.quantity * bar.close)
            weight_delta = order_value / float(equity)

            if order.symbol in symbols:
                idx = symbols.index(order.symbol)
                if order.side == OrderSide.BUY:
                    proposed[idx] += weight_delta
                else:
                    proposed[idx] -= weight_delta

        return proposed
