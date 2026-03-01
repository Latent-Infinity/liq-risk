"""EWMA covariance + parametric VaR/CVaR risk model.

Provides portfolio risk estimation using Exponentially Weighted Moving Average
(EWMA) covariance and parametric (normal) VaR/CVaR calculations.

Mathematical formulation:
    EWMA weights: w_i = lambda^(T-1-i), normalized to sum to 1
    EWMA covariance: Sigma = sum(w_i * (r_i - mu_w)(r_i - mu_w)^T)
    Portfolio sigma: sigma_p = sqrt(w^T Sigma w)
    Parametric VaR: mu_p - z_alpha * sigma_p
    Parametric CVaR: mu_p - sigma_p * phi(z_alpha) / (1 - alpha)

Where:
    lambda is the decay factor (typical 0.94 for daily data)
    z_alpha = normal ppf(alpha)
    phi = normal pdf
    w = portfolio weights vector
    r_i = return vector at time i
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


@dataclass(frozen=True)
class RiskModelOutput:
    """Output of a risk model computation.

    Attributes:
        sigma: Portfolio standard deviation (annualized or per-period,
            matching the input return frequency).
        var: Value at Risk. Negative value represents a loss threshold.
            At the given confidence level, losses are not expected to
            exceed this magnitude.
        cvar: Conditional Value at Risk (Expected Shortfall). More negative
            than VaR; represents the expected loss given that VaR is exceeded.
        mdd_pred: Predicted maximum drawdown over the configured horizon.
            Parametric approximation: ``sigma * sqrt(horizon)``.
            Zero-drift Brownian motion assumption (conservative for
            positive-drift portfolios).
        covariance_matrix: The EWMA covariance matrix, if computed.
            Shape (n_assets, n_assets). None when insufficient data.
    """

    sigma: float
    var: float
    cvar: float
    mdd_pred: float = 0.0
    covariance_matrix: NDArray[np.float64] | None = field(default=None, repr=False)


class EWMARiskModel:
    """EWMA covariance + parametric VaR/CVaR risk model.

    Computes portfolio risk using exponentially weighted covariance
    estimation and normal-distribution-based VaR/CVaR.

    Args:
        decay: EWMA decay factor lambda in (0, 1). Higher values weight
            history more evenly; lower values emphasize recent data.
            Typical: 0.94 for daily returns (RiskMetrics).
        alpha: Confidence level for VaR/CVaR in (0, 1).
            0.95 means "95% confidence that losses won't exceed VaR".
        min_periods: Minimum number of return observations required to
            produce a valid estimate. Below this, returns safe fallback.

    Example:
        >>> model = EWMARiskModel(decay=0.94, alpha=0.95)
        >>> returns = np.random.normal(0, 0.02, size=(100, 3))
        >>> weights = np.array([0.4, 0.3, 0.3])
        >>> result = model.compute(returns, weights)
        >>> result.sigma  # portfolio std dev
        >>> result.var    # VaR (negative = loss)
        >>> result.cvar   # CVaR (more negative than VaR)
    """

    __slots__ = ("_decay", "_alpha", "_min_periods", "_horizon", "_z_alpha", "_phi_z_alpha")

    def __init__(
        self,
        decay: float = 0.94,
        alpha: float = 0.95,
        min_periods: int = 10,
        horizon: int = 21,
    ) -> None:
        if not (0.0 < decay < 1.0):
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        if not (0.0 < alpha < 1.0):
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if min_periods < 1:
            raise ValueError(f"min_periods must be >= 1, got {min_periods}")
        if horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {horizon}")

        self._decay = decay
        self._alpha = alpha
        self._min_periods = min_periods
        self._horizon = horizon

        # Pre-compute z-score and pdf value for the confidence level
        self._z_alpha: float = float(norm.ppf(alpha))
        self._phi_z_alpha: float = float(norm.pdf(self._z_alpha))

    @property
    def decay(self) -> float:
        """EWMA decay factor lambda."""
        return self._decay

    @property
    def alpha(self) -> float:
        """Confidence level for VaR/CVaR."""
        return self._alpha

    @property
    def min_periods(self) -> int:
        """Minimum observations required for valid estimation."""
        return self._min_periods

    @property
    def horizon(self) -> int:
        """Horizon (in periods) for predicted maximum drawdown."""
        return self._horizon

    def compute(
        self,
        returns: NDArray[np.float64],
        weights: NDArray[np.float64],
    ) -> RiskModelOutput:
        """Compute portfolio risk metrics.

        Args:
            returns: Return time series. Shape (T, N) where T is the number
                of time periods and N is the number of assets. Also accepts
                1D array (T,) for single-asset portfolios.
            weights: Portfolio weight vector. Shape (N,) where N is the
                number of assets. Positive for long, negative for short.
                Need not sum to 1 (allows leveraged portfolios).

        Returns:
            RiskModelOutput with sigma, var, cvar, mdd_pred, and
            covariance_matrix. Returns safe fallback (all zeros) for
            insufficient/invalid data.

        Raises:
            ValueError: If weights length doesn't match number of assets.
        """
        returns = np.asarray(returns, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)

        # Handle 1D returns (single asset)
        if returns.ndim == 1:
            returns = returns.reshape(-1, 1)

        # Validate shapes
        if returns.ndim != 2:
            return _safe_fallback()

        t_obs, n_assets = returns.shape

        if weights.shape != (n_assets,):
            raise ValueError(
                f"weights length ({weights.shape[0]}) must match number of assets ({n_assets})"
            )

        # Check for insufficient data
        if t_obs < self._min_periods:
            return _safe_fallback()

        # Check for non-finite values (NaN, Inf)
        if not np.all(np.isfinite(returns)):
            # Drop rows with any non-finite values
            finite_mask = np.all(np.isfinite(returns), axis=1)
            returns = returns[finite_mask]
            t_obs = returns.shape[0]
            if t_obs < self._min_periods:
                return _safe_fallback()

        # Compute EWMA covariance matrix
        cov_matrix = self._ewma_covariance(returns)

        # Portfolio variance: w^T Sigma w
        port_variance = float(weights @ cov_matrix @ weights)

        # Guard against numerical issues
        if port_variance < 0.0:
            port_variance = 0.0

        sigma_p = np.sqrt(port_variance)

        # Portfolio mean return (EWMA-weighted)
        mu_p = self._ewma_portfolio_mean(returns, weights)

        # Parametric VaR and CVaR
        if sigma_p == 0.0:
            return RiskModelOutput(
                sigma=0.0,
                var=0.0,
                cvar=0.0,
                mdd_pred=0.0,
                covariance_matrix=cov_matrix,
            )

        var = mu_p - self._z_alpha * sigma_p
        cvar = mu_p - sigma_p * self._phi_z_alpha / (1.0 - self._alpha)
        mdd_pred = float(sigma_p * np.sqrt(self._horizon))

        return RiskModelOutput(
            sigma=float(sigma_p),
            var=float(var),
            cvar=float(cvar),
            mdd_pred=mdd_pred,
            covariance_matrix=cov_matrix,
        )

    def _ewma_weights(self, t_obs: int) -> NDArray[np.float64]:
        """Compute normalized EWMA weights for T observations.

        Weights are ordered chronologically: w[0] is the oldest observation,
        w[T-1] is the most recent (highest weight for decay < 1).

        Args:
            t_obs: Number of time observations.

        Returns:
            Normalized weight vector of shape (T,) summing to 1.
        """
        # Exponents: oldest gets highest exponent (decays most)
        exponents = np.arange(t_obs - 1, -1, -1, dtype=np.float64)
        raw_weights = self._decay**exponents
        return raw_weights / raw_weights.sum()

    def _ewma_covariance(self, returns: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute EWMA covariance matrix.

        Args:
            returns: Shape (T, N) return matrix.

        Returns:
            Covariance matrix of shape (N, N).
        """
        t_obs, n_assets = returns.shape
        ewma_w = self._ewma_weights(t_obs)

        # Weighted mean
        mu = ewma_w @ returns  # shape (N,)

        # De-mean
        demeaned = returns - mu  # shape (T, N)

        # Weighted covariance: sum_i w_i * (r_i - mu)(r_i - mu)^T
        # Efficient: (demeaned.T * sqrt(w)) @ (demeaned * sqrt(w)).T
        # But simpler: weight each row and matrix multiply
        weighted = demeaned * ewma_w[:, np.newaxis]  # shape (T, N)
        cov = demeaned.T @ weighted  # shape (N, N)

        return cov

    def _ewma_portfolio_mean(
        self,
        returns: NDArray[np.float64],
        port_weights: NDArray[np.float64],
    ) -> float:
        """Compute EWMA-weighted portfolio mean return.

        Args:
            returns: Shape (T, N) return matrix.
            port_weights: Shape (N,) portfolio weight vector.

        Returns:
            Scalar portfolio mean return.
        """
        t_obs = returns.shape[0]
        ewma_w = self._ewma_weights(t_obs)

        # Portfolio returns per period
        port_returns = returns @ port_weights  # shape (T,)

        # EWMA-weighted mean
        return float(ewma_w @ port_returns)


def _safe_fallback() -> RiskModelOutput:
    """Return a safe zero-value fallback for insufficient data."""
    return RiskModelOutput(sigma=0.0, var=0.0, cvar=0.0)
