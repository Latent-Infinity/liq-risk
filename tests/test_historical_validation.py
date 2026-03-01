"""Track B exit criterion: 3-month historical risk validation.

Validates that EWMARiskModel produces valid sigma/VaR/CVaR for >= 3 months
of historical data (63 trading days). All tests use seeded RNG for
determinism and have no external data dependencies.
"""

import numpy as np

from liq.risk.var_model import EWMARiskModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TRADING_DAYS_3_MONTHS = 63
TRADING_DAYS_6_MONTHS = 126


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_returns(
    rng: np.random.Generator,
    n_days: int,
    n_assets: int,
    mu: float = 0.0,
    sigma: float = 0.02,
) -> np.ndarray:
    """Generate synthetic daily returns."""
    return rng.normal(mu, sigma, size=(n_days, n_assets))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestThreeMonthHistoricalValidation:
    """EWMARiskModel produces valid risk metrics over >= 3 months of data."""

    def test_single_asset_3_months(self) -> None:
        """Single asset, 63 trading days: sigma > 0, VaR < 0, CVaR <= VaR."""
        rng = np.random.default_rng(42)
        returns = _generate_returns(rng, TRADING_DAYS_3_MONTHS, 1)
        weights = np.array([1.0])

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        result = model.compute(returns, weights)

        assert result.sigma > 0.0
        assert result.var < 0.0
        assert result.cvar <= result.var
        assert np.isfinite(result.sigma)
        assert np.isfinite(result.var)
        assert np.isfinite(result.cvar)

    def test_multi_asset_3_months(self) -> None:
        """5 assets, 63 days: valid metrics + covariance matrix properties."""
        rng = np.random.default_rng(123)
        n_assets = 5
        returns = _generate_returns(rng, TRADING_DAYS_3_MONTHS, n_assets)
        weights = np.array([0.3, 0.2, 0.2, 0.15, 0.15])

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        result = model.compute(returns, weights)

        # Core metrics
        assert result.sigma > 0.0
        assert result.var < 0.0
        assert result.cvar <= result.var

        # Covariance matrix shape and symmetry
        cov = result.covariance_matrix
        assert cov is not None
        assert cov.shape == (n_assets, n_assets)
        np.testing.assert_allclose(cov, cov.T, atol=1e-14)

        # Positive semi-definite: all eigenvalues >= 0
        eigenvalues = np.linalg.eigvalsh(cov)
        assert np.all(eigenvalues >= -1e-12)

    def test_six_months_stability(self) -> None:
        """126 days, 3 assets: no NaN/Inf, outputs in reasonable range."""
        rng = np.random.default_rng(77)
        n_assets = 3
        returns = _generate_returns(rng, TRADING_DAYS_6_MONTHS, n_assets, sigma=0.015)
        weights = np.array([0.4, 0.3, 0.3])

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        result = model.compute(returns, weights)

        # All outputs finite
        assert np.isfinite(result.sigma)
        assert np.isfinite(result.var)
        assert np.isfinite(result.cvar)

        # Reasonable ranges for daily returns with ~1.5% vol
        assert 0.0 < result.sigma < 0.10  # daily sigma < 10%
        assert -0.20 < result.var < 0.0  # daily VaR bounded
        assert result.cvar <= result.var

        # Covariance finite
        cov = result.covariance_matrix
        assert cov is not None
        assert np.all(np.isfinite(cov))

    def test_regime_shift_mid_window(self) -> None:
        """Low-vol first half, high-vol second half: model responds."""
        rng = np.random.default_rng(999)
        n_assets = 2
        half = TRADING_DAYS_3_MONTHS // 2

        low_vol = _generate_returns(rng, half, n_assets, sigma=0.005)
        high_vol = _generate_returns(rng, TRADING_DAYS_3_MONTHS - half, n_assets, sigma=0.04)
        returns = np.vstack([low_vol, high_vol])

        weights = np.array([0.5, 0.5])
        model = EWMARiskModel(decay=0.94, alpha=0.95)

        # Full window result
        result_full = model.compute(returns, weights)

        # Low-vol only result
        result_low = model.compute(low_vol, weights)

        # Model should reflect the regime shift: full-window sigma > low-vol sigma
        # because EWMA with decay=0.94 emphasizes recent (high-vol) data
        assert result_full.sigma > result_low.sigma

    def test_correlated_assets_3_months(self) -> None:
        """Known correlation structure: covariance captures it."""
        rng = np.random.default_rng(2024)
        n_assets = 3

        # Create correlated returns via Cholesky
        target_corr = np.array(
            [
                [1.0, 0.8, 0.2],
                [0.8, 1.0, 0.3],
                [0.2, 0.3, 1.0],
            ]
        )
        sigmas = np.array([0.02, 0.025, 0.015])
        target_cov = np.outer(sigmas, sigmas) * target_corr
        chol = np.linalg.cholesky(target_cov)

        raw = rng.standard_normal((TRADING_DAYS_3_MONTHS, n_assets))
        returns = raw @ chol.T

        weights = np.array([1.0 / 3, 1.0 / 3, 1.0 / 3])
        model = EWMARiskModel(decay=0.94, alpha=0.95)
        result = model.compute(returns, weights)

        cov = result.covariance_matrix
        assert cov is not None

        # Off-diagonal (0,1) should be substantially positive (high corr)
        assert cov[0, 1] > 0
        # Off-diagonal (0,2) should be positive but smaller
        assert cov[0, 2] > 0
        assert cov[0, 1] > cov[0, 2]

    def test_all_assets_positive_sigma_individually(self) -> None:
        """Each single-asset slice produces sigma > 0."""
        rng = np.random.default_rng(555)
        n_assets = 4
        returns = _generate_returns(rng, TRADING_DAYS_3_MONTHS, n_assets)

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        for i in range(n_assets):
            single = returns[:, i : i + 1]
            result = model.compute(single, np.array([1.0]))
            assert result.sigma > 0.0, f"Asset {i} has zero sigma"
