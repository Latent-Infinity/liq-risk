"""Tests for EWMA covariance + parametric VaR/CVaR risk model.

Following TDD: RED phase - write failing tests first.

Mathematical formulas under test:
    EWMA covariance: Sigma = sum(w_i * (r_i - mu)(r_i - mu)^T) with exponentially decaying weights
    Portfolio variance: sigma_p^2 = w^T Sigma w
    Parametric VaR = mu_p - z_alpha * sigma_p  (normal assumption)
    Parametric CVaR = mu_p - sigma_p * phi(z_alpha) / (1 - alpha)  (expected shortfall)

Where:
    w_i = lambda^i * (1 - lambda) / (1 - lambda^T)  (EWMA weights, normalized)
    z_alpha = normal ppf(alpha)
    phi = normal pdf
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm


class TestRiskModelOutputStructure:
    """Test RiskModelOutput dataclass structure and validation."""

    def test_risk_model_output_has_required_fields(self) -> None:
        """RiskModelOutput must have sigma, var, cvar fields."""
        from liq.risk.var_model import RiskModelOutput

        output = RiskModelOutput(sigma=0.01, var=-0.02, cvar=-0.03)
        assert hasattr(output, "sigma")
        assert hasattr(output, "var")
        assert hasattr(output, "cvar")

    def test_risk_model_output_is_frozen(self) -> None:
        """RiskModelOutput should be immutable."""
        from liq.risk.var_model import RiskModelOutput

        output = RiskModelOutput(sigma=0.01, var=-0.02, cvar=-0.03)
        with pytest.raises(AttributeError):
            output.sigma = 0.5  # type: ignore[misc]

    def test_risk_model_output_stores_values(self) -> None:
        """RiskModelOutput must store provided values accurately."""
        from liq.risk.var_model import RiskModelOutput

        output = RiskModelOutput(sigma=0.015, var=-0.0247, cvar=-0.031)
        assert output.sigma == pytest.approx(0.015)
        assert output.var == pytest.approx(-0.0247)
        assert output.cvar == pytest.approx(-0.031)

    def test_risk_model_output_optional_covariance_matrix(self) -> None:
        """RiskModelOutput should optionally store the covariance matrix."""
        from liq.risk.var_model import RiskModelOutput

        cov = np.array([[0.01, 0.002], [0.002, 0.02]])
        output = RiskModelOutput(sigma=0.01, var=-0.02, cvar=-0.03, covariance_matrix=cov)
        assert output.covariance_matrix is not None
        np.testing.assert_array_equal(output.covariance_matrix, cov)

    def test_risk_model_output_covariance_matrix_defaults_none(self) -> None:
        """RiskModelOutput covariance_matrix should default to None."""
        from liq.risk.var_model import RiskModelOutput

        output = RiskModelOutput(sigma=0.01, var=-0.02, cvar=-0.03)
        assert output.covariance_matrix is None

    def test_risk_model_output_has_mdd_pred_field(self) -> None:
        """RiskModelOutput must have mdd_pred field."""
        from liq.risk.var_model import RiskModelOutput

        output = RiskModelOutput(sigma=0.01, var=-0.02, cvar=-0.03, mdd_pred=0.05)
        assert output.mdd_pred == pytest.approx(0.05)

    def test_risk_model_output_mdd_pred_defaults_zero(self) -> None:
        """RiskModelOutput mdd_pred should default to 0.0."""
        from liq.risk.var_model import RiskModelOutput

        output = RiskModelOutput(sigma=0.01, var=-0.02, cvar=-0.03)
        assert output.mdd_pred == 0.0


class TestEWMARiskModelInstantiation:
    """Test EWMARiskModel construction."""

    def test_default_construction(self) -> None:
        """EWMARiskModel can be created with defaults."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel()
        assert model is not None

    def test_custom_parameters(self) -> None:
        """EWMARiskModel accepts custom decay, alpha, min_periods."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.97, alpha=0.99, min_periods=30)
        assert model.decay == 0.97
        assert model.alpha == 0.99
        assert model.min_periods == 30


class TestEWMARiskModelDeterminism:
    """Verify deterministic output for identical inputs."""

    def test_same_input_same_output(self) -> None:
        """Identical inputs must produce identical outputs."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=(100, 3))
        weights = np.array([0.4, 0.3, 0.3])

        result1 = model.compute(returns, weights)
        result2 = model.compute(returns, weights)

        assert result1.sigma == pytest.approx(result2.sigma)
        assert result1.var == pytest.approx(result2.var)
        assert result1.cvar == pytest.approx(result2.cvar)

    def test_deterministic_across_instances(self) -> None:
        """Different model instances with same params must produce same output."""
        from liq.risk.var_model import EWMARiskModel

        rng = np.random.default_rng(99)
        returns = rng.normal(0.0005, 0.015, size=(50, 2))
        weights = np.array([0.6, 0.4])

        model_a = EWMARiskModel(decay=0.94, alpha=0.95)
        model_b = EWMARiskModel(decay=0.94, alpha=0.95)

        result_a = model_a.compute(returns, weights)
        result_b = model_b.compute(returns, weights)

        assert result_a.sigma == pytest.approx(result_b.sigma)
        assert result_a.var == pytest.approx(result_b.var)
        assert result_a.cvar == pytest.approx(result_b.cvar)


class TestEWMARiskModelOutputValidity:
    """Verify sigma, VaR, CVaR have correct sign and reasonable range."""

    def test_sigma_is_non_negative(self) -> None:
        """Portfolio standard deviation must be >= 0."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=(100, 3))
        weights = np.array([0.4, 0.3, 0.3])

        result = model.compute(returns, weights)
        assert result.sigma >= 0.0

    def test_var_is_negative_or_zero_for_typical_alpha(self) -> None:
        """VaR at 95% confidence should be negative for typical returns (loss measure).

        VaR = mu - z_alpha * sigma. For alpha >= 0.5, z_alpha > 0, so if
        sigma > 0 and returns are not abnormally positive, VaR should be negative.
        """
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        # Use zero-mean returns to guarantee VaR < 0
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(100, 2))
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)
        # With zero-mean returns and 95% confidence, VaR should be negative
        assert result.var < 0.0

    def test_cvar_is_worse_than_var(self) -> None:
        """CVaR (expected shortfall) should be <= VaR (more extreme loss).

        CVaR is the expected loss given that the loss exceeds VaR,
        so it must represent a worse outcome (more negative).
        """
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=(200, 3))
        weights = np.array([0.4, 0.3, 0.3])

        result = model.compute(returns, weights)
        # CVaR should be more negative than VaR (worse loss)
        assert result.cvar <= result.var

    def test_sigma_in_reasonable_range(self) -> None:
        """Portfolio sigma from daily returns should be in reasonable range."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        # Typical daily equity returns: mean ~0, std ~2%
        returns = rng.normal(0.0, 0.02, size=(100, 3))
        weights = np.array([1 / 3, 1 / 3, 1 / 3])

        result = model.compute(returns, weights)
        # Portfolio sigma should be in the ballpark of individual asset sigmas
        # With diversification, should be less than max individual sigma
        assert 0.0 < result.sigma < 0.10  # Less than 10% daily

    def test_var_bounded_by_returns(self) -> None:
        """VaR should not be more extreme than the worst portfolio return."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(100, 2))
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)
        # Portfolio returns
        port_returns = returns @ weights
        worst_return = port_returns.min()
        # VaR at 95% should not be more extreme than the worst observed return
        # (parametric can exceed empirical, but for normal data it should be close)
        # Use a generous bound: worst return minus 2 sigma
        assert result.var > worst_return - 0.1


class TestEWMACovarianceComputation:
    """Test the EWMA covariance matrix computation directly."""

    def test_covariance_matrix_is_symmetric(self) -> None:
        """EWMA covariance matrix must be symmetric."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(100, 3))
        weights = np.array([0.4, 0.3, 0.3])

        result = model.compute(returns, weights)
        cov = result.covariance_matrix
        assert cov is not None
        np.testing.assert_array_almost_equal(cov, cov.T, decimal=12)

    def test_covariance_matrix_is_positive_semidefinite(self) -> None:
        """EWMA covariance matrix must be positive semi-definite."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(100, 3))
        weights = np.array([0.4, 0.3, 0.3])

        result = model.compute(returns, weights)
        cov = result.covariance_matrix
        assert cov is not None
        eigenvalues = np.linalg.eigvalsh(cov)
        # All eigenvalues should be >= 0 (or very close to 0 for numerical precision)
        assert np.all(eigenvalues >= -1e-10)

    def test_covariance_diagonal_matches_variances(self) -> None:
        """Diagonal of covariance matrix should approximate individual asset variances."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        # Single asset: covariance should equal variance
        returns = rng.normal(0.0, 0.02, size=(200, 1))
        weights = np.array([1.0])

        result = model.compute(returns, weights)
        cov = result.covariance_matrix
        assert cov is not None
        # Portfolio sigma should match sqrt of the single variance
        assert result.sigma == pytest.approx(np.sqrt(cov[0, 0]), rel=1e-6)

    def test_covariance_shape_matches_assets(self) -> None:
        """Covariance matrix shape must be (n_assets, n_assets)."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        n_assets = 5
        returns = rng.normal(0.0, 0.02, size=(100, n_assets))
        weights = np.ones(n_assets) / n_assets

        result = model.compute(returns, weights)
        cov = result.covariance_matrix
        assert cov is not None
        assert cov.shape == (n_assets, n_assets)


class TestEWMAWeightDecay:
    """Test that EWMA weights decay correctly."""

    def test_higher_decay_means_more_weight_on_recent(self) -> None:
        """Higher decay lambda should give more weight to recent observations.

        With a regime change (low vol then high vol), a higher decay should
        estimate higher sigma since it weights the recent high-vol period more.
        """
        from liq.risk.var_model import EWMARiskModel

        # First 80 observations: low vol, last 20: high vol
        rng = np.random.default_rng(42)
        low_vol = rng.normal(0.0, 0.005, size=(80, 2))
        high_vol = rng.normal(0.0, 0.05, size=(20, 2))
        returns = np.vstack([low_vol, high_vol])
        weights = np.array([0.5, 0.5])

        model_high_decay = EWMARiskModel(decay=0.98, alpha=0.95)
        model_low_decay = EWMARiskModel(decay=0.80, alpha=0.95)

        result_high_decay = model_high_decay.compute(returns, weights)
        result_low_decay = model_low_decay.compute(returns, weights)

        # Lower decay => more weight on recent high-vol data => higher sigma
        assert result_low_decay.sigma > result_high_decay.sigma

    def test_decay_affects_var(self) -> None:
        """Different decay parameters should produce different VaR values."""
        from liq.risk.var_model import EWMARiskModel

        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(100, 2))
        weights = np.array([0.5, 0.5])

        model_a = EWMARiskModel(decay=0.90, alpha=0.95)
        model_b = EWMARiskModel(decay=0.99, alpha=0.95)

        result_a = model_a.compute(returns, weights)
        result_b = model_b.compute(returns, weights)

        # Different decays should produce different results
        assert result_a.var != pytest.approx(result_b.var, abs=1e-10)


class TestParametricVaRFormula:
    """Test the parametric VaR/CVaR formulas directly."""

    def test_var_formula_known_values(self) -> None:
        """VaR formula: mu - z_alpha * sigma, verified with known z-scores."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        # Create returns with known properties:
        # Constant returns to get zero variance -> known result
        # Instead, use a well-understood case
        rng = np.random.default_rng(42)
        n = 10000
        returns = rng.normal(0.001, 0.02, size=(n, 1))
        weights = np.array([1.0])

        result = model.compute(returns, weights)

        # For single asset, sigma_p = sigma_asset
        # VaR = mu - z_0.95 * sigma
        z_95 = norm.ppf(0.95)  # ~1.6449
        expected_var = result.sigma * (-z_95)  # mu is approximately 0 for EWMA-weighted
        # The VaR should be close to -z * sigma (with mu close to 0)
        # Allow some tolerance for the mean term
        assert result.var == pytest.approx(expected_var, abs=result.sigma * 0.5)

    def test_cvar_formula_known_values(self) -> None:
        """CVaR formula: mu - sigma * phi(z_alpha) / (1 - alpha)."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(10000, 1))
        weights = np.array([1.0])

        result = model.compute(returns, weights)

        # CVaR = mu - sigma * phi(z_alpha) / (1 - alpha)
        z_95 = norm.ppf(0.95)
        phi_z = norm.pdf(z_95)
        # With zero-mean, CVaR ~ -sigma * phi(z) / (1 - alpha)
        expected_cvar = -result.sigma * phi_z / (1 - 0.95)
        assert result.cvar == pytest.approx(expected_cvar, abs=result.sigma * 0.5)

    def test_alpha_99_more_extreme_than_95(self) -> None:
        """VaR at 99% confidence should be more extreme than at 95%."""
        from liq.risk.var_model import EWMARiskModel

        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(200, 2))
        weights = np.array([0.5, 0.5])

        model_95 = EWMARiskModel(decay=0.94, alpha=0.95)
        model_99 = EWMARiskModel(decay=0.94, alpha=0.99)

        result_95 = model_95.compute(returns, weights)
        result_99 = model_99.compute(returns, weights)

        # 99% VaR should be more negative than 95% VaR
        assert result_99.var < result_95.var


class TestEdgeCasesEmptyWindow:
    """Test handling of empty or too-short return windows."""

    def test_empty_returns_array(self) -> None:
        """Empty returns should return safe fallback."""
        from liq.risk.var_model import EWMARiskModel, RiskModelOutput

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        returns = np.empty((0, 2))
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)

        assert isinstance(result, RiskModelOutput)
        assert result.sigma == 0.0
        assert result.var == 0.0
        assert result.cvar == 0.0

    def test_single_observation(self) -> None:
        """Single observation (T=1) should return safe fallback (insufficient data)."""
        from liq.risk.var_model import EWMARiskModel, RiskModelOutput

        model = EWMARiskModel(decay=0.94, alpha=0.95, min_periods=2)
        returns = np.array([[0.01, -0.02]])
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)

        assert isinstance(result, RiskModelOutput)
        assert result.sigma == 0.0
        assert result.var == 0.0
        assert result.cvar == 0.0

    def test_insufficient_data_below_min_periods(self) -> None:
        """Returns shorter than min_periods should return safe fallback."""
        from liq.risk.var_model import EWMARiskModel, RiskModelOutput

        model = EWMARiskModel(decay=0.94, alpha=0.95, min_periods=20)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(10, 2))  # Only 10 < 20
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)

        assert isinstance(result, RiskModelOutput)
        assert result.sigma == 0.0
        assert result.var == 0.0
        assert result.cvar == 0.0

    def test_exactly_min_periods_works(self) -> None:
        """Returns exactly equal to min_periods should produce valid output."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95, min_periods=10)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(10, 2))
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)

        assert result.sigma > 0.0


class TestEdgeCasesSingleAsset:
    """Test handling of single-asset portfolios."""

    def test_single_asset_portfolio(self) -> None:
        """Single asset: covariance is just variance."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=(100, 1))
        weights = np.array([1.0])

        result = model.compute(returns, weights)

        assert result.sigma > 0.0
        assert result.var < 0.0
        assert result.cvar < result.var

    def test_single_asset_sigma_approximates_std(self) -> None:
        """For single asset with weight=1, sigma should approximate asset volatility."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        # Generate returns with known std
        target_std = 0.02
        returns = rng.normal(0.0, target_std, size=(1000, 1))
        weights = np.array([1.0])

        result = model.compute(returns, weights)

        # EWMA sigma should be in the ballpark of the true std
        assert result.sigma == pytest.approx(target_std, rel=0.3)

    def test_single_asset_1d_returns(self) -> None:
        """1D returns array should be handled (reshaped internally)."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns_1d = rng.normal(0.0, 0.02, size=100)
        weights = np.array([1.0])

        result = model.compute(returns_1d, weights)

        assert result.sigma > 0.0


class TestEdgeCasesExtremeValues:
    """Test handling of extreme input values."""

    def test_zero_returns(self) -> None:
        """All-zero returns should produce zero sigma and zero VaR/CVaR."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        returns = np.zeros((100, 2))
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)

        assert result.sigma == 0.0
        assert result.var == 0.0
        assert result.cvar == 0.0

    def test_constant_returns(self) -> None:
        """Constant non-zero returns should produce zero sigma."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        returns = np.full((100, 2), 0.01)
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)

        assert result.sigma == pytest.approx(0.0, abs=1e-10)

    def test_very_large_returns(self) -> None:
        """Very large returns should not cause overflow, NaN, or Inf."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 1.0, size=(100, 2))  # 100% daily std!
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)

        assert np.isfinite(result.sigma)
        assert np.isfinite(result.var)
        assert np.isfinite(result.cvar)

    def test_very_small_returns(self) -> None:
        """Very small returns should produce small but valid results."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 1e-8, size=(100, 2))
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)

        assert np.isfinite(result.sigma)
        assert result.sigma >= 0.0
        assert np.isfinite(result.var)
        assert np.isfinite(result.cvar)

    def test_nan_in_returns_handled(self) -> None:
        """NaN values in returns should not crash; model should return fallback."""
        from liq.risk.var_model import EWMARiskModel, RiskModelOutput

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        returns = np.array([[0.01, 0.02], [np.nan, 0.01], [0.03, -0.01]])
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)

        assert isinstance(result, RiskModelOutput)
        assert np.isfinite(result.sigma)
        assert np.isfinite(result.var)
        assert np.isfinite(result.cvar)

    def test_inf_in_returns_handled(self) -> None:
        """Inf values in returns should not crash; model should return fallback."""
        from liq.risk.var_model import EWMARiskModel, RiskModelOutput

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        returns = np.array([[0.01, 0.02], [np.inf, 0.01], [0.03, -0.01]])
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)

        assert isinstance(result, RiskModelOutput)
        assert np.isfinite(result.sigma)
        assert np.isfinite(result.var)
        assert np.isfinite(result.cvar)


class TestEdgeCasesWeights:
    """Test handling of edge cases in portfolio weights."""

    def test_zero_weights(self) -> None:
        """All-zero weights should produce zero outputs."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(100, 2))
        weights = np.array([0.0, 0.0])

        result = model.compute(returns, weights)

        assert result.sigma == 0.0
        assert result.var == 0.0
        assert result.cvar == 0.0

    def test_concentrated_weight(self) -> None:
        """100% weight in one asset should match single-asset result."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(100, 3))

        # Weight all in first asset
        weights_concentrated = np.array([1.0, 0.0, 0.0])
        result_concentrated = model.compute(returns, weights_concentrated)

        # Single asset result (just first column)
        weights_single = np.array([1.0])
        result_single = model.compute(returns[:, :1], weights_single)

        assert result_concentrated.sigma == pytest.approx(result_single.sigma, rel=1e-10)
        assert result_concentrated.var == pytest.approx(result_single.var, rel=1e-10)

    def test_negative_weights_short_positions(self) -> None:
        """Negative weights (short positions) should work correctly."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(100, 2))
        # Long first asset, short second asset (dollar-neutral)
        weights = np.array([0.5, -0.5])

        result = model.compute(returns, weights)

        assert result.sigma > 0.0
        assert np.isfinite(result.var)
        assert np.isfinite(result.cvar)

    def test_weights_not_summing_to_one(self) -> None:
        """Weights not summing to 1 should still produce valid results.

        Weights represent portfolio dollar allocations, not necessarily
        normalized probabilities. A leveraged portfolio might sum to > 1.
        """
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(100, 2))
        # Leveraged portfolio
        weights = np.array([0.8, 0.8])

        result = model.compute(returns, weights)

        assert result.sigma > 0.0
        assert np.isfinite(result.var)
        assert np.isfinite(result.cvar)


class TestEWMAMultiAssetPortfolio:
    """Test multi-asset portfolio computations."""

    def test_diversification_reduces_risk(self) -> None:
        """An equally-weighted portfolio of uncorrelated assets should have lower
        sigma than the average individual asset sigma."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        n_assets = 5
        rng = np.random.default_rng(42)
        # Generate uncorrelated returns
        returns = rng.normal(0.0, 0.02, size=(200, n_assets))
        weights = np.ones(n_assets) / n_assets

        result = model.compute(returns, weights)

        # Compute individual sigmas
        individual_sigmas = []
        for i in range(n_assets):
            single_result = model.compute(returns[:, i : i + 1], np.array([1.0]))
            individual_sigmas.append(single_result.sigma)

        avg_individual_sigma = np.mean(individual_sigmas)
        # Portfolio sigma should be less than average individual sigma
        # (due to diversification of uncorrelated assets)
        assert result.sigma < avg_individual_sigma

    def test_perfectly_correlated_no_diversification(self) -> None:
        """Perfectly correlated assets should show no diversification benefit."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        base = rng.normal(0.0, 0.02, size=(100, 1))
        # All assets are perfectly correlated (identical returns)
        returns = np.hstack([base, base, base])
        weights = np.array([1 / 3, 1 / 3, 1 / 3])

        result_portfolio = model.compute(returns, weights)
        result_single = model.compute(base, np.array([1.0]))

        # Portfolio sigma should equal single asset sigma
        assert result_portfolio.sigma == pytest.approx(result_single.sigma, rel=1e-6)

    def test_many_assets_portfolio(self) -> None:
        """Should handle portfolios with many assets without issues."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        n_assets = 50
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(100, n_assets))
        weights = np.ones(n_assets) / n_assets

        result = model.compute(returns, weights)

        assert result.sigma > 0.0
        assert np.isfinite(result.var)
        assert np.isfinite(result.cvar)
        assert result.covariance_matrix is not None
        assert result.covariance_matrix.shape == (n_assets, n_assets)


class TestInputValidation:
    """Test input validation and shape mismatches."""

    def test_weights_length_mismatch_raises(self) -> None:
        """Weights length must match number of assets in returns."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(100, 3))
        weights = np.array([0.5, 0.5])  # 2 weights for 3 assets

        with pytest.raises(ValueError, match="weights"):
            model.compute(returns, weights)

    def test_decay_must_be_in_valid_range(self) -> None:
        """Decay lambda must be in (0, 1)."""
        from liq.risk.var_model import EWMARiskModel

        with pytest.raises(ValueError, match="decay"):
            EWMARiskModel(decay=0.0)

        with pytest.raises(ValueError, match="decay"):
            EWMARiskModel(decay=1.0)

        with pytest.raises(ValueError, match="decay"):
            EWMARiskModel(decay=1.5)

        with pytest.raises(ValueError, match="decay"):
            EWMARiskModel(decay=-0.1)

    def test_alpha_must_be_in_valid_range(self) -> None:
        """Confidence level alpha must be in (0, 1)."""
        from liq.risk.var_model import EWMARiskModel

        with pytest.raises(ValueError, match="alpha"):
            EWMARiskModel(alpha=0.0)

        with pytest.raises(ValueError, match="alpha"):
            EWMARiskModel(alpha=1.0)

    def test_min_periods_must_be_positive(self) -> None:
        """min_periods must be >= 1."""
        from liq.risk.var_model import EWMARiskModel

        with pytest.raises(ValueError, match="min_periods"):
            EWMARiskModel(min_periods=0)

        with pytest.raises(ValueError, match="min_periods"):
            EWMARiskModel(min_periods=-1)


class TestMDDPrediction:
    """Test predicted maximum drawdown computation."""

    def test_mdd_pred_is_non_negative(self) -> None:
        """Predicted MDD must be >= 0."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.001, 0.02, size=(100, 3))
        weights = np.array([0.4, 0.3, 0.3])

        result = model.compute(returns, weights)
        assert result.mdd_pred >= 0.0

    def test_mdd_pred_scales_with_sqrt_horizon(self) -> None:
        """MDD at horizon 63 / MDD at horizon 21 should approximate sqrt(3)."""
        from liq.risk.var_model import EWMARiskModel

        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(200, 2))
        weights = np.array([0.5, 0.5])

        model_21 = EWMARiskModel(decay=0.94, alpha=0.95, horizon=21)
        model_63 = EWMARiskModel(decay=0.94, alpha=0.95, horizon=63)

        result_21 = model_21.compute(returns, weights)
        result_63 = model_63.compute(returns, weights)

        ratio = result_63.mdd_pred / result_21.mdd_pred
        assert ratio == pytest.approx(np.sqrt(3), rel=1e-6)

    def test_mdd_pred_equals_sigma_times_sqrt_horizon(self) -> None:
        """mdd_pred should equal sigma * sqrt(horizon)."""
        from liq.risk.var_model import EWMARiskModel

        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(200, 2))
        weights = np.array([0.5, 0.5])

        model = EWMARiskModel(decay=0.94, alpha=0.95, horizon=21)
        result = model.compute(returns, weights)

        expected_mdd = result.sigma * np.sqrt(21)
        assert result.mdd_pred == pytest.approx(expected_mdd, rel=1e-10)

    def test_mdd_pred_zero_for_zero_sigma(self) -> None:
        """Zero-vol returns should produce mdd_pred=0.0."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95)
        returns = np.zeros((100, 2))
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)
        assert result.mdd_pred == 0.0

    def test_mdd_pred_zero_for_insufficient_data(self) -> None:
        """Insufficient data (under min_periods) should return mdd_pred=0.0."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(decay=0.94, alpha=0.95, min_periods=50)
        rng = np.random.default_rng(42)
        returns = rng.normal(0.0, 0.02, size=(10, 2))
        weights = np.array([0.5, 0.5])

        result = model.compute(returns, weights)
        assert result.mdd_pred == 0.0

    def test_default_horizon_is_21(self) -> None:
        """EWMARiskModel default horizon should be 21."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel()
        assert model.horizon == 21

    def test_custom_horizon(self) -> None:
        """EWMARiskModel should accept custom horizon."""
        from liq.risk.var_model import EWMARiskModel

        model = EWMARiskModel(horizon=63)
        assert model.horizon == 63

    def test_horizon_must_be_positive(self) -> None:
        """horizon=0 and negative values should raise ValueError."""
        from liq.risk.var_model import EWMARiskModel

        with pytest.raises(ValueError, match="horizon"):
            EWMARiskModel(horizon=0)

        with pytest.raises(ValueError, match="horizon"):
            EWMARiskModel(horizon=-5)

    def test_higher_vol_means_higher_mdd_pred(self) -> None:
        """Shifting weight to a high-vol asset should increase mdd_pred."""
        from liq.risk.var_model import EWMARiskModel

        rng = np.random.default_rng(42)
        returns = np.column_stack(
            [
                rng.normal(0.0, 0.01, size=200),  # Low vol
                rng.normal(0.0, 0.10, size=200),  # High vol
            ]
        )

        model = EWMARiskModel(decay=0.94, alpha=0.95)

        result_low = model.compute(returns, np.array([0.9, 0.1]))
        result_high = model.compute(returns, np.array([0.1, 0.9]))

        assert result_high.mdd_pred > result_low.mdd_pred
