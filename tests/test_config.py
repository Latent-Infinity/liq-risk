"""Tests for liq-risk configuration types.

Following TDD: RED phase - write failing tests first.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from liq.core import Bar
from pydantic import ValidationError


class TestRiskConfig:
    """Tests for RiskConfig dataclass."""

    def test_default_construction(self) -> None:
        """RiskConfig can be constructed with no arguments (zero-config)."""
        from liq.risk import RiskConfig

        config = RiskConfig()

        # Verify sensible defaults exist
        assert config.max_position_pct > 0
        assert config.max_position_pct <= 1.0
        assert config.max_positions > 0
        assert config.risk_per_trade > 0

    def test_default_values(self) -> None:
        """RiskConfig has sensible default values."""
        from liq.risk import RiskConfig

        config = RiskConfig()

        # Position limits
        assert config.max_position_pct == 0.05  # 5%
        assert config.max_positions == 50
        assert config.min_position_value == Decimal("100")

        # Exposure limits
        assert config.max_sector_pct == 0.30  # 30%
        assert config.max_gross_leverage == 1.0
        assert config.max_net_leverage == 1.0

        # Sizing parameters
        assert config.risk_per_trade == 0.01  # 1%
        assert config.kelly_fraction == 0.25  # 25%

        # Risk controls
        assert config.stop_loss_atr_mult == 2.0
        assert config.max_drawdown_halt == 0.15  # 15%

    def test_custom_values(self) -> None:
        """RiskConfig accepts custom values."""
        from liq.risk import RiskConfig

        config = RiskConfig(
            max_position_pct=0.10,
            max_positions=20,
            risk_per_trade=0.02,
            max_drawdown_halt=0.20,
        )

        assert config.max_position_pct == 0.10
        assert config.max_positions == 20
        assert config.risk_per_trade == 0.02
        assert config.max_drawdown_halt == 0.20

    def test_optional_fields_default_none(self) -> None:
        """Optional fields default to None."""
        from liq.risk import RiskConfig

        config = RiskConfig()

        assert config.vol_target is None
        assert config.take_profit_atr_mult is None
        assert config.max_daily_loss_halt is None
        assert config.max_correlation is None

    def test_optional_fields_can_be_set(self) -> None:
        """Optional fields can be set to values."""
        from liq.risk import RiskConfig

        config = RiskConfig(
            vol_target=0.15,
            take_profit_atr_mult=3.0,
            max_daily_loss_halt=0.05,
            max_correlation=0.70,
        )

        assert config.vol_target == 0.15
        assert config.take_profit_atr_mult == 3.0
        assert config.max_daily_loss_halt == 0.05
        assert config.max_correlation == 0.70

    def test_immutable(self) -> None:
        """RiskConfig should be frozen/immutable."""
        from liq.risk import RiskConfig

        config = RiskConfig()

        with pytest.raises((TypeError, AttributeError, ValidationError)):
            config.max_position_pct = 0.50  # type: ignore[misc]


class TestRiskConfigValidation:
    """Tests for RiskConfig validation."""

    def test_max_position_pct_must_be_positive(self) -> None:
        """max_position_pct must be > 0."""
        from liq.risk import RiskConfig

        with pytest.raises(ValueError, match="max_position_pct"):
            RiskConfig(max_position_pct=0.0)

        with pytest.raises(ValueError, match="max_position_pct"):
            RiskConfig(max_position_pct=-0.1)

    def test_max_position_pct_must_be_at_most_one(self) -> None:
        """max_position_pct must be <= 1.0."""
        from liq.risk import RiskConfig

        with pytest.raises(ValueError, match="max_position_pct"):
            RiskConfig(max_position_pct=1.5)

    def test_max_positions_must_be_positive(self) -> None:
        """max_positions must be > 0."""
        from liq.risk import RiskConfig

        with pytest.raises(ValueError, match="max_positions"):
            RiskConfig(max_positions=0)

        with pytest.raises(ValueError, match="max_positions"):
            RiskConfig(max_positions=-5)

    def test_risk_per_trade_must_be_positive(self) -> None:
        """risk_per_trade must be > 0."""
        from liq.risk import RiskConfig

        with pytest.raises(ValueError, match="risk_per_trade"):
            RiskConfig(risk_per_trade=0.0)

    def test_risk_per_trade_must_be_at_most_one(self) -> None:
        """risk_per_trade must be <= 1.0."""
        from liq.risk import RiskConfig

        with pytest.raises(ValueError, match="risk_per_trade"):
            RiskConfig(risk_per_trade=1.5)

    def test_max_drawdown_halt_must_be_positive(self) -> None:
        """max_drawdown_halt must be > 0."""
        from liq.risk import RiskConfig

        with pytest.raises(ValueError, match="max_drawdown_halt"):
            RiskConfig(max_drawdown_halt=0.0)

    def test_max_drawdown_halt_must_be_at_most_one(self) -> None:
        """max_drawdown_halt must be <= 1.0."""
        from liq.risk import RiskConfig

        with pytest.raises(ValueError, match="max_drawdown_halt"):
            RiskConfig(max_drawdown_halt=1.5)

    def test_kelly_fraction_must_be_in_valid_range(self) -> None:
        """kelly_fraction must be in (0, 1]."""
        from liq.risk import RiskConfig

        with pytest.raises(ValueError, match="kelly_fraction"):
            RiskConfig(kelly_fraction=0.0)

        with pytest.raises(ValueError, match="kelly_fraction"):
            RiskConfig(kelly_fraction=1.5)

        # Edge case: 1.0 is valid (full Kelly, though aggressive)
        config = RiskConfig(kelly_fraction=1.0)
        assert config.kelly_fraction == 1.0


class TestMarketState:
    """Tests for MarketState dataclass."""

    def test_construction_with_required_fields(self) -> None:
        """MarketState can be constructed with required fields."""
        from liq.risk import MarketState

        now = datetime.now(UTC)
        state = MarketState(
            current_bars={},
            volatility={},
            liquidity={},
            timestamp=now,
        )

        assert state.current_bars == {}
        assert state.volatility == {}
        assert state.liquidity == {}
        assert state.timestamp == now

    def test_optional_fields_default_none(self) -> None:
        """Optional fields default to None."""
        from liq.risk import MarketState

        now = datetime.now(UTC)
        state = MarketState(
            current_bars={},
            volatility={},
            liquidity={},
            timestamp=now,
        )

        assert state.sector_map is None
        assert state.correlations is None
        assert state.regime is None

    def test_with_bar_data(self) -> None:
        """MarketState with actual bar data."""
        from liq.risk import MarketState

        now = datetime.now(UTC)
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("150.00"),
            high=Decimal("152.00"),
            low=Decimal("149.00"),
            close=Decimal("151.50"),
            volume=Decimal("1000000"),
        )

        state = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.50")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )

        assert "AAPL" in state.current_bars
        assert state.current_bars["AAPL"].close == Decimal("151.50")
        assert state.volatility["AAPL"] == Decimal("2.50")

    def test_with_sector_map(self) -> None:
        """MarketState with sector mapping."""
        from liq.risk import MarketState

        now = datetime.now(UTC)
        state = MarketState(
            current_bars={},
            volatility={},
            liquidity={},
            sector_map={"AAPL": "Technology", "JPM": "Financials"},
            timestamp=now,
        )

        assert state.sector_map is not None
        assert state.sector_map["AAPL"] == "Technology"
        assert state.sector_map["JPM"] == "Financials"

    def test_with_regime(self) -> None:
        """MarketState with market regime label."""
        from liq.risk import MarketState

        now = datetime.now(UTC)
        state = MarketState(
            current_bars={},
            volatility={},
            liquidity={},
            regime="high_volatility",
            timestamp=now,
        )

        assert state.regime == "high_volatility"

    def test_timestamp_must_be_timezone_aware(self) -> None:
        """MarketState timestamp must be timezone-aware."""
        from liq.risk import MarketState

        naive_time = datetime(2024, 1, 1, 12, 0, 0)  # No timezone

        with pytest.raises(ValueError, match="timezone"):
            MarketState(
                current_bars={},
                volatility={},
                liquidity={},
                timestamp=naive_time,
            )

    def test_immutable(self) -> None:
        """MarketState should be frozen/immutable."""
        from liq.risk import MarketState

        now = datetime.now(UTC)
        state = MarketState(
            current_bars={},
            volatility={},
            liquidity={},
            timestamp=now,
        )

        with pytest.raises((TypeError, AttributeError, ValidationError)):
            state.regime = "changed"  # type: ignore[misc]
