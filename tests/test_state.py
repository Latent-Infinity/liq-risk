"""Tests for liq-risk state types.

Tests PriceState, RiskFactors, AssetMetadata, and ExecutionState.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from liq.core import OrderRequest, OrderSide, OrderType
from liq.core.bar import Bar

from liq.risk.enums import PriceReference


class TestPriceState:
    """Tests for PriceState dataclass."""

    def test_creation_basic(self):
        """Test basic PriceState creation."""
        from liq.risk.state import PriceState

        now = datetime.now(UTC)
        bar = Bar(
            symbol="AAPL",
            timestamp=now,
            open=Decimal("150.00"),
            high=Decimal("155.00"),
            low=Decimal("148.00"),
            close=Decimal("152.00"),
            volume=Decimal("1000000"),
        )

        state = PriceState(
            current_bars={"AAPL": bar},
            timestamp=now,
        )

        assert state.timestamp == now
        assert "AAPL" in state.current_bars
        assert state.current_bars["AAPL"].close == Decimal("152.00")

    def test_get_price_midrange(self):
        """Test get_price with MIDRANGE reference."""
        from liq.risk.state import PriceState

        now = datetime.now(UTC)
        bar = Bar(
            symbol="AAPL",
            timestamp=now,
            open=Decimal("150.00"),
            high=Decimal("160.00"),
            low=Decimal("140.00"),
            close=Decimal("155.00"),
            volume=Decimal("1000000"),
        )

        state = PriceState(current_bars={"AAPL": bar}, timestamp=now)

        # Midrange = (160 + 140) / 2 = 150
        price = state.get_price("AAPL", PriceReference.MIDRANGE)
        assert price == Decimal("150.00")

    def test_get_price_close(self):
        """Test get_price with CLOSE reference."""
        from liq.risk.state import PriceState

        now = datetime.now(UTC)
        bar = Bar(
            symbol="AAPL",
            timestamp=now,
            open=Decimal("150.00"),
            high=Decimal("160.00"),
            low=Decimal("140.00"),
            close=Decimal("155.00"),
            volume=Decimal("1000000"),
        )

        state = PriceState(current_bars={"AAPL": bar}, timestamp=now)

        price = state.get_price("AAPL", PriceReference.CLOSE)
        assert price == Decimal("155.00")

    def test_get_price_vwap_fallback(self):
        """Test get_price with VWAP falls back to close."""
        from liq.risk.state import PriceState

        now = datetime.now(UTC)
        bar = Bar(
            symbol="AAPL",
            timestamp=now,
            open=Decimal("150.00"),
            high=Decimal("160.00"),
            low=Decimal("140.00"),
            close=Decimal("155.00"),
            volume=Decimal("1000000"),
        )

        state = PriceState(current_bars={"AAPL": bar}, timestamp=now)

        # VWAP not available, should fall back to close
        price = state.get_price("AAPL", PriceReference.VWAP)
        assert price == Decimal("155.00")

    def test_get_price_missing_symbol(self):
        """Test get_price returns None for missing symbol."""
        from liq.risk.state import PriceState

        now = datetime.now(UTC)
        state = PriceState(current_bars={}, timestamp=now)

        price = state.get_price("AAPL", PriceReference.CLOSE)
        assert price is None

    def test_frozen_immutability(self):
        """Test that PriceState is immutable."""
        from liq.risk.state import PriceState

        now = datetime.now(UTC)
        state = PriceState(current_bars={}, timestamp=now)

        with pytest.raises(AttributeError):
            state.timestamp = datetime.now(UTC)  # type: ignore


class TestRiskFactors:
    """Tests for RiskFactors dataclass."""

    def test_creation_basic(self):
        """Test basic RiskFactors creation."""
        from liq.risk.state import RiskFactors

        factors = RiskFactors(
            volatility={"AAPL": 2.5, "GOOGL": 3.2},
        )

        assert factors.volatility["AAPL"] == 2.5
        assert factors.volatility["GOOGL"] == 3.2
        assert factors.correlations is None
        assert factors.regime is None

    def test_creation_with_all_fields(self):
        """Test RiskFactors with all optional fields."""
        from liq.risk.state import RiskFactors

        factors = RiskFactors(
            volatility={"AAPL": 2.5},
            correlations=None,  # Would be polars DataFrame in real use
            regime="high_volatility",
        )

        assert factors.regime == "high_volatility"

    def test_frozen_immutability(self):
        """Test that RiskFactors is immutable."""
        from liq.risk.state import RiskFactors

        factors = RiskFactors(volatility={})

        with pytest.raises(AttributeError):
            factors.regime = "low_vol"  # type: ignore


class TestAssetMetadata:
    """Tests for AssetMetadata dataclass."""

    def test_creation_basic(self):
        """Test basic AssetMetadata creation."""
        from liq.risk.state import AssetMetadata

        metadata = AssetMetadata()

        assert metadata.sector_map is None
        assert metadata.group_map is None
        assert metadata.liquidity is None
        assert metadata.borrow_rates is None

    def test_creation_with_sector_map(self):
        """Test AssetMetadata with sector mapping."""
        from liq.risk.state import AssetMetadata

        metadata = AssetMetadata(
            sector_map={"AAPL": "Technology", "JPM": "Financials"},
        )

        assert metadata.sector_map is not None
        assert metadata.sector_map["AAPL"] == "Technology"
        assert metadata.sector_map["JPM"] == "Financials"

    def test_creation_with_group_map(self):
        """Test AssetMetadata with generic group mapping."""
        from liq.risk.state import AssetMetadata

        metadata = AssetMetadata(
            group_map={
                "AAPL": {"sector": "Technology", "country": "US", "size": "large"},
                "TSM": {"sector": "Technology", "country": "TW", "size": "large"},
            },
        )

        assert metadata.group_map is not None
        assert metadata.group_map["AAPL"]["country"] == "US"
        assert metadata.group_map["TSM"]["country"] == "TW"

    def test_creation_with_liquidity(self):
        """Test AssetMetadata with liquidity data."""
        from liq.risk.state import AssetMetadata

        metadata = AssetMetadata(
            liquidity={"AAPL": 50_000_000.0, "GOOGL": 20_000_000.0},
        )

        assert metadata.liquidity is not None
        assert metadata.liquidity["AAPL"] == 50_000_000.0

    def test_creation_with_borrow_rates(self):
        """Test AssetMetadata with borrow rates."""
        from liq.risk.state import AssetMetadata

        metadata = AssetMetadata(
            borrow_rates={"GME": 0.15, "AMC": 0.20},  # 15% and 20% annual
        )

        assert metadata.borrow_rates is not None
        assert metadata.borrow_rates["GME"] == 0.15

    def test_frozen_immutability(self):
        """Test that AssetMetadata is immutable."""
        from liq.risk.state import AssetMetadata

        metadata = AssetMetadata()

        with pytest.raises(AttributeError):
            metadata.sector_map = {}  # type: ignore


class TestExecutionState:
    """Tests for ExecutionState dataclass."""

    def test_creation_basic(self):
        """Test basic ExecutionState creation."""
        from liq.risk.state import ExecutionState

        state = ExecutionState(
            open_orders=[],
            reserved_capital=Decimal("0"),
        )

        assert state.open_orders == []
        assert state.reserved_capital == Decimal("0")

    def test_creation_with_orders(self):
        """Test ExecutionState with open orders."""
        from liq.risk.state import ExecutionState

        now = datetime.now(UTC)
        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
            timestamp=now,
        )

        state = ExecutionState(
            open_orders=[order],
            reserved_capital=Decimal("15000"),  # 100 * $150
        )

        assert len(state.open_orders) == 1
        assert state.reserved_capital == Decimal("15000")

    def test_reserved_by_symbol_empty(self):
        """Test reserved_by_symbol with no orders."""
        from liq.risk.state import ExecutionState

        state = ExecutionState(open_orders=[], reserved_capital=Decimal("0"))

        reserved = state.reserved_by_symbol
        assert reserved == {}

    def test_reserved_by_symbol_single_order(self):
        """Test reserved_by_symbol with one order."""
        from liq.risk.state import ExecutionState

        now = datetime.now(UTC)
        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
            timestamp=now,
        )

        state = ExecutionState(
            open_orders=[order],
            reserved_capital=Decimal("15000"),
        )

        reserved = state.reserved_by_symbol
        assert "AAPL" in reserved
        assert reserved["AAPL"] == Decimal("15000")

    def test_reserved_by_symbol_multiple_orders(self):
        """Test reserved_by_symbol with multiple orders."""
        from liq.risk.state import ExecutionState

        now = datetime.now(UTC)
        order1 = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
            timestamp=now,
        )
        order2 = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("50"),
            limit_price=Decimal("148.00"),
            timestamp=now,
        )
        order3 = OrderRequest(
            symbol="GOOGL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("10"),
            limit_price=Decimal("2800.00"),
            timestamp=now,
        )

        state = ExecutionState(
            open_orders=[order1, order2, order3],
            reserved_capital=Decimal("50400"),  # 15000 + 7400 + 28000
        )

        reserved = state.reserved_by_symbol
        assert reserved["AAPL"] == Decimal("22400")  # 15000 + 7400
        assert reserved["GOOGL"] == Decimal("28000")

    def test_reserved_by_symbol_sell_orders_excluded(self):
        """Test that sell orders don't reserve capital."""
        from liq.risk.state import ExecutionState

        now = datetime.now(UTC)
        buy_order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.LIMIT,
            quantity=Decimal("100"),
            limit_price=Decimal("150.00"),
            timestamp=now,
        )
        sell_order = OrderRequest(
            symbol="GOOGL",
            side=OrderSide.SELL,
            order_type=OrderType.LIMIT,
            quantity=Decimal("50"),
            limit_price=Decimal("2800.00"),
            timestamp=now,
        )

        state = ExecutionState(
            open_orders=[buy_order, sell_order],
            reserved_capital=Decimal("15000"),  # Only buy order
        )

        reserved = state.reserved_by_symbol
        assert reserved["AAPL"] == Decimal("15000")
        assert "GOOGL" not in reserved  # Sell doesn't reserve


class TestEnums:
    """Tests for risk enums."""

    def test_halt_mode_values(self):
        """Test HaltMode enum values."""
        from liq.risk.enums import HaltMode

        assert HaltMode.HALT_BUYS_ONLY.value == "halt_buys_only"
        assert HaltMode.HALT_ALL_RISK_INCREASING.value == "halt_risk_inc"
        assert HaltMode.HALT_ALL_TRADES.value == "halt_all"

    def test_sizing_mode_values(self):
        """Test SizingMode enum values."""
        from liq.risk.enums import SizingMode

        assert SizingMode.INCREMENTAL.value == "incremental"
        assert SizingMode.REBALANCE.value == "rebalance"
        assert SizingMode.REPLACE.value == "replace"

    def test_price_reference_values(self):
        """Test PriceReference enum values."""
        from liq.risk.enums import PriceReference

        assert PriceReference.MIDRANGE.value == "midrange"
        assert PriceReference.CLOSE.value == "close"
        assert PriceReference.VWAP.value == "vwap"
