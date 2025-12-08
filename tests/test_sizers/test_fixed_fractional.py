"""Tests for FixedFractionalSizer.

Following TDD: RED phase - write failing tests first.

FixedFractionalSizer Formula:
    quantity = (equity * fraction) / price

Simple sizing: allocate X% of equity to each signal.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from liq.core import Bar, OrderSide, OrderType, PortfolioState
from liq.signals import Signal

from liq.risk import MarketState, RiskConfig
from liq.risk.protocols import PositionSizer


class TestFixedFractionalSizerProtocol:
    """Test that FixedFractionalSizer conforms to PositionSizer protocol."""

    def test_conforms_to_protocol(self) -> None:
        """FixedFractionalSizer must implement PositionSizer protocol."""
        from liq.risk.sizers import FixedFractionalSizer

        sizer = FixedFractionalSizer()
        assert isinstance(sizer, PositionSizer)


class TestFixedFractionalSizerBasic:
    """Basic functionality tests for FixedFractionalSizer."""

    def test_default_fraction(self) -> None:
        """Default fraction should be 0.02 (2%)."""
        from liq.risk.sizers import FixedFractionalSizer

        sizer = FixedFractionalSizer()
        assert sizer.fraction == 0.02

    def test_custom_fraction(self) -> None:
        """Can set custom fraction."""
        from liq.risk.sizers import FixedFractionalSizer

        sizer = FixedFractionalSizer(fraction=0.05)
        assert sizer.fraction == 0.05

    def test_empty_signals_returns_empty_orders(self) -> None:
        """No signals should produce no orders."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        sizer = FixedFractionalSizer()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        market = MarketState(
            current_bars={},
            volatility={},
            liquidity={},
            timestamp=now,
        )

        orders = sizer.size_positions([], portfolio, market, config)

        assert orders == []

    def test_flat_signals_produce_no_orders(self) -> None:
        """Flat signals should not produce orders."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        sizer = FixedFractionalSizer()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("150"),
            high=Decimal("152"),
            low=Decimal("149"),
            close=Decimal("151"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.50")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="flat", strength=0.5)]

        orders = sizer.size_positions(signals, portfolio, market, config)

        assert orders == []

    def test_long_signal_produces_buy_order(self) -> None:
        """Long signal should produce a BUY order."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        sizer = FixedFractionalSizer(fraction=0.02)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("150"),
            high=Decimal("152"),
            low=Decimal("148"),
            close=Decimal("150"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.8)]

        orders = sizer.size_positions(signals, portfolio, market, config)

        assert len(orders) == 1
        assert orders[0].symbol == "AAPL"
        assert orders[0].side == OrderSide.BUY
        assert orders[0].order_type == OrderType.MARKET

    def test_short_signal_produces_sell_order(self) -> None:
        """Short signal should produce a SELL order."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        sizer = FixedFractionalSizer(fraction=0.02)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("150"),
            high=Decimal("152"),
            low=Decimal("148"),
            close=Decimal("150"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="short", strength=0.8)]

        orders = sizer.size_positions(signals, portfolio, market, config)

        assert len(orders) == 1
        assert orders[0].symbol == "AAPL"
        assert orders[0].side == OrderSide.SELL


class TestFixedFractionalSizerFormula:
    """Tests for the fixed fractional sizing formula."""

    def test_sizing_formula_calculation(self) -> None:
        """Verify the fixed fractional formula: qty = (equity * fraction) / price."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        # Formula: qty = (100000 * 0.02) / 100 = 2000 / 100 = 20 shares
        sizer = FixedFractionalSizer(fraction=0.02)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),  # $100k equity
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),  # close = 100
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.50")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)]

        orders = sizer.size_positions(signals, portfolio, market, config)

        assert len(orders) == 1
        # qty = (100000 * 0.02) / 100 = 20
        assert orders[0].quantity == Decimal("20")

    def test_higher_fraction_larger_position(self) -> None:
        """Higher fraction should result in larger position size."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)]

        sizer_small = FixedFractionalSizer(fraction=0.01)
        sizer_large = FixedFractionalSizer(fraction=0.05)

        orders_small = sizer_small.size_positions(signals, portfolio, market, config)
        orders_large = sizer_large.size_positions(signals, portfolio, market, config)

        assert orders_large[0].quantity > orders_small[0].quantity

    def test_higher_equity_larger_position(self) -> None:
        """Higher equity should result in larger position size."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        sizer = FixedFractionalSizer(fraction=0.02)
        config = RiskConfig()

        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)]

        portfolio_small = PortfolioState(
            cash=Decimal("50000"),
            positions={},
            timestamp=now,
        )
        portfolio_large = PortfolioState(
            cash=Decimal("200000"),
            positions={},
            timestamp=now,
        )

        orders_small = sizer.size_positions(signals, portfolio_small, market, config)
        orders_large = sizer.size_positions(signals, portfolio_large, market, config)

        assert orders_large[0].quantity > orders_small[0].quantity

    def test_higher_price_smaller_position(self) -> None:
        """Higher price should result in smaller position size (fewer shares)."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        sizer = FixedFractionalSizer(fraction=0.02)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )

        bar_cheap = Bar(
            timestamp=now,
            symbol="CHEAP",
            open=Decimal("50"),
            high=Decimal("52"),
            low=Decimal("48"),
            close=Decimal("50"),
            volume=Decimal("1000000"),
        )
        bar_expensive = Bar(
            timestamp=now,
            symbol="EXPENSIVE",
            open=Decimal("500"),
            high=Decimal("510"),
            low=Decimal("490"),
            close=Decimal("500"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"CHEAP": bar_cheap, "EXPENSIVE": bar_expensive},
            volatility={"CHEAP": Decimal("2.00"), "EXPENSIVE": Decimal("20.00")},
            liquidity={"CHEAP": Decimal("50000000"), "EXPENSIVE": Decimal("50000000")},
            timestamp=now,
        )

        signals_cheap = [Signal(symbol="CHEAP", timestamp=now, direction="long", strength=1.0)]
        signals_expensive = [Signal(symbol="EXPENSIVE", timestamp=now, direction="long", strength=1.0)]

        orders_cheap = sizer.size_positions(signals_cheap, portfolio, market, config)
        orders_expensive = sizer.size_positions(signals_expensive, portfolio, market, config)

        assert orders_cheap[0].quantity > orders_expensive[0].quantity

    def test_uses_close_price(self) -> None:
        """Should use close price for sizing."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        sizer = FixedFractionalSizer(fraction=0.02)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        # Bar with close = 50
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("45"),
            high=Decimal("60"),
            low=Decimal("40"),
            close=Decimal("50"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.50")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)]

        orders = sizer.size_positions(signals, portfolio, market, config)

        # qty = (100000 * 0.02) / 50 = 2000 / 50 = 40
        assert orders[0].quantity == Decimal("40")


class TestFixedFractionalSizerEdgeCases:
    """Edge case tests for FixedFractionalSizer."""

    def test_missing_bar_data_skips_signal(self) -> None:
        """Signal for symbol without bar data should be skipped."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        sizer = FixedFractionalSizer()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        market = MarketState(
            current_bars={},  # No bar for AAPL
            volatility={"AAPL": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.8)]

        orders = sizer.size_positions(signals, portfolio, market, config)

        assert orders == []

    def test_quantity_rounded_down_to_whole_shares(self) -> None:
        """Quantity should be rounded down to whole shares."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        sizer = FixedFractionalSizer(fraction=0.02)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        # Setup to get fractional result: (100000 * 0.02) / 33 = 2000 / 33 = 60.606...
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("33"),
            high=Decimal("34"),
            low=Decimal("32"),
            close=Decimal("33"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("1.00")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)]

        orders = sizer.size_positions(signals, portfolio, market, config)

        # Should round down to 60 (not 61)
        assert orders[0].quantity == Decimal("60")

    def test_quantity_less_than_one_skips_signal(self) -> None:
        """If calculated quantity < 1, signal should be skipped."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        sizer = FixedFractionalSizer(fraction=0.001)  # Very small fraction
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("1000"),  # Small portfolio
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("500"),
            high=Decimal("510"),
            low=Decimal("490"),
            close=Decimal("500"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("10.00")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)]

        orders = sizer.size_positions(signals, portfolio, market, config)

        # qty = (1000 * 0.001) / 500 = 1 / 500 = 0.002 -> rounds to 0
        assert orders == []

    def test_multiple_signals_processed(self) -> None:
        """Multiple signals should produce multiple orders."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        sizer = FixedFractionalSizer(fraction=0.02)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        bar_aapl = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        bar_googl = Bar(
            timestamp=now,
            symbol="GOOGL",
            open=Decimal("140"),
            high=Decimal("142"),
            low=Decimal("138"),
            close=Decimal("140"),
            volume=Decimal("500000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar_aapl, "GOOGL": bar_googl},
            volatility={"AAPL": Decimal("2.00"), "GOOGL": Decimal("3.00")},
            liquidity={"AAPL": Decimal("50000000"), "GOOGL": Decimal("20000000")},
            timestamp=now,
        )
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.8),
            Signal(symbol="GOOGL", timestamp=now, direction="short", strength=0.6),
        ]

        orders = sizer.size_positions(signals, portfolio, market, config)

        assert len(orders) == 2
        symbols = {o.symbol for o in orders}
        assert symbols == {"AAPL", "GOOGL"}

    def test_fraction_validation(self) -> None:
        """Fraction must be in valid range (0, 1]."""
        from liq.risk.sizers import FixedFractionalSizer

        with pytest.raises(ValueError, match="fraction"):
            FixedFractionalSizer(fraction=0.0)

        with pytest.raises(ValueError, match="fraction"):
            FixedFractionalSizer(fraction=-0.1)

        with pytest.raises(ValueError, match="fraction"):
            FixedFractionalSizer(fraction=1.5)

        # Edge case: 1.0 is valid (all equity per position)
        sizer = FixedFractionalSizer(fraction=1.0)
        assert sizer.fraction == 1.0


class TestFixedFractionalSizerPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        equity=st.decimals(min_value=1000, max_value=10000000, places=2, allow_nan=False, allow_infinity=False),
        fraction=st.floats(min_value=0.001, max_value=1.0),
        price=st.decimals(min_value=1, max_value=10000, places=2, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_quantity_always_positive_or_zero(
        self,
        equity: Decimal,
        fraction: float,
        price: Decimal,
    ) -> None:
        """Calculated quantity should always be >= 0."""
        from liq.risk.sizers import FixedFractionalSizer

        now = datetime.now(UTC)
        sizer = FixedFractionalSizer(fraction=fraction)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=equity,
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="TEST",
            open=price,
            high=price * Decimal("1.01"),
            low=price * Decimal("0.99"),
            close=price,
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"TEST": bar},
            volatility={"TEST": Decimal("2.00")},
            liquidity={"TEST": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="TEST", timestamp=now, direction="long", strength=1.0)]

        orders = sizer.size_positions(signals, portfolio, market, config)

        # Either no order (qty < 1) or positive quantity
        if orders:
            assert orders[0].quantity >= 1

    @given(
        fraction1=st.floats(min_value=0.01, max_value=0.5),
        fraction2=st.floats(min_value=0.01, max_value=0.5),
    )
    @settings(max_examples=50)
    def test_higher_fraction_never_smaller_position(
        self,
        fraction1: float,
        fraction2: float,
    ) -> None:
        """Higher fraction should never result in smaller position."""
        from liq.risk.sizers import FixedFractionalSizer

        if abs(fraction1 - fraction2) < 0.001:
            return  # Skip nearly equal fractions

        now = datetime.now(UTC)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="TEST",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"TEST": bar},
            volatility={"TEST": Decimal("2.00")},
            liquidity={"TEST": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="TEST", timestamp=now, direction="long", strength=1.0)]

        sizer1 = FixedFractionalSizer(fraction=fraction1)
        sizer2 = FixedFractionalSizer(fraction=fraction2)

        orders1 = sizer1.size_positions(signals, portfolio, market, config)
        orders2 = sizer2.size_positions(signals, portfolio, market, config)

        qty1 = orders1[0].quantity if orders1 else Decimal("0")
        qty2 = orders2[0].quantity if orders2 else Decimal("0")

        # Higher fraction should mean larger or equal position
        if fraction1 > fraction2:
            assert qty1 >= qty2
        else:
            assert qty2 >= qty1
