"""Tests for VolatilitySizer.

Following TDD: RED phase - write failing tests first.

VolatilitySizer Formula:
    quantity = (equity * risk_per_trade) / (price * atr_multiple * atr)

Higher volatility → smaller position (inverse relationship).

Note: Sizers now return TargetPosition instead of OrderRequest.
TargetPosition has target_quantity (positive for long, negative for short)
and direction ("long", "short", "flat").
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st
from liq.core import Bar, PortfolioState
from liq.signals import Signal

from liq.risk import MarketState, RiskConfig
from liq.risk.protocols import PositionSizer


class TestVolatilitySizerProtocol:
    """Test that VolatilitySizer conforms to PositionSizer protocol."""

    def test_conforms_to_protocol(self) -> None:
        """VolatilitySizer must implement PositionSizer protocol."""
        from liq.risk.sizers import VolatilitySizer

        sizer = VolatilitySizer()
        assert isinstance(sizer, PositionSizer)


class TestVolatilitySizerBasic:
    """Basic functionality tests for VolatilitySizer."""

    def test_empty_signals_returns_empty_orders(self) -> None:
        """No signals should produce no orders."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        sizer = VolatilitySizer()
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
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        sizer = VolatilitySizer()
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

    def test_long_signal_produces_long_target(self) -> None:
        """Long signal should produce a long TargetPosition."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        sizer = VolatilitySizer()
        config = RiskConfig(risk_per_trade=0.01)
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
            volatility={"AAPL": Decimal("2.00")},  # ATR = $2
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.8)]

        targets = sizer.size_positions(signals, portfolio, market, config)

        assert len(targets) == 1
        assert targets[0].symbol == "AAPL"
        assert targets[0].direction == "long"
        assert targets[0].target_quantity > 0

    def test_short_signal_produces_short_target(self) -> None:
        """Short signal should produce a short TargetPosition."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        sizer = VolatilitySizer()
        config = RiskConfig(risk_per_trade=0.01)
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

        targets = sizer.size_positions(signals, portfolio, market, config)

        assert len(targets) == 1
        assert targets[0].symbol == "AAPL"
        assert targets[0].direction == "short"
        assert targets[0].target_quantity < 0  # Negative for short


class TestVolatilitySizerFormula:
    """Tests for the volatility sizing formula."""

    def test_sizing_formula_calculation(self) -> None:
        """Verify the volatility sizing formula: qty = (equity * risk) / (price * atr_mult * atr)."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        # Formula: qty = (100000 * 0.01) / (100 * 2.0 * 2.50)
        #              = 1000 / 500 = 2 shares
        sizer = VolatilitySizer(atr_multiple=2.0)
        config = RiskConfig(risk_per_trade=0.01)  # 1% risk
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
            close=Decimal("100"),  # midrange = 100
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.50")},  # ATR = $2.50
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)]

        targets = sizer.size_positions(signals, portfolio, market, config)

        assert len(targets) == 1
        # qty = (100000 * 0.01) / (100 * 2.0 * 2.50) = 1000 / 500 = 2
        assert targets[0].target_quantity == Decimal("2")

    def test_higher_volatility_smaller_position(self) -> None:
        """Higher volatility should result in smaller position size."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        sizer = VolatilitySizer()
        config = RiskConfig(risk_per_trade=0.01)
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )

        # Low volatility
        bar_low_vol = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("101"),
            low=Decimal("99"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        market_low_vol = MarketState(
            current_bars={"AAPL": bar_low_vol},
            volatility={"AAPL": Decimal("1.00")},  # Low ATR
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )

        # High volatility
        bar_high_vol = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        market_high_vol = MarketState(
            current_bars={"AAPL": bar_high_vol},
            volatility={"AAPL": Decimal("5.00")},  # High ATR
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )

        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)]

        targets_low_vol = sizer.size_positions(signals, portfolio, market_low_vol, config)
        targets_high_vol = sizer.size_positions(signals, portfolio, market_high_vol, config)

        assert targets_low_vol[0].target_quantity > targets_high_vol[0].target_quantity

    def test_higher_equity_larger_position(self) -> None:
        """Higher equity should result in larger position size."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        sizer = VolatilitySizer()
        config = RiskConfig(risk_per_trade=0.01)

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

        # Small portfolio
        portfolio_small = PortfolioState(
            cash=Decimal("50000"),
            positions={},
            timestamp=now,
        )
        # Large portfolio
        portfolio_large = PortfolioState(
            cash=Decimal("200000"),
            positions={},
            timestamp=now,
        )

        targets_small = sizer.size_positions(signals, portfolio_small, market, config)
        targets_large = sizer.size_positions(signals, portfolio_large, market, config)

        assert targets_large[0].target_quantity > targets_small[0].target_quantity

    def test_uses_midrange_price_by_default(self) -> None:
        """By default, should use midrange price for sizing."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        sizer = VolatilitySizer(use_midrange_price=True, atr_multiple=2.0)
        config = RiskConfig(risk_per_trade=0.01)
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        # Bar with midrange = (110 + 90) / 2 = 100
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("95"),
            high=Decimal("110"),
            low=Decimal("90"),
            close=Decimal("105"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.50")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # qty = (100000 * 0.01) / (100 * 2.0 * 2.50) = 1000 / 500 = 2
        assert targets[0].target_quantity == Decimal("2")

    def test_can_use_close_price(self) -> None:
        """Can configure to use close price instead of midrange."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        sizer = VolatilitySizer(use_midrange_price=False, atr_multiple=2.0)
        config = RiskConfig(risk_per_trade=0.01)
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        # Bar with close = 50, midrange = (60 + 40) / 2 = 50
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

        targets = sizer.size_positions(signals, portfolio, market, config)

        # qty = (100000 * 0.01) / (50 * 2.0 * 2.50) = 1000 / 250 = 4
        assert targets[0].target_quantity == Decimal("4")


class TestVolatilitySizerEdgeCases:
    """Edge case tests for VolatilitySizer."""

    def test_missing_bar_data_skips_signal(self) -> None:
        """Signal for symbol without bar data should be skipped."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        sizer = VolatilitySizer()
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

    def test_missing_volatility_skips_signal(self) -> None:
        """Signal for symbol without volatility data should be skipped."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        sizer = VolatilitySizer()
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
            volatility={},  # No volatility for AAPL
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.8)]

        orders = sizer.size_positions(signals, portfolio, market, config)

        assert orders == []

    def test_zero_volatility_skips_signal(self) -> None:
        """Signal with zero volatility should be skipped (division by zero)."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        sizer = VolatilitySizer()
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
            high=Decimal("150"),
            low=Decimal("150"),
            close=Decimal("150"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("0")},  # Zero volatility
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.8)]

        orders = sizer.size_positions(signals, portfolio, market, config)

        assert orders == []

    def test_quantity_rounded_down_to_whole_shares(self) -> None:
        """Quantity should be rounded down to whole shares when configured."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        # Configure for whole-share trading (equities)
        sizer = VolatilitySizer(
            atr_multiple=2.0,
            min_quantity=Decimal("1"),
            quantize_step=Decimal("1"),
        )
        config = RiskConfig(risk_per_trade=0.01)
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        # Setup to get fractional result: (100000 * 0.01) / (100 * 2.0 * 3.0) = 1000 / 600 = 1.666...
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
            volatility={"AAPL": Decimal("3.00")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # Should round down to 1 (not 2)
        assert targets[0].target_quantity == Decimal("1")

    def test_quantity_less_than_one_skips_signal(self) -> None:
        """If calculated quantity < 1, signal should be skipped for whole-share mode."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        # Configure for whole-share trading (equities)
        sizer = VolatilitySizer(
            atr_multiple=2.0,
            min_quantity=Decimal("1"),
            quantize_step=Decimal("1"),
        )
        config = RiskConfig(risk_per_trade=0.001)  # Very small risk
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
            volatility={"AAPL": Decimal("10.00")},  # High vol
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)]

        orders = sizer.size_positions(signals, portfolio, market, config)

        # qty = (1000 * 0.001) / (500 * 2.0 * 10) = 1 / 10000 = 0.0001 -> rounds to 0
        assert orders == []

    def test_multiple_signals_processed(self) -> None:
        """Multiple signals should produce multiple orders."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        sizer = VolatilitySizer()
        config = RiskConfig(risk_per_trade=0.01)
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


class TestVolatilitySizerPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        equity=st.decimals(min_value=1000, max_value=10000000, places=2, allow_nan=False, allow_infinity=False),
        risk_pct=st.floats(min_value=0.001, max_value=0.10),
        price=st.decimals(min_value=1, max_value=10000, places=2, allow_nan=False, allow_infinity=False),
        volatility=st.decimals(min_value=Decimal("0.01"), max_value=100, places=2, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=100)
    def test_quantity_always_positive_or_zero(
        self,
        equity: Decimal,
        risk_pct: float,
        price: Decimal,
        volatility: Decimal,
    ) -> None:
        """Calculated quantity should always be >= 0 for long positions."""
        from liq.risk.sizers import VolatilitySizer

        now = datetime.now(UTC)
        sizer = VolatilitySizer()
        config = RiskConfig(risk_per_trade=risk_pct)
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
            volatility={"TEST": volatility},
            liquidity={"TEST": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="TEST", timestamp=now, direction="long", strength=1.0)]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # Either no target (qty < min_quantity) or positive target_quantity for long
        if targets:
            assert targets[0].target_quantity > 0

    @given(
        vol1=st.decimals(min_value=Decimal("0.1"), max_value=50, places=2, allow_nan=False, allow_infinity=False),
        vol2=st.decimals(min_value=Decimal("0.1"), max_value=50, places=2, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_higher_vol_never_larger_position(
        self,
        vol1: Decimal,
        vol2: Decimal,
    ) -> None:
        """Higher volatility should never result in larger position."""
        from liq.risk.sizers import VolatilitySizer

        if vol1 == vol2:
            return  # Skip equal volatilities

        now = datetime.now(UTC)
        sizer = VolatilitySizer()
        config = RiskConfig(risk_per_trade=0.01)
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

        market1 = MarketState(
            current_bars={"TEST": bar},
            volatility={"TEST": vol1},
            liquidity={"TEST": Decimal("50000000")},
            timestamp=now,
        )
        market2 = MarketState(
            current_bars={"TEST": bar},
            volatility={"TEST": vol2},
            liquidity={"TEST": Decimal("50000000")},
            timestamp=now,
        )
        signals = [Signal(symbol="TEST", timestamp=now, direction="long", strength=1.0)]

        targets1 = sizer.size_positions(signals, portfolio, market1, config)
        targets2 = sizer.size_positions(signals, portfolio, market2, config)

        # Get quantities (0 if no target)
        qty1 = targets1[0].target_quantity if targets1 else Decimal("0")
        qty2 = targets2[0].target_quantity if targets2 else Decimal("0")

        # Higher volatility should mean smaller or equal position
        if vol1 > vol2:
            assert qty1 <= qty2
        else:
            assert qty2 <= qty1
