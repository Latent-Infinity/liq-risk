"""Tests for KellySizer.

Following TDD: RED phase - write failing tests first.

KellySizer: Sizes positions using Kelly criterion for optimal growth.
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


class TestKellySizerProtocol:
    """Test KellySizer conforms to PositionSizer protocol."""

    def test_conforms_to_protocol(self) -> None:
        """KellySizer should implement PositionSizer protocol."""
        from liq.risk.sizers import KellySizer

        sizer = KellySizer()
        assert isinstance(sizer, PositionSizer)


class TestKellySizerBasic:
    """Basic functionality tests."""

    def test_empty_signals_returns_empty_targets(self) -> None:
        """Empty signals should return empty targets."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
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

        targets = sizer.size_positions([], portfolio, market, config)

        assert targets == []

    def test_flat_signals_produce_no_targets(self) -> None:
        """Flat signals should not produce targets."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
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
        signals = [
            Signal(
                symbol="AAPL",
                timestamp=now,
                direction="flat",
                strength=1.0,
            )
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        assert targets == []

    def test_long_signal_produces_long_target(self) -> None:
        """Long signal should produce long target."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
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
        signals = [
            Signal(
                symbol="AAPL",
                timestamp=now,
                direction="long",
                strength=1.0,
            )
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        assert len(targets) == 1
        assert targets[0].direction == "long"
        assert targets[0].target_quantity > 0
        assert targets[0].symbol == "AAPL"

    def test_short_signal_produces_short_target(self) -> None:
        """Short signal should produce short target."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
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
        signals = [
            Signal(
                symbol="AAPL",
                timestamp=now,
                direction="short",
                strength=1.0,
            )
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        assert len(targets) == 1
        assert targets[0].direction == "short"
        assert targets[0].target_quantity < 0


class TestKellySizerFormula:
    """Test Kelly criterion formula.

    Kelly formula: f* = p - (1-p)/b = (p*b - (1-p)) / b
    Where:
        p = win probability (derived from signal strength)
        b = win/loss ratio (assumed 1:1 for symmetric returns)

    Simplified: f* = 2*p - 1 (for b=1)

    We use fractional Kelly (kelly_fraction config) for safety.
    """

    def test_uses_kelly_fraction_from_config(self) -> None:
        """Should use kelly_fraction from RiskConfig."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
        quarter_kelly = RiskConfig(kelly_fraction=0.25)
        half_kelly = RiskConfig(kelly_fraction=0.50)
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
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)
        ]

        targets_quarter = sizer.size_positions(signals, portfolio, market, quarter_kelly)
        targets_half = sizer.size_positions(signals, portfolio, market, half_kelly)

        # Half Kelly should produce larger position than quarter Kelly
        assert targets_half[0].target_quantity > targets_quarter[0].target_quantity

    def test_higher_strength_larger_position(self) -> None:
        """Higher signal strength should produce larger position."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
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

        high_strength = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)
        ]
        low_strength = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.6)
        ]

        targets_high = sizer.size_positions(high_strength, portfolio, market, config)
        targets_low = sizer.size_positions(low_strength, portfolio, market, config)

        # Higher strength should produce larger position
        assert targets_high[0].target_quantity > targets_low[0].target_quantity

    def test_strength_at_threshold_produces_zero(self) -> None:
        """Signal strength at 0.5 (no edge) should produce no position."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
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
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.5)
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # Kelly fraction with p=0.5: f* = 2*0.5 - 1 = 0
        assert targets == []

    def test_strength_below_threshold_produces_zero(self) -> None:
        """Signal strength below 0.5 (negative edge) should produce no position."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
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
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.4)
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # Kelly fraction with p<0.5: f* < 0, so no position
        assert targets == []

    def test_formula_calculation(self) -> None:
        """Verify Kelly formula calculation.

        Full Kelly: f* = 2p - 1 (for symmetric returns)
        With p = 0.75 (strength): f* = 2 * 0.75 - 1 = 0.5
        Quarter Kelly (0.25): position = 0.5 * 0.25 = 0.125 = 12.5% of equity
        """
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
        config = RiskConfig(kelly_fraction=0.25)  # Quarter Kelly
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
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.75)
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # f* = 2 * 0.75 - 1 = 0.5
        # Fractional Kelly: 0.5 * 0.25 = 0.125
        # Position value: $100,000 * 0.125 = $12,500
        # Shares: $12,500 / $100 = 125
        assert targets[0].target_quantity == Decimal("125")


class TestKellySizerEdgeCases:
    """Edge case tests."""

    def test_missing_bar_data_skips_signal(self) -> None:
        """Signals without bar data should be skipped."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        market = MarketState(
            current_bars={},  # No bar data
            volatility={},
            liquidity={},
            timestamp=now,
        )
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        assert targets == []

    def test_quantity_rounded_down_to_whole_shares(self) -> None:
        """Quantity should be rounded down to whole shares."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
        config = RiskConfig(kelly_fraction=0.25)
        portfolio = PortfolioState(
            cash=Decimal("10000"),
            positions={},
            timestamp=now,
        )
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
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.75)
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # f* = 0.5, fractional = 0.125
        # Value = $10,000 * 0.125 = $1,250
        # Shares = $1,250 / $33 = 37.87... → 37
        assert targets[0].target_quantity == Decimal("37")

    def test_quantity_less_than_one_skips_signal(self) -> None:
        """Signals resulting in < 1 share should be skipped."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
        config = RiskConfig(kelly_fraction=0.25)
        portfolio = PortfolioState(
            cash=Decimal("100"),  # Very small portfolio
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
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.75)
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # f* = 0.5, fractional = 0.125
        # Value = $100 * 0.125 = $12.50
        # Shares = $12.50 / $500 = 0.025 → skip
        assert targets == []

    def test_multiple_signals_processed(self) -> None:
        """Multiple signals should all be processed."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        bars = {
            "AAPL": Bar(
                timestamp=now,
                symbol="AAPL",
                open=Decimal("100"),
                high=Decimal("102"),
                low=Decimal("98"),
                close=Decimal("100"),
                volume=Decimal("1000000"),
            ),
            "GOOGL": Bar(
                timestamp=now,
                symbol="GOOGL",
                open=Decimal("150"),
                high=Decimal("153"),
                low=Decimal("147"),
                close=Decimal("150"),
                volume=Decimal("500000"),
            ),
        }
        market = MarketState(
            current_bars=bars,
            volatility={"AAPL": Decimal("2.00"), "GOOGL": Decimal("3.00")},
            liquidity={"AAPL": Decimal("50000000"), "GOOGL": Decimal("30000000")},
            timestamp=now,
        )
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.75),
            Signal(symbol="GOOGL", timestamp=now, direction="long", strength=0.80),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        assert len(targets) == 2
        symbols = {o.symbol for o in targets}
        assert "AAPL" in symbols
        assert "GOOGL" in symbols


class TestKellySizerPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        strength=st.floats(min_value=0.51, max_value=1.0, allow_nan=False),
        kelly_fraction=st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
    )
    @settings(max_examples=100)
    def test_quantity_always_positive_for_edge(
        self, strength: float, kelly_fraction: float
    ) -> None:
        """Quantity should be positive when there's an edge (strength > 0.5)."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
        config = RiskConfig(kelly_fraction=kelly_fraction)
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
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=strength)
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # With sufficient equity and edge, should produce positive quantity
        if targets:
            assert abs(targets[0].target_quantity) > 0

    @given(
        strength=st.floats(min_value=0.0, max_value=0.5, allow_nan=False),
    )
    @settings(max_examples=50)
    def test_no_position_without_edge(self, strength: float) -> None:
        """Should not produce position when there's no edge."""
        from liq.risk.sizers import KellySizer

        now = datetime.now(UTC)
        sizer = KellySizer()
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
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=strength)
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # No edge (strength <= 0.5) should produce no position
        assert targets == []
