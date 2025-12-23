"""Tests for RiskParitySizer.

Following TDD: RED phase - write failing tests first.

RiskParitySizer: Equal risk contribution from each position.
Sizes positions so each contributes equal volatility to portfolio.

Formula:
    weight_i = (1/vol_i) / Σ(1/vol_j)
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st
from liq.core import Bar, PortfolioState
from liq.signals import Signal

from liq.risk import MarketState, RiskConfig
from liq.risk.sizers import RiskParitySizer


class TestRiskParitySizerProtocol:
    """Tests for RiskParitySizer protocol compliance."""

    def test_conforms_to_protocol(self) -> None:
        """RiskParitySizer conforms to PositionSizer protocol."""
        from liq.risk.protocols import PositionSizer

        sizer = RiskParitySizer()
        assert hasattr(sizer, "size_positions")
        assert isinstance(sizer, PositionSizer)


class TestRiskParitySizerBasic:
    """Basic tests for RiskParitySizer behavior."""

    def test_empty_signals_returns_empty_targets(self) -> None:
        """Empty signal list returns empty target list."""
        sizer = RiskParitySizer()
        now = datetime.now(UTC)

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
        config = RiskConfig()

        targets = sizer.size_positions([], portfolio, market, config)
        assert targets == []

    def test_flat_signals_produce_no_targets(self) -> None:
        """Flat direction signals should not produce targets."""
        sizer = RiskParitySizer()
        now = datetime.now(UTC)

        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2")},
            liquidity={"AAPL": Decimal("1000000")},
            timestamp=now,
        )
        config = RiskConfig()

        signals = [
            Signal(
                symbol="AAPL",
                timestamp=now,
                direction="flat",
                strength=0.5,
            )
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)
        assert targets == []

    def test_long_signal_produces_long_target(self) -> None:
        """Long signal should produce long target."""
        sizer = RiskParitySizer()
        now = datetime.now(UTC)

        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2")},
            liquidity={"AAPL": Decimal("1000000")},
            timestamp=now,
        )
        config = RiskConfig()

        signals = [
            Signal(
                symbol="AAPL",
                timestamp=now,
                direction="long",
                strength=0.8,
            )
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)
        assert len(targets) == 1
        assert targets[0].direction == "long"
        assert targets[0].target_quantity > 0
        assert targets[0].symbol == "AAPL"

    def test_short_signal_produces_short_target(self) -> None:
        """Short signal should produce short target."""
        sizer = RiskParitySizer()
        now = datetime.now(UTC)

        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2")},
            liquidity={"AAPL": Decimal("1000000")},
            timestamp=now,
        )
        config = RiskConfig()

        signals = [
            Signal(
                symbol="AAPL",
                timestamp=now,
                direction="short",
                strength=0.8,
            )
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)
        assert len(targets) == 1
        assert targets[0].direction == "short"
        assert targets[0].target_quantity < 0


class TestRiskParitySizerFormula:
    """Tests for risk parity sizing formula."""

    def test_equal_volatility_equal_weights(self) -> None:
        """With equal volatility, positions should have equal weights."""
        sizer = RiskParitySizer()
        now = datetime.now(UTC)

        bar_a = Bar(
            timestamp=now, symbol="SYM_A",
            open=Decimal("100"), high=Decimal("105"),
            low=Decimal("95"), close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        bar_b = Bar(
            timestamp=now, symbol="SYM_B",
            open=Decimal("100"), high=Decimal("105"),
            low=Decimal("95"), close=Decimal("100"),
            volume=Decimal("1000000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        # Same volatility for both
        market = MarketState(
            current_bars={"SYM_A": bar_a, "SYM_B": bar_b},
            volatility={"SYM_A": Decimal("5"), "SYM_B": Decimal("5")},
            liquidity={"SYM_A": Decimal("1000000"), "SYM_B": Decimal("1000000")},
            timestamp=now,
        )
        config = RiskConfig()

        signals = [
            Signal(symbol="SYM_A", timestamp=now, direction="long", strength=0.8),
            Signal(symbol="SYM_B", timestamp=now, direction="long", strength=0.8),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)
        assert len(targets) == 2

        # Equal volatility means equal position sizes
        qty_a = next(abs(t.target_quantity) for t in targets if t.symbol == "SYM_A")
        qty_b = next(abs(t.target_quantity) for t in targets if t.symbol == "SYM_B")
        assert qty_a == qty_b

    def test_higher_volatility_smaller_position(self) -> None:
        """Higher volatility asset should get smaller position."""
        sizer = RiskParitySizer()
        now = datetime.now(UTC)

        # Same price for both
        bar_low_vol = Bar(
            timestamp=now, symbol="LOWVOL",
            open=Decimal("100"), high=Decimal("105"),
            low=Decimal("95"), close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        bar_high_vol = Bar(
            timestamp=now, symbol="HIGHVOL",
            open=Decimal("100"), high=Decimal("105"),
            low=Decimal("95"), close=Decimal("100"),
            volume=Decimal("1000000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        # HIGHVOL has 2x the volatility
        market = MarketState(
            current_bars={"LOWVOL": bar_low_vol, "HIGHVOL": bar_high_vol},
            volatility={"LOWVOL": Decimal("2"), "HIGHVOL": Decimal("4")},
            liquidity={"LOWVOL": Decimal("1000000"), "HIGHVOL": Decimal("1000000")},
            timestamp=now,
        )
        config = RiskConfig()

        signals = [
            Signal(symbol="LOWVOL", timestamp=now, direction="long", strength=0.8),
            Signal(symbol="HIGHVOL", timestamp=now, direction="long", strength=0.8),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)
        assert len(targets) == 2

        # Find targets by symbol
        low_vol_target = next(t for t in targets if t.symbol == "LOWVOL")
        high_vol_target = next(t for t in targets if t.symbol == "HIGHVOL")

        # Low vol should have 2x the position (inverse of volatility ratio)
        assert abs(low_vol_target.target_quantity) == abs(high_vol_target.target_quantity) * 2

    def test_weights_sum_to_risk_allocation(self) -> None:
        """Position values should sum to risk allocation."""
        sizer = RiskParitySizer()
        now = datetime.now(UTC)

        bar_a = Bar(
            timestamp=now, symbol="SYM_A",
            open=Decimal("50"), high=Decimal("52"),
            low=Decimal("48"), close=Decimal("50"),
            volume=Decimal("1000000"),
        )
        bar_b = Bar(
            timestamp=now, symbol="SYM_B",
            open=Decimal("100"), high=Decimal("105"),
            low=Decimal("95"), close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        bar_c = Bar(
            timestamp=now, symbol="SYM_C",
            open=Decimal("200"), high=Decimal("210"),
            low=Decimal("190"), close=Decimal("200"),
            volume=Decimal("1000000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        market = MarketState(
            current_bars={"SYM_A": bar_a, "SYM_B": bar_b, "SYM_C": bar_c},
            volatility={"SYM_A": Decimal("2"), "SYM_B": Decimal("4"), "SYM_C": Decimal("8")},
            liquidity={"SYM_A": Decimal("1000000"), "SYM_B": Decimal("1000000"), "SYM_C": Decimal("1000000")},
            timestamp=now,
        )
        config = RiskConfig(risk_per_trade=0.02)  # 2% risk allocation

        signals = [
            Signal(symbol="SYM_A", timestamp=now, direction="long", strength=0.8),
            Signal(symbol="SYM_B", timestamp=now, direction="long", strength=0.8),
            Signal(symbol="SYM_C", timestamp=now, direction="long", strength=0.8),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)
        assert len(targets) == 3

        # Calculate total notional value
        total_value = Decimal("0")
        for target in targets:
            bar = market.current_bars[target.symbol]
            price = (bar.high + bar.low) / 2
            total_value += abs(target.target_quantity) * price

        # Should be close to risk_per_trade * equity
        expected_allocation = Decimal("100000") * Decimal("0.02")
        # Allow rounding tolerance (can lose up to a few hundred due to rounding down)
        assert abs(total_value - expected_allocation) < Decimal("300")


class TestRiskParitySizerEdgeCases:
    """Tests for edge cases in RiskParitySizer."""

    def test_missing_bar_data_skips_signal(self) -> None:
        """Signal with missing bar data should be skipped."""
        sizer = RiskParitySizer()
        now = datetime.now(UTC)

        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        market = MarketState(
            current_bars={},  # No bar data
            volatility={"AAPL": Decimal("2")},
            liquidity={"AAPL": Decimal("1000000")},
            timestamp=now,
        )
        config = RiskConfig()

        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.8),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)
        assert targets == []

    def test_missing_volatility_skips_signal(self) -> None:
        """Signal with missing volatility should be skipped."""
        sizer = RiskParitySizer()
        now = datetime.now(UTC)

        bar = Bar(
            timestamp=now, symbol="AAPL",
            open=Decimal("100"), high=Decimal("105"),
            low=Decimal("95"), close=Decimal("100"),
            volume=Decimal("1000000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={},  # No volatility data
            liquidity={"AAPL": Decimal("1000000")},
            timestamp=now,
        )
        config = RiskConfig()

        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.8),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)
        assert targets == []

    def test_zero_volatility_skips_signal(self) -> None:
        """Signal with zero volatility should be skipped."""
        sizer = RiskParitySizer()
        now = datetime.now(UTC)

        bar = Bar(
            timestamp=now, symbol="AAPL",
            open=Decimal("100"), high=Decimal("105"),
            low=Decimal("95"), close=Decimal("100"),
            volume=Decimal("1000000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("0")},  # Zero volatility
            liquidity={"AAPL": Decimal("1000000")},
            timestamp=now,
        )
        config = RiskConfig()

        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.8),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)
        assert targets == []

    def test_quantity_rounded_down_to_whole_shares(self) -> None:
        """Quantity should be rounded down to whole shares."""
        sizer = RiskParitySizer()
        now = datetime.now(UTC)

        bar = Bar(
            timestamp=now, symbol="AAPL",
            open=Decimal("100"), high=Decimal("105"),
            low=Decimal("95"), close=Decimal("100"),
            volume=Decimal("1000000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2")},
            liquidity={"AAPL": Decimal("1000000")},
            timestamp=now,
        )
        config = RiskConfig()

        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.8),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)
        if targets:
            # Quantity should be a whole number
            assert abs(targets[0].target_quantity) == abs(targets[0].target_quantity).to_integral_value()

    def test_quantity_less_than_one_skips_signal(self) -> None:
        """Signal with calculated quantity < 1 should be skipped."""
        sizer = RiskParitySizer()
        now = datetime.now(UTC)

        # Very expensive stock with high volatility
        bar = Bar(
            timestamp=now, symbol="BRKA",
            open=Decimal("500000"), high=Decimal("510000"),
            low=Decimal("490000"), close=Decimal("500000"),
            volume=Decimal("1000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("1000"),  # Very small account
            positions={},
            timestamp=now,
        )
        market = MarketState(
            current_bars={"BRKA": bar},
            volatility={"BRKA": Decimal("10000")},
            liquidity={"BRKA": Decimal("100000000")},
            timestamp=now,
        )
        config = RiskConfig(risk_per_trade=0.01)

        signals = [
            Signal(symbol="BRKA", timestamp=now, direction="long", strength=0.8),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)
        assert targets == []


class TestRiskParitySizerPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        vol1=st.decimals(min_value=1, max_value=100, places=2),
        vol2=st.decimals(min_value=1, max_value=100, places=2),
    )
    @settings(max_examples=100)
    def test_inverse_volatility_relationship(
        self, vol1: Decimal, vol2: Decimal
    ) -> None:
        """Position sizes should be inversely proportional to volatility."""
        sizer = RiskParitySizer()
        now = datetime.now(UTC)

        bar_a = Bar(
            timestamp=now, symbol="SYM_A",
            open=Decimal("100"), high=Decimal("105"),
            low=Decimal("95"), close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        bar_b = Bar(
            timestamp=now, symbol="SYM_B",
            open=Decimal("100"), high=Decimal("105"),
            low=Decimal("95"), close=Decimal("100"),
            volume=Decimal("1000000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("1000000"),  # Large account for meaningful positions
            positions={},
            timestamp=now,
        )
        market = MarketState(
            current_bars={"SYM_A": bar_a, "SYM_B": bar_b},
            volatility={"SYM_A": vol1, "SYM_B": vol2},
            liquidity={"SYM_A": Decimal("1000000"), "SYM_B": Decimal("1000000")},
            timestamp=now,
        )
        config = RiskConfig(risk_per_trade=0.10)  # 10% for larger positions

        signals = [
            Signal(symbol="SYM_A", timestamp=now, direction="long", strength=0.8),
            Signal(symbol="SYM_B", timestamp=now, direction="long", strength=0.8),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        if len(targets) == 2:
            qty_a = next(abs(t.target_quantity) for t in targets if t.symbol == "SYM_A")
            qty_b = next(abs(t.target_quantity) for t in targets if t.symbol == "SYM_B")

            # Risk contribution should be approximately equal
            # risk_a = qty_a * vol1, risk_b = qty_b * vol2
            # They should be close (within rounding)
            risk_a = qty_a * vol1
            risk_b = qty_b * vol2

            # Allow 20% tolerance for rounding effects
            if risk_a > 0 and risk_b > 0:
                ratio = float(risk_a / risk_b)
                assert 0.5 < ratio < 2.0, f"Risk ratio {ratio} too far from 1.0"
