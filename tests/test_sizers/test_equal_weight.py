"""Tests for EqualWeightSizer.

Following TDD: RED phase - write failing tests first.

EqualWeightSizer: Allocates equal weight to each signal.
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


class TestEqualWeightSizerProtocol:
    """Test EqualWeightSizer conforms to PositionSizer protocol."""

    def test_conforms_to_protocol(self) -> None:
        """EqualWeightSizer should implement PositionSizer protocol."""
        from liq.risk.sizers import EqualWeightSizer

        sizer = EqualWeightSizer()
        assert isinstance(sizer, PositionSizer)


class TestEqualWeightSizerBasic:
    """Basic functionality tests."""

    def test_empty_signals_returns_empty_targets(self) -> None:
        """Empty signals should return empty targets."""
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
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
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
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
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
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
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
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


class TestEqualWeightSizerFormula:
    """Test equal weight allocation formula."""

    def test_single_signal_gets_full_allocation(self) -> None:
        """Single signal should get full allocation."""
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
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
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # Single signal gets 100% allocation
        # $100,000 / $100 = 1000 shares
        assert targets[0].target_quantity == Decimal("1000")

    def test_two_signals_split_equally(self) -> None:
        """Two signals should each get 50% allocation."""
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
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
                open=Decimal("100"),
                high=Decimal("102"),
                low=Decimal("98"),
                close=Decimal("100"),
                volume=Decimal("500000"),
            ),
        }
        market = MarketState(
            current_bars=bars,
            volatility={"AAPL": Decimal("2.00"), "GOOGL": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000"), "GOOGL": Decimal("30000000")},
            timestamp=now,
        )
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0),
            Signal(symbol="GOOGL", timestamp=now, direction="long", strength=1.0),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # Each gets 50% = $50,000 / $100 = 500 shares
        assert len(targets) == 2
        for target in targets:
            assert target.target_quantity == Decimal("500")

    def test_three_signals_split_equally(self) -> None:
        """Three signals should each get ~33% allocation."""
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("99000"),  # Divisible by 3
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
                open=Decimal("100"),
                high=Decimal("102"),
                low=Decimal("98"),
                close=Decimal("100"),
                volume=Decimal("500000"),
            ),
            "MSFT": Bar(
                timestamp=now,
                symbol="MSFT",
                open=Decimal("100"),
                high=Decimal("102"),
                low=Decimal("98"),
                close=Decimal("100"),
                volume=Decimal("800000"),
            ),
        }
        market = MarketState(
            current_bars=bars,
            volatility={
                "AAPL": Decimal("2.00"),
                "GOOGL": Decimal("2.00"),
                "MSFT": Decimal("2.00"),
            },
            liquidity={
                "AAPL": Decimal("50000000"),
                "GOOGL": Decimal("30000000"),
                "MSFT": Decimal("40000000"),
            },
            timestamp=now,
        )
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0),
            Signal(symbol="GOOGL", timestamp=now, direction="long", strength=1.0),
            Signal(symbol="MSFT", timestamp=now, direction="long", strength=1.0),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # Each gets 33.33% = $33,000 / $100 = 330 shares
        assert len(targets) == 3
        for target in targets:
            assert target.target_quantity == Decimal("330")

    def test_different_prices_same_dollar_allocation(self) -> None:
        """Different priced stocks should get same dollar allocation."""
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        bars = {
            "CHEAP": Bar(
                timestamp=now,
                symbol="CHEAP",
                open=Decimal("10"),
                high=Decimal("11"),
                low=Decimal("9"),
                close=Decimal("10"),
                volume=Decimal("1000000"),
            ),
            "EXPENSIVE": Bar(
                timestamp=now,
                symbol="EXPENSIVE",
                open=Decimal("500"),
                high=Decimal("510"),
                low=Decimal("490"),
                close=Decimal("500"),
                volume=Decimal("100000"),
            ),
        }
        market = MarketState(
            current_bars=bars,
            volatility={"CHEAP": Decimal("1.00"), "EXPENSIVE": Decimal("10.00")},
            liquidity={"CHEAP": Decimal("10000000"), "EXPENSIVE": Decimal("50000000")},
            timestamp=now,
        )
        signals = [
            Signal(symbol="CHEAP", timestamp=now, direction="long", strength=1.0),
            Signal(symbol="EXPENSIVE", timestamp=now, direction="long", strength=1.0),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # Each gets $50,000
        # CHEAP: $50,000 / $10 = 5000 shares
        # EXPENSIVE: $50,000 / $500 = 100 shares
        cheap_target = next(t for t in targets if t.symbol == "CHEAP")
        expensive_target = next(t for t in targets if t.symbol == "EXPENSIVE")

        assert cheap_target.target_quantity == Decimal("5000")
        assert expensive_target.target_quantity == Decimal("100")


class TestEqualWeightSizerMaxPositions:
    """Test max_positions limit respect."""

    def test_respects_max_positions_config(self) -> None:
        """Should only allocate to max_positions signals."""
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
        config = RiskConfig(max_positions=2)  # Only 2 positions
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
                open=Decimal("100"),
                high=Decimal("102"),
                low=Decimal("98"),
                close=Decimal("100"),
                volume=Decimal("500000"),
            ),
            "MSFT": Bar(
                timestamp=now,
                symbol="MSFT",
                open=Decimal("100"),
                high=Decimal("102"),
                low=Decimal("98"),
                close=Decimal("100"),
                volume=Decimal("800000"),
            ),
        }
        market = MarketState(
            current_bars=bars,
            volatility={
                "AAPL": Decimal("2.00"),
                "GOOGL": Decimal("2.00"),
                "MSFT": Decimal("2.00"),
            },
            liquidity={
                "AAPL": Decimal("50000000"),
                "GOOGL": Decimal("30000000"),
                "MSFT": Decimal("40000000"),
            },
            timestamp=now,
        )
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0),
            Signal(symbol="GOOGL", timestamp=now, direction="long", strength=0.8),
            Signal(symbol="MSFT", timestamp=now, direction="long", strength=0.6),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # Should only produce 2 targets (highest strength signals)
        assert len(targets) == 2
        symbols = {t.symbol for t in targets}
        assert "AAPL" in symbols  # Highest strength
        assert "GOOGL" in symbols  # Second highest

    def test_allocation_based_on_max_positions(self) -> None:
        """Allocation should be based on max_positions, not signal count."""
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
        config = RiskConfig(max_positions=2)
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
                open=Decimal("100"),
                high=Decimal("102"),
                low=Decimal("98"),
                close=Decimal("100"),
                volume=Decimal("500000"),
            ),
            "MSFT": Bar(
                timestamp=now,
                symbol="MSFT",
                open=Decimal("100"),
                high=Decimal("102"),
                low=Decimal("98"),
                close=Decimal("100"),
                volume=Decimal("800000"),
            ),
        }
        market = MarketState(
            current_bars=bars,
            volatility={
                "AAPL": Decimal("2.00"),
                "GOOGL": Decimal("2.00"),
                "MSFT": Decimal("2.00"),
            },
            liquidity={
                "AAPL": Decimal("50000000"),
                "GOOGL": Decimal("30000000"),
                "MSFT": Decimal("40000000"),
            },
            timestamp=now,
        )
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0),
            Signal(symbol="GOOGL", timestamp=now, direction="long", strength=0.8),
            Signal(symbol="MSFT", timestamp=now, direction="long", strength=0.6),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # Each of 2 positions gets 50% = $50,000 / $100 = 500 shares
        for target in targets:
            assert target.target_quantity == Decimal("500")


class TestEqualWeightSizerEdgeCases:
    """Edge case tests."""

    def test_missing_bar_data_skips_signal(self) -> None:
        """Signals without bar data should be skipped."""
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
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
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
        config = RiskConfig()
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
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # $10,000 / $33 = 303.03... → 303 shares
        assert targets[0].target_quantity == Decimal("303")

    def test_quantity_less_than_one_skips_signal(self) -> None:
        """Signals resulting in < 1 share should be skipped."""
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
        config = RiskConfig()
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
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        # $100 / $500 = 0.2 shares → skip
        assert targets == []


class TestEqualWeightSizerPropertyBased:
    """Property-based tests using Hypothesis."""

    @given(
        equity=st.decimals(
            min_value=1000, max_value=10000000, places=2, allow_nan=False, allow_infinity=False
        ),
        price=st.decimals(
            min_value=1, max_value=10000, places=2, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=100)
    def test_quantity_always_positive_or_zero(
        self, equity: Decimal, price: Decimal
    ) -> None:
        """Quantity should always be positive or signal skipped."""
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=equity,
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=price,
            high=price * Decimal("1.02"),
            low=price * Decimal("0.98"),
            close=price,
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

        targets = sizer.size_positions(signals, portfolio, market, config)

        for target in targets:
            assert abs(target.target_quantity) > 0

    @given(
        n_signals=st.integers(min_value=1, max_value=10),
    )
    @settings(max_examples=50)
    def test_total_allocation_never_exceeds_equity(self, n_signals: int) -> None:
        """Total allocation should never exceed equity."""
        from liq.risk.sizers import EqualWeightSizer

        now = datetime.now(UTC)
        sizer = EqualWeightSizer()
        config = RiskConfig()
        equity = Decimal("100000")
        portfolio = PortfolioState(
            cash=equity,
            positions={},
            timestamp=now,
        )

        # Create bars and signals for n symbols
        bars = {}
        volatility = {}
        liquidity = {}
        signals = []
        price = Decimal("100")

        for i in range(n_signals):
            symbol = f"SYM{i}"
            bars[symbol] = Bar(
                timestamp=now,
                symbol=symbol,
                open=price,
                high=price * Decimal("1.02"),
                low=price * Decimal("0.98"),
                close=price,
                volume=Decimal("1000000"),
            )
            volatility[symbol] = Decimal("2.00")
            liquidity[symbol] = Decimal("50000000")
            signals.append(
                Signal(symbol=symbol, timestamp=now, direction="long", strength=1.0)
            )

        market = MarketState(
            current_bars=bars,
            volatility=volatility,
            liquidity=liquidity,
            timestamp=now,
        )

        targets = sizer.size_positions(signals, portfolio, market, config)

        # Total value should not exceed equity
        total_value = sum(abs(t.target_quantity) * price for t in targets)
        assert total_value <= equity
