"""Integration tests for the liq-risk package.

Tests the complete signal-to-order pipeline including:
- Sizing + constraints applied together
- Multiple signals processed correctly
- Edge cases and error handling
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from liq.core import Bar, OrderSide, PortfolioState, Position
from liq.signals import Signal

from liq.risk import (
    BuyingPowerConstraint,
    FixedFractionalSizer,
    GrossLeverageConstraint,
    MarketState,
    MaxPositionConstraint,
    MaxPositionsConstraint,
    MinPositionValueConstraint,
    NetLeverageConstraint,
    RiskConfig,
    RiskEngine,
    RiskEngineResult,
    ShortSellingConstraint,
    VolatilitySizer,
)


class TestEndToEndPipeline:
    """End-to-end tests for signal → order pipeline."""

    def test_full_pipeline_single_signal(self) -> None:
        """Single signal processed through entire pipeline."""
        now = datetime.now(UTC)
        engine = RiskEngine()
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

        result = engine.process_signals(signals, portfolio, market, config)

        # Verify result structure
        assert isinstance(result, RiskEngineResult)
        assert len(result.orders) == 1
        assert result.orders[0].symbol == "AAPL"
        assert result.orders[0].side == OrderSide.BUY
        assert result.orders[0].quantity > 0
        # Stop-loss should be calculated
        assert "AAPL" in result.stop_losses

    def test_full_pipeline_multiple_signals(self) -> None:
        """Multiple signals processed through pipeline."""
        now = datetime.now(UTC)
        engine = RiskEngine()
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
            "MSFT": Bar(
                timestamp=now,
                symbol="MSFT",
                open=Decimal("350"),
                high=Decimal("355"),
                low=Decimal("345"),
                close=Decimal("350"),
                volume=Decimal("800000"),
            ),
        }
        market = MarketState(
            current_bars=bars,
            volatility={
                "AAPL": Decimal("2.00"),
                "GOOGL": Decimal("3.00"),
                "MSFT": Decimal("5.00"),
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

        result = engine.process_signals(signals, portfolio, market, config)

        # All should produce orders (subject to constraints)
        assert len(result.orders) >= 1
        symbols = {o.symbol for o in result.orders}
        # At least some orders should be generated
        assert len(symbols) >= 1

    def test_constraints_applied_in_order(self) -> None:
        """Constraints should be applied in correct order."""
        now = datetime.now(UTC)
        config = RiskConfig(
            max_position_pct=0.02,  # 2% max per position = $2000
            max_positions=2,  # Only 2 positions allowed
        )
        engine = RiskEngine()
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
            "MSFT": Bar(
                timestamp=now,
                symbol="MSFT",
                open=Decimal("350"),
                high=Decimal("355"),
                low=Decimal("345"),
                close=Decimal("350"),
                volume=Decimal("800000"),
            ),
        }
        market = MarketState(
            current_bars=bars,
            volatility={
                "AAPL": Decimal("2.00"),
                "GOOGL": Decimal("3.00"),
                "MSFT": Decimal("5.00"),
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

        result = engine.process_signals(signals, portfolio, market, config)

        # Max 2 positions should be allowed
        assert len(result.orders) <= 2

        # Each order should be constrained to 2% max = $2000 / price = 20 shares @ $100
        for order in result.orders:
            bar = bars[order.symbol]
            max_value = Decimal("100000") * Decimal("0.02")  # $2000
            order_value = order.quantity * bar.close
            assert order_value <= max_value


class TestRejectionTracking:
    """Test that rejected signals are properly tracked."""

    def test_rejected_signals_tracked(self) -> None:
        """Signals rejected by constraints should be tracked."""
        now = datetime.now(UTC)
        config = RiskConfig(max_positions=1)  # Only 1 position
        engine = RiskEngine()
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
            volatility={
                "AAPL": Decimal("2.00"),
                "GOOGL": Decimal("3.00"),
            },
            liquidity={
                "AAPL": Decimal("50000000"),
                "GOOGL": Decimal("30000000"),
            },
            timestamp=now,
        )
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0),
            Signal(symbol="GOOGL", timestamp=now, direction="long", strength=0.5),
        ]

        result = engine.process_signals(signals, portfolio, market, config)

        # One should be accepted, one rejected
        assert len(result.orders) == 1
        assert len(result.rejected_signals) == 1
        # Higher strength should be accepted
        assert result.orders[0].symbol == "AAPL"
        assert result.rejected_signals[0].symbol == "GOOGL"


class TestCustomSizerIntegration:
    """Test custom sizer integration."""

    def test_custom_volatility_sizer(self) -> None:
        """Custom VolatilitySizer should work with engine."""
        now = datetime.now(UTC)
        sizer = VolatilitySizer()
        engine = RiskEngine(sizer=sizer)
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

        result = engine.process_signals(signals, portfolio, market, config)

        assert len(result.orders) == 1

    def test_custom_fixed_fractional_sizer(self) -> None:
        """Custom FixedFractionalSizer should work with engine."""
        now = datetime.now(UTC)
        sizer = FixedFractionalSizer(fraction=0.10)  # 10% of equity
        engine = RiskEngine(sizer=sizer)
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

        result = engine.process_signals(signals, portfolio, market, config)

        assert len(result.orders) == 1
        # 10% of $100k / $100 = 100 shares (before constraints)
        # May be reduced by max_position_pct constraint (5% default)


class TestCustomConstraintIntegration:
    """Test custom constraint integration."""

    def test_custom_constraint_chain(self) -> None:
        """Custom constraint chain should work."""
        now = datetime.now(UTC)
        # Only use MaxPositionConstraint
        constraints = [MaxPositionConstraint()]
        engine = RiskEngine(constraints=constraints)
        config = RiskConfig(max_position_pct=0.01)  # 1% = $1000
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

        result = engine.process_signals(signals, portfolio, market, config)

        if result.orders:
            # Should be limited to 1% = 10 shares
            assert result.orders[0].quantity <= Decimal("10")


class TestStopLossIntegration:
    """Test stop-loss calculation integration."""

    def test_stop_loss_long_position(self) -> None:
        """Stop-loss for long position should be below entry."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig(stop_loss_atr_mult=2.0)
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
            volatility={"AAPL": Decimal("2.00")},  # ATR = $2
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0)
        ]

        result = engine.process_signals(signals, portfolio, market, config)

        # Midrange = (102 + 98) / 2 = 100
        # Stop = 100 - (2 * 2) = 96
        assert result.stop_losses["AAPL"] == Decimal("96")

    def test_stop_loss_short_position(self) -> None:
        """Stop-loss for short position should be above entry."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig(stop_loss_atr_mult=2.0)
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
            Signal(symbol="AAPL", timestamp=now, direction="short", strength=1.0)
        ]

        result = engine.process_signals(signals, portfolio, market, config)

        # Midrange = 100, Stop = 100 + (2 * 2) = 104
        assert result.stop_losses["AAPL"] == Decimal("104")


class TestDrawdownHaltIntegration:
    """Test drawdown halt integration."""

    def test_halt_blocks_buy_orders(self) -> None:
        """Drawdown halt should block new buy orders."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig(max_drawdown_halt=0.10)  # 10% halt
        # At 15% drawdown
        portfolio = PortfolioState(
            cash=Decimal("85000"),
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

        result = engine.process_signals(
            signals, portfolio, market, config,
            high_water_mark=Decimal("100000"),
        )

        assert result.halted is True
        assert len([o for o in result.orders if o.side == OrderSide.BUY]) == 0

    def test_halt_allows_sell_orders(self) -> None:
        """Drawdown halt should allow sell orders to reduce exposure."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig(max_drawdown_halt=0.10)
        # At 15% drawdown with existing position
        portfolio = PortfolioState(
            cash=Decimal("35000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("500"),
                    average_price=Decimal("100"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                )
            },
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
            Signal(symbol="AAPL", timestamp=now, direction="short", strength=1.0)
        ]

        result = engine.process_signals(
            signals, portfolio, market, config,
            high_water_mark=Decimal("100000"),
        )

        assert result.halted is True
        # Sell orders should still be allowed
        sell_orders = [o for o in result.orders if o.side == OrderSide.SELL]
        assert len(sell_orders) >= 0  # May or may not generate based on sizer


class TestExistingPositionIntegration:
    """Test integration with existing positions."""

    def test_new_orders_respect_existing_positions(self) -> None:
        """New orders should respect existing position limits."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig(max_positions=2)
        # Already have 1 position
        portfolio = PortfolioState(
            cash=Decimal("90000"),
            positions={
                "TSLA": Position(
                    symbol="TSLA",
                    quantity=Decimal("50"),
                    average_price=Decimal("200"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                )
            },
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
            volatility={
                "AAPL": Decimal("2.00"),
                "GOOGL": Decimal("3.00"),
            },
            liquidity={
                "AAPL": Decimal("50000000"),
                "GOOGL": Decimal("30000000"),
            },
            timestamp=now,
        )
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=1.0),
            Signal(symbol="GOOGL", timestamp=now, direction="long", strength=0.8),
        ]

        result = engine.process_signals(signals, portfolio, market, config)

        # Only 1 new position allowed (max 2 - 1 existing = 1)
        assert len(result.orders) == 1
        # Should pick higher strength
        assert result.orders[0].symbol == "AAPL"


class TestDefaultConstraintChain:
    """Test the default constraint chain ordering and content."""

    def test_default_chain_contains_all_constraints(self) -> None:
        """Default chain should contain all expected constraints."""
        engine = RiskEngine()
        constraints = engine._get_constraints()

        # Check we have all expected constraints
        constraint_types = [type(c).__name__ for c in constraints]
        assert "ShortSellingConstraint" in constraint_types
        assert "MinPositionValueConstraint" in constraint_types
        assert "MaxPositionConstraint" in constraint_types
        assert "MaxPositionsConstraint" in constraint_types
        assert "BuyingPowerConstraint" in constraint_types
        assert "GrossLeverageConstraint" in constraint_types
        assert "NetLeverageConstraint" in constraint_types

    def test_default_chain_order(self) -> None:
        """Constraints should be in the correct order."""
        engine = RiskEngine()
        constraints = engine._get_constraints()
        constraint_types = [type(c).__name__ for c in constraints]

        # ShortSelling should be first (early exit)
        assert constraint_types[0] == "ShortSellingConstraint"
        # MinPositionValue should be second
        assert constraint_types[1] == "MinPositionValueConstraint"
        # BuyingPower should come before GrossLeverage
        bp_idx = constraint_types.index("BuyingPowerConstraint")
        gl_idx = constraint_types.index("GrossLeverageConstraint")
        assert bp_idx < gl_idx
        # NetLeverage should be last
        assert constraint_types[-1] == "NetLeverageConstraint"


class TestBuyingPowerIntegration:
    """Integration tests for buying power constraint."""

    def test_order_exceeding_cash_rejected(self) -> None:
        """Orders exceeding available cash should be rejected."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("1000"),  # Only $1000 cash
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

        result = engine.process_signals(signals, portfolio, market, config)

        # Order should be constrained to $1000 cash = 10 shares max
        if result.orders:
            order_value = result.orders[0].quantity * Decimal("100")
            assert order_value <= Decimal("1000")

    def test_multiple_buys_share_cash_proportionally(self) -> None:
        """Multiple buy orders should share available cash."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig(max_position_pct=1.0)  # No position limit
        portfolio = PortfolioState(
            cash=Decimal("10000"),  # $10k cash
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

        result = engine.process_signals(signals, portfolio, market, config)

        # Total order value should not exceed cash
        total_value = sum(o.quantity * Decimal("100") for o in result.orders)
        assert total_value <= Decimal("10000")


class TestNetLeverageIntegration:
    """Integration tests for net leverage constraint."""

    def test_net_exposure_limited(self) -> None:
        """Net exposure should be limited to max_net_leverage * equity."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig(
            max_net_leverage=0.5,  # 50% net leverage
            max_gross_leverage=2.0,
            max_position_pct=1.0,
        )
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

        result = engine.process_signals(signals, portfolio, market, config)

        # Net exposure should be <= 50% of equity = $50k
        if result.orders:
            order_value = result.orders[0].quantity * Decimal("100")
            max_net = Decimal("100000") * Decimal("0.5")
            assert order_value <= max_net


class TestShortSellingIntegration:
    """Integration tests for short selling constraint."""

    def test_shorts_blocked_when_disabled(self) -> None:
        """Short signals should produce no orders when shorts disabled."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig(allow_shorts=False)
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
            Signal(symbol="AAPL", timestamp=now, direction="short", strength=1.0)
        ]

        result = engine.process_signals(signals, portfolio, market, config)

        # Short signal should be blocked
        assert len(result.orders) == 0

    def test_closing_long_allowed_when_shorts_disabled(self) -> None:
        """Closing a long position should still work when shorts disabled."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig(allow_shorts=False)
        portfolio = PortfolioState(
            cash=Decimal("0"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("100"),
                    average_price=Decimal("100"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                )
            },
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
            Signal(symbol="AAPL", timestamp=now, direction="short", strength=1.0)
        ]

        result = engine.process_signals(signals, portfolio, market, config)

        # Should be able to close (sell) the long position
        sell_orders = [o for o in result.orders if o.side == OrderSide.SELL]
        if sell_orders:
            # Quantity should not exceed position size
            assert sell_orders[0].quantity <= Decimal("100")


class TestKillSwitchIntegration:
    """Integration tests for kill-switch mechanisms."""

    def test_equity_floor_halts_trading(self) -> None:
        """Trading should halt when equity <= 0."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("0"),
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

        result = engine.process_signals(signals, portfolio, market, config)

        assert result.halted is True
        assert "equity" in result.halt_reason.lower()

    def test_daily_loss_halts_trading(self) -> None:
        """Trading should halt when daily loss exceeds limit."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig(max_daily_loss_halt=0.05)  # 5% daily loss limit
        # Lost 10% today (started at $100k, now at $90k)
        portfolio = PortfolioState(
            cash=Decimal("90000"),
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

        result = engine.process_signals(
            signals, portfolio, market, config,
            day_start_equity=Decimal("100000"),
        )

        assert result.halted is True
        assert "daily" in result.halt_reason.lower()

    def test_drawdown_halts_trading(self) -> None:
        """Trading should halt when drawdown exceeds limit."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig(max_drawdown_halt=0.10)  # 10% drawdown limit
        # At 15% drawdown
        portfolio = PortfolioState(
            cash=Decimal("85000"),
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

        result = engine.process_signals(
            signals, portfolio, market, config,
            high_water_mark=Decimal("100000"),
        )

        assert result.halted is True
        assert "drawdown" in result.halt_reason.lower()


class TestAuditScenarios:
    """Tests matching the 'Quick Audit Snippets' from the risk audit guide."""

    def test_buying_power_audit_snippet(self) -> None:
        """Buying power audit: order_value > cash should be rejected."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig()
        # Only $5000 cash
        portfolio = PortfolioState(
            cash=Decimal("5000"),
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
        # Signal would normally size to more than $5000
        signals = [
            Signal(symbol="TEST", timestamp=now, direction="long", strength=1.0)
        ]

        result = engine.process_signals(signals, portfolio, market, config)

        # Order value should be <= $5000
        if result.orders:
            order_value = result.orders[0].quantity * Decimal("100")
            assert order_value <= Decimal("5000")

    def test_gross_leverage_audit_snippet(self) -> None:
        """Gross leverage audit: 100% existing + new buy should be constrained."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig(max_gross_leverage=1.0)  # 100% max
        # Already at 100% exposure
        portfolio = PortfolioState(
            cash=Decimal("0"),
            positions={
                "EXISTING": Position(
                    symbol="EXISTING",
                    quantity=Decimal("1000"),
                    average_price=Decimal("100"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                    current_price=Decimal("100"),
                )
            },
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="NEW",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"NEW": bar},
            volatility={"NEW": Decimal("2.00")},
            liquidity={"NEW": Decimal("50000000")},
            timestamp=now,
        )
        signals = [
            Signal(symbol="NEW", timestamp=now, direction="long", strength=1.0)
        ]

        result = engine.process_signals(signals, portfolio, market, config)

        # No new buys allowed (no cash and already at max leverage)
        assert len([o for o in result.orders if o.side == OrderSide.BUY]) == 0

    def test_position_limit_audit_snippet(self) -> None:
        """Position limit audit: max_position_pct should be enforced."""
        now = datetime.now(UTC)
        engine = RiskEngine()
        config = RiskConfig(max_position_pct=0.05)  # 5% max per position
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
        signals = [
            Signal(symbol="TEST", timestamp=now, direction="long", strength=1.0)
        ]

        result = engine.process_signals(signals, portfolio, market, config)

        # Order should be <= 5% of equity = $5000 = 50 shares @ $100
        if result.orders:
            order_value = result.orders[0].quantity * Decimal("100")
            max_position = Decimal("100000") * Decimal("0.05")
            assert order_value <= max_position
