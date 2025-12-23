"""Tests for PyramidingConstraint.

Following TDD: Tests for pyramiding (position scaling) limits.

PyramidingConstraint: Limits how many times you can add to a position.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState, Position

from liq.risk import MarketState, RiskConfig
from liq.risk.constraints import PyramidingConstraint, PyramidingState
from liq.risk.protocols import Constraint


class TestPyramidingConstraintProtocol:
    """Test that PyramidingConstraint conforms to Constraint protocol."""

    def test_conforms_to_protocol(self) -> None:
        """PyramidingConstraint must implement Constraint protocol."""
        constraint = PyramidingConstraint()
        assert isinstance(constraint, Constraint)


class TestPyramidingConstraintInit:
    """Test PyramidingConstraint initialization."""

    def test_default_values(self) -> None:
        """Default values should be sensible."""
        constraint = PyramidingConstraint()
        assert constraint.max_pyramid_adds == 3
        assert constraint.max_add_pct == Decimal("0.5")

    def test_custom_values(self) -> None:
        """Custom values should be accepted."""
        constraint = PyramidingConstraint(max_pyramid_adds=5, max_add_pct=0.25)
        assert constraint.max_pyramid_adds == 5
        assert constraint.max_add_pct == Decimal("0.25")

    def test_invalid_max_pyramid_adds(self) -> None:
        """Negative max_pyramid_adds should raise ValueError."""
        with pytest.raises(ValueError, match="max_pyramid_adds must be >= 0"):
            PyramidingConstraint(max_pyramid_adds=-1)

    def test_invalid_max_add_pct_zero(self) -> None:
        """Zero max_add_pct should raise ValueError."""
        with pytest.raises(ValueError, match="max_add_pct must be in"):
            PyramidingConstraint(max_add_pct=0)

    def test_invalid_max_add_pct_over_one(self) -> None:
        """max_add_pct > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="max_add_pct must be in"):
            PyramidingConstraint(max_add_pct=1.5)


class TestPyramidingConstraintBasic:
    """Basic functionality tests for PyramidingConstraint."""

    def test_empty_orders_returns_empty(self) -> None:
        """No orders should return no orders."""
        now = datetime.now(UTC)
        constraint = PyramidingConstraint()
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

        result = constraint.apply([], portfolio, market, config)

        assert result.orders == []
        assert result.rejected == []

    def test_initial_entry_always_passes(self) -> None:
        """Initial position entry should always pass."""
        now = datetime.now(UTC)
        constraint = PyramidingConstraint(max_pyramid_adds=0)  # No adds allowed
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},  # No existing position
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
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result.orders) == 1
        assert result.orders[0].quantity == Decimal("100")

    def test_risk_reducing_order_always_passes(self) -> None:
        """Orders that reduce risk should always pass."""
        now = datetime.now(UTC)
        constraint = PyramidingConstraint(max_pyramid_adds=0)
        config = RiskConfig()
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
        # Sell to close long position
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result.orders) == 1
        assert result.orders[0].quantity == Decimal("50")


class TestPyramidingConstraintAddLimits:
    """Tests for pyramiding add count limits."""

    def test_add_blocked_when_limit_reached(self) -> None:
        """Adding to position blocked when add limit reached."""
        now = datetime.now(UTC)
        # Pre-populate state showing 3 adds already made
        state = {
            "AAPL": PyramidingState(
                add_count=3,
                initial_quantity=Decimal("100"),
                total_added=Decimal("150"),
            )
        }
        constraint = PyramidingConstraint(max_pyramid_adds=3, pyramiding_state=state)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("50000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("250"),  # Initial 100 + 3 adds
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
        # Try to add more
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert result.orders == []
        assert len(result.rejected) == 1
        assert "Pyramiding limit reached" in result.rejected[0].reason

    def test_add_allowed_when_under_limit(self) -> None:
        """Adding to position allowed when under limit."""
        now = datetime.now(UTC)
        # Pre-populate state showing 2 adds (under limit of 3)
        state = {
            "AAPL": PyramidingState(
                add_count=2,
                initial_quantity=Decimal("100"),
                total_added=Decimal("100"),
            )
        }
        constraint = PyramidingConstraint(max_pyramid_adds=3, pyramiding_state=state)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("50000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("200"),
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
        # Try to add (within limits)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),  # 50% of initial
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result.orders) == 1
        assert result.orders[0].quantity == Decimal("50")


class TestPyramidingConstraintAddSize:
    """Tests for pyramiding add size limits."""

    def test_add_scaled_when_exceeds_max_pct(self) -> None:
        """Add scaled down when it exceeds max percentage of initial position."""
        now = datetime.now(UTC)
        # Initial quantity 100, max add 50% = 50 shares max per add
        state = {
            "AAPL": PyramidingState(
                add_count=0,
                initial_quantity=Decimal("100"),
                total_added=Decimal("0"),
            )
        }
        constraint = PyramidingConstraint(max_pyramid_adds=3, max_add_pct=0.5, pyramiding_state=state)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("50000"),
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
        # Try to add 75 shares (exceeds 50% of 100)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("75"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        # Should be scaled to 50 (50% of initial 100)
        assert len(result.orders) == 1
        assert result.orders[0].quantity == Decimal("50")
        assert len(result.rejected) == 1
        assert "Scaled from 75 to 50" in result.rejected[0].reason

    def test_add_within_max_pct_passes_full(self) -> None:
        """Add within max percentage passes unchanged."""
        now = datetime.now(UTC)
        state = {
            "AAPL": PyramidingState(
                add_count=0,
                initial_quantity=Decimal("100"),
                total_added=Decimal("0"),
            )
        }
        constraint = PyramidingConstraint(max_pyramid_adds=3, max_add_pct=0.5, pyramiding_state=state)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("50000"),
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
        # Add exactly 50% of initial
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result.orders) == 1
        assert result.orders[0].quantity == Decimal("50")
        assert result.rejected == []


class TestPyramidingConstraintStateReset:
    """Tests for state reset when position is closed."""

    def test_state_reset_when_position_fully_closed(self) -> None:
        """State should reset when position is fully closed."""
        now = datetime.now(UTC)
        state = {
            "AAPL": PyramidingState(
                add_count=3,
                initial_quantity=Decimal("100"),
                total_added=Decimal("150"),
            )
        }
        constraint = PyramidingConstraint(max_pyramid_adds=3, pyramiding_state=state)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("0"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("250"),
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
        # Close entire position
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("250"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result.orders) == 1
        # State should be reset
        assert "AAPL" not in constraint._state


class TestPyramidingConstraintShortPositions:
    """Tests for pyramiding with short positions."""

    def test_add_to_short_blocked_when_limit_reached(self) -> None:
        """Adding to short position blocked when limit reached."""
        now = datetime.now(UTC)
        state = {
            "AAPL": PyramidingState(
                add_count=3,
                initial_quantity=Decimal("100"),
                total_added=Decimal("150"),
            )
        }
        constraint = PyramidingConstraint(max_pyramid_adds=3, pyramiding_state=state)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("-250"),  # Short position
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
        # Try to sell more (add to short)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert result.orders == []
        assert len(result.rejected) == 1
        assert "Pyramiding limit reached" in result.rejected[0].reason

    def test_covering_short_always_passes(self) -> None:
        """Buying to cover short should always pass."""
        now = datetime.now(UTC)
        state = {
            "AAPL": PyramidingState(
                add_count=3,
                initial_quantity=Decimal("100"),
                total_added=Decimal("150"),
            )
        }
        constraint = PyramidingConstraint(max_pyramid_adds=3, pyramiding_state=state)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("-250"),
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
        # Buy to cover short
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result.orders) == 1
        assert result.orders[0].quantity == Decimal("100")


class TestPyramidingConstraintClassifyRisk:
    """Tests for classify_risk method."""

    def test_buy_no_position_is_risk_increasing(self) -> None:
        """Buying with no position is risk increasing."""
        now = datetime.now(UTC)
        constraint = PyramidingConstraint()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            timestamp=now,
        )

        assert constraint.classify_risk(order, portfolio) is True

    def test_buy_with_long_is_risk_increasing(self) -> None:
        """Buying when already long is risk increasing."""
        now = datetime.now(UTC)
        constraint = PyramidingConstraint()
        portfolio = PortfolioState(
            cash=Decimal("50000"),
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
        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("50"),
            timestamp=now,
        )

        assert constraint.classify_risk(order, portfolio) is True

    def test_buy_covering_short_is_risk_reducing(self) -> None:
        """Buying to cover short is risk reducing."""
        now = datetime.now(UTC)
        constraint = PyramidingConstraint()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("-100"),
                    average_price=Decimal("100"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                )
            },
            timestamp=now,
        )
        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("50"),
            timestamp=now,
        )

        assert constraint.classify_risk(order, portfolio) is False

    def test_sell_closing_long_is_risk_reducing(self) -> None:
        """Selling to close long is risk reducing."""
        now = datetime.now(UTC)
        constraint = PyramidingConstraint()
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
        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("50"),
            timestamp=now,
        )

        assert constraint.classify_risk(order, portfolio) is False


class TestPyramidingConstraintRecordFill:
    """Tests for record_fill method."""

    def test_record_initial_fill(self) -> None:
        """Recording initial fill sets up state correctly."""
        constraint = PyramidingConstraint()
        constraint.record_fill("AAPL", Decimal("100"), is_add=False)

        state = constraint.get_state("AAPL")
        assert state.initial_quantity == Decimal("100")
        assert state.add_count == 0
        assert state.total_added == Decimal("0")

    def test_record_add_fill(self) -> None:
        """Recording add fill updates state correctly."""
        constraint = PyramidingConstraint()
        constraint.record_fill("AAPL", Decimal("100"), is_add=False)  # Initial
        constraint.record_fill("AAPL", Decimal("50"), is_add=True)  # Add

        state = constraint.get_state("AAPL")
        assert state.initial_quantity == Decimal("100")
        assert state.add_count == 1
        assert state.total_added == Decimal("50")

    def test_multiple_adds_tracked(self) -> None:
        """Multiple adds are tracked correctly."""
        constraint = PyramidingConstraint()
        constraint.record_fill("AAPL", Decimal("100"), is_add=False)
        constraint.record_fill("AAPL", Decimal("50"), is_add=True)
        constraint.record_fill("AAPL", Decimal("30"), is_add=True)
        constraint.record_fill("AAPL", Decimal("20"), is_add=True)

        state = constraint.get_state("AAPL")
        assert state.add_count == 3
        assert state.total_added == Decimal("100")


class TestPyramidingConstraintMultipleSymbols:
    """Tests for handling multiple symbols."""

    def test_independent_state_per_symbol(self) -> None:
        """Each symbol has independent pyramiding state."""
        now = datetime.now(UTC)
        state = {
            "AAPL": PyramidingState(
                add_count=3,
                initial_quantity=Decimal("100"),
                total_added=Decimal("150"),
            ),
            # GOOGL has no state - fresh entry
        }
        constraint = PyramidingConstraint(max_pyramid_adds=3, pyramiding_state=state)
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("250"),
                    average_price=Decimal("100"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                ),
            },
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
            open=Decimal("150"),
            high=Decimal("152"),
            low=Decimal("148"),
            close=Decimal("150"),
            volume=Decimal("500000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar_aapl, "GOOGL": bar_googl},
            volatility={"AAPL": Decimal("2.00"), "GOOGL": Decimal("3.00")},
            liquidity={"AAPL": Decimal("50000000"), "GOOGL": Decimal("20000000")},
            timestamp=now,
        )
        orders = [
            # AAPL add should be blocked (limit reached)
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),
                timestamp=now,
            ),
            # GOOGL initial entry should pass
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            ),
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result.orders) == 1
        assert result.orders[0].symbol == "GOOGL"
        assert len(result.rejected) == 1
        assert result.rejected[0].order.symbol == "AAPL"
