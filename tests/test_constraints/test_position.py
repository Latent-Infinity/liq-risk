"""Tests for position-related constraints.

Following TDD: RED phase - write failing tests first.

MaxPositionConstraint: Limits individual position size as % of equity.
MaxPositionsConstraint: Limits total number of concurrent positions.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st
from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState, Position

from liq.risk import MarketState, RiskConfig
from liq.risk.protocols import Constraint


class TestMaxPositionConstraintProtocol:
    """Test that MaxPositionConstraint conforms to Constraint protocol."""

    def test_conforms_to_protocol(self) -> None:
        """MaxPositionConstraint must implement Constraint protocol."""
        from liq.risk.constraints import MaxPositionConstraint

        constraint = MaxPositionConstraint()
        assert isinstance(constraint, Constraint)


class TestMaxPositionConstraintBasic:
    """Basic functionality tests for MaxPositionConstraint."""

    def test_empty_orders_returns_empty(self) -> None:
        """No orders should return no orders."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
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

        constraint_result = constraint.apply([], portfolio, market, config)

        assert constraint_result.orders == []

    def test_order_within_limit_passes(self) -> None:
        """Order within position limit should pass unchanged."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        config = RiskConfig(max_position_pct=0.05)  # 5% limit
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
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        # Order for $4000 = 4% of equity (within 5% limit)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("40"),  # 40 * $100 = $4000
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("40")

    def test_order_exceeding_limit_reduced(self) -> None:
        """Order exceeding position limit should be reduced."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        config = RiskConfig(max_position_pct=0.05)  # 5% limit = $5000
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
        # Order for $10000 = 10% of equity (exceeds 5% limit)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),  # 100 * $100 = $10000
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        # Should be reduced to 50 shares ($5000 = 5%)
        assert constraint_result.orders[0].quantity == Decimal("50")

    def test_existing_position_considered(self) -> None:
        """Existing position should be considered when checking limit."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        config = RiskConfig(max_position_pct=0.05)  # 5% limit = $5000
        portfolio = PortfolioState(
            cash=Decimal("97000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("30"),  # Already have 30 shares
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
        # Want to buy 50 more shares, but already have 30
        # Total would be 80 * $100 = $8000 = 8% (exceeds 5%)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        # Can only add 20 more shares to reach 50 total ($5000)
        assert constraint_result.orders[0].quantity == Decimal("20")

    def test_order_reduced_to_zero_filtered_out(self) -> None:
        """Order reduced to zero should be filtered out."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        config = RiskConfig(max_position_pct=0.05)  # 5% limit
        portfolio = PortfolioState(
            cash=Decimal("95000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("50"),  # Already at limit
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
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        # Order should be filtered out completely
        assert constraint_result.orders == []

    def test_sell_closing_long_not_limited(self) -> None:
        """Sell orders closing long positions should not be limited."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        config = RiskConfig(max_position_pct=0.05)
        portfolio = PortfolioState(
            cash=Decimal("90000"),
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
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),  # Selling (closing), not buying
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("50")

    def test_sell_initiating_short_constrained(self) -> None:
        """Sell orders initiating short positions should be constrained."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        config = RiskConfig(max_position_pct=0.05)  # 5% limit = $5000
        portfolio = PortfolioState(
            cash=Decimal("100000"),  # No existing positions
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
        # Sell 100 shares = $10000 short (exceeds 5% limit)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        # Should be reduced to 50 shares ($5000 = 5%)
        assert constraint_result.orders[0].quantity == Decimal("50")

    def test_sell_increasing_short_constrained(self) -> None:
        """Sell orders increasing existing short should be constrained."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        config = RiskConfig(max_position_pct=0.05)  # 5% limit = $5000
        # Existing short: -30 shares @ $100 = -$3000 market value
        # Cash: $100k + $3k proceeds = $103k, Equity = $103k - $3k = $100k
        portfolio = PortfolioState(
            cash=Decimal("103000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("-30"),  # Short position
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
        # Current short: $3000 (|-30 * 100|)
        # Max position: $5000
        # Room: $5000 - $3000 = $2000 = 20 shares
        # Want to sell 50 more shares ($5000) - exceeds remaining $2000
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        # Should be reduced to 20 shares ($2000 remaining)
        assert constraint_result.orders[0].quantity == Decimal("20")

    def test_sell_flipping_long_to_short_constrained(self) -> None:
        """Sell order flipping long to short should constrain the short portion."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        config = RiskConfig(max_position_pct=0.05)  # 5% limit = $5000
        # 30 shares long @ $100 = $3000 position
        # Cash: $97k, Equity: $100k
        portfolio = PortfolioState(
            cash=Decimal("97000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("30"),
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
        # Sell 130 shares: first 30 close long, remaining 100 go short ($10k)
        # Short portion: 100 shares = $10000 (exceeds $5000 limit)
        # Maximum allowed: close 30 + short 50 = 80 total
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("130"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        # Should allow: 30 (close long) + 50 (max short) = 80
        assert constraint_result.orders[0].quantity == Decimal("80")

    def test_missing_bar_data_filters_order(self) -> None:
        """Order without bar data should be filtered."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
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
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert constraint_result.orders == []


class TestMaxPositionsConstraintProtocol:
    """Test that MaxPositionsConstraint conforms to Constraint protocol."""

    def test_conforms_to_protocol(self) -> None:
        """MaxPositionsConstraint must implement Constraint protocol."""
        from liq.risk.constraints import MaxPositionsConstraint

        constraint = MaxPositionsConstraint()
        assert isinstance(constraint, Constraint)


class TestMaxPositionsConstraintBasic:
    """Basic functionality tests for MaxPositionsConstraint."""

    def test_empty_orders_returns_empty(self) -> None:
        """No orders should return no orders."""
        from liq.risk.constraints import MaxPositionsConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionsConstraint()
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

        constraint_result = constraint.apply([], portfolio, market, config)

        assert constraint_result.orders == []

    def test_orders_within_limit_pass(self) -> None:
        """Orders within position count limit should pass."""
        from liq.risk.constraints import MaxPositionsConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionsConstraint()
        config = RiskConfig(max_positions=5)
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},  # No existing positions
            timestamp=now,
        )
        market = MarketState(
            current_bars={},
            volatility={},
            liquidity={},
            timestamp=now,
        )
        # 3 new positions (within limit of 5)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("5"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="MSFT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("8"),
                timestamp=now,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 3

    def test_orders_exceeding_limit_filtered(self) -> None:
        """Orders exceeding position count limit should be filtered."""
        from liq.risk.constraints import MaxPositionsConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionsConstraint()
        config = RiskConfig(max_positions=2)
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
        # 3 orders but limit is 2
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                timestamp=now,
                confidence=0.9,  # Highest priority
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("5"),
                timestamp=now,
                confidence=0.7,
            ),
            OrderRequest(
                symbol="MSFT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("8"),
                timestamp=now,
                confidence=0.5,  # Lowest priority - should be dropped
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 2
        symbols = {o.symbol for o in constraint_result.orders}
        assert symbols == {"AAPL", "GOOGL"}

    def test_existing_positions_counted(self) -> None:
        """Existing positions should count toward limit."""
        from liq.risk.constraints import MaxPositionsConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionsConstraint()
        config = RiskConfig(max_positions=3)
        portfolio = PortfolioState(
            cash=Decimal("90000"),
            positions={
                "TSLA": Position(
                    symbol="TSLA",
                    quantity=Decimal("10"),
                    average_price=Decimal("200"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                ),
                "NVDA": Position(
                    symbol="NVDA",
                    quantity=Decimal("5"),
                    average_price=Decimal("400"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                ),
            },
            timestamp=now,
        )
        market = MarketState(
            current_bars={},
            volatility={},
            liquidity={},
            timestamp=now,
        )
        # 2 existing + 2 new = 4 (exceeds 3)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                timestamp=now,
                confidence=0.9,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("5"),
                timestamp=now,
                confidence=0.7,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        # Can only add 1 new position (3 - 2 existing = 1)
        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].symbol == "AAPL"  # Highest confidence

    def test_order_for_existing_position_not_counted_as_new(self) -> None:
        """Order for existing position shouldn't count as new position."""
        from liq.risk.constraints import MaxPositionsConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionsConstraint()
        config = RiskConfig(max_positions=2)
        portfolio = PortfolioState(
            cash=Decimal("95000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("10"),
                    average_price=Decimal("100"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                ),
                "GOOGL": Position(
                    symbol="GOOGL",
                    quantity=Decimal("5"),
                    average_price=Decimal("140"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                ),
            },
            timestamp=now,
        )
        market = MarketState(
            current_bars={},
            volatility={},
            liquidity={},
            timestamp=now,
        )
        # Adding to existing positions (not new)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("5"),
                timestamp=now,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        # Both should pass - adding to existing positions
        assert len(constraint_result.orders) == 2

    def test_sell_closing_long_always_passes(self) -> None:
        """Sell orders closing long positions should always pass."""
        from liq.risk.constraints import MaxPositionsConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionsConstraint()
        config = RiskConfig(max_positions=1)
        portfolio = PortfolioState(
            cash=Decimal("90000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("50"),
                    average_price=Decimal("100"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                ),
                "GOOGL": Position(
                    symbol="GOOGL",
                    quantity=Decimal("30"),
                    average_price=Decimal("140"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                ),
            },
            timestamp=now,
        )
        market = MarketState(
            current_bars={},
            volatility={},
            liquidity={},
            timestamp=now,
        )
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("30"),
                timestamp=now,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        # Both sells closing longs should pass
        assert len(constraint_result.orders) == 2

    def test_sell_initiating_short_counts_as_new_position(self) -> None:
        """Sell orders initiating short should count as new position."""
        from liq.risk.constraints import MaxPositionsConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionsConstraint()
        config = RiskConfig(max_positions=2)
        portfolio = PortfolioState(
            cash=Decimal("90000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("50"),  # Existing long
                    average_price=Decimal("100"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                ),
            },
            timestamp=now,
        )
        market = MarketState(
            current_bars={},
            volatility={},
            liquidity={},
            timestamp=now,
        )
        # Sell orders for symbols without positions create new short positions
        # Only room for 1 new position (2 max - 1 existing)
        orders = [
            OrderRequest(
                symbol="GOOGL",  # New short position
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("30"),
                timestamp=now,
                confidence=0.9,
            ),
            OrderRequest(
                symbol="MSFT",  # Another new short position
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("20"),
                timestamp=now,
                confidence=0.7,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        # Only 1 new position allowed
        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].symbol == "GOOGL"  # Higher confidence

    def test_sell_into_existing_short_not_new_position(self) -> None:
        """Sell orders into existing short should not count as new position."""
        from liq.risk.constraints import MaxPositionsConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionsConstraint()
        config = RiskConfig(max_positions=2)
        portfolio = PortfolioState(
            cash=Decimal("110000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("50"),  # Existing long
                    average_price=Decimal("100"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                ),
                "GOOGL": Position(
                    symbol="GOOGL",
                    quantity=Decimal("-20"),  # Existing short
                    average_price=Decimal("140"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                ),
            },
            timestamp=now,
        )
        market = MarketState(
            current_bars={},
            volatility={},
            liquidity={},
            timestamp=now,
        )
        # Sell more into existing short - should pass (not a new position)
        orders = [
            OrderRequest(
                symbol="GOOGL",  # Adding to existing short
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                timestamp=now,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        # Should pass - adding to existing position
        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].symbol == "GOOGL"

    def test_priority_by_confidence(self) -> None:
        """Orders should be prioritized by confidence when filtering."""
        from liq.risk.constraints import MaxPositionsConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionsConstraint()
        config = RiskConfig(max_positions=2)
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
        orders = [
            OrderRequest(
                symbol="LOW",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                timestamp=now,
                confidence=0.3,
            ),
            OrderRequest(
                symbol="HIGH",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("5"),
                timestamp=now,
                confidence=0.9,
            ),
            OrderRequest(
                symbol="MED",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("8"),
                timestamp=now,
                confidence=0.6,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 2
        symbols = [o.symbol for o in constraint_result.orders]
        # Should keep HIGH and MED (top 2 by confidence)
        assert symbols == ["HIGH", "MED"]


class TestMaxPositionConstraintPropertyBased:
    """Property-based tests for MaxPositionConstraint."""

    @given(
        max_pct=st.floats(min_value=0.01, max_value=1.0),
        equity=st.decimals(min_value=10000, max_value=1000000, places=2, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_resulting_position_never_exceeds_limit(
        self,
        max_pct: float,
        equity: Decimal,
    ) -> None:
        """Resulting position should never exceed limit."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        config = RiskConfig(max_position_pct=max_pct)
        portfolio = PortfolioState(
            cash=equity,
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
        # Try to buy way more than limit
        orders = [
            OrderRequest(
                symbol="TEST",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10000"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        if constraint_result.orders:
            max_value = equity * Decimal(str(max_pct))
            result_value = constraint_result.orders[0].quantity * Decimal("100")
            assert result_value <= max_value + Decimal("1")  # Allow rounding tolerance


class TestMaxPositionConstraintClassifyRisk:
    """Tests for MaxPositionConstraint classify_risk method."""

    def test_buy_no_position_is_risk_increasing(self) -> None:
        """Buying with no existing position increases risk."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
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

    def test_buy_with_long_position_is_risk_increasing(self) -> None:
        """Buying more when already long increases risk."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("50"),  # Already long
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
            quantity=Decimal("100"),
            timestamp=now,
        )

        assert constraint.classify_risk(order, portfolio) is True

    def test_buy_covering_short_is_risk_reducing(self) -> None:
        """Buying to cover a short position reduces risk."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("-50"),  # Short position
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
        """Selling to close a long position reduces risk."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("100"),  # Long position
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

    def test_sell_initiating_short_is_risk_increasing(self) -> None:
        """Selling to initiate short increases risk."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("50"),
            timestamp=now,
        )

        assert constraint.classify_risk(order, portfolio) is True

    def test_sell_increasing_short_is_risk_increasing(self) -> None:
        """Selling more when already short increases risk."""
        from liq.risk.constraints import MaxPositionConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionConstraint()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("-50"),  # Already short
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

        assert constraint.classify_risk(order, portfolio) is True


class TestMaxPositionsConstraintClassifyRisk:
    """Tests for MaxPositionsConstraint classify_risk method."""

    def test_buy_no_position_is_risk_increasing(self) -> None:
        """Buying with no existing position increases risk."""
        from liq.risk.constraints import MaxPositionsConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionsConstraint()
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

    def test_sell_closing_long_is_risk_reducing(self) -> None:
        """Selling to close a long position reduces risk."""
        from liq.risk.constraints import MaxPositionsConstraint

        now = datetime.now(UTC)
        constraint = MaxPositionsConstraint()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
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
