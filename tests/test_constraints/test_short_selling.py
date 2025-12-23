"""Tests for ShortSellingConstraint.

Following TDD: RED phase - write failing tests first.

ShortSellingConstraint: Filters short sell orders when shorts are disabled.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState, Position

from liq.risk import MarketState, RiskConfig
from liq.risk.protocols import Constraint


class TestShortSellingConstraintProtocol:
    """Test that ShortSellingConstraint conforms to Constraint protocol."""

    def test_conforms_to_protocol(self) -> None:
        """ShortSellingConstraint must implement Constraint protocol."""
        from liq.risk.constraints import ShortSellingConstraint

        constraint = ShortSellingConstraint()
        assert isinstance(constraint, Constraint)


class TestShortSellingConstraintBasic:
    """Basic functionality tests for ShortSellingConstraint."""

    def test_empty_orders_returns_empty(self) -> None:
        """No orders should return no orders."""
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
        config = RiskConfig(allow_shorts=False)
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

    def test_buy_orders_pass_when_shorts_disabled(self) -> None:
        """Buy orders should always pass regardless of short permission."""
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
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
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("100")

    def test_sell_closing_long_passes_when_shorts_disabled(self) -> None:
        """Sell orders closing a long position pass even when shorts disabled."""
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
        config = RiskConfig(allow_shorts=False)
        portfolio = PortfolioState(
            cash=Decimal("0"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("200"),  # Long 200 shares
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
        # Selling 100 of 200 shares - partial close, not going short
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
        assert constraint_result.orders[0].quantity == Decimal("100")

    def test_short_sell_blocked_when_shorts_disabled(self) -> None:
        """Short sell orders should be blocked when shorts disabled."""
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
        config = RiskConfig(allow_shorts=False)
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},  # No position - selling would go short
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
        # No position, so this would create a short
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

        assert constraint_result.orders == []

    def test_short_sell_allowed_when_shorts_enabled(self) -> None:
        """Short sell orders should pass when shorts enabled."""
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
        config = RiskConfig(allow_shorts=True)  # Shorts allowed (default)
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},  # No position - selling would go short
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
                quantity=Decimal("100"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("100")


class TestShortSellingConstraintPartialClose:
    """Tests for sell orders that would partially close long positions."""

    def test_sell_exceeding_position_trimmed(self) -> None:
        """Sell order exceeding position should be trimmed when shorts disabled."""
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
        config = RiskConfig(allow_shorts=False)
        portfolio = PortfolioState(
            cash=Decimal("0"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("100"),  # Long 100 shares
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
        # Trying to sell 150 shares when only 100 long
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("150"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        # Should be trimmed to 100 (can only close, not go short)
        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("100")

    def test_sell_exceeding_position_full_when_shorts_enabled(self) -> None:
        """Sell order exceeding position should be full when shorts enabled."""
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
        config = RiskConfig(allow_shorts=True)
        portfolio = PortfolioState(
            cash=Decimal("0"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("100"),  # Long 100 shares
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
        # Trying to sell 150 shares - 100 closes, 50 goes short
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("150"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        # Full 150 should pass when shorts enabled
        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("150")


class TestShortSellingConstraintMultipleOrders:
    """Tests with multiple orders."""

    def test_mixed_orders_filtered_correctly(self) -> None:
        """Buy orders pass, short sells filtered when shorts disabled."""
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
        config = RiskConfig(allow_shorts=False)
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("50"),
                    average_price=Decimal("100"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                )
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
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=Decimal("500000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar_aapl, "GOOGL": bar_googl},
            volatility={"AAPL": Decimal("2.00"), "GOOGL": Decimal("3.00")},
            liquidity={"AAPL": Decimal("50000000"), "GOOGL": Decimal("20000000")},
            timestamp=now,
        )
        orders = [
            # Buy order - should pass
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            ),
            # Close long - should pass
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),
                timestamp=now,
            ),
            # Short sell (no position) - should be blocked
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 2
        symbols = {o.symbol for o in constraint_result.orders}
        assert symbols == {"AAPL"}
        # One buy, one sell for AAPL
        sides = {o.side for o in constraint_result.orders}
        assert sides == {OrderSide.BUY, OrderSide.SELL}


class TestShortSellingConstraintExistingShort:
    """Tests when existing short positions exist."""

    def test_adding_to_short_blocked_when_shorts_disabled(self) -> None:
        """Adding to an existing short position should be blocked when shorts disabled."""
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
        config = RiskConfig(allow_shorts=False)
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("-100"),  # Already short 100 shares
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
        # Trying to sell more (add to short)
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

        # Should be blocked - can't add to short
        assert constraint_result.orders == []

    def test_covering_short_allowed_when_shorts_disabled(self) -> None:
        """Buying to cover a short should always be allowed."""
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
        config = RiskConfig(allow_shorts=False)
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("-100"),  # Short 100 shares
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
        # Buying to cover short
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        # Should pass - covering short is always allowed
        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("100")


class TestShortSellingConstraintClassifyRisk:
    """Tests for ShortSellingConstraint classify_risk method."""

    def test_buy_no_position_is_risk_increasing(self) -> None:
        """Buying with no existing position increases risk."""
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
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

    def test_buy_covering_short_is_risk_reducing(self) -> None:
        """Buying to cover a short position reduces risk."""
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("-50"),
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
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
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

    def test_sell_initiating_short_is_risk_increasing(self) -> None:
        """Selling to initiate short increases risk."""
        from liq.risk.constraints import ShortSellingConstraint

        now = datetime.now(UTC)
        constraint = ShortSellingConstraint()
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
