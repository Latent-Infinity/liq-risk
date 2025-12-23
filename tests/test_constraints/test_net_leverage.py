"""Tests for NetLeverageConstraint.

Following TDD: RED phase - write failing tests first.

NetLeverageConstraint: Limits net exposure (longs - shorts) to equity multiple.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st
from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState, Position

from liq.risk import MarketState, RiskConfig
from liq.risk.protocols import Constraint


class TestNetLeverageConstraintProtocol:
    """Test that NetLeverageConstraint conforms to Constraint protocol."""

    def test_conforms_to_protocol(self) -> None:
        """NetLeverageConstraint must implement Constraint protocol."""
        from liq.risk.constraints import NetLeverageConstraint

        constraint = NetLeverageConstraint()
        assert isinstance(constraint, Constraint)


class TestNetLeverageConstraintBasic:
    """Basic functionality tests for NetLeverageConstraint."""

    def test_empty_orders_returns_empty(self) -> None:
        """No orders should return no orders."""
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
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

    def test_order_within_net_leverage_passes(self) -> None:
        """Order within net leverage limit should pass."""
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
        config = RiskConfig(max_net_leverage=1.0)  # 1x net leverage
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
        # Buy 500 shares = $50k net long (0.5x net leverage)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("500"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("500")

    def test_net_long_exceeding_limit_scaled(self) -> None:
        """Net long exceeding limit should be scaled."""
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
        config = RiskConfig(max_net_leverage=1.0)  # 1x = $100k max net
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
        # Buy 1500 shares = $150k net long (1.5x exceeds 1x)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1500"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        # Should be scaled to 1000 shares ($100k = 1x)
        assert constraint_result.orders[0].quantity == Decimal("1000")

    def test_net_short_exceeding_limit_scaled(self) -> None:
        """Net short exceeding limit should be scaled."""
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
        config = RiskConfig(max_net_leverage=1.0)  # 1x = $100k max net
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
        # Short 1500 shares = -$150k net (1.5x exceeds 1x)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("1500"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        # Should be scaled to 1000 shares ($100k = 1x)
        assert constraint_result.orders[0].quantity == Decimal("1000")

    def test_existing_positions_counted(self) -> None:
        """Existing positions should count toward net leverage."""
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
        config = RiskConfig(max_net_leverage=1.0)  # 1x = $100k max net
        # Already $50k long
        portfolio = PortfolioState(
            cash=Decimal("50000"),
            positions={
                "TSLA": Position(
                    symbol="TSLA",
                    quantity=Decimal("250"),  # 250 * $200 = $50k long
                    average_price=Decimal("200"),
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
        # Want to buy $80k more, but only $50k room left (1x - 0.5x)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("800"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        # Should be scaled to 500 shares ($50k remaining)
        assert constraint_result.orders[0].quantity == Decimal("500")

    def test_balanced_long_short_allows_more(self) -> None:
        """Balanced long/short should have room for more positions."""
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
        config = RiskConfig(max_net_leverage=1.0)
        # $50k long TSLA, $50k short MSFT = $0 net
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "TSLA": Position(
                    symbol="TSLA",
                    quantity=Decimal("250"),  # $50k long
                    average_price=Decimal("200"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                ),
                "MSFT": Position(
                    symbol="MSFT",
                    quantity=Decimal("-500"),  # $50k short at $100
                    average_price=Decimal("100"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                ),
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
        bar_msft = Bar(
            timestamp=now,
            symbol="MSFT",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar, "MSFT": bar_msft},
            volatility={"AAPL": Decimal("2.00"), "MSFT": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000"), "MSFT": Decimal("50000000")},
            timestamp=now,
        )
        # Net exposure is 0, so can add up to $100k more
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1000"),  # $100k
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("1000")

    def test_sell_reducing_long_passes(self) -> None:
        """Sell order reducing long position should pass (reduces net exposure)."""
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
        config = RiskConfig(max_net_leverage=0.5)  # Very restrictive
        # Already at max net leverage
        portfolio = PortfolioState(
            cash=Decimal("50000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("500"),  # $50k long = 0.5x
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
        # Sell to reduce long - should pass
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("200"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("200")

    def test_buy_reducing_short_passes(self) -> None:
        """Buy order reducing short position should pass (reduces net exposure)."""
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
        config = RiskConfig(max_net_leverage=0.5)  # Very restrictive
        # Already at max net (short) leverage
        portfolio = PortfolioState(
            cash=Decimal("150000"),  # Includes short proceeds
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("-500"),  # $50k short = 0.5x net short
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
        # Buy to reduce short - should pass
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("200"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("200")


class TestNetLeverageConstraintEdgeCases:
    """Edge case tests for NetLeverageConstraint."""

    def test_missing_bar_data_filters_order(self) -> None:
        """Order without bar data should be filtered."""
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
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

        assert constraint_result.orders == []

    def test_order_scaled_to_zero_filtered(self) -> None:
        """Order scaled to zero should be filtered."""
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
        config = RiskConfig(max_net_leverage=1.0)
        # Already at max net leverage
        portfolio = PortfolioState(
            cash=Decimal("0"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("1000"),  # $100k = 1x
                    average_price=Decimal("100"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                )
            },
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="GOOGL",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"GOOGL": bar},
            volatility={"GOOGL": Decimal("2.00")},
            liquidity={"GOOGL": Decimal("50000000")},
            timestamp=now,
        )
        # Try to add more long exposure - should be blocked
        orders = [
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert constraint_result.orders == []

    def test_multiple_orders_scaled_proportionally(self) -> None:
        """Multiple orders exceeding limit should be scaled proportionally."""
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
        config = RiskConfig(max_net_leverage=1.0)  # $100k max net
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
        # Total long: 1000 * $100 + 500 * $100 = $150k (exceeds 1x)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1000"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("500"),
                timestamp=now,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 2
        total_value = sum(o.quantity * Decimal("100") for o in constraint_result.orders)
        assert total_value <= Decimal("100000")


class TestNetLeverageConstraintPropertyBased:
    """Property-based tests for NetLeverageConstraint."""

    @given(
        max_net_leverage=st.floats(min_value=0.1, max_value=5.0),
        equity=st.decimals(min_value=10000, max_value=1000000, places=2, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_resulting_net_leverage_never_exceeds_limit(
        self,
        max_net_leverage: float,
        equity: Decimal,
    ) -> None:
        """Resulting net exposure should never exceed limit."""
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
        # Ensure gross leverage >= net leverage
        config = RiskConfig(max_net_leverage=max_net_leverage, max_gross_leverage=max(max_net_leverage, 1.0))
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
                quantity=Decimal("100000"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        if constraint_result.orders:
            max_net_exposure = equity * Decimal(str(max_net_leverage))
            result_exposure = constraint_result.orders[0].quantity * Decimal("100")
            # Allow small tolerance for rounding
            assert result_exposure <= max_net_exposure + Decimal("100")


class TestNetLeverageConstraintClassifyRisk:
    """Tests for NetLeverageConstraint classify_risk method."""

    def test_buy_no_position_is_risk_increasing(self) -> None:
        """Buying with no existing position increases risk."""
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
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
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
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
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
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
        from liq.risk.constraints import NetLeverageConstraint

        now = datetime.now(UTC)
        constraint = NetLeverageConstraint()
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
