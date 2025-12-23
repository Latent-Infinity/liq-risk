"""Tests for BuyingPowerConstraint.

Following TDD: RED phase - write failing tests first.

BuyingPowerConstraint: Limits buy orders to available cash.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st
from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState, Position

from liq.risk import MarketState, RiskConfig
from liq.risk.protocols import Constraint


class TestBuyingPowerConstraintProtocol:
    """Test that BuyingPowerConstraint conforms to Constraint protocol."""

    def test_conforms_to_protocol(self) -> None:
        """BuyingPowerConstraint must implement Constraint protocol."""
        from liq.risk.constraints import BuyingPowerConstraint

        constraint = BuyingPowerConstraint()
        assert isinstance(constraint, Constraint)


class TestBuyingPowerConstraintBasic:
    """Basic functionality tests for BuyingPowerConstraint."""

    def test_empty_orders_returns_empty(self) -> None:
        """No orders should return no orders."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
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

    def test_order_within_cash_passes(self) -> None:
        """Order within available cash should pass."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
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
        # Order value: 500 * $100 = $50000 (within $100k cash)
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

    def test_order_exceeding_cash_scaled_down(self) -> None:
        """Order exceeding available cash should be scaled down."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("50000"),  # Only $50k cash
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
        # Order value: 1000 * $100 = $100000 (exceeds $50k cash)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1000"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        # Should be scaled to 500 shares ($50k)
        assert constraint_result.orders[0].quantity == Decimal("500")

    def test_sell_order_passes_freely(self) -> None:
        """Sell orders should pass without cash check."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("0"),  # No cash!
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("1000"),
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
        # Sell order should pass even with $0 cash
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("500"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("500")

    def test_multiple_buy_orders_share_cash(self) -> None:
        """Multiple buy orders should share available cash proportionally."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
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
        # Total: 1000 * $100 + 500 * $100 = $150k (exceeds $100k)
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

        # Both should be scaled by same factor (100k / 150k = 0.667)
        assert len(constraint_result.orders) == 2
        total_value = sum(o.quantity * Decimal("100") for o in constraint_result.orders)
        assert total_value <= Decimal("100000")


class TestBuyingPowerConstraintWithFees:
    """Test fee deduction from buying power."""

    def test_commission_reduces_buying_power(self) -> None:
        """Commission should reduce available buying power."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
        # 1% commission
        config = RiskConfig(default_commission_pct=0.01)
        portfolio = PortfolioState(
            cash=Decimal("10000"),  # $10k cash
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
        # Want to buy 100 shares @ $100 = $10000
        # With 1% commission, total cost = $10000 * 1.01 = $10100
        # Should be constrained to fewer shares
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
        # With 1% commission, max = $10000 / 1.01 = ~$9900.99
        # So max shares = 99
        assert constraint_result.orders[0].quantity == Decimal("99")

    def test_slippage_reduces_buying_power(self) -> None:
        """Slippage should reduce available buying power."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
        # 0.5% slippage
        config = RiskConfig(default_slippage_pct=0.005)
        portfolio = PortfolioState(
            cash=Decimal("10000"),
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
        # Want to buy 100 shares @ $100 = $10000
        # With 0.5% slippage, total cost = $10000 * 1.005 = $10050
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
        # Max affordable with 0.5% slippage = $10000 / 1.005 = ~$9950
        # So max shares = 99
        assert constraint_result.orders[0].quantity == Decimal("99")

    def test_combined_fees_and_slippage(self) -> None:
        """Combined commission and slippage should reduce buying power."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
        # 1% commission + 0.5% slippage = 1.5% total cost
        config = RiskConfig(default_commission_pct=0.01, default_slippage_pct=0.005)
        portfolio = PortfolioState(
            cash=Decimal("10000"),
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
        # Max affordable = $10000 / 1.015 = ~$9852
        # Max shares = 98
        assert constraint_result.orders[0].quantity == Decimal("98")


class TestBuyingPowerConstraintEdgeCases:
    """Edge case tests for BuyingPowerConstraint."""

    def test_zero_cash_blocks_all_buys(self) -> None:
        """With zero cash, all buy orders should be blocked."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
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

    def test_missing_bar_data_filters_order(self) -> None:
        """Order without bar data should be filtered."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
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
                quantity=Decimal("100"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert constraint_result.orders == []

    def test_order_scaled_to_zero_filtered(self) -> None:
        """Order scaled to zero quantity should be filtered."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=Decimal("50"),  # Only $50 cash
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),  # $100 per share
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

        # Can't buy even 1 share at $100 with $50
        assert constraint_result.orders == []

    def test_buy_covering_short_uses_cash(self) -> None:
        """Buy order covering a short position still requires cash."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
        config = RiskConfig()
        # Short position: -100 shares
        # Cash includes short proceeds: $100k + $10k = $110k
        portfolio = PortfolioState(
            cash=Decimal("5000"),  # Only $5k cash available
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
        # Want to buy 200 shares to cover short (100) and go long (100)
        # Requires $20k but only have $5k
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
        # Can only buy 50 shares with $5k
        assert constraint_result.orders[0].quantity == Decimal("50")


class TestBuyingPowerConstraintPropertyBased:
    """Property-based tests for BuyingPowerConstraint."""

    @given(
        cash=st.decimals(min_value=100, max_value=1000000, places=2, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_resulting_order_value_never_exceeds_cash(
        self,
        cash: Decimal,
    ) -> None:
        """Resulting order value should never exceed available cash."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
        config = RiskConfig()
        portfolio = PortfolioState(
            cash=cash,
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
        # Try to buy way more than cash
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
            result_value = constraint_result.orders[0].quantity * Decimal("100")
            # Allow small tolerance for rounding
            assert result_value <= cash + Decimal("100")


class TestBuyingPowerConstraintClassifyRisk:
    """Tests for BuyingPowerConstraint classify_risk method."""

    def test_buy_no_position_is_risk_increasing(self) -> None:
        """Buying with no existing position increases risk."""
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
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
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
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
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
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
        from liq.risk.constraints import BuyingPowerConstraint

        now = datetime.now(UTC)
        constraint = BuyingPowerConstraint()
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
