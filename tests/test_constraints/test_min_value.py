"""Tests for MinPositionValueConstraint.

Following TDD: RED phase - write failing tests first.

MinPositionValueConstraint: Filters out orders below minimum notional value.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st
from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState

from liq.risk import MarketState, RiskConfig
from liq.risk.protocols import Constraint


class TestMinPositionValueConstraintProtocol:
    """Test that MinPositionValueConstraint conforms to Constraint protocol."""

    def test_conforms_to_protocol(self) -> None:
        """MinPositionValueConstraint must implement Constraint protocol."""
        from liq.risk.constraints import MinPositionValueConstraint

        constraint = MinPositionValueConstraint()
        assert isinstance(constraint, Constraint)


class TestMinPositionValueConstraintBasic:
    """Basic functionality tests for MinPositionValueConstraint."""

    def test_empty_orders_returns_empty(self) -> None:
        """No orders should return no orders."""
        from liq.risk.constraints import MinPositionValueConstraint

        now = datetime.now(UTC)
        constraint = MinPositionValueConstraint()
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

        assert result == []

    def test_order_above_minimum_passes(self) -> None:
        """Order above minimum value should pass."""
        from liq.risk.constraints import MinPositionValueConstraint

        now = datetime.now(UTC)
        constraint = MinPositionValueConstraint()
        config = RiskConfig(min_position_value=Decimal("100"))
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
        # Order value: 10 * $150 = $1500 (above $100 minimum)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result) == 1
        assert result[0].quantity == Decimal("10")

    def test_order_below_minimum_filtered(self) -> None:
        """Order below minimum value should be filtered."""
        from liq.risk.constraints import MinPositionValueConstraint

        now = datetime.now(UTC)
        constraint = MinPositionValueConstraint()
        config = RiskConfig(min_position_value=Decimal("500"))
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("50"),
            high=Decimal("52"),
            low=Decimal("48"),
            close=Decimal("50"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        # Order value: 5 * $50 = $250 (below $500 minimum)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("5"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert result == []

    def test_order_exactly_at_minimum_passes(self) -> None:
        """Order exactly at minimum value should pass."""
        from liq.risk.constraints import MinPositionValueConstraint

        now = datetime.now(UTC)
        constraint = MinPositionValueConstraint()
        config = RiskConfig(min_position_value=Decimal("500"))
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("50"),
            high=Decimal("52"),
            low=Decimal("48"),
            close=Decimal("50"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        # Order value: 10 * $50 = $500 (exactly at minimum)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result) == 1

    def test_multiple_orders_filtered_independently(self) -> None:
        """Multiple orders should be filtered independently."""
        from liq.risk.constraints import MinPositionValueConstraint

        now = datetime.now(UTC)
        constraint = MinPositionValueConstraint()
        config = RiskConfig(min_position_value=Decimal("200"))
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
        bar_cheap = Bar(
            timestamp=now,
            symbol="CHEAP",
            open=Decimal("10"),
            high=Decimal("11"),
            low=Decimal("9"),
            close=Decimal("10"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar_aapl, "CHEAP": bar_cheap},
            volatility={"AAPL": Decimal("2.00"), "CHEAP": Decimal("1.00")},
            liquidity={"AAPL": Decimal("50000000"), "CHEAP": Decimal("10000000")},
            timestamp=now,
        )
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("5"),  # 5 * $100 = $500 - passes
                timestamp=now,
            ),
            OrderRequest(
                symbol="CHEAP",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),  # 10 * $10 = $100 - filtered
                timestamp=now,
            ),
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result) == 1
        assert result[0].symbol == "AAPL"

    def test_sell_orders_always_pass(self) -> None:
        """Sell orders should always pass (reducing position)."""
        from liq.risk.constraints import MinPositionValueConstraint

        now = datetime.now(UTC)
        constraint = MinPositionValueConstraint()
        config = RiskConfig(min_position_value=Decimal("1000"))
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("50"),
            high=Decimal("52"),
            low=Decimal("48"),
            close=Decimal("50"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        # Sell order value: 1 * $50 = $50 (below minimum but sell)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result) == 1

    def test_missing_bar_data_filters_order(self) -> None:
        """Order without bar data should be filtered."""
        from liq.risk.constraints import MinPositionValueConstraint

        now = datetime.now(UTC)
        constraint = MinPositionValueConstraint()
        config = RiskConfig(min_position_value=Decimal("100"))
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

        result = constraint.apply(orders, portfolio, market, config)

        assert result == []

    def test_zero_minimum_passes_all(self) -> None:
        """Zero minimum should pass all orders."""
        from liq.risk.constraints import MinPositionValueConstraint

        now = datetime.now(UTC)
        constraint = MinPositionValueConstraint()
        config = RiskConfig(min_position_value=Decimal("0"))
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("0.01"),
            high=Decimal("0.02"),
            low=Decimal("0.01"),
            close=Decimal("0.01"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("0.001")},
            liquidity={"AAPL": Decimal("50000000")},
            timestamp=now,
        )
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),  # $0.01 value
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result) == 1


class TestMinPositionValueConstraintPropertyBased:
    """Property-based tests for MinPositionValueConstraint."""

    @given(
        min_value=st.decimals(min_value=0, max_value=10000, places=2, allow_nan=False, allow_infinity=False),
        price=st.decimals(min_value=Decimal("0.01"), max_value=1000, places=2, allow_nan=False, allow_infinity=False),
        quantity=st.integers(min_value=1, max_value=1000),
    )
    @settings(max_examples=50)
    def test_filtered_orders_always_below_minimum(
        self,
        min_value: Decimal,
        price: Decimal,
        quantity: int,
    ) -> None:
        """Filtered buy orders should always be below minimum."""
        from liq.risk.constraints import MinPositionValueConstraint

        now = datetime.now(UTC)
        constraint = MinPositionValueConstraint()
        config = RiskConfig(min_position_value=min_value)
        portfolio = PortfolioState(
            cash=Decimal("100000"),
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
            volatility={"TEST": Decimal("2.00")},
            liquidity={"TEST": Decimal("50000000")},
            timestamp=now,
        )
        orders = [
            OrderRequest(
                symbol="TEST",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal(str(quantity)),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)
        order_value = price * Decimal(str(quantity))

        # If filtered, order value must be below minimum
        if not result:
            assert order_value < min_value
        # If passed, order value must be >= minimum
        else:
            assert order_value >= min_value
