"""Tests for GrossLeverageConstraint.

Following TDD: RED phase - write failing tests first.

GrossLeverageConstraint: Limits total gross exposure / equity ratio.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from hypothesis import given, settings
from hypothesis import strategies as st
from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState, Position

from liq.risk import MarketState, RiskConfig
from liq.risk.protocols import Constraint


class TestGrossLeverageConstraintProtocol:
    """Test that GrossLeverageConstraint conforms to Constraint protocol."""

    def test_conforms_to_protocol(self) -> None:
        """GrossLeverageConstraint must implement Constraint protocol."""
        from liq.risk.constraints import GrossLeverageConstraint

        constraint = GrossLeverageConstraint()
        assert isinstance(constraint, Constraint)


class TestGrossLeverageConstraintBasic:
    """Basic functionality tests for GrossLeverageConstraint."""

    def test_empty_orders_returns_empty(self) -> None:
        """No orders should return no orders."""
        from liq.risk.constraints import GrossLeverageConstraint

        now = datetime.now(UTC)
        constraint = GrossLeverageConstraint()
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

    def test_order_within_leverage_limit_passes(self) -> None:
        """Order within leverage limit should pass."""
        from liq.risk.constraints import GrossLeverageConstraint

        now = datetime.now(UTC)
        constraint = GrossLeverageConstraint()
        config = RiskConfig(max_gross_leverage=1.0)  # 1x leverage
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
        # Order value: 500 * $100 = $50000 = 0.5x (within 1x limit)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("500"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result) == 1
        assert result[0].quantity == Decimal("500")

    def test_order_exceeding_leverage_scaled_down(self) -> None:
        """Order exceeding leverage limit should be scaled down."""
        from liq.risk.constraints import GrossLeverageConstraint

        now = datetime.now(UTC)
        constraint = GrossLeverageConstraint()
        config = RiskConfig(max_gross_leverage=1.0)  # 1x leverage = $100k max
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
        # Order value: 1500 * $100 = $150000 = 1.5x (exceeds 1x limit)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1500"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result) == 1
        # Should be scaled to 1000 shares ($100k = 1x)
        assert result[0].quantity == Decimal("1000")

    def test_existing_positions_counted(self) -> None:
        """Existing positions should count toward leverage."""
        from liq.risk.constraints import GrossLeverageConstraint

        now = datetime.now(UTC)
        constraint = GrossLeverageConstraint()
        config = RiskConfig(max_gross_leverage=1.0)  # 1x leverage
        portfolio = PortfolioState(
            cash=Decimal("50000"),
            positions={
                "TSLA": Position(
                    symbol="TSLA",
                    quantity=Decimal("250"),
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
        # Want to buy $80k more, but only $50k room left (1x - 0.5x existing)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("800"),  # 800 * $100 = $80000
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result) == 1
        # Should be scaled to 500 shares ($50k)
        assert result[0].quantity == Decimal("500")

    def test_short_positions_count_gross(self) -> None:
        """Short positions should count toward gross leverage (absolute)."""
        from liq.risk.constraints import GrossLeverageConstraint

        now = datetime.now(UTC)
        constraint = GrossLeverageConstraint()
        config = RiskConfig(max_gross_leverage=1.0)
        # Equity = cash + market_value = $50k + $50k = $100k
        # With short position, market_value is abs(quantity) * price
        portfolio = PortfolioState(
            cash=Decimal("50000"),
            positions={
                "TSLA": Position(
                    symbol="TSLA",
                    quantity=Decimal("-250"),  # Short position, market_value = $50k
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
        # Equity = $100k, max_exposure = $100k (1x)
        # Current gross exposure = |short| = $50k
        # Remaining capacity = $100k - $50k = $50k
        # Order for 800 * $100 = $80k exceeds $50k remaining
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("800"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result) == 1
        # Should be scaled to 500 shares ($50k remaining)
        assert result[0].quantity == Decimal("500")

    def test_multiple_orders_proportionally_scaled(self) -> None:
        """Multiple orders should be proportionally scaled."""
        from liq.risk.constraints import GrossLeverageConstraint

        now = datetime.now(UTC)
        constraint = GrossLeverageConstraint()
        config = RiskConfig(max_gross_leverage=1.0)  # $100k max
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
        # Total: 1000 * $100 + 500 * $100 = $150k = 1.5x (exceeds 1x)
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

        result = constraint.apply(orders, portfolio, market, config)

        # Both should be scaled by same factor (100k / 150k = 0.667)
        assert len(result) == 2
        total_value = sum(o.quantity * Decimal("100") for o in result)
        assert total_value <= Decimal("100000")

    def test_sell_orders_reduce_exposure(self) -> None:
        """Sell orders should reduce gross exposure."""
        from liq.risk.constraints import GrossLeverageConstraint

        now = datetime.now(UTC)
        constraint = GrossLeverageConstraint()
        config = RiskConfig(max_gross_leverage=0.5)  # Very restrictive
        portfolio = PortfolioState(
            cash=Decimal("50000"),
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
        # Sell orders should pass even if over leverage (reducing exposure)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("200"),
                timestamp=now,
            )
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result) == 1
        assert result[0].quantity == Decimal("200")

    def test_order_scaled_to_zero_filtered(self) -> None:
        """Order scaled to zero should be filtered."""
        from liq.risk.constraints import GrossLeverageConstraint

        now = datetime.now(UTC)
        constraint = GrossLeverageConstraint()
        config = RiskConfig(max_gross_leverage=1.0)
        portfolio = PortfolioState(
            cash=Decimal("0"),  # No equity to support positions
            positions={
                "TSLA": Position(
                    symbol="TSLA",
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

        # Already at 1x leverage, no room for more
        assert result == []

    def test_missing_bar_data_filters_order(self) -> None:
        """Order without bar data should be filtered."""
        from liq.risk.constraints import GrossLeverageConstraint

        now = datetime.now(UTC)
        constraint = GrossLeverageConstraint()
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

        result = constraint.apply(orders, portfolio, market, config)

        assert result == []

    def test_higher_leverage_limit_allows_more(self) -> None:
        """Higher leverage limit should allow more exposure."""
        from liq.risk.constraints import GrossLeverageConstraint

        now = datetime.now(UTC)
        constraint = GrossLeverageConstraint()
        config_1x = RiskConfig(max_gross_leverage=1.0)
        config_2x = RiskConfig(max_gross_leverage=2.0)
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
                quantity=Decimal("1500"),
                timestamp=now,
            )
        ]

        result_1x = constraint.apply(orders, portfolio, market, config_1x)
        result_2x = constraint.apply(orders, portfolio, market, config_2x)

        # 2x should allow more
        assert result_2x[0].quantity > result_1x[0].quantity
        assert result_1x[0].quantity == Decimal("1000")  # 1x = $100k
        assert result_2x[0].quantity == Decimal("1500")  # 1.5x within 2x


class TestGrossLeverageConstraintPropertyBased:
    """Property-based tests for GrossLeverageConstraint."""

    @given(
        max_leverage=st.floats(min_value=0.1, max_value=5.0),
        equity=st.decimals(min_value=10000, max_value=1000000, places=2, allow_nan=False, allow_infinity=False),
    )
    @settings(max_examples=50)
    def test_resulting_leverage_never_exceeds_limit(
        self,
        max_leverage: float,
        equity: Decimal,
    ) -> None:
        """Resulting leverage should never exceed limit."""
        from liq.risk.constraints import GrossLeverageConstraint

        now = datetime.now(UTC)
        constraint = GrossLeverageConstraint()
        config = RiskConfig(max_gross_leverage=max_leverage)
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

        result = constraint.apply(orders, portfolio, market, config)

        if result:
            max_exposure = equity * Decimal(str(max_leverage))
            result_exposure = result[0].quantity * Decimal("100")
            # Allow small tolerance for rounding
            assert result_exposure <= max_exposure + Decimal("100")
