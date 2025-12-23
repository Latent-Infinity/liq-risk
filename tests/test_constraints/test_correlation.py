"""Tests for CorrelationConstraint.

Following TDD: RED phase - write failing tests first.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState, Position, TimeInForce

from liq.risk import MarketState, RiskConfig
from liq.risk.constraints import CorrelationConstraint


class TestCorrelationConstraintProtocol:
    """Tests for CorrelationConstraint protocol compliance."""

    def test_conforms_to_constraint_protocol(self) -> None:
        """CorrelationConstraint conforms to Constraint protocol."""
        from liq.risk.protocols import Constraint

        constraint = CorrelationConstraint()
        assert hasattr(constraint, "apply")
        assert isinstance(constraint, Constraint)


class TestCorrelationConstraintBasic:
    """Basic tests for CorrelationConstraint behavior."""

    def test_empty_orders_returns_empty(self) -> None:
        """Empty order list returns empty list."""
        constraint = CorrelationConstraint()
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

        constraint_result = constraint.apply([], portfolio, market, config)
        assert constraint_result.orders == []

    def test_no_correlations_passes_all_orders(self) -> None:
        """Without correlation data, all orders pass through."""
        constraint = CorrelationConstraint()
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
            correlations=None,  # No correlation data
            timestamp=now,
        )
        config = RiskConfig(max_correlation=0.7)

        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
            time_in_force=TimeInForce.DAY,
            timestamp=now,
        )

        constraint_result = constraint.apply([order], portfolio, market, config)
        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].symbol == "AAPL"

    def test_no_max_correlation_config_passes_all(self) -> None:
        """Without max_correlation config, all orders pass through."""
        constraint = CorrelationConstraint()
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
        # Correlation matrix where AAPL and MSFT are highly correlated
        correlations = {
            ("AAPL", "MSFT"): 0.95,
            ("MSFT", "AAPL"): 0.95,
        }
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2")},
            liquidity={"AAPL": Decimal("1000000")},
            correlations=correlations,
            timestamp=now,
        )
        config = RiskConfig(max_correlation=None)  # No limit

        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
            time_in_force=TimeInForce.DAY,
            timestamp=now,
        )

        constraint_result = constraint.apply([order], portfolio, market, config)
        assert len(constraint_result.orders) == 1

    def test_sell_orders_always_pass(self) -> None:
        """Sell orders should always pass (reduce exposure)."""
        constraint = CorrelationConstraint()
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
        # Very high correlation
        correlations = {
            ("AAPL", "MSFT"): 0.99,
            ("MSFT", "AAPL"): 0.99,
        }
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2")},
            liquidity={"AAPL": Decimal("1000000")},
            correlations=correlations,
            timestamp=now,
        )
        config = RiskConfig(max_correlation=0.5)  # Low threshold

        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
            time_in_force=TimeInForce.DAY,
            timestamp=now,
        )

        constraint_result = constraint.apply([order], portfolio, market, config)
        assert len(constraint_result.orders) == 1


class TestCorrelationConstraintFiltering:
    """Tests for correlation-based filtering."""

    def test_single_order_no_existing_positions_passes(self) -> None:
        """Single order with no existing positions passes."""
        constraint = CorrelationConstraint()
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
        correlations = {
            ("AAPL", "MSFT"): 0.95,
            ("MSFT", "AAPL"): 0.95,
        }
        market = MarketState(
            current_bars={"AAPL": bar},
            volatility={"AAPL": Decimal("2")},
            liquidity={"AAPL": Decimal("1000000")},
            correlations=correlations,
            timestamp=now,
        )
        config = RiskConfig(max_correlation=0.7)

        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
            time_in_force=TimeInForce.DAY,
            timestamp=now,
        )

        constraint_result = constraint.apply([order], portfolio, market, config)
        assert len(constraint_result.orders) == 1

    def test_rejects_order_highly_correlated_to_existing_position(self) -> None:
        """Order rejected if highly correlated to existing position."""
        from liq.core import Position

        constraint = CorrelationConstraint()
        now = datetime.now(UTC)

        bar_aapl = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        bar_msft = Bar(
            timestamp=now,
            symbol="MSFT",
            open=Decimal("300"),
            high=Decimal("310"),
            low=Decimal("290"),
            close=Decimal("300"),
            volume=Decimal("1000000"),
        )

        # Existing MSFT position
        position = Position(
            symbol="MSFT",
            quantity=Decimal("100"),
            average_price=Decimal("300"),
            realized_pnl=Decimal("0"),
            timestamp=now,
        )

        portfolio = PortfolioState(
            cash=Decimal("70000"),
            positions={"MSFT": position},
            timestamp=now,
        )

        # AAPL and MSFT highly correlated
        correlations = {
            ("AAPL", "MSFT"): 0.95,
            ("MSFT", "AAPL"): 0.95,
        }
        market = MarketState(
            current_bars={"AAPL": bar_aapl, "MSFT": bar_msft},
            volatility={"AAPL": Decimal("2"), "MSFT": Decimal("5")},
            liquidity={"AAPL": Decimal("1000000"), "MSFT": Decimal("1000000")},
            correlations=correlations,
            timestamp=now,
        )
        config = RiskConfig(max_correlation=0.7)

        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
            time_in_force=TimeInForce.DAY,
            timestamp=now,
        )

        constraint_result = constraint.apply([order], portfolio, market, config)
        # Should be rejected due to high correlation with MSFT position
        assert len(constraint_result.orders) == 0

    def test_accepts_order_low_correlation_to_existing_position(self) -> None:
        """Order accepted if low correlation to existing positions."""
        from liq.core import Position

        constraint = CorrelationConstraint()
        now = datetime.now(UTC)

        bar_aapl = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        bar_xom = Bar(
            timestamp=now,
            symbol="XOM",
            open=Decimal("80"),
            high=Decimal("82"),
            low=Decimal("78"),
            close=Decimal("80"),
            volume=Decimal("1000000"),
        )

        # Existing XOM (energy) position
        position = Position(
            symbol="XOM",
            quantity=Decimal("100"),
            average_price=Decimal("80"),
            realized_pnl=Decimal("0"),
            timestamp=now,
        )

        portfolio = PortfolioState(
            cash=Decimal("92000"),
            positions={"XOM": position},
            timestamp=now,
        )

        # AAPL and XOM have low correlation (tech vs energy)
        correlations = {
            ("AAPL", "XOM"): 0.3,
            ("XOM", "AAPL"): 0.3,
        }
        market = MarketState(
            current_bars={"AAPL": bar_aapl, "XOM": bar_xom},
            volatility={"AAPL": Decimal("2"), "XOM": Decimal("3")},
            liquidity={"AAPL": Decimal("1000000"), "XOM": Decimal("1000000")},
            correlations=correlations,
            timestamp=now,
        )
        config = RiskConfig(max_correlation=0.7)

        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
            time_in_force=TimeInForce.DAY,
            timestamp=now,
        )

        constraint_result = constraint.apply([order], portfolio, market, config)
        # Should be accepted due to low correlation
        assert len(constraint_result.orders) == 1


class TestCorrelationConstraintMultipleOrders:
    """Tests for multiple orders with correlation filtering."""

    def test_filters_multiple_correlated_orders(self) -> None:
        """Multiple orders - keeps first, rejects subsequent correlated ones."""
        constraint = CorrelationConstraint()
        now = datetime.now(UTC)

        bar_aapl = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        bar_msft = Bar(
            timestamp=now,
            symbol="MSFT",
            open=Decimal("300"),
            high=Decimal("310"),
            low=Decimal("290"),
            close=Decimal("300"),
            volume=Decimal("1000000"),
        )
        bar_googl = Bar(
            timestamp=now,
            symbol="GOOGL",
            open=Decimal("150"),
            high=Decimal("155"),
            low=Decimal("145"),
            close=Decimal("150"),
            volume=Decimal("1000000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )

        # All tech stocks highly correlated
        correlations = {
            ("AAPL", "MSFT"): 0.9,
            ("MSFT", "AAPL"): 0.9,
            ("AAPL", "GOOGL"): 0.85,
            ("GOOGL", "AAPL"): 0.85,
            ("MSFT", "GOOGL"): 0.88,
            ("GOOGL", "MSFT"): 0.88,
        }
        market = MarketState(
            current_bars={"AAPL": bar_aapl, "MSFT": bar_msft, "GOOGL": bar_googl},
            volatility={"AAPL": Decimal("2"), "MSFT": Decimal("5"), "GOOGL": Decimal("4")},
            liquidity={"AAPL": Decimal("1000000"), "MSFT": Decimal("1000000"), "GOOGL": Decimal("1000000")},
            correlations=correlations,
            timestamp=now,
        )
        config = RiskConfig(max_correlation=0.7)

        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                time_in_force=TimeInForce.DAY,
                timestamp=now,
            ),
            OrderRequest(
                symbol="MSFT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("5"),
                time_in_force=TimeInForce.DAY,
                timestamp=now,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("8"),
                time_in_force=TimeInForce.DAY,
                timestamp=now,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)
        # Should keep AAPL, reject MSFT and GOOGL due to high correlation
        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].symbol == "AAPL"

    def test_keeps_uncorrelated_orders(self) -> None:
        """Orders for uncorrelated assets are kept."""
        constraint = CorrelationConstraint()
        now = datetime.now(UTC)

        bar_aapl = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        bar_xom = Bar(
            timestamp=now,
            symbol="XOM",
            open=Decimal("80"),
            high=Decimal("82"),
            low=Decimal("78"),
            close=Decimal("80"),
            volume=Decimal("1000000"),
        )
        bar_gld = Bar(
            timestamp=now,
            symbol="GLD",
            open=Decimal("180"),
            high=Decimal("182"),
            low=Decimal("178"),
            close=Decimal("180"),
            volume=Decimal("500000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )

        # Low correlations across different sectors/asset classes
        correlations = {
            ("AAPL", "XOM"): 0.3,
            ("XOM", "AAPL"): 0.3,
            ("AAPL", "GLD"): -0.1,
            ("GLD", "AAPL"): -0.1,
            ("XOM", "GLD"): 0.2,
            ("GLD", "XOM"): 0.2,
        }
        market = MarketState(
            current_bars={"AAPL": bar_aapl, "XOM": bar_xom, "GLD": bar_gld},
            volatility={"AAPL": Decimal("2"), "XOM": Decimal("3"), "GLD": Decimal("1")},
            liquidity={"AAPL": Decimal("1000000"), "XOM": Decimal("1000000"), "GLD": Decimal("500000")},
            correlations=correlations,
            timestamp=now,
        )
        config = RiskConfig(max_correlation=0.7)

        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                time_in_force=TimeInForce.DAY,
                timestamp=now,
            ),
            OrderRequest(
                symbol="XOM",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("15"),
                time_in_force=TimeInForce.DAY,
                timestamp=now,
            ),
            OrderRequest(
                symbol="GLD",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("5"),
                time_in_force=TimeInForce.DAY,
                timestamp=now,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)
        # All should be kept - diversified portfolio
        assert len(constraint_result.orders) == 3


class TestCorrelationConstraintEdgeCases:
    """Tests for edge cases in correlation constraint."""

    def test_missing_correlation_pair_passes(self) -> None:
        """Orders pass if correlation data is missing for pair."""
        constraint = CorrelationConstraint()
        now = datetime.now(UTC)

        bar_aapl = Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("100"),
            high=Decimal("105"),
            low=Decimal("95"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        bar_xyz = Bar(
            timestamp=now,
            symbol="XYZ",
            open=Decimal("50"),
            high=Decimal("52"),
            low=Decimal("48"),
            close=Decimal("50"),
            volume=Decimal("100000"),
        )

        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )

        # No correlation data for XYZ
        correlations = {
            ("AAPL", "MSFT"): 0.9,
            ("MSFT", "AAPL"): 0.9,
        }
        market = MarketState(
            current_bars={"AAPL": bar_aapl, "XYZ": bar_xyz},
            volatility={"AAPL": Decimal("2"), "XYZ": Decimal("5")},
            liquidity={"AAPL": Decimal("1000000"), "XYZ": Decimal("100000")},
            correlations=correlations,
            timestamp=now,
        )
        config = RiskConfig(max_correlation=0.7)

        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                time_in_force=TimeInForce.DAY,
                timestamp=now,
            ),
            OrderRequest(
                symbol="XYZ",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("20"),
                time_in_force=TimeInForce.DAY,
                timestamp=now,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)
        # Both should pass - XYZ has no correlation data
        assert len(constraint_result.orders) == 2

    def test_negative_correlation_allowed(self) -> None:
        """Negatively correlated assets are always allowed."""
        from liq.core import Position

        constraint = CorrelationConstraint()
        now = datetime.now(UTC)

        bar_spy = Bar(
            timestamp=now,
            symbol="SPY",
            open=Decimal("450"),
            high=Decimal("455"),
            low=Decimal("445"),
            close=Decimal("450"),
            volume=Decimal("10000000"),
        )
        bar_vix = Bar(
            timestamp=now,
            symbol="VIX",
            open=Decimal("20"),
            high=Decimal("22"),
            low=Decimal("18"),
            close=Decimal("20"),
            volume=Decimal("1000000"),
        )

        # Existing SPY position
        position = Position(
            symbol="SPY",
            quantity=Decimal("100"),
            average_price=Decimal("450"),
            realized_pnl=Decimal("0"),
            timestamp=now,
        )

        portfolio = PortfolioState(
            cash=Decimal("55000"),
            positions={"SPY": position},
            timestamp=now,
        )

        # SPY and VIX negatively correlated
        correlations = {
            ("SPY", "VIX"): -0.8,
            ("VIX", "SPY"): -0.8,
        }
        market = MarketState(
            current_bars={"SPY": bar_spy, "VIX": bar_vix},
            volatility={"SPY": Decimal("10"), "VIX": Decimal("5")},
            liquidity={"SPY": Decimal("10000000"), "VIX": Decimal("1000000")},
            correlations=correlations,
            timestamp=now,
        )
        config = RiskConfig(max_correlation=0.7)

        order = OrderRequest(
            symbol="VIX",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("50"),
            time_in_force=TimeInForce.DAY,
            timestamp=now,
        )

        constraint_result = constraint.apply([order], portfolio, market, config)
        # Should be allowed - VIX is negatively correlated to SPY (hedging)
        assert len(constraint_result.orders) == 1


class TestCorrelationConstraintClassifyRisk:
    """Tests for CorrelationConstraint classify_risk method."""

    def test_buy_no_position_is_risk_increasing(self) -> None:
        """Buying with no existing position increases risk."""
        from liq.risk.constraints import CorrelationConstraint

        now = datetime.now(UTC)
        constraint = CorrelationConstraint()
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
        from liq.risk.constraints import CorrelationConstraint

        now = datetime.now(UTC)
        constraint = CorrelationConstraint()
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
        from liq.risk.constraints import CorrelationConstraint

        now = datetime.now(UTC)
        constraint = CorrelationConstraint()
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
        from liq.risk.constraints import CorrelationConstraint

        now = datetime.now(UTC)
        constraint = CorrelationConstraint()
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
