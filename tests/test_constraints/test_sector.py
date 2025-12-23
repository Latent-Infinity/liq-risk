"""Tests for SectorExposureConstraint.

Following TDD: RED phase - write failing tests first.

SectorExposureConstraint: Limits exposure to any single sector.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState, Position

from liq.risk import MarketState, RiskConfig
from liq.risk.protocols import Constraint


class TestSectorExposureConstraintProtocol:
    """Test SectorExposureConstraint conforms to Constraint protocol."""

    def test_conforms_to_protocol(self) -> None:
        """SectorExposureConstraint should implement Constraint protocol."""
        from liq.risk.constraints import SectorExposureConstraint

        constraint = SectorExposureConstraint()
        assert isinstance(constraint, Constraint)


class TestSectorExposureConstraintBasic:
    """Basic functionality tests."""

    def test_empty_orders_returns_empty(self) -> None:
        """Empty orders should return empty."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
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

    def test_orders_pass_when_no_sector_map(self) -> None:
        """Orders should pass through when no sector map provided."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
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
            sector_map=None,  # No sector map
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

        # Should pass through unchanged
        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("100")

    def test_sell_orders_always_pass(self) -> None:
        """Sell orders should always pass (reduce exposure)."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
        config = RiskConfig(max_sector_pct=0.10)  # Very low limit
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
            sector_map={"AAPL": "Technology"},
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

        # Sell order should pass even though sector is already over limit
        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("100")


class TestSectorExposureConstraintLimit:
    """Test sector exposure limit enforcement."""

    def test_order_within_sector_limit_passes(self) -> None:
        """Order within sector limit should pass unchanged."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
        config = RiskConfig(max_sector_pct=0.30)  # 30% limit
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
            sector_map={"AAPL": "Technology"},
            timestamp=now,
        )
        # Order for $20,000 (20% of $100k equity) - within 30% limit
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("200"),  # 200 * $100 = $20,000
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("200")

    def test_order_exceeding_sector_limit_scaled(self) -> None:
        """Order exceeding sector limit should be scaled down."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
        config = RiskConfig(max_sector_pct=0.20)  # 20% limit = $20,000
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
            sector_map={"AAPL": "Technology"},
            timestamp=now,
        )
        # Order for $50,000 (50% of equity) - exceeds 20% limit
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("500"),  # 500 * $100 = $50,000
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        # Should be scaled to 200 shares ($20,000 / $100)
        assert constraint_result.orders[0].quantity == Decimal("200")

    def test_existing_sector_exposure_considered(self) -> None:
        """Existing positions in sector should be considered."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
        config = RiskConfig(max_sector_pct=0.30)  # 30% limit
        # Already have $20k in Technology (AAPL position)
        portfolio = PortfolioState(
            cash=Decimal("80000"),
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
            volatility={"AAPL": Decimal("2.00"), "MSFT": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000"), "MSFT": Decimal("40000000")},
            sector_map={"AAPL": "Technology", "MSFT": "Technology"},
            timestamp=now,
        )
        # Equity = $80k + $20k = $100k
        # 30% limit = $30k, already have $20k, so $10k remaining
        # Order for $20,000 in same sector (Technology)
        orders = [
            OrderRequest(
                symbol="MSFT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("200"),  # 200 * $100 = $20,000
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        # Should be scaled to 100 shares ($10,000 remaining / $100)
        assert constraint_result.orders[0].quantity == Decimal("100")

    def test_sector_at_limit_rejects_new_orders(self) -> None:
        """New buy orders should be rejected when sector at limit."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
        config = RiskConfig(max_sector_pct=0.20)  # 20% limit
        # Already at 20% in Technology
        portfolio = PortfolioState(
            cash=Decimal("80000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("200"),  # $20k = 20%
                    average_price=Decimal("100"),
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
            volatility={"AAPL": Decimal("2.00"), "MSFT": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000"), "MSFT": Decimal("40000000")},
            sector_map={"AAPL": "Technology", "MSFT": "Technology"},
            timestamp=now,
        )
        orders = [
            OrderRequest(
                symbol="MSFT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        # Order should be rejected (no room in sector)
        assert len(constraint_result.orders) == 0


class TestSectorExposureConstraintMultipleSectors:
    """Test handling of multiple sectors."""

    def test_different_sectors_independent(self) -> None:
        """Different sectors should be tracked independently."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
        config = RiskConfig(max_sector_pct=0.20)  # 20% limit per sector
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
            "XOM": Bar(
                timestamp=now,
                symbol="XOM",
                open=Decimal("100"),
                high=Decimal("102"),
                low=Decimal("98"),
                close=Decimal("100"),
                volume=Decimal("500000"),
            ),
        }
        market = MarketState(
            current_bars=bars,
            volatility={"AAPL": Decimal("2.00"), "XOM": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000"), "XOM": Decimal("30000000")},
            sector_map={"AAPL": "Technology", "XOM": "Energy"},
            timestamp=now,
        )
        # Order for 20% in each sector
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("200"),  # $20k = 20%
                timestamp=now,
            ),
            OrderRequest(
                symbol="XOM",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("200"),  # $20k = 20%
                timestamp=now,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        # Both should pass (different sectors)
        assert len(constraint_result.orders) == 2
        assert constraint_result.orders[0].quantity == Decimal("200")
        assert constraint_result.orders[1].quantity == Decimal("200")

    def test_multiple_orders_same_sector_cumulative(self) -> None:
        """Multiple orders in same sector should be tracked cumulatively."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
        config = RiskConfig(max_sector_pct=0.30)  # 30% limit = $30k
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
            volatility={"AAPL": Decimal("2.00"), "MSFT": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000"), "MSFT": Decimal("40000000")},
            sector_map={"AAPL": "Technology", "MSFT": "Technology"},
            timestamp=now,
        )
        # Two orders in same sector: AAPL $20k + MSFT $20k = $40k > 30% limit
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("200"),  # $20k
                timestamp=now,
            ),
            OrderRequest(
                symbol="MSFT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("200"),  # $20k
                timestamp=now,
            ),
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        # First order should pass, second should be scaled
        assert len(constraint_result.orders) == 2
        assert constraint_result.orders[0].quantity == Decimal("200")  # AAPL full
        assert constraint_result.orders[1].quantity == Decimal("100")  # MSFT scaled to $10k remaining


class TestSectorExposureConstraintEdgeCases:
    """Edge case tests."""

    def test_unknown_sector_passes_through(self) -> None:
        """Symbols not in sector map should pass through."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
        config = RiskConfig(max_sector_pct=0.10)
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        bar = Bar(
            timestamp=now,
            symbol="UNKNOWN",
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        market = MarketState(
            current_bars={"UNKNOWN": bar},
            volatility={"UNKNOWN": Decimal("2.00")},
            liquidity={"UNKNOWN": Decimal("50000000")},
            sector_map={"AAPL": "Technology"},  # UNKNOWN not in map
            timestamp=now,
        )
        orders = [
            OrderRequest(
                symbol="UNKNOWN",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("500"),
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        # Should pass through (unknown sector)
        assert len(constraint_result.orders) == 1
        assert constraint_result.orders[0].quantity == Decimal("500")

    def test_missing_bar_data_skips_order(self) -> None:
        """Orders without bar data should be skipped."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
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
            sector_map={"AAPL": "Technology"},
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

        # Order skipped due to missing bar data
        assert len(constraint_result.orders) == 0

    def test_order_scaled_to_zero_filtered(self) -> None:
        """Orders scaled to < 1 share should be filtered out."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
        config = RiskConfig(max_sector_pct=0.001)  # 0.1% limit = $100
        portfolio = PortfolioState(
            cash=Decimal("100000"),
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
            sector_map={"AAPL": "Technology"},
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

        # $100 limit / $500 price = 0.2 shares → filtered
        assert len(constraint_result.orders) == 0

    def test_existing_position_uses_current_price(self) -> None:
        """Existing position value should use current market price."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
        config = RiskConfig(max_sector_pct=0.30)  # 30% limit
        # Position bought at $50, now worth $100 per share
        # Note: PortfolioState.equity uses Position.market_value which uses average_price
        # So equity = $80k + (200 * $50) = $90k
        # But sector exposure uses current bar price: 200 * $100 = $20k
        portfolio = PortfolioState(
            cash=Decimal("80000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("200"),
                    average_price=Decimal("50"),  # Bought at $50
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
                close=Decimal("100"),  # Now at $100
                volume=Decimal("1000000"),
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
            volatility={"AAPL": Decimal("2.00"), "MSFT": Decimal("2.00")},
            liquidity={"AAPL": Decimal("50000000"), "MSFT": Decimal("40000000")},
            sector_map={"AAPL": "Technology", "MSFT": "Technology"},
            timestamp=now,
        )
        # Equity = $80k + $10k (position at avg price) = $90k
        # 30% limit = $27k max sector
        # Current sector exposure = 200 * $100 (bar price) = $20k
        # Remaining = $27k - $20k = $7k
        orders = [
            OrderRequest(
                symbol="MSFT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("200"),  # $20k requested
                timestamp=now,
            )
        ]

        constraint_result = constraint.apply(orders, portfolio, market, config)

        assert len(constraint_result.orders) == 1
        # Should be scaled to $7k remaining / $100 = 70 shares
        assert constraint_result.orders[0].quantity == Decimal("70")


class TestSectorExposureConstraintClassifyRisk:
    """Tests for SectorExposureConstraint classify_risk method."""

    def test_buy_no_position_is_risk_increasing(self) -> None:
        """Buying with no existing position increases risk."""
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
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
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
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
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
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
        from liq.risk.constraints import SectorExposureConstraint

        now = datetime.now(UTC)
        constraint = SectorExposureConstraint()
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
