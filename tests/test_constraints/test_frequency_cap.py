"""Tests for FrequencyCapConstraint.

Following TDD: Tests for trade frequency limits.

FrequencyCapConstraint: Limits trade frequency to prevent over-trading.
Supports N trades per Y timeframe (minute, hour, day, etc.).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState, Position

from liq.risk import MarketState, RiskConfig
from liq.risk.constraints import (
    FrequencyCapConfig,
    FrequencyCapConstraint,
    Timeframe,
    create_frequency_cap,
)
from liq.risk.constraints.frequency_cap import TradeRecord
from liq.risk.protocols import Constraint


class TestTimeframeEnum:
    """Tests for Timeframe enum."""

    def test_timeframe_values(self) -> None:
        """Timeframe values should be in seconds."""
        assert Timeframe.SECOND.value == 1
        assert Timeframe.MINUTE.value == 60
        assert Timeframe.HOUR.value == 3600
        assert Timeframe.DAY.value == 86400
        assert Timeframe.WEEK.value == 604800
        assert Timeframe.MONTH.value == 2592000

    def test_from_string_full_names(self) -> None:
        """Full timeframe names should parse correctly."""
        assert Timeframe.from_string("second") == Timeframe.SECOND
        assert Timeframe.from_string("minute") == Timeframe.MINUTE
        assert Timeframe.from_string("hour") == Timeframe.HOUR
        assert Timeframe.from_string("day") == Timeframe.DAY
        assert Timeframe.from_string("week") == Timeframe.WEEK
        assert Timeframe.from_string("month") == Timeframe.MONTH

    def test_from_string_short_names(self) -> None:
        """Short timeframe names should parse correctly."""
        assert Timeframe.from_string("s") == Timeframe.SECOND
        assert Timeframe.from_string("m") == Timeframe.MINUTE
        assert Timeframe.from_string("h") == Timeframe.HOUR
        assert Timeframe.from_string("d") == Timeframe.DAY
        assert Timeframe.from_string("w") == Timeframe.WEEK
        assert Timeframe.from_string("mo") == Timeframe.MONTH

    def test_from_string_with_number(self) -> None:
        """Timeframe with number prefix should parse correctly."""
        assert Timeframe.from_string("1s") == Timeframe.SECOND
        assert Timeframe.from_string("1m") == Timeframe.MINUTE
        assert Timeframe.from_string("1h") == Timeframe.HOUR
        assert Timeframe.from_string("1d") == Timeframe.DAY
        assert Timeframe.from_string("1w") == Timeframe.WEEK
        assert Timeframe.from_string("1mo") == Timeframe.MONTH

    def test_from_string_case_insensitive(self) -> None:
        """Parsing should be case insensitive."""
        assert Timeframe.from_string("HOUR") == Timeframe.HOUR
        assert Timeframe.from_string("Hour") == Timeframe.HOUR
        assert Timeframe.from_string("  hour  ") == Timeframe.HOUR

    def test_from_string_invalid(self) -> None:
        """Invalid strings should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown timeframe"):
            Timeframe.from_string("invalid")

    def test_to_timedelta(self) -> None:
        """Timeframe should convert to timedelta."""
        assert Timeframe.MINUTE.to_timedelta() == timedelta(minutes=1)
        assert Timeframe.HOUR.to_timedelta() == timedelta(hours=1)
        assert Timeframe.DAY.to_timedelta() == timedelta(days=1)


class TestFrequencyCapConfig:
    """Tests for FrequencyCapConfig dataclass."""

    def test_create_config(self) -> None:
        """Config should store values correctly."""
        cap = FrequencyCapConfig(
            max_trades=5,
            timeframe=Timeframe.HOUR,
            per_symbol=True,
        )
        assert cap.max_trades == 5
        assert cap.timeframe == Timeframe.HOUR
        assert cap.per_symbol is True

    def test_default_per_symbol(self) -> None:
        """Default per_symbol should be True."""
        cap = FrequencyCapConfig(max_trades=10, timeframe=Timeframe.MINUTE)
        assert cap.per_symbol is True


class TestCreateFrequencyCapHelper:
    """Tests for create_frequency_cap helper function."""

    def test_create_with_string_timeframe(self) -> None:
        """Helper should accept string timeframe."""
        cap = create_frequency_cap(5, "hour")
        assert cap.max_trades == 5
        assert cap.timeframe == Timeframe.HOUR
        assert cap.per_symbol is True

    def test_create_with_enum_timeframe(self) -> None:
        """Helper should accept enum timeframe."""
        cap = create_frequency_cap(10, Timeframe.DAY, per_symbol=False)
        assert cap.max_trades == 10
        assert cap.timeframe == Timeframe.DAY
        assert cap.per_symbol is False


class TestFrequencyCapConstraintProtocol:
    """Test that FrequencyCapConstraint conforms to Constraint protocol."""

    def test_conforms_to_protocol(self) -> None:
        """FrequencyCapConstraint must implement Constraint protocol."""
        constraint = FrequencyCapConstraint()
        assert isinstance(constraint, Constraint)


class TestFrequencyCapConstraintInit:
    """Tests for FrequencyCapConstraint initialization."""

    def test_default_caps(self) -> None:
        """Default should be 10 trades per minute per symbol."""
        constraint = FrequencyCapConstraint()
        assert len(constraint.caps) == 1
        assert constraint.caps[0].max_trades == 10
        assert constraint.caps[0].timeframe == Timeframe.MINUTE
        assert constraint.caps[0].per_symbol is True

    def test_custom_caps(self) -> None:
        """Custom caps should be accepted."""
        caps = [
            FrequencyCapConfig(max_trades=5, timeframe=Timeframe.HOUR),
            FrequencyCapConfig(max_trades=100, timeframe=Timeframe.DAY, per_symbol=False),
        ]
        constraint = FrequencyCapConstraint(caps=caps)
        assert len(constraint.caps) == 2

    def test_invalid_max_trades(self) -> None:
        """max_trades < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="max_trades must be >= 1"):
            FrequencyCapConstraint(
                caps=[FrequencyCapConfig(max_trades=0, timeframe=Timeframe.HOUR)]
            )


class TestFrequencyCapConstraintBasic:
    """Basic functionality tests for FrequencyCapConstraint."""

    def test_empty_orders_returns_empty(self) -> None:
        """No orders should return no orders."""
        now = datetime.now(UTC)
        constraint = FrequencyCapConstraint()
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

    def test_orders_pass_when_under_limit(self) -> None:
        """Orders should pass when under frequency limit."""
        now = datetime.now(UTC)
        constraint = FrequencyCapConstraint(
            caps=[FrequencyCapConfig(max_trades=5, timeframe=Timeframe.HOUR)]
        )
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
        assert result.rejected == []


class TestFrequencyCapConstraintPerSymbol:
    """Tests for per-symbol frequency caps."""

    def test_order_blocked_when_symbol_limit_reached(self) -> None:
        """Orders should be blocked when per-symbol limit reached."""
        now = datetime.now(UTC)
        # Create history with 5 trades for AAPL in the last hour
        history = [
            TradeRecord(
                symbol="AAPL",
                timestamp=now - timedelta(minutes=30),
                side=OrderSide.BUY,
                quantity=Decimal("10"),
            )
            for _ in range(5)
        ]
        constraint = FrequencyCapConstraint(
            caps=[FrequencyCapConfig(max_trades=5, timeframe=Timeframe.HOUR)],
            trade_history=history,
        )
        config = RiskConfig()
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

        assert result.orders == []
        assert len(result.rejected) == 1
        assert "Frequency cap exceeded for AAPL" in result.rejected[0].reason

    def test_other_symbol_not_affected(self) -> None:
        """Other symbols should not be affected by one symbol's limit."""
        now = datetime.now(UTC)
        # Create history with 5 trades for AAPL only
        history = [
            TradeRecord(
                symbol="AAPL",
                timestamp=now - timedelta(minutes=30),
                side=OrderSide.BUY,
                quantity=Decimal("10"),
            )
            for _ in range(5)
        ]
        constraint = FrequencyCapConstraint(
            caps=[FrequencyCapConfig(max_trades=5, timeframe=Timeframe.HOUR)],
            trade_history=history,
        )
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
            # AAPL should be blocked
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            ),
            # GOOGL should pass
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),
                timestamp=now,
            ),
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result.orders) == 1
        assert result.orders[0].symbol == "GOOGL"
        assert len(result.rejected) == 1
        assert result.rejected[0].order.symbol == "AAPL"


class TestFrequencyCapConstraintGlobal:
    """Tests for global frequency caps."""

    def test_global_cap_blocks_all_symbols(self) -> None:
        """Global cap should block orders regardless of symbol."""
        now = datetime.now(UTC)
        # Create history with 10 trades spread across symbols
        history = [
            TradeRecord(
                symbol=f"SYM{i}",
                timestamp=now - timedelta(minutes=30),
                side=OrderSide.BUY,
                quantity=Decimal("10"),
            )
            for i in range(10)
        ]
        constraint = FrequencyCapConstraint(
            caps=[FrequencyCapConfig(max_trades=10, timeframe=Timeframe.HOUR, per_symbol=False)],
            trade_history=history,
        )
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
        # Even a new symbol should be blocked
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

        assert result.orders == []
        assert len(result.rejected) == 1
        assert "Global frequency cap exceeded" in result.rejected[0].reason


class TestFrequencyCapConstraintTimeWindows:
    """Tests for different time windows."""

    def test_old_trades_not_counted(self) -> None:
        """Trades outside the window should not be counted."""
        now = datetime.now(UTC)
        # Create old history (2 hours ago - outside 1 hour window)
        history = [
            TradeRecord(
                symbol="AAPL",
                timestamp=now - timedelta(hours=2),
                side=OrderSide.BUY,
                quantity=Decimal("10"),
            )
            for _ in range(10)
        ]
        constraint = FrequencyCapConstraint(
            caps=[FrequencyCapConfig(max_trades=5, timeframe=Timeframe.HOUR)],
            trade_history=history,
        )
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

        # Should pass - old trades don't count
        assert len(result.orders) == 1

    def test_minute_cap(self) -> None:
        """Minute-level cap should work correctly."""
        now = datetime.now(UTC)
        # Create 3 trades in the last 30 seconds
        history = [
            TradeRecord(
                symbol="AAPL",
                timestamp=now - timedelta(seconds=30),
                side=OrderSide.BUY,
                quantity=Decimal("10"),
            )
            for _ in range(3)
        ]
        constraint = FrequencyCapConstraint(
            caps=[FrequencyCapConfig(max_trades=3, timeframe=Timeframe.MINUTE)],
            trade_history=history,
        )
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

        # Should be blocked - already 3 trades in minute
        assert result.orders == []
        assert "minute" in result.rejected[0].reason.lower()

    def test_day_cap(self) -> None:
        """Day-level cap should work correctly."""
        now = datetime.now(UTC)
        # Create 50 trades in the last 12 hours
        history = [
            TradeRecord(
                symbol="AAPL",
                timestamp=now - timedelta(hours=12),
                side=OrderSide.BUY,
                quantity=Decimal("10"),
            )
            for _ in range(50)
        ]
        constraint = FrequencyCapConstraint(
            caps=[FrequencyCapConfig(max_trades=50, timeframe=Timeframe.DAY)],
            trade_history=history,
        )
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

        # Should be blocked - already 50 trades in day
        assert result.orders == []
        assert "day" in result.rejected[0].reason.lower()


class TestFrequencyCapConstraintMultipleCaps:
    """Tests for multiple simultaneous caps."""

    def test_most_restrictive_cap_applies(self) -> None:
        """All caps must be satisfied."""
        now = datetime.now(UTC)
        # Create 3 trades in the last minute (within minute limit but not hour)
        history = [
            TradeRecord(
                symbol="AAPL",
                timestamp=now - timedelta(seconds=30),
                side=OrderSide.BUY,
                quantity=Decimal("10"),
            )
            for _ in range(3)
        ]
        # Also some trades from earlier in the hour
        history.extend([
            TradeRecord(
                symbol="AAPL",
                timestamp=now - timedelta(minutes=30),
                side=OrderSide.BUY,
                quantity=Decimal("10"),
            )
            for _ in range(7)
        ])

        constraint = FrequencyCapConstraint(
            caps=[
                FrequencyCapConfig(max_trades=5, timeframe=Timeframe.MINUTE),  # OK - only 3
                FrequencyCapConfig(max_trades=10, timeframe=Timeframe.HOUR),  # BLOCKED - 10 trades
            ],
            trade_history=history,
        )
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

        # Should be blocked by hour cap
        assert result.orders == []
        assert "hour" in result.rejected[0].reason.lower()


class TestFrequencyCapConstraintBatchOrders:
    """Tests for batch order handling."""

    def test_batch_orders_counted_correctly(self) -> None:
        """Multiple orders in one batch should be counted correctly."""
        now = datetime.now(UTC)
        constraint = FrequencyCapConstraint(
            caps=[FrequencyCapConfig(max_trades=3, timeframe=Timeframe.HOUR)]
        )
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
        # 5 orders in one batch
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                timestamp=now,
            )
            for _ in range(5)
        ]

        result = constraint.apply(orders, portfolio, market, config)

        # Only first 3 should pass
        assert len(result.orders) == 3
        assert len(result.rejected) == 2


class TestFrequencyCapConstraintRecordTrade:
    """Tests for record_trade method."""

    def test_record_trade_adds_to_history(self) -> None:
        """Recording a trade should add to history."""
        constraint = FrequencyCapConstraint()
        now = datetime.now(UTC)

        constraint.record_trade(
            symbol="AAPL",
            timestamp=now,
            side=OrderSide.BUY,
            quantity=Decimal("100"),
        )

        assert constraint.get_trade_count() == 1
        assert constraint.get_trade_count(symbol="AAPL") == 1

    def test_get_trade_count_filters_correctly(self) -> None:
        """get_trade_count should filter by symbol and time."""
        constraint = FrequencyCapConstraint()
        now = datetime.now(UTC)

        constraint.record_trade("AAPL", now, OrderSide.BUY, Decimal("100"))
        constraint.record_trade("AAPL", now - timedelta(hours=2), OrderSide.BUY, Decimal("100"))
        constraint.record_trade("GOOGL", now, OrderSide.BUY, Decimal("100"))

        assert constraint.get_trade_count() == 3
        assert constraint.get_trade_count(symbol="AAPL") == 2
        assert constraint.get_trade_count(symbol="GOOGL") == 1
        assert constraint.get_trade_count(since=now - timedelta(hours=1)) == 2

    def test_clear_history(self) -> None:
        """Clearing history should remove all trades."""
        constraint = FrequencyCapConstraint()
        now = datetime.now(UTC)

        constraint.record_trade("AAPL", now, OrderSide.BUY, Decimal("100"))
        constraint.record_trade("GOOGL", now, OrderSide.BUY, Decimal("100"))

        assert constraint.get_trade_count() == 2

        constraint.clear_history()

        assert constraint.get_trade_count() == 0


class TestFrequencyCapConstraintClassifyRisk:
    """Tests for classify_risk method."""

    def test_buy_no_position_is_risk_increasing(self) -> None:
        """Buying with no position is risk increasing."""
        now = datetime.now(UTC)
        constraint = FrequencyCapConstraint()
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
        """Selling to close long is risk reducing."""
        now = datetime.now(UTC)
        constraint = FrequencyCapConstraint()
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


class TestFrequencyCapConstraintAllTimeframes:
    """Integration tests for all supported timeframes."""

    @pytest.mark.parametrize(
        "timeframe,expected_duration",
        [
            (Timeframe.SECOND, timedelta(seconds=1)),
            (Timeframe.MINUTE, timedelta(minutes=1)),
            (Timeframe.HOUR, timedelta(hours=1)),
            (Timeframe.DAY, timedelta(days=1)),
            (Timeframe.WEEK, timedelta(weeks=1)),
            (Timeframe.MONTH, timedelta(days=30)),
        ],
    )
    def test_timeframe_durations(
        self, timeframe: Timeframe, expected_duration: timedelta
    ) -> None:
        """All timeframes should have correct durations."""
        assert timeframe.to_timedelta() == expected_duration

    @pytest.mark.parametrize(
        "timeframe_str",
        ["second", "minute", "hour", "day", "week", "month"],
    )
    def test_all_timeframes_work_in_constraint(self, timeframe_str: str) -> None:
        """Constraint should work with all timeframes."""
        now = datetime.now(UTC)
        timeframe = Timeframe.from_string(timeframe_str)
        constraint = FrequencyCapConstraint(
            caps=[FrequencyCapConfig(max_trades=5, timeframe=timeframe)]
        )
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

        # Should pass - no prior history
        assert len(result.orders) == 1
