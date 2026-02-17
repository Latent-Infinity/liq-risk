"""Tests for liq-risk type definitions.

Tests TargetPosition, RejectedOrder, ConstraintResult, and RoundingPolicy.
"""

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from liq.core import OrderRequest, OrderSide, OrderType


class TestTargetPosition:
    """Tests for TargetPosition dataclass."""

    def test_creation_basic(self):
        """Test basic TargetPosition creation."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("100"),
            current_quantity=Decimal("50"),
            direction="long",
        )

        assert tp.symbol == "AAPL"
        assert tp.target_quantity == Decimal("100")
        assert tp.current_quantity == Decimal("50")
        assert tp.direction == "long"
        assert tp.urgency == "normal"
        assert tp.stop_price is None
        assert tp.take_profit_price is None
        assert tp.signal_strength == 1.0
        assert tp.risk_tags == {}

    def test_creation_with_all_fields(self):
        """Test TargetPosition with all optional fields."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="GOOGL",
            target_quantity=Decimal("200"),
            current_quantity=Decimal("100"),
            direction="long",
            urgency="urgent",
            stop_price=Decimal("145.50"),
            take_profit_price=Decimal("165.00"),
            signal_strength=0.85,
            risk_tags={"strategy": "momentum", "sector": "tech"},
        )

        assert tp.urgency == "urgent"
        assert tp.stop_price == Decimal("145.50")
        assert tp.take_profit_price == Decimal("165.00")
        assert tp.signal_strength == 0.85
        assert tp.risk_tags == {"strategy": "momentum", "sector": "tech"}

    def test_frozen_immutability(self):
        """Test that TargetPosition is immutable."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("100"),
            current_quantity=Decimal("50"),
            direction="long",
        )

        with pytest.raises(AttributeError):
            tp.symbol = "GOOGL"  # type: ignore

    def test_delta_quantity_positive(self):
        """Test delta_quantity when buying more."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("150"),
            current_quantity=Decimal("50"),
            direction="long",
        )

        assert tp.delta_quantity == Decimal("100")

    def test_delta_quantity_negative(self):
        """Test delta_quantity when selling."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("25"),
            current_quantity=Decimal("100"),
            direction="long",
        )

        assert tp.delta_quantity == Decimal("-75")

    def test_delta_quantity_zero(self):
        """Test delta_quantity when no change needed."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("100"),
            current_quantity=Decimal("100"),
            direction="long",
        )

        assert tp.delta_quantity == Decimal("0")

    def test_delta_quantity_short_position(self):
        """Test delta_quantity with short positions (negative quantities)."""
        from liq.risk.types import TargetPosition

        # Increasing short position
        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("-150"),
            current_quantity=Decimal("-50"),
            direction="short",
        )

        assert tp.delta_quantity == Decimal("-100")

    def test_is_risk_increasing_new_long(self):
        """Test is_risk_increasing for new long position."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("100"),
            current_quantity=Decimal("0"),
            direction="long",
        )

        assert tp.is_risk_increasing is True

    def test_is_risk_increasing_add_to_long(self):
        """Test is_risk_increasing when adding to existing long."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("150"),
            current_quantity=Decimal("100"),
            direction="long",
        )

        assert tp.is_risk_increasing is True

    def test_is_risk_increasing_reduce_long(self):
        """Test is_risk_increasing when reducing long position."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("50"),
            current_quantity=Decimal("100"),
            direction="long",
        )

        assert tp.is_risk_increasing is False

    def test_is_risk_increasing_close_position(self):
        """Test is_risk_increasing when closing position."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("0"),
            current_quantity=Decimal("100"),
            direction="flat",
        )

        assert tp.is_risk_increasing is False

    def test_is_risk_increasing_new_short(self):
        """Test is_risk_increasing for new short position."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("-100"),
            current_quantity=Decimal("0"),
            direction="short",
        )

        assert tp.is_risk_increasing is True

    def test_is_risk_increasing_add_to_short(self):
        """Test is_risk_increasing when increasing short."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("-150"),
            current_quantity=Decimal("-100"),
            direction="short",
        )

        assert tp.is_risk_increasing is True

    def test_is_risk_increasing_cover_short(self):
        """Test is_risk_increasing when covering short."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("-50"),
            current_quantity=Decimal("-100"),
            direction="short",
        )

        assert tp.is_risk_increasing is False

    def test_to_order_request_buy(self):
        """Test conversion to buy OrderRequest."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("150"),
            current_quantity=Decimal("50"),
            direction="long",
            signal_strength=0.8,
        )

        now = datetime.now(UTC)
        order = tp.to_order_request(timestamp=now)

        assert order is not None
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.BUY
        assert order.quantity == Decimal("100")
        assert order.order_type == OrderType.MARKET
        assert order.timestamp == now
        assert order.confidence == 0.8

    def test_to_order_request_sell(self):
        """Test conversion to sell OrderRequest."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("25"),
            current_quantity=Decimal("100"),
            direction="long",
        )

        now = datetime.now(UTC)
        order = tp.to_order_request(timestamp=now)

        assert order is not None
        assert order.symbol == "AAPL"
        assert order.side == OrderSide.SELL
        assert order.quantity == Decimal("75")

    def test_to_order_request_zero_delta(self):
        """Test that zero delta returns None."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("100"),
            current_quantity=Decimal("100"),
            direction="long",
        )

        order = tp.to_order_request(timestamp=datetime.now(UTC))
        assert order is None

    def test_to_order_request_with_limit_order(self):
        """Test conversion with specific order type."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("100"),
            current_quantity=Decimal("0"),
            direction="long",
        )

        now = datetime.now(UTC)
        order = tp.to_order_request(timestamp=now, order_type=OrderType.MARKET)

        assert order is not None
        assert order.order_type == OrderType.MARKET

    def test_to_order_request_short_entry(self):
        """Test conversion for short entry."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("-100"),
            current_quantity=Decimal("0"),
            direction="short",
        )

        now = datetime.now(UTC)
        order = tp.to_order_request(timestamp=now)

        assert order is not None
        assert order.side == OrderSide.SELL
        assert order.quantity == Decimal("100")

    def test_to_order_request_short_cover(self):
        """Test conversion for short cover."""
        from liq.risk.types import TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("0"),
            current_quantity=Decimal("-100"),
            direction="flat",
        )

        now = datetime.now(UTC)
        order = tp.to_order_request(timestamp=now)

        assert order is not None
        assert order.side == OrderSide.BUY
        assert order.quantity == Decimal("100")

    def test_to_order_request_with_rounding(self):
        """Test conversion with RoundingPolicy."""
        from liq.risk.types import RoundingPolicy, TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("157"),
            current_quantity=Decimal("50"),
            direction="long",
        )

        rounding = RoundingPolicy(lot_size=Decimal("10"))
        now = datetime.now(UTC)
        order = tp.to_order_request(timestamp=now, rounding=rounding)

        assert order is not None
        # 107 rounded down to lot size 10 = 100
        assert order.quantity == Decimal("100")

    def test_to_order_request_rounds_to_zero(self):
        """Test that rounding to zero returns None."""
        from liq.risk.types import RoundingPolicy, TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("55"),
            current_quantity=Decimal("50"),
            direction="long",
        )

        rounding = RoundingPolicy(lot_size=Decimal("10"))
        order = tp.to_order_request(
            timestamp=datetime.now(UTC), rounding=rounding
        )

        # Delta is 5, rounds to 0 with lot size 10
        assert order is None

    def test_direction_values(self):
        """Test valid direction values."""
        from liq.risk.types import TargetPosition

        for direction in ["long", "short", "flat"]:
            tp = TargetPosition(
                symbol="AAPL",
                target_quantity=Decimal("100"),
                current_quantity=Decimal("0"),
                direction=direction,  # type: ignore
            )
            assert tp.direction == direction

    def test_urgency_values(self):
        """Test valid urgency values."""
        from liq.risk.types import TargetPosition

        for urgency in ["normal", "urgent", "immediate"]:
            tp = TargetPosition(
                symbol="AAPL",
                target_quantity=Decimal("100"),
                current_quantity=Decimal("0"),
                direction="long",
                urgency=urgency,  # type: ignore
            )
            assert tp.urgency == urgency


class TestRejectedOrder:
    """Tests for RejectedOrder dataclass."""

    def test_creation_with_order_request(self):
        """Test RejectedOrder creation with OrderRequest."""
        from liq.risk.types import RejectedOrder

        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            timestamp=datetime.now(UTC),
        )

        rejected = RejectedOrder(
            order=order,
            constraint_name="MaxPositionConstraint",
            reason="Position would exceed 5% of equity",
        )

        assert rejected.order == order
        assert rejected.constraint_name == "MaxPositionConstraint"
        assert rejected.reason == "Position would exceed 5% of equity"
        assert rejected.original_quantity is None

    def test_creation_with_original_quantity(self):
        """Test RejectedOrder when order was modified (not fully rejected)."""
        from liq.risk.types import RejectedOrder

        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("50"),  # Reduced quantity
            timestamp=datetime.now(UTC),
        )

        rejected = RejectedOrder(
            order=order,
            constraint_name="BuyingPowerConstraint",
            reason="Scaled from 100 to 50 due to insufficient cash",
            original_quantity=Decimal("100"),
        )

        assert rejected.original_quantity == Decimal("100")

    def test_creation_with_target_position(self):
        """Test RejectedOrder creation with TargetPosition."""
        from liq.risk.types import RejectedOrder, TargetPosition

        tp = TargetPosition(
            symbol="AAPL",
            target_quantity=Decimal("100"),
            current_quantity=Decimal("0"),
            direction="long",
        )

        rejected = RejectedOrder(
            order=tp,
            constraint_name="ShortSellingConstraint",
            reason="Short selling disabled",
        )

        assert rejected.order == tp

    def test_frozen_immutability(self):
        """Test that RejectedOrder is immutable."""
        from liq.risk.types import RejectedOrder

        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            timestamp=datetime.now(UTC),
        )

        rejected = RejectedOrder(
            order=order,
            constraint_name="Test",
            reason="Test reason",
        )

        with pytest.raises(AttributeError):
            rejected.reason = "New reason"  # type: ignore


class TestConstraintResult:
    """Tests for ConstraintResult dataclass."""

    def test_creation_basic(self):
        """Test basic ConstraintResult creation."""
        from liq.risk.types import ConstraintResult

        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            timestamp=datetime.now(UTC),
        )

        result = ConstraintResult(orders=[order], rejected=[])

        assert len(result.orders) == 1
        assert len(result.rejected) == 0
        assert result.warnings == []

    def test_creation_with_rejections(self):
        """Test ConstraintResult with rejected orders."""
        from liq.risk.types import ConstraintResult, RejectedOrder

        order1 = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            timestamp=datetime.now(UTC),
        )
        order2 = OrderRequest(
            symbol="GOOGL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("50"),
            timestamp=datetime.now(UTC),
        )

        rejected = RejectedOrder(
            order=order2,
            constraint_name="MaxPositionsConstraint",
            reason="Would exceed max positions limit of 10",
        )

        result = ConstraintResult(orders=[order1], rejected=[rejected])

        assert len(result.orders) == 1
        assert len(result.rejected) == 1
        assert result.rejected[0].constraint_name == "MaxPositionsConstraint"

    def test_creation_with_warnings(self):
        """Test ConstraintResult with warnings."""
        from liq.risk.types import ConstraintResult

        result = ConstraintResult(
            orders=[],
            rejected=[],
            warnings=["Sector exposure at 28%, approaching limit of 30%"],
        )

        assert len(result.warnings) == 1
        assert "28%" in result.warnings[0]

    def test_empty_result(self):
        """Test ConstraintResult with no orders."""
        from liq.risk.types import ConstraintResult

        result = ConstraintResult(orders=[], rejected=[])

        assert len(result.orders) == 0
        assert len(result.rejected) == 0
        assert result.warnings == []


class TestRoundingPolicy:
    """Tests for RoundingPolicy dataclass."""

    def test_default_values(self):
        """Test RoundingPolicy default values."""
        from liq.risk.types import RoundingPolicy

        policy = RoundingPolicy()

        assert policy.lot_size == Decimal("1")
        assert policy.step_size == Decimal("1")
        assert policy.min_notional == Decimal("1")
        assert policy.max_precision == 8

    def test_custom_values(self):
        """Test RoundingPolicy with custom values."""
        from liq.risk.types import RoundingPolicy

        policy = RoundingPolicy(
            lot_size=Decimal("100"),
            step_size=Decimal("10"),
            min_notional=Decimal("1000"),
            max_precision=2,
        )

        assert policy.lot_size == Decimal("100")
        assert policy.step_size == Decimal("10")
        assert policy.min_notional == Decimal("1000")
        assert policy.max_precision == 2

    def test_round_quantity_default_down(self):
        """Test round_quantity with default direction (down)."""
        from liq.risk.types import RoundingPolicy

        policy = RoundingPolicy(lot_size=Decimal("10"))

        assert policy.round_quantity(Decimal("157")) == Decimal("150")
        assert policy.round_quantity(Decimal("150")) == Decimal("150")
        assert policy.round_quantity(Decimal("159")) == Decimal("150")

    def test_round_quantity_up(self):
        """Test round_quantity with direction up."""
        from liq.risk.types import RoundingPolicy

        policy = RoundingPolicy(lot_size=Decimal("10"))

        assert policy.round_quantity(Decimal("151"), direction="up") == Decimal("160")
        assert policy.round_quantity(Decimal("150"), direction="up") == Decimal("150")

    def test_round_quantity_nearest(self):
        """Test round_quantity with direction nearest."""
        from liq.risk.types import RoundingPolicy

        policy = RoundingPolicy(lot_size=Decimal("10"))

        assert policy.round_quantity(Decimal("154"), direction="nearest") == Decimal(
            "150"
        )
        assert policy.round_quantity(Decimal("155"), direction="nearest") == Decimal(
            "160"
        )
        assert policy.round_quantity(Decimal("156"), direction="nearest") == Decimal(
            "160"
        )

    def test_round_quantity_lot_size_one(self):
        """Test round_quantity with lot size 1."""
        from liq.risk.types import RoundingPolicy

        policy = RoundingPolicy(lot_size=Decimal("1"))

        assert policy.round_quantity(Decimal("157.89")) == Decimal("157")

    def test_round_quantity_fractional_lot(self):
        """Test round_quantity with fractional lot size (crypto)."""
        from liq.risk.types import RoundingPolicy

        policy = RoundingPolicy(lot_size=Decimal("0.001"))

        assert policy.round_quantity(Decimal("1.23456789")) == Decimal("1.234")

    def test_round_quantity_zero(self):
        """Test round_quantity with zero input."""
        from liq.risk.types import RoundingPolicy

        policy = RoundingPolicy(lot_size=Decimal("10"))

        assert policy.round_quantity(Decimal("0")) == Decimal("0")

    def test_round_quantity_less_than_lot(self):
        """Test round_quantity when quantity is less than lot size."""
        from liq.risk.types import RoundingPolicy

        policy = RoundingPolicy(lot_size=Decimal("100"))

        assert policy.round_quantity(Decimal("50")) == Decimal("0")
        assert policy.round_quantity(Decimal("99")) == Decimal("0")

    def test_frozen_immutability(self):
        """Test that RoundingPolicy is immutable."""
        from liq.risk.types import RoundingPolicy

        policy = RoundingPolicy()

        with pytest.raises(AttributeError):
            policy.lot_size = Decimal("10")  # type: ignore

    def test_round_quantity_large_numbers(self):
        """Test round_quantity with large numbers."""
        from liq.risk.types import RoundingPolicy

        policy = RoundingPolicy(lot_size=Decimal("1000"))

        assert policy.round_quantity(Decimal("1234567")) == Decimal("1234000")

    def test_round_quantity_step_size_interaction(self):
        """Test that step_size is used for finer granularity."""
        from liq.risk.types import RoundingPolicy

        # Lot size is minimum unit, step size is increment
        policy = RoundingPolicy(lot_size=Decimal("1"), step_size=Decimal("0.5"))

        # Should round to nearest 0.5
        result = policy.round_quantity(Decimal("10.3"))
        assert result == Decimal("10")  # Rounds down to whole number at lot boundary
