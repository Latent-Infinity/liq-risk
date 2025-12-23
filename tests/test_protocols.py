"""Tests for liq-risk protocols.

Following TDD: RED phase - write failing tests first.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

from liq.core import OrderRequest, OrderSide, OrderType, PortfolioState
from liq.signals import Signal

if TYPE_CHECKING:
    from liq.risk import MarketState, RiskConfig


class TestPositionSizerProtocol:
    """Tests for PositionSizer protocol compliance."""

    def test_protocol_defines_size_positions_method(self) -> None:
        """PositionSizer protocol must define size_positions method."""
        from liq.risk.protocols import PositionSizer

        # Protocol should have size_positions in its annotations
        assert hasattr(PositionSizer, "size_positions")

    def test_custom_sizer_conforms_to_protocol(self) -> None:
        """A class implementing size_positions conforms to PositionSizer."""
        from liq.risk.protocols import PositionSizer

        class MockSizer:
            """Mock sizer that conforms to protocol."""

            def size_positions(
                self,
                signals: list[Signal],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> list[OrderRequest]:
                return []

        sizer = MockSizer()
        # Should be recognized as implementing the protocol
        assert isinstance(sizer, PositionSizer)

    def test_non_conforming_class_rejected(self) -> None:
        """A class without size_positions does not conform to PositionSizer."""
        from liq.risk.protocols import PositionSizer

        class NotASizer:
            """Class that does not implement the protocol."""

            def some_other_method(self) -> None:
                pass

        not_sizer = NotASizer()
        assert not isinstance(not_sizer, PositionSizer)

    def test_sizer_returns_order_requests(self) -> None:
        """PositionSizer.size_positions returns list of OrderRequest."""
        from liq.risk import MarketState, RiskConfig

        now = datetime.now(UTC)

        class SimpleSizer:
            """Simple sizer that returns one order per signal."""

            def size_positions(
                self,
                signals: list[Signal],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> list[OrderRequest]:
                orders = []
                for signal in signals:
                    if signal.direction == "long":
                        side = OrderSide.BUY
                    elif signal.direction == "short":
                        side = OrderSide.SELL
                    else:
                        continue
                    orders.append(
                        OrderRequest(
                            symbol=signal.symbol,
                            side=side,
                            order_type=OrderType.MARKET,
                            quantity=Decimal("100"),
                            timestamp=signal.timestamp,
                        )
                    )
                return orders

        sizer = SimpleSizer()
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
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.8),
        ]

        orders = sizer.size_positions(signals, portfolio, market, config)

        assert len(orders) == 1
        assert orders[0].symbol == "AAPL"
        assert orders[0].side == OrderSide.BUY
        assert orders[0].quantity == Decimal("100")


class TestConstraintProtocol:
    """Tests for Constraint protocol compliance."""

    def test_protocol_defines_apply_method(self) -> None:
        """Constraint protocol must define apply method."""
        from liq.risk.protocols import Constraint

        assert hasattr(Constraint, "apply")

    def test_custom_constraint_conforms_to_protocol(self) -> None:
        """A class implementing apply conforms to Constraint."""
        from liq.risk.protocols import Constraint

        class MockConstraint:
            """Mock constraint that conforms to protocol."""

            def apply(
                self,
                orders: list[OrderRequest],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> list[OrderRequest]:
                return orders

        constraint = MockConstraint()
        assert isinstance(constraint, Constraint)

    def test_non_conforming_class_rejected(self) -> None:
        """A class without apply does not conform to Constraint."""
        from liq.risk.protocols import Constraint

        class NotAConstraint:
            """Class that does not implement the protocol."""

            def check(self) -> bool:
                return True

        not_constraint = NotAConstraint()
        assert not isinstance(not_constraint, Constraint)

    def test_constraint_can_filter_orders(self) -> None:
        """Constraint.apply can remove orders from the list."""
        from liq.risk import MarketState, RiskConfig

        now = datetime.now(UTC)

        class MaxQuantityConstraint:
            """Constraint that filters orders above max quantity."""

            def __init__(self, max_quantity: Decimal) -> None:
                self.max_quantity = max_quantity

            def apply(
                self,
                orders: list[OrderRequest],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> list[OrderRequest]:
                return [o for o in orders if o.quantity <= self.max_quantity]

        constraint = MaxQuantityConstraint(max_quantity=Decimal("50"))
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

        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("25"),
                timestamp=now,
            ),
        ]

        filtered = constraint.apply(orders, portfolio, market, config)

        assert len(filtered) == 1
        assert filtered[0].symbol == "GOOGL"

    def test_constraint_can_modify_orders(self) -> None:
        """Constraint.apply can return modified orders."""
        from liq.risk import MarketState, RiskConfig

        now = datetime.now(UTC)

        class HalveQuantityConstraint:
            """Constraint that halves all order quantities."""

            def apply(
                self,
                orders: list[OrderRequest],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> list[OrderRequest]:
                return [
                    OrderRequest(
                        symbol=o.symbol,
                        side=o.side,
                        order_type=o.order_type,
                        quantity=o.quantity / 2,
                        timestamp=o.timestamp,
                    )
                    for o in orders
                ]

        constraint = HalveQuantityConstraint()
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

        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            ),
        ]

        modified = constraint.apply(orders, portfolio, market, config)

        assert len(modified) == 1
        assert modified[0].quantity == Decimal("50")


class TestTargetPositionSizerProtocol:
    """Tests for TargetPositionSizer protocol compliance."""

    def test_protocol_defines_size_positions_method(self) -> None:
        """TargetPositionSizer protocol must define size_positions method."""
        from liq.risk.protocols import TargetPositionSizer

        assert hasattr(TargetPositionSizer, "size_positions")

    def test_custom_sizer_conforms_to_protocol(self) -> None:
        """A class implementing size_positions with TargetPosition return conforms."""
        from liq.risk.protocols import TargetPositionSizer
        from liq.risk.types import TargetPosition

        class MockTargetSizer:
            """Mock sizer that returns TargetPosition."""

            def size_positions(
                self,
                signals: list[Signal],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> list[TargetPosition]:
                return []

        sizer = MockTargetSizer()
        assert isinstance(sizer, TargetPositionSizer)

    def test_non_conforming_class_rejected(self) -> None:
        """A class without size_positions does not conform."""
        from liq.risk.protocols import TargetPositionSizer

        class NotASizer:
            """Class that does not implement the protocol."""

            def some_method(self) -> None:
                pass

        not_sizer = NotASizer()
        assert not isinstance(not_sizer, TargetPositionSizer)

    def test_sizer_returns_target_positions(self) -> None:
        """TargetPositionSizer.size_positions returns list of TargetPosition."""
        from liq.risk import MarketState, RiskConfig
        from liq.risk.types import TargetPosition

        now = datetime.now(UTC)

        class SimpleTargetSizer:
            """Simple sizer that returns TargetPositions."""

            def size_positions(
                self,
                signals: list[Signal],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> list[TargetPosition]:
                targets = []
                for signal in signals:
                    if signal.direction == "long":
                        direction = "long"
                        target_qty = Decimal("100")
                    elif signal.direction == "short":
                        direction = "short"
                        target_qty = Decimal("-100")
                    else:
                        direction = "flat"
                        target_qty = Decimal("0")
                    # Get current position
                    current = portfolio_state.positions.get(signal.symbol)
                    current_qty = current.quantity if current else Decimal("0")
                    targets.append(
                        TargetPosition(
                            symbol=signal.symbol,
                            target_quantity=target_qty,
                            current_quantity=current_qty,
                            direction=direction,
                            signal_strength=signal.strength,
                        )
                    )
                return targets

        sizer = SimpleTargetSizer()
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
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.8),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        assert len(targets) == 1
        assert targets[0].symbol == "AAPL"
        assert targets[0].target_quantity == Decimal("100")
        assert targets[0].direction == "long"
        assert targets[0].signal_strength == 0.8

    def test_target_sizer_calculates_delta(self) -> None:
        """TargetPositionSizer result includes correct delta calculation."""
        from liq.core import Position

        from liq.risk import MarketState, RiskConfig
        from liq.risk.types import TargetPosition

        now = datetime.now(UTC)

        class DeltaAwareSizer:
            """Sizer that considers existing positions."""

            def size_positions(
                self,
                signals: list[Signal],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> list[TargetPosition]:
                targets = []
                for signal in signals:
                    current = portfolio_state.positions.get(signal.symbol)
                    current_qty = current.quantity if current else Decimal("0")
                    targets.append(
                        TargetPosition(
                            symbol=signal.symbol,
                            target_quantity=Decimal("150"),
                            current_quantity=current_qty,
                            direction="long",
                        )
                    )
                return targets

        sizer = DeltaAwareSizer()
        # Portfolio has existing position of 50
        portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("50"),
                    average_price=Decimal("150.00"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                )
            },
            timestamp=now,
        )
        market = MarketState(
            current_bars={},
            volatility={},
            liquidity={},
            timestamp=now,
        )
        config = RiskConfig()
        signals = [
            Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.9),
        ]

        targets = sizer.size_positions(signals, portfolio, market, config)

        assert len(targets) == 1
        assert targets[0].target_quantity == Decimal("150")
        assert targets[0].current_quantity == Decimal("50")
        assert targets[0].delta_quantity == Decimal("100")


class TestStructuredConstraintProtocol:
    """Tests for StructuredConstraint protocol compliance."""

    def test_protocol_defines_required_members(self) -> None:
        """StructuredConstraint protocol must define name, apply, and classify_risk."""
        from liq.risk.protocols import StructuredConstraint

        assert hasattr(StructuredConstraint, "name")
        assert hasattr(StructuredConstraint, "apply")
        assert hasattr(StructuredConstraint, "classify_risk")

    def test_custom_constraint_conforms_to_protocol(self) -> None:
        """A class implementing all required members conforms."""
        from liq.risk.protocols import StructuredConstraint
        from liq.risk.types import ConstraintResult

        class MockStructuredConstraint:
            """Mock constraint that returns ConstraintResult."""

            @property
            def name(self) -> str:
                return "MockConstraint"

            def apply(
                self,
                orders: list[OrderRequest],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> ConstraintResult:
                return ConstraintResult(orders=orders, rejected=[])

            def classify_risk(
                self,
                order: OrderRequest,
                portfolio_state: PortfolioState,
            ) -> bool:
                return True

        constraint = MockStructuredConstraint()
        assert isinstance(constraint, StructuredConstraint)

    def test_missing_name_not_conforming(self) -> None:
        """A class without name property does not conform."""
        from liq.risk.protocols import StructuredConstraint
        from liq.risk.types import ConstraintResult

        class MissingName:
            """Constraint without name property."""

            def apply(
                self,
                orders: list[OrderRequest],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> ConstraintResult:
                return ConstraintResult(orders=orders, rejected=[])

            def classify_risk(
                self,
                order: OrderRequest,
                portfolio_state: PortfolioState,
            ) -> bool:
                return True

        constraint = MissingName()
        assert not isinstance(constraint, StructuredConstraint)

    def test_missing_classify_risk_not_conforming(self) -> None:
        """A class without classify_risk method does not conform."""
        from liq.risk.protocols import StructuredConstraint
        from liq.risk.types import ConstraintResult

        class MissingClassifyRisk:
            """Constraint without classify_risk method."""

            @property
            def name(self) -> str:
                return "MissingClassifyRisk"

            def apply(
                self,
                orders: list[OrderRequest],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> ConstraintResult:
                return ConstraintResult(orders=orders, rejected=[])

        constraint = MissingClassifyRisk()
        assert not isinstance(constraint, StructuredConstraint)

    def test_constraint_returns_constraint_result(self) -> None:
        """StructuredConstraint.apply returns ConstraintResult."""
        from liq.risk import MarketState, RiskConfig
        from liq.risk.types import ConstraintResult, RejectedOrder

        now = datetime.now(UTC)

        class MaxQuantityStructuredConstraint:
            """Constraint that tracks rejections."""

            def __init__(self, max_quantity: Decimal) -> None:
                self.max_quantity = max_quantity

            @property
            def name(self) -> str:
                return "MaxQuantityConstraint"

            def apply(
                self,
                orders: list[OrderRequest],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> ConstraintResult:
                passed = []
                rejected = []
                for order in orders:
                    if order.quantity <= self.max_quantity:
                        passed.append(order)
                    else:
                        rejected.append(
                            RejectedOrder(
                                order=order,
                                constraint_name=self.name,
                                reason=f"Quantity {order.quantity} exceeds max {self.max_quantity}",
                            )
                        )
                return ConstraintResult(orders=passed, rejected=rejected)

            def classify_risk(
                self,
                order: OrderRequest,
                portfolio_state: PortfolioState,
            ) -> bool:
                # Buy orders are risk-increasing
                return order.side == OrderSide.BUY

        constraint = MaxQuantityStructuredConstraint(max_quantity=Decimal("50"))
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

        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("25"),
                timestamp=now,
            ),
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert isinstance(result, ConstraintResult)
        assert len(result.orders) == 1
        assert result.orders[0].symbol == "GOOGL"
        assert len(result.rejected) == 1
        assert result.rejected[0].order.symbol == "AAPL"
        assert result.rejected[0].constraint_name == "MaxQuantityConstraint"
        assert "exceeds max" in result.rejected[0].reason

    def test_classify_risk_identifies_risk_increasing(self) -> None:
        """classify_risk correctly identifies risk-increasing orders."""
        from liq.core import Position

        from liq.risk.types import ConstraintResult

        now = datetime.now(UTC)

        class PositionAwareConstraint:
            """Constraint that checks position direction."""

            @property
            def name(self) -> str:
                return "PositionAwareConstraint"

            def apply(
                self,
                orders: list[OrderRequest],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> ConstraintResult:
                return ConstraintResult(orders=orders, rejected=[])

            def classify_risk(
                self,
                order: OrderRequest,
                portfolio_state: PortfolioState,
            ) -> bool:
                """Risk-increasing if adding to position or opening new."""
                position = portfolio_state.positions.get(order.symbol)
                if position is None:
                    # New position is always risk-increasing
                    return True
                # Adding to long position with buy = risk-increasing
                if position.quantity > 0 and order.side == OrderSide.BUY:
                    return True
                # Adding to short position with sell = risk-increasing
                if position.quantity < 0 and order.side == OrderSide.SELL:
                    return True
                # Reducing position is risk-reducing
                return False

        constraint = PositionAwareConstraint()

        # Empty portfolio - any order is risk-increasing
        empty_portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={},
            timestamp=now,
        )
        buy_order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            timestamp=now,
        )
        assert constraint.classify_risk(buy_order, empty_portfolio) is True

        # Long position - sell is risk-reducing
        long_portfolio = PortfolioState(
            cash=Decimal("100000"),
            positions={
                "AAPL": Position(
                    symbol="AAPL",
                    quantity=Decimal("100"),
                    average_price=Decimal("150.00"),
                    realized_pnl=Decimal("0"),
                    timestamp=now,
                )
            },
            timestamp=now,
        )
        sell_order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("50"),
            timestamp=now,
        )
        assert constraint.classify_risk(sell_order, long_portfolio) is False

        # Long position - buy is risk-increasing
        assert constraint.classify_risk(buy_order, long_portfolio) is True

    def test_constraint_with_warnings(self) -> None:
        """StructuredConstraint can include warnings in result."""
        from liq.risk import MarketState, RiskConfig
        from liq.risk.types import ConstraintResult

        now = datetime.now(UTC)

        class WarningConstraint:
            """Constraint that adds warnings when approaching limits."""

            @property
            def name(self) -> str:
                return "WarningConstraint"

            def apply(
                self,
                orders: list[OrderRequest],
                portfolio_state: PortfolioState,
                market_state: MarketState,
                risk_config: RiskConfig,
            ) -> ConstraintResult:
                warnings = []
                # Add warning if total quantity is high
                total_qty = sum(o.quantity for o in orders)
                if total_qty > Decimal("500"):
                    warnings.append(f"High total quantity: {total_qty}")
                return ConstraintResult(
                    orders=orders, rejected=[], warnings=warnings
                )

            def classify_risk(
                self,
                order: OrderRequest,
                portfolio_state: PortfolioState,
            ) -> bool:
                return order.side == OrderSide.BUY

        constraint = WarningConstraint()
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

        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("300"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("300"),
                timestamp=now,
            ),
        ]

        result = constraint.apply(orders, portfolio, market, config)

        assert len(result.orders) == 2
        assert len(result.rejected) == 0
        assert len(result.warnings) == 1
        assert "High total quantity" in result.warnings[0]
