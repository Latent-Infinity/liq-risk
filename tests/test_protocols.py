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
