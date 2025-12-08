"""Common test fixtures for liq-risk tests."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState, Position
from liq.signals import Signal

from liq.risk import MarketState, RiskConfig


@pytest.fixture
def now() -> datetime:
    """Current UTC timestamp for tests."""
    return datetime.now(UTC)


@pytest.fixture
def default_risk_config() -> RiskConfig:
    """Default RiskConfig with standard settings."""
    return RiskConfig()


@pytest.fixture
def conservative_risk_config() -> RiskConfig:
    """Conservative RiskConfig with tighter limits."""
    return RiskConfig(
        max_position_pct=0.02,
        max_positions=10,
        risk_per_trade=0.005,
        max_drawdown_halt=0.10,
    )


@pytest.fixture
def aggressive_risk_config() -> RiskConfig:
    """Aggressive RiskConfig with looser limits."""
    return RiskConfig(
        max_position_pct=0.10,
        max_positions=100,
        risk_per_trade=0.02,
        max_drawdown_halt=0.25,
    )


@pytest.fixture
def empty_portfolio(now: datetime) -> PortfolioState:
    """Empty portfolio with $100k cash."""
    return PortfolioState(
        cash=Decimal("100000"),
        positions={},
        timestamp=now,
    )


@pytest.fixture
def portfolio_with_positions(now: datetime) -> PortfolioState:
    """Portfolio with existing positions."""
    return PortfolioState(
        cash=Decimal("50000"),
        positions={
            "AAPL": Position(
                symbol="AAPL",
                quantity=Decimal("100"),
                average_cost=Decimal("150.00"),
                market_value=Decimal("15000"),
                timestamp=now,
            ),
            "GOOGL": Position(
                symbol="GOOGL",
                quantity=Decimal("50"),
                average_cost=Decimal("140.00"),
                market_value=Decimal("7500"),
                timestamp=now,
            ),
        },
        timestamp=now,
    )


@pytest.fixture
def sample_bar_aapl(now: datetime) -> Bar:
    """Sample AAPL bar."""
    return Bar(
        timestamp=now,
        symbol="AAPL",
        open=Decimal("150.00"),
        high=Decimal("152.00"),
        low=Decimal("149.00"),
        close=Decimal("151.50"),
        volume=Decimal("1000000"),
    )


@pytest.fixture
def sample_bar_googl(now: datetime) -> Bar:
    """Sample GOOGL bar."""
    return Bar(
        timestamp=now,
        symbol="GOOGL",
        open=Decimal("140.00"),
        high=Decimal("142.00"),
        low=Decimal("138.00"),
        close=Decimal("141.00"),
        volume=Decimal("500000"),
    )


@pytest.fixture
def sample_bar_msft(now: datetime) -> Bar:
    """Sample MSFT bar."""
    return Bar(
        timestamp=now,
        symbol="MSFT",
        open=Decimal("380.00"),
        high=Decimal("385.00"),
        low=Decimal("378.00"),
        close=Decimal("382.50"),
        volume=Decimal("800000"),
    )


@pytest.fixture
def market_state_single(
    now: datetime,
    sample_bar_aapl: Bar,
) -> MarketState:
    """MarketState with single symbol."""
    return MarketState(
        current_bars={"AAPL": sample_bar_aapl},
        volatility={"AAPL": Decimal("2.50")},
        liquidity={"AAPL": Decimal("50000000")},
        timestamp=now,
    )


@pytest.fixture
def market_state_multi(
    now: datetime,
    sample_bar_aapl: Bar,
    sample_bar_googl: Bar,
    sample_bar_msft: Bar,
) -> MarketState:
    """MarketState with multiple symbols."""
    return MarketState(
        current_bars={
            "AAPL": sample_bar_aapl,
            "GOOGL": sample_bar_googl,
            "MSFT": sample_bar_msft,
        },
        volatility={
            "AAPL": Decimal("2.50"),
            "GOOGL": Decimal("3.20"),
            "MSFT": Decimal("4.00"),
        },
        liquidity={
            "AAPL": Decimal("50000000"),
            "GOOGL": Decimal("20000000"),
            "MSFT": Decimal("30000000"),
        },
        sector_map={
            "AAPL": "Technology",
            "GOOGL": "Technology",
            "MSFT": "Technology",
        },
        timestamp=now,
    )


@pytest.fixture
def long_signal_aapl(now: datetime) -> Signal:
    """Long signal for AAPL."""
    return Signal(
        symbol="AAPL",
        timestamp=now,
        direction="long",
        strength=0.8,
    )


@pytest.fixture
def long_signal_googl(now: datetime) -> Signal:
    """Long signal for GOOGL."""
    return Signal(
        symbol="GOOGL",
        timestamp=now,
        direction="long",
        strength=0.6,
    )


@pytest.fixture
def short_signal_msft(now: datetime) -> Signal:
    """Short signal for MSFT."""
    return Signal(
        symbol="MSFT",
        timestamp=now,
        direction="short",
        strength=0.7,
    )


@pytest.fixture
def flat_signal_aapl(now: datetime) -> Signal:
    """Flat signal for AAPL."""
    return Signal(
        symbol="AAPL",
        timestamp=now,
        direction="flat",
        strength=0.0,
    )


@pytest.fixture
def sample_order_aapl(now: datetime) -> OrderRequest:
    """Sample market buy order for AAPL."""
    return OrderRequest(
        symbol="AAPL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("100"),
        timestamp=now,
    )


@pytest.fixture
def sample_order_googl(now: datetime) -> OrderRequest:
    """Sample market buy order for GOOGL."""
    return OrderRequest(
        symbol="GOOGL",
        side=OrderSide.BUY,
        order_type=OrderType.MARKET,
        quantity=Decimal("50"),
        timestamp=now,
    )
