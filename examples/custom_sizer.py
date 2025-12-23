"""Custom sizer example.

This example demonstrates how to:
1. Use different built-in sizers (Kelly, Equal Weight)
2. Compare sizing results across strategies
"""

from datetime import UTC, datetime
from decimal import Decimal

from liq.core import Bar, PortfolioState
from liq.signals import Signal

from liq.risk import (
    EqualWeightSizer,
    KellySizer,
    MarketState,
    RiskConfig,
    RiskEngine,
    RiskParitySizer,
    VolatilitySizer,
)


def main() -> None:
    """Compare different sizing strategies."""
    now = datetime.now(UTC)

    # Market data
    bars = {
        "AAPL": Bar(
            timestamp=now,
            symbol="AAPL",
            open=Decimal("150"),
            high=Decimal("152"),
            low=Decimal("148"),
            close=Decimal("151"),
            volume=Decimal("50000000"),
        ),
        "MSFT": Bar(
            timestamp=now,
            symbol="MSFT",
            open=Decimal("380"),
            high=Decimal("385"),
            low=Decimal("375"),
            close=Decimal("382"),
            volume=Decimal("30000000"),
        ),
        "GOOGL": Bar(
            timestamp=now,
            symbol="GOOGL",
            open=Decimal("140"),
            high=Decimal("142"),
            low=Decimal("138"),
            close=Decimal("141"),
            volume=Decimal("20000000"),
        ),
    }

    market_state = MarketState(
        current_bars=bars,
        volatility={
            "AAPL": Decimal("2.50"),
            "MSFT": Decimal("5.00"),
            "GOOGL": Decimal("3.20"),
        },
        liquidity={
            "AAPL": Decimal("50000000"),
            "MSFT": Decimal("30000000"),
            "GOOGL": Decimal("20000000"),
        },
        timestamp=now,
    )

    portfolio = PortfolioState(
        cash=Decimal("100000"),
        positions={},
        timestamp=now,
    )

    config = RiskConfig(
        max_position_pct=0.10,  # Allow larger positions for demo
        kelly_fraction=0.25,  # Quarter Kelly
    )

    signals = [
        Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.75),
        Signal(symbol="MSFT", timestamp=now, direction="long", strength=0.65),
        Signal(symbol="GOOGL", timestamp=now, direction="long", strength=0.55),
    ]

    # Compare different sizers
    sizers = [
        ("Volatility Sizer", VolatilitySizer()),
        ("Equal Weight Sizer", EqualWeightSizer()),
        ("Kelly Sizer", KellySizer()),
        ("Risk Parity Sizer", RiskParitySizer()),
    ]

    print("=" * 70)
    print("Custom Sizer Comparison")
    print("=" * 70)
    print()

    for name, sizer in sizers:
        engine = RiskEngine(sizer=sizer)
        result = engine.process_signals(signals, portfolio, market_state, config)

        print(f"{name}:")
        print("-" * 50)
        total_value = Decimal("0")
        for order in result.orders:
            price = bars[order.symbol].close
            value = order.quantity * price
            total_value += value
            print(f"  {order.symbol}: {order.quantity:>6} shares (${value:,.2f})")

        print(f"  Total: ${total_value:,.2f} ({total_value / portfolio.equity * 100:.1f}%)")
        print()

    print("=" * 70)
    print("Note: Different sizers produce different position sizes based on")
    print("their algorithms (volatility-adjusted, equal weight, or Kelly criterion).")
    print("=" * 70)


if __name__ == "__main__":
    main()
