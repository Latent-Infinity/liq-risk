"""Basic position sizing example.

This example demonstrates the core workflow:
1. Create market state with current prices and volatility
2. Configure risk parameters
3. Process signals through the risk engine
4. Receive sized orders with stop-losses
"""

from datetime import UTC, datetime
from decimal import Decimal

from liq.core import Bar, PortfolioState
from liq.signals import Signal

from liq.risk import MarketState, RiskConfig, RiskEngine


def main() -> None:
    """Run basic sizing example."""
    now = datetime.now(UTC)

    # Current market data
    bar_aapl = Bar(
        timestamp=now,
        symbol="AAPL",
        open=Decimal("150.00"),
        high=Decimal("152.00"),
        low=Decimal("148.00"),
        close=Decimal("151.00"),
        volume=Decimal("50000000"),
    )

    bar_googl = Bar(
        timestamp=now,
        symbol="GOOGL",
        open=Decimal("140.00"),
        high=Decimal("142.00"),
        low=Decimal("138.00"),
        close=Decimal("141.00"),
        volume=Decimal("20000000"),
    )

    # Market state with volatility (ATR)
    market_state = MarketState(
        current_bars={"AAPL": bar_aapl, "GOOGL": bar_googl},
        volatility={"AAPL": Decimal("2.50"), "GOOGL": Decimal("3.20")},
        liquidity={"AAPL": Decimal("50000000"), "GOOGL": Decimal("20000000")},
        timestamp=now,
    )

    # Current portfolio
    portfolio = PortfolioState(
        cash=Decimal("100000"),
        positions={},
        timestamp=now,
    )

    # Risk configuration (or use defaults with RiskConfig())
    config = RiskConfig(
        max_position_pct=0.05,  # Max 5% per position
        max_positions=20,  # Max 20 positions
        risk_per_trade=0.01,  # Risk 1% per trade
        stop_loss_atr_mult=2.0,  # 2x ATR stop
        max_drawdown_halt=0.15,  # Halt at 15% drawdown
    )

    # Trading signals from your model
    signals = [
        Signal(
            symbol="AAPL",
            timestamp=now,
            direction="long",
            strength=0.8,  # High confidence
        ),
        Signal(
            symbol="GOOGL",
            timestamp=now,
            direction="long",
            strength=0.6,  # Medium confidence
        ),
    ]

    # Create risk engine and process signals
    engine = RiskEngine()
    result = engine.process_signals(signals, portfolio, market_state, config)

    # Display results
    print("=" * 60)
    print("Basic Sizing Example")
    print("=" * 60)
    print()

    if result.halted:
        print(f"TRADING HALTED: {result.halt_reason}")
        return

    print("Generated Orders:")
    print("-" * 40)
    for order in result.orders:
        stop = result.stop_losses.get(order.symbol, "N/A")
        print(f"  {order.side.value:4} {order.quantity:>6} {order.symbol}")
        print(f"       Stop-loss: ${stop}")
        print()

    if result.rejected_signals:
        print("Rejected Signals:")
        print("-" * 40)
        for signal in result.rejected_signals:
            print(f"  {signal.symbol}: {signal.direction}")

    print("=" * 60)


if __name__ == "__main__":
    main()
