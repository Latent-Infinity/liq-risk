"""Performance benchmarks for liq-risk.

Targets:
- Size 1000 signals: < 10ms
- Apply constraint chain: < 5ms
"""

from __future__ import annotations

import time
from datetime import UTC, datetime
from decimal import Decimal

from liq.core import Bar, PortfolioState

from liq.risk import MarketState, RiskConfig, RiskEngine
from liq.risk.constraints import (
    CorrelationConstraint,
    GrossLeverageConstraint,
    MaxPositionConstraint,
    MaxPositionsConstraint,
    MinPositionValueConstraint,
    SectorExposureConstraint,
)


def create_signals(n: int, now: datetime) -> list:
    """Create n test signals."""
    from liq.signals import Signal

    symbols = [f"SYM{i:04d}" for i in range(n)]
    return [
        Signal(
            symbol=symbol,
            timestamp=now,
            direction="long",
            strength=0.7,
        )
        for symbol in symbols
    ]


def create_market_state(symbols: list[str], now: datetime) -> MarketState:
    """Create market state with bars and volatility for all symbols."""
    bars = {}
    volatility = {}
    liquidity = {}
    sector_map = {}

    sectors = ["Technology", "Healthcare", "Financials", "Energy", "Consumer"]

    for i, symbol in enumerate(symbols):
        bars[symbol] = Bar(
            timestamp=now,
            symbol=symbol,
            open=Decimal("100"),
            high=Decimal("102"),
            low=Decimal("98"),
            close=Decimal("100"),
            volume=Decimal("1000000"),
        )
        volatility[symbol] = Decimal("2.00")
        liquidity[symbol] = Decimal("50000000")
        sector_map[symbol] = sectors[i % len(sectors)]

    return MarketState(
        current_bars=bars,
        volatility=volatility,
        liquidity=liquidity,
        sector_map=sector_map,
        timestamp=now,
    )


def benchmark_size_signals(n_signals: int = 1000, iterations: int = 100) -> float:
    """Benchmark sizing n signals through the engine.

    Returns average time in milliseconds.
    """
    now = datetime.now(UTC)
    signals = create_signals(n_signals, now)
    symbols = [s.symbol for s in signals]
    market_state = create_market_state(symbols, now)
    portfolio = PortfolioState(
        cash=Decimal("10000000"),  # $10M
        positions={},
        timestamp=now,
    )
    config = RiskConfig(max_positions=500)
    engine = RiskEngine()

    # Warmup
    engine.process_signals(signals, portfolio, market_state, config)

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        engine.process_signals(signals, portfolio, market_state, config)
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)  # Convert to ms

    avg_ms = sum(times) / len(times)
    return avg_ms


def benchmark_constraint_chain(n_orders: int = 1000, iterations: int = 100) -> float:
    """Benchmark applying constraint chain to n orders.

    Returns average time in milliseconds.
    """
    from liq.core import OrderRequest, OrderSide, OrderType

    now = datetime.now(UTC)
    symbols = [f"SYM{i:04d}" for i in range(n_orders)]
    market_state = create_market_state(symbols, now)
    portfolio = PortfolioState(
        cash=Decimal("10000000"),
        positions={},
        timestamp=now,
    )
    config = RiskConfig(
        max_position_pct=0.05,
        max_positions=500,
        max_sector_pct=0.30,
        max_correlation=0.8,
    )

    orders = [
        OrderRequest(
            symbol=symbol,
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("100"),
            timestamp=now,
        )
        for symbol in symbols
    ]

    constraints = [
        MinPositionValueConstraint(),
        MaxPositionConstraint(),
        MaxPositionsConstraint(),
        GrossLeverageConstraint(),
        SectorExposureConstraint(),
        CorrelationConstraint(),
    ]

    # Warmup
    result = orders
    for c in constraints:
        output = c.apply(result, portfolio, market_state, config)
        if isinstance(output, list):
            result = output
        else:
            result = output.orders

    # Benchmark
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
    result = orders
    for c in constraints:
        output = c.apply(result, portfolio, market_state, config)
        if isinstance(output, list):
            result = output
        else:
            result = output.orders
        elapsed = time.perf_counter() - start
        times.append(elapsed * 1000)

    avg_ms = sum(times) / len(times)
    return avg_ms


def main() -> None:
    """Run benchmarks and report results."""
    print("liq-risk Performance Benchmarks")
    print("=" * 50)
    print()

    # Benchmark 1: Size 1000 signals
    print("Benchmark 1: Size 1000 signals through RiskEngine")
    avg_ms = benchmark_size_signals(n_signals=1000, iterations=100)
    status = "PASS" if avg_ms < 10 else "FAIL"
    print(f"  Average: {avg_ms:.2f}ms (target: <10ms) [{status}]")
    print()

    # Benchmark 2: Apply constraint chain
    print("Benchmark 2: Apply constraint chain to 1000 orders")
    avg_ms = benchmark_constraint_chain(n_orders=1000, iterations=100)
    status = "PASS" if avg_ms < 5 else "FAIL"
    print(f"  Average: {avg_ms:.2f}ms (target: <5ms) [{status}]")
    print()

    # Additional benchmarks
    print("Additional Benchmarks:")
    print()

    # Scaling test
    for n in [100, 500, 1000, 2000]:
        avg_ms = benchmark_size_signals(n_signals=n, iterations=50)
        print(f"  Size {n:4d} signals: {avg_ms:.2f}ms")

    print()
    print("=" * 50)


if __name__ == "__main__":
    main()
