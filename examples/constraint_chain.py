"""Constraint chain example.

This example demonstrates:
1. How constraints filter and modify orders
2. Custom constraint chains
3. Sector exposure limiting
4. Correlation-based diversification
"""

from datetime import UTC, datetime
from decimal import Decimal

from liq.core import Bar, PortfolioState
from liq.signals import Signal

from liq.risk import (
    CorrelationConstraint,
    GrossLeverageConstraint,
    MarketState,
    MaxPositionConstraint,
    MaxPositionsConstraint,
    MinPositionValueConstraint,
    RiskConfig,
    RiskEngine,
    SectorExposureConstraint,
)


def main() -> None:
    """Demonstrate constraint chain behavior."""
    now = datetime.now(UTC)

    # Market data for tech stocks
    bars = {
        "AAPL": Bar(
            timestamp=now, symbol="AAPL",
            open=Decimal("150"), high=Decimal("152"),
            low=Decimal("148"), close=Decimal("150"),
            volume=Decimal("50000000"),
        ),
        "MSFT": Bar(
            timestamp=now, symbol="MSFT",
            open=Decimal("380"), high=Decimal("385"),
            low=Decimal("375"), close=Decimal("380"),
            volume=Decimal("30000000"),
        ),
        "GOOGL": Bar(
            timestamp=now, symbol="GOOGL",
            open=Decimal("140"), high=Decimal("142"),
            low=Decimal("138"), close=Decimal("140"),
            volume=Decimal("20000000"),
        ),
        "XOM": Bar(
            timestamp=now, symbol="XOM",
            open=Decimal("100"), high=Decimal("102"),
            low=Decimal("98"), close=Decimal("100"),
            volume=Decimal("15000000"),
        ),
        "JPM": Bar(
            timestamp=now, symbol="JPM",
            open=Decimal("180"), high=Decimal("182"),
            low=Decimal("178"), close=Decimal("180"),
            volume=Decimal("10000000"),
        ),
    }

    # Sector mapping
    sector_map = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "GOOGL": "Technology",
        "XOM": "Energy",
        "JPM": "Financials",
    }

    # Correlation matrix (tech stocks are highly correlated)
    correlations = {
        ("AAPL", "MSFT"): 0.85,
        ("MSFT", "AAPL"): 0.85,
        ("AAPL", "GOOGL"): 0.80,
        ("GOOGL", "AAPL"): 0.80,
        ("MSFT", "GOOGL"): 0.82,
        ("GOOGL", "MSFT"): 0.82,
        ("AAPL", "XOM"): 0.30,
        ("XOM", "AAPL"): 0.30,
        ("AAPL", "JPM"): 0.45,
        ("JPM", "AAPL"): 0.45,
        ("XOM", "JPM"): 0.35,
        ("JPM", "XOM"): 0.35,
    }

    market_state = MarketState(
        current_bars=bars,
        volatility={s: Decimal("3") for s in bars},
        liquidity={s: Decimal("10000000") for s in bars},
        sector_map=sector_map,
        correlations=correlations,
        timestamp=now,
    )

    portfolio = PortfolioState(
        cash=Decimal("100000"),
        positions={},
        timestamp=now,
    )

    # Restrictive config for demo
    config = RiskConfig(
        max_position_pct=0.10,  # 10% max per position
        max_positions=3,  # Only allow 3 positions
        max_sector_pct=0.25,  # 25% max per sector
        max_correlation=0.70,  # Filter highly correlated assets
        min_position_value=100,  # Min $100 per position
        risk_per_trade=0.05,  # Higher risk per trade for demo
    )

    # Signals for all 5 stocks
    signals = [
        Signal(symbol="AAPL", timestamp=now, direction="long", strength=0.9),
        Signal(symbol="MSFT", timestamp=now, direction="long", strength=0.85),
        Signal(symbol="GOOGL", timestamp=now, direction="long", strength=0.8),
        Signal(symbol="XOM", timestamp=now, direction="long", strength=0.7),
        Signal(symbol="JPM", timestamp=now, direction="long", strength=0.65),
    ]

    # Build custom constraint chain
    constraints = [
        MinPositionValueConstraint(),  # Filter tiny orders first
        MaxPositionConstraint(),  # Limit individual positions
        MaxPositionsConstraint(),  # Limit total count
        SectorExposureConstraint(),  # Limit sector concentration
        CorrelationConstraint(),  # Ensure diversification
        GrossLeverageConstraint(),  # Final leverage check
    ]

    engine = RiskEngine(constraints=constraints)
    result = engine.process_signals(signals, portfolio, market_state, config)

    print("=" * 70)
    print("Constraint Chain Example")
    print("=" * 70)
    print()
    print("Input: 5 signals (AAPL, MSFT, GOOGL, XOM, JPM)")
    print("Constraints applied:")
    print("  - Max 3 positions")
    print("  - Max 25% per sector")
    print("  - Max 70% correlation allowed")
    print()

    print("Resulting Orders:")
    print("-" * 50)
    for order in result.orders:
        sector = sector_map[order.symbol]
        print(f"  {order.symbol}: {order.quantity} shares (Sector: {sector})")
    print()

    if result.rejected_signals:
        print("Rejected Signals:")
        print("-" * 50)
        for signal in result.rejected_signals:
            print(f"  {signal.symbol} (strength: {signal.strength})")
        print()

    if result.constraint_violations:
        print("Constraint Violations:")
        print("-" * 50)
        for constraint, symbols in result.constraint_violations.items():
            print(f"  {constraint}: {', '.join(symbols)}")
        print()

    print("=" * 70)
    print("Note: The constraint chain ensures a diversified portfolio by")
    print("limiting sector concentration and filtering correlated assets.")
    print("=" * 70)


if __name__ == "__main__":
    main()
