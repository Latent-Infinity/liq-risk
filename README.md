# liq-risk

Position sizing and risk management for the LIQ Stack.

**liq-risk** serves as "The Bridge Between Prediction and Execution" — transforming trading signals into sized order requests while enforcing risk constraints.

## Installation

```bash
pip install liq-risk
```

For development:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from datetime import datetime, timezone
from decimal import Decimal

from liq.core import Bar, PortfolioState, Position
from liq.signals import Signal
from liq.risk import RiskEngine, RiskConfig, MarketState

# Create market state
now = datetime.now(timezone.utc)
bar = Bar(
    timestamp=now,
    symbol="AAPL",
    open=Decimal("150.00"),
    high=Decimal("152.00"),
    low=Decimal("148.00"),
    close=Decimal("151.00"),
    volume=Decimal("1000000"),
)

market_state = MarketState(
    current_bars={"AAPL": bar},
    volatility={"AAPL": Decimal("2.50")},  # ATR
    liquidity={"AAPL": Decimal("50000000")},
    timestamp=now,
)

# Create portfolio state
portfolio = PortfolioState(
    cash=Decimal("100000"),
    positions={},
    timestamp=now,
)

# Create risk config (or use defaults)
config = RiskConfig(
    max_position_pct=0.05,      # Max 5% per position
    max_positions=20,           # Max 20 concurrent positions
    risk_per_trade=0.01,        # Risk 1% per trade
    max_drawdown_halt=0.15,     # Halt at 15% drawdown
)

# Create a signal
signal = Signal(
    symbol="AAPL",
    direction="long",
    strength=0.75,
    timestamp=now,
)

# Process through risk engine
engine = RiskEngine()
result = engine.process_signals([signal], portfolio, market_state, config)

# Execute resulting orders
for order in result.orders:
    print(f"Order: {order.side} {order.quantity} {order.symbol}")

# Check for halt conditions
if result.halted:
    print(f"Trading halted: {result.halt_reason}")
```

## Core Concepts

### RiskEngine

The `RiskEngine` is the main orchestrator that transforms signals into orders:

1. **Check drawdown halt** - Block new buys if drawdown exceeds threshold
2. **Size positions** - Convert signals to orders using a sizing algorithm
3. **Apply constraints** - Filter/scale orders through constraint chain
4. **Calculate stop-losses** - ATR-based stop prices for risk management

```python
from liq.risk import RiskEngine
from liq.risk.sizers import KellySizer
from liq.risk.constraints import MaxPositionConstraint, SectorExposureConstraint

# Custom sizer and constraints
engine = RiskEngine(
    sizer=KellySizer(),
    constraints=[MaxPositionConstraint(), SectorExposureConstraint()],
)

result = engine.process_signals(signals, portfolio, market, config)
```

### Position Sizers

Sizers implement the `PositionSizer` protocol to determine order quantities.

#### VolatilitySizer (Default)

Sizes positions based on ATR/volatility to normalize risk across instruments.

```python
from liq.risk.sizers import VolatilitySizer

sizer = VolatilitySizer()
# Position = (equity × risk_per_trade) / (ATR × price)
```

#### EqualWeightSizer

Divides capital equally among signals, regardless of volatility.

```python
from liq.risk.sizers import EqualWeightSizer

sizer = EqualWeightSizer()
# Position = equity / n_signals / price
```

#### KellySizer

Uses the Kelly Criterion for optimal position sizing based on signal strength.

```python
from liq.risk.sizers import KellySizer

sizer = KellySizer()
# Full Kelly: f* = 2p - 1 (where p = signal.strength)
# Fractional Kelly: f* × kelly_fraction (default 0.25)
```

#### FixedFractionalSizer

Allocates a fixed percentage of equity per position.

```python
from liq.risk.sizers import FixedFractionalSizer

sizer = FixedFractionalSizer()
# Position = equity × risk_per_trade / price
```

#### RiskParitySizer

Equal risk contribution from each position (inverse volatility weighting).

```python
from liq.risk.sizers import RiskParitySizer

sizer = RiskParitySizer()
# weight_i = (1/vol_i) / Σ(1/vol_j)
# Higher volatility assets get smaller positions
```

#### CryptoFractionalSizer

Designed for cryptocurrency with fractional quantities and step sizes.

```python
from decimal import Decimal
from liq.risk.sizers import CryptoFractionalSizer

sizer = CryptoFractionalSizer(
    fraction=0.02,                    # 2% of equity per position
    min_qty=Decimal("0.0001"),        # Minimum quantity
    step_qty=Decimal("0.0001"),       # Step size for rounding
)
```

### Constraints

Constraints implement the `Constraint` protocol to filter/scale orders.

#### MaxPositionConstraint

Limits individual position size as percentage of equity.

```python
from liq.risk.constraints import MaxPositionConstraint

# Scales orders to fit within max_position_pct (default 5%)
constraint = MaxPositionConstraint()
```

#### MaxPositionsConstraint

Limits total number of concurrent positions.

```python
from liq.risk.constraints import MaxPositionsConstraint

# Drops lowest-strength orders beyond max_positions (default 50)
constraint = MaxPositionsConstraint()
```

#### GrossLeverageConstraint

Limits total portfolio exposure relative to equity.

```python
from liq.risk.constraints import GrossLeverageConstraint

# Scales orders to fit within max_gross_leverage (default 1.0)
constraint = GrossLeverageConstraint()
```

#### MinPositionValueConstraint

Filters out orders below minimum notional value.

```python
from liq.risk.constraints import MinPositionValueConstraint

# Drops orders below min_position_value (default $100)
constraint = MinPositionValueConstraint()
```

#### SectorExposureConstraint

Limits exposure to any single sector.

```python
from liq.risk.constraints import SectorExposureConstraint

# Requires sector_map in MarketState
market_state = MarketState(
    current_bars=bars,
    volatility=volatility,
    liquidity=liquidity,
    sector_map={"AAPL": "Technology", "JPM": "Financials"},
    timestamp=now,
)

# Scales orders to fit within max_sector_pct (default 30%)
constraint = SectorExposureConstraint()
```

#### CorrelationConstraint

Filters highly correlated assets for diversification.

```python
from liq.risk.constraints import CorrelationConstraint

# Requires correlation_matrix in MarketState
market_state = MarketState(
    current_bars=bars,
    volatility=volatility,
    liquidity=liquidity,
    correlation_matrix={
        ("AAPL", "MSFT"): Decimal("0.85"),
        ("AAPL", "GOOGL"): Decimal("0.72"),
    },
    timestamp=now,
)

# Drops orders correlated above max_correlation (default 0.8)
constraint = CorrelationConstraint()
```

### Short Position Handling

All constraints correctly handle short positions:

- **SELL orders that close/reduce long positions**: Pass freely (reduce exposure)
- **SELL orders that initiate/increase short positions**: Constrained like buys

```python
# Short signals generate SELL orders
signal = Signal(symbol="AAPL", direction="short", strength=0.75, timestamp=now)

# Constraints treat new short positions like new long positions
result = engine.process_signals([signal], portfolio, market, config)
# MaxPositionConstraint: Short position limited to max_position_pct
# GrossLeverageConstraint: Short exposure counts toward gross leverage
# MaxPositionsConstraint: New short counts as new position
```

### Configuration

#### RiskConfig

All risk parameters with sensible defaults (zero-config friendly):

```python
from liq.risk import RiskConfig

# Use all defaults
config = RiskConfig()

# Or customize
config = RiskConfig(
    # Position limits
    max_position_pct=0.05,       # Max 5% per position
    max_positions=50,            # Max 50 positions
    min_position_value=100,      # Min $100 per order

    # Exposure limits
    max_sector_pct=0.30,         # Max 30% per sector
    max_gross_leverage=1.0,      # Max 100% gross exposure
    max_net_leverage=1.0,        # Max 100% net exposure

    # Sizing parameters
    risk_per_trade=0.01,         # Risk 1% per trade
    kelly_fraction=0.25,         # Use quarter Kelly

    # Risk controls
    stop_loss_atr_mult=2.0,      # Stop at 2× ATR
    max_drawdown_halt=0.15,      # Halt at 15% drawdown

    # Trading costs (for cost-aware sizing)
    default_borrow_rate=0.02,    # 2% annual borrow for shorts
    default_slippage_pct=0.001,  # 0.1% slippage
    default_commission_pct=0.0005,  # 0.05% commission
)
```

#### MarketState

Current market conditions for sizing decisions:

```python
from liq.risk import MarketState

state = MarketState(
    current_bars={"AAPL": bar},           # Latest price bars
    volatility={"AAPL": Decimal("2.50")}, # ATR per symbol
    liquidity={"AAPL": Decimal("50M")},   # Daily volume
    sector_map={"AAPL": "Technology"},    # Optional sector mapping
    borrow_rates={"GME": Decimal("0.50")},  # Optional per-symbol borrow rates
    regime="normal",                       # Optional market regime
    timestamp=datetime.now(timezone.utc),  # Must be timezone-aware
)
```

## Stop-Loss and Take-Profit Calculation

The engine calculates ATR-based stops and targets:

```python
# Stop-loss (always calculated)
# Long: stop = entry - (ATR × stop_loss_atr_mult)
# Short: stop = entry + (ATR × stop_loss_atr_mult)

# Take-profit (optional, set take_profit_atr_mult)
# Long: target = entry + (ATR × take_profit_atr_mult)
# Short: target = entry - (ATR × take_profit_atr_mult)

config = RiskConfig(
    stop_loss_atr_mult=2.0,    # Stop at 2× ATR
    take_profit_atr_mult=3.0,  # Take profit at 3× ATR (risk:reward = 1:1.5)
)

result = engine.process_signals(signals, portfolio, market, config)

for symbol, stop_price in result.stop_losses.items():
    print(f"{symbol} stop-loss: {stop_price}")

for symbol, target_price in result.take_profits.items():
    print(f"{symbol} take-profit: {target_price}")
```

## Drawdown Halt

Trading can be automatically halted during drawdowns:

```python
# Pass high water mark to enable drawdown tracking
result = engine.process_signals(
    signals, portfolio, market, config,
    high_water_mark=Decimal("120000"),  # Peak equity
)

if result.halted:
    # Only sell orders allowed during halt
    print(f"HALTED: {result.halt_reason}")
```

## Development

Run tests:

```bash
pytest tests/ -v
```

Run linting and type checks:

```bash
ruff check src/ tests/
mypy src/liq/risk --ignore-missing-imports
```

## License

MIT
