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

from liq.risk import RiskConfig, MarketState

# Use defaults (zero-config)
config = RiskConfig()

# Or customize
config = RiskConfig(
    max_position_pct=0.05,      # Max 5% per position
    max_positions=20,           # Max 20 concurrent positions
    risk_per_trade=0.01,        # Risk 1% per trade
    max_drawdown_halt=0.15,     # Halt at 15% drawdown
)

# Market state for sizing decisions
market_state = MarketState(
    current_bars={"AAPL": bar},
    volatility={"AAPL": Decimal("2.50")},
    liquidity={"AAPL": Decimal("50000000")},
    timestamp=datetime.now(timezone.utc),
)
```

## Core Concepts

### Protocols

- **PositionSizer**: Protocol for sizing algorithms that transform signals into orders
- **Constraint**: Protocol for risk constraints that filter/modify orders

### Configuration

- **RiskConfig**: All risk parameters with sensible defaults
- **MarketState**: Current market conditions for sizing decisions

## Development

Run tests:

```bash
pytest tests/ -v
```

Run linting and type checks:

```bash
ruff check src/ tests/
mypy src/liq/risk --strict
```

## License

MIT
