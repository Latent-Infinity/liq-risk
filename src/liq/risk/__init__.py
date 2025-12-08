"""Position sizing and risk management for the LIQ Stack.

liq-risk serves as 'The Bridge Between Prediction and Execution',
transforming trading signals into sized order requests while enforcing
risk constraints.

Core Concepts:
    - PositionSizer: Protocol for sizing algorithms
    - Constraint: Protocol for risk constraints
    - RiskEngine: Orchestrates sizing and constraint application
    - RiskConfig: Configuration for risk parameters
    - MarketState: Current market conditions for sizing decisions
"""

__all__ = [
    # Configuration
    "RiskConfig",
    "MarketState",
    # Protocols
    "PositionSizer",
    "Constraint",
    # Sizers
    "VolatilitySizer",
    "FixedFractionalSizer",
    # Constraints
    "MaxPositionConstraint",
    "MaxPositionsConstraint",
    "MinPositionValueConstraint",
    "GrossLeverageConstraint",
]

from liq.risk.config import MarketState, RiskConfig
from liq.risk.constraints import (
    GrossLeverageConstraint,
    MaxPositionConstraint,
    MaxPositionsConstraint,
    MinPositionValueConstraint,
)
from liq.risk.protocols import Constraint, PositionSizer
from liq.risk.sizers import FixedFractionalSizer, VolatilitySizer

__version__ = "0.1.0"
