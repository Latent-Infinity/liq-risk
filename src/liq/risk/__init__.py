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
    - TargetPosition: Execution-agnostic position target
    - ConstraintResult: Structured constraint output with audit trail
"""

__all__ = [
    # Core Types
    "TargetPosition",
    "RejectedOrder",
    "ConstraintResult",
    "RoundingPolicy",
    # Enums
    "HaltMode",
    "SizingMode",
    "PriceReference",
    # Configuration
    "RiskConfig",
    "MarketState",
    # Layered State
    "PriceState",
    "RiskFactors",
    "AssetMetadata",
    "ExecutionState",
    # Protocols
    "PositionSizer",
    "TargetPositionSizer",
    "Constraint",
    "StructuredConstraint",
    # Engine
    "RiskEngine",
    "RiskEngineResult",
    # Exceptions
    "RiskError",
    "InsufficientBuyingPowerError",
    "LeverageExceededError",
    "EquityFloorBreachedError",
    "TradingHaltedError",
    # Sizers
    "VolatilitySizer",
    "FixedFractionalSizer",
    "EqualWeightSizer",
    "KellySizer",
    "RiskParitySizer",
    # Constraints
    "Constraint",
    "MaxPositionConstraint",
    "MaxPositionsConstraint",
    "MinPositionValueConstraint",
    "GrossLeverageConstraint",
    "NetLeverageConstraint",
    "SectorExposureConstraint",
    "CorrelationConstraint",
    "BuyingPowerConstraint",
    "ShortSellingConstraint",
]

from liq.risk.config import MarketState, RiskConfig
from liq.risk.constraints import (
    BuyingPowerConstraint,
    CorrelationConstraint,
    GrossLeverageConstraint,
    MaxPositionConstraint,
    MaxPositionsConstraint,
    MinPositionValueConstraint,
    NetLeverageConstraint,
    SectorExposureConstraint,
    ShortSellingConstraint,
)
from liq.risk.engine import RiskEngine, RiskEngineResult
from liq.risk.enums import HaltMode, PriceReference, SizingMode
from liq.risk.exceptions import (
    EquityFloorBreachedError,
    InsufficientBuyingPowerError,
    LeverageExceededError,
    RiskError,
    TradingHaltedError,
)
from liq.risk.protocols import Constraint, PositionSizer, StructuredConstraint, TargetPositionSizer
from liq.risk.sizers import (
    EqualWeightSizer,
    FixedFractionalSizer,
    KellySizer,
    RiskParitySizer,
    VolatilitySizer,
)
from liq.risk.state import AssetMetadata, ExecutionState, PriceState, RiskFactors
from liq.risk.types import ConstraintResult, RejectedOrder, RoundingPolicy, TargetPosition

__version__ = "0.1.0"
