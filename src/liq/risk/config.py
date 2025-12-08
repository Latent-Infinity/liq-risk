"""Configuration types for liq-risk.

Provides RiskConfig for risk parameters and MarketState for
current market conditions.
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)


class RiskConfig(BaseModel):
    """Risk parameters for position sizing.

    All percentages are expressed as decimals (0.05 = 5%).

    This class supports zero-config usage: all fields have sensible
    defaults that provide a conservative starting point.

    Attributes:
        max_position_pct: Maximum position size as fraction of equity.
        max_positions: Maximum number of concurrent positions.
        min_position_value: Minimum order value (filters tiny orders).
        max_sector_pct: Maximum exposure to any single sector.
        max_gross_leverage: Maximum gross exposure / equity ratio.
        max_net_leverage: Maximum net exposure / equity ratio.
        max_correlation: Maximum average pairwise correlation (optional).
        risk_per_trade: Fraction of equity to risk per trade.
        kelly_fraction: Fractional Kelly multiplier for safety.
        vol_target: Target portfolio volatility (optional).
        stop_loss_atr_mult: Stop-loss distance in ATR multiples.
        take_profit_atr_mult: Take-profit distance in ATR multiples (optional).
        max_drawdown_halt: Halt trading at this drawdown level.
        max_daily_loss_halt: Halt trading at this daily loss level (optional).

    Example:
        >>> config = RiskConfig()  # Use all defaults
        >>> config.max_position_pct
        0.05

        >>> config = RiskConfig(max_position_pct=0.10, max_positions=20)
        >>> config.max_position_pct
        0.10
    """

    model_config = ConfigDict(frozen=True)

    # Position limits
    max_position_pct: float = Field(
        default=0.05,
        gt=0.0,
        le=1.0,
        description="Max position size as fraction of equity (0.05 = 5%)",
    )
    max_positions: int = Field(
        default=50,
        gt=0,
        description="Maximum number of concurrent positions",
    )
    min_position_value: Decimal = Field(
        default=Decimal("100"),
        ge=Decimal("0"),
        description="Minimum order notional value",
    )

    # Exposure limits
    max_sector_pct: float = Field(
        default=0.30,
        gt=0.0,
        le=1.0,
        description="Max exposure to any single sector (0.30 = 30%)",
    )
    max_gross_leverage: float = Field(
        default=1.0,
        gt=0.0,
        description="Max gross exposure / equity ratio",
    )
    max_net_leverage: float = Field(
        default=1.0,
        gt=0.0,
        description="Max net exposure / equity ratio",
    )
    max_correlation: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Max average pairwise correlation",
    )

    # Sizing parameters
    risk_per_trade: float = Field(
        default=0.01,
        gt=0.0,
        le=1.0,
        description="Fraction of equity to risk per trade (0.01 = 1%)",
    )
    kelly_fraction: float = Field(
        default=0.25,
        gt=0.0,
        le=1.0,
        description="Fractional Kelly multiplier (0.25 = quarter Kelly)",
    )
    vol_target: float | None = Field(
        default=None,
        gt=0.0,
        description="Target portfolio volatility (annualized)",
    )

    # Risk controls
    stop_loss_atr_mult: float = Field(
        default=2.0,
        gt=0.0,
        description="Stop-loss distance in ATR multiples",
    )
    take_profit_atr_mult: float | None = Field(
        default=None,
        gt=0.0,
        description="Take-profit distance in ATR multiples",
    )
    max_drawdown_halt: float = Field(
        default=0.15,
        gt=0.0,
        le=1.0,
        description="Halt new buys at this drawdown level (0.15 = 15%)",
    )
    max_daily_loss_halt: float | None = Field(
        default=None,
        gt=0.0,
        le=1.0,
        description="Halt trading at this daily loss level",
    )


class MarketState(BaseModel):
    """Current market conditions for sizing decisions.

    Provides the context needed for volatility-based sizing
    and constraint checking.

    Attributes:
        current_bars: Most recent bar for each symbol.
        volatility: ATR or range-based volatility per symbol.
        liquidity: Average daily volume per symbol.
        sector_map: Symbol to sector mapping (optional).
        correlations: Pairwise correlation matrix (optional).
        regime: Market regime label (optional).
        timestamp: State snapshot time (UTC, timezone-aware).

    Example:
        >>> from datetime import datetime, timezone
        >>> now = datetime.now(timezone.utc)
        >>> state = MarketState(
        ...     current_bars={"AAPL": bar},
        ...     volatility={"AAPL": Decimal("2.50")},
        ...     liquidity={"AAPL": Decimal("50000000")},
        ...     timestamp=now,
        ... )
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    current_bars: dict[str, Any] = Field(
        description="Most recent bar for each symbol (symbol -> Bar)",
    )
    volatility: dict[str, Decimal] = Field(
        description="ATR or range-based volatility per symbol",
    )
    liquidity: dict[str, Decimal] = Field(
        description="Average daily volume per symbol",
    )
    sector_map: dict[str, str] | None = Field(
        default=None,
        description="Symbol to sector mapping",
    )
    correlations: Any | None = Field(
        default=None,
        description="Pairwise correlation matrix (polars.DataFrame)",
    )
    regime: str | None = Field(
        default=None,
        description="Market regime label (e.g., 'high_volatility', 'low_volatility')",
    )
    timestamp: datetime = Field(
        description="State snapshot time (UTC, timezone-aware)",
    )

    @field_validator("timestamp")
    @classmethod
    def validate_timestamp_timezone(cls, v: datetime) -> datetime:
        """Ensure timestamp is timezone-aware."""
        if v.tzinfo is None or v.tzinfo.utcoffset(v) is None:
            raise ValueError("timestamp must be timezone-aware (UTC expected)")
        return v
