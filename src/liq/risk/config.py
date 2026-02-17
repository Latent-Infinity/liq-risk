"""Configuration types for liq-risk.

Provides RiskConfig for risk parameters and MarketState for
current market conditions.
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from liq.risk.enums import HaltMode, PriceReference, SizingMode

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


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
        sizing_mode: How to handle existing positions (INCREMENTAL, REBALANCE, REPLACE).
        price_reference: Which price to use for sizing (MIDRANGE, CLOSE, VWAP).
        stop_loss_atr_mult: Stop-loss distance in ATR multiples.
        take_profit_atr_mult: Take-profit distance in ATR multiples (optional).
        max_drawdown_halt: Halt trading at this drawdown level.
        max_daily_loss_halt: Halt trading at this daily loss level (optional).
        halt_mode: What "halt" means - which orders to block.
        allow_shorts: Allow short selling (False for long-only strategies).
        allow_leverage: Allow gross leverage > 1.0.

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

    # Sizing behavior
    sizing_mode: SizingMode = Field(
        default=SizingMode.REBALANCE,
        description="How to handle existing positions when sizing",
    )
    price_reference: PriceReference = Field(
        default=PriceReference.MIDRANGE,
        description="Which price to use for sizing calculations",
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
    halt_mode: HaltMode = Field(
        default=HaltMode.HALT_BUYS_ONLY,
        description="What 'halt' means - which orders to block",
    )

    # Trading permissions
    allow_shorts: bool = Field(
        default=True,
        description="Allow short selling (False = long-only strategy)",
    )
    allow_leverage: bool = Field(
        default=False,
        description="Allow gross leverage > 1.0",
    )

    # Trading costs (for cost-aware sizing)
    default_borrow_rate: float = Field(
        default=0.0,
        ge=0.0,
        description="Default annualized borrow rate for shorts (0.02 = 2%)",
    )
    default_slippage_pct: float = Field(
        default=0.0,
        ge=0.0,
        description="Default slippage estimate as fraction (0.001 = 0.1%)",
    )
    default_commission_pct: float = Field(
        default=0.0,
        ge=0.0,
        description="Default commission rate as fraction (0.001 = 0.1%)",
    )

    @model_validator(mode="after")
    def validate_leverage_consistency(self) -> RiskConfig:
        """Validate that leverage settings are consistent."""
        # Net leverage should not exceed gross leverage
        if self.max_net_leverage > self.max_gross_leverage:
            raise ValueError(
                f"max_net_leverage ({self.max_net_leverage}) cannot exceed "
                f"max_gross_leverage ({self.max_gross_leverage})"
            )

        # Warn if max_position_pct * max_positions > max_gross_leverage
        max_theoretical = self.max_position_pct * self.max_positions
        if max_theoretical > self.max_gross_leverage:
            warnings.warn(
                f"max_position_pct ({self.max_position_pct}) * max_positions "
                f"({self.max_positions}) = {max_theoretical:.2f} exceeds "
                f"max_gross_leverage ({self.max_gross_leverage}). "
                f"Consider adjusting limits.",
                UserWarning,
                stacklevel=2,
            )
            logger.warning(
                "Config warning: max_position_pct * max_positions = %.2f "
                "exceeds max_gross_leverage = %.2f",
                max_theoretical,
                self.max_gross_leverage,
            )

        return self


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
        borrow_rates: Per-symbol annualized borrow rates (optional).
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
    borrow_rates: dict[str, Decimal] | None = Field(
        default=None,
        description="Per-symbol annualized borrow rates for shorts",
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
