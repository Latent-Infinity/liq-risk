"""Enum reason codes for the sizing pipeline.

Every transformation in the ordered pipeline that changes ``size_vol``
emits a :class:`ReasonCode`. The code's ``stage`` identifies which step of
the pipeline fired it; the ``severity`` distinguishes informational codes
(``info``, ``ok``) from conservative-fallback events (``warn``,
``conservative``) that the sizing-layer audit must surface.

The code constants are the canonical strings; consumers may compare with
string equality or via the :class:`ReasonCode` instances. The catalog below
covers the F5 (sizing pipeline) steps. F5H will add the shock-override
codes; ``MODEL_STALE_FALLBACK`` already ships in ``liq-models``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Severity = Literal["ok", "info", "warn", "conservative"]
Stage = Literal[
    "select",
    "combination",
    "fallback",
    "uncertainty",
    "floor",
    "cap",
    "smoothing",
    "shock",
]


@dataclass(frozen=True)
class ReasonCode:
    """Single sigil emitted by a sizing-pipeline step."""

    code: str
    stage: Stage
    severity: Severity
    description: str

    def __str__(self) -> str:  # pragma: no cover - convenience
        return self.code


SELECT_MEASUREMENT = ReasonCode(
    code="SELECT_MEASUREMENT",
    stage="select",
    severity="ok",
    description="Policy A: raw_vol = vol_t from the measurement track.",
)
SELECT_FORECAST = ReasonCode(
    code="SELECT_FORECAST",
    stage="select",
    severity="ok",
    description="Policy B: raw_vol = sqrt(forecast_var_per_session_equiv).",
)
SELECT_HYBRID = ReasonCode(
    code="SELECT_HYBRID",
    stage="select",
    severity="ok",
    description="Policy C: conservative hybrid raw_vol = max(measurement, forecast).",
)
SELECT_UNCERTAINTY_ADJUSTED = ReasonCode(
    code="SELECT_UNCERTAINTY_ADJUSTED",
    stage="select",
    severity="ok",
    description=("Policy D: hybrid raw_vol followed by an uncertainty-aware multiplier."),
)
CONSERVATIVE_COMBINATION_APPLIED = ReasonCode(
    code="CONSERVATIVE_COMBINATION_APPLIED",
    stage="combination",
    severity="info",
    description="Hybrid raw_vol = max(measurement_vol, forecast_vol).",
)
DATA_QUALITY_FALLBACK = ReasonCode(
    code="DATA_QUALITY_FALLBACK",
    stage="fallback",
    severity="conservative",
    description=(
        "Measurement or forecast input failed a data-quality check; the "
        "conservative fallback path overrode raw_vol."
    ),
)
UNCERTAINTY_APPLIED = ReasonCode(
    code="UNCERTAINTY_APPLIED",
    stage="uncertainty",
    severity="info",
    description=("Uncertainty multiplier (>= 1.0) scaled raw_vol upward (Concept #8)."),
)
FLOOR_APPLIED = ReasonCode(
    code="FLOOR_APPLIED",
    stage="floor",
    severity="info",
    description="Bounded_vol clipped up to policy_floor.",
)
EMERGENCY_FLOOR_APPLIED = ReasonCode(
    code="EMERGENCY_FLOOR_APPLIED",
    stage="floor",
    severity="conservative",
    description=(
        "Bounded_vol clipped up to the symbol/bucket EMERGENCY_FLOOR; the absolute fail-safe."
    ),
)
CAP_APPLIED = ReasonCode(
    code="CAP_APPLIED",
    stage="cap",
    severity="warn",
    description="Bounded_vol clipped down to cap_vol.",
)
SMOOTHING_APPLIED = ReasonCode(
    code="SMOOTHING_APPLIED",
    stage="smoothing",
    severity="info",
    description="EWMA smoothing carried previous_size_vol into size_vol.",
)
HYSTERESIS_APPLIED = ReasonCode(
    code="HYSTERESIS_APPLIED",
    stage="smoothing",
    severity="info",
    description=(
        "Per-step max-daily-decrease cap limited size_vol's fall versus the previous value."
    ),
)
SHOCK_OVERRIDE_APPLIED = ReasonCode(
    code="SHOCK_OVERRIDE_APPLIED",
    stage="shock",
    severity="warn",
    description=(
        "Shock-override bypassed smoothing: size_vol = "
        "max(size_vol_smoothed, shock_floor_multiplier * raw_vol)."
    ),
)
FALLBACK_HOLD = ReasonCode(
    code="FALLBACK_HOLD",
    stage="fallback",
    severity="conservative",
    description=(
        "Monotonic-conservatism HOLD: size_vol >= last_trusted_size_vol while "
        "the primary source is invalid."
    ),
)
FALLBACK_RECOVERY_DECAY = ReasonCode(
    code="FALLBACK_RECOVERY_DECAY",
    stage="fallback",
    severity="info",
    description=(
        "Release valve: source recovered, size_vol may decay per smoothing "
        "policy bounded by max daily decrease."
    ),
)
SOURCE_RECOVERED = ReasonCode(
    code="SOURCE_RECOVERED",
    stage="fallback",
    severity="ok",
    description=(
        "Fallback source passed N consecutive valid observations and the release valve is now open."
    ),
)
ATR_FALLBACK = ReasonCode(
    code="ATR_FALLBACK",
    stage="fallback",
    severity="conservative",
    description="Sizing routed to legacy ATR-based fallback (tier 5).",
)
TRADE_BLOCKED = ReasonCode(
    code="TRADE_BLOCKED",
    stage="fallback",
    severity="conservative",
    description=("All upstream fallback tiers failed; trade blocked / no-size (tier 6)."),
)

REASON_CODES: tuple[ReasonCode, ...] = (
    SELECT_MEASUREMENT,
    SELECT_FORECAST,
    SELECT_HYBRID,
    SELECT_UNCERTAINTY_ADJUSTED,
    CONSERVATIVE_COMBINATION_APPLIED,
    DATA_QUALITY_FALLBACK,
    UNCERTAINTY_APPLIED,
    FLOOR_APPLIED,
    EMERGENCY_FLOOR_APPLIED,
    CAP_APPLIED,
    SMOOTHING_APPLIED,
    HYSTERESIS_APPLIED,
    SHOCK_OVERRIDE_APPLIED,
    FALLBACK_HOLD,
    FALLBACK_RECOVERY_DECAY,
    SOURCE_RECOVERED,
    ATR_FALLBACK,
    TRADE_BLOCKED,
)


__all__ = [
    "ATR_FALLBACK",
    "CAP_APPLIED",
    "CONSERVATIVE_COMBINATION_APPLIED",
    "DATA_QUALITY_FALLBACK",
    "EMERGENCY_FLOOR_APPLIED",
    "FALLBACK_HOLD",
    "FALLBACK_RECOVERY_DECAY",
    "FLOOR_APPLIED",
    "HYSTERESIS_APPLIED",
    "REASON_CODES",
    "ReasonCode",
    "SELECT_FORECAST",
    "SELECT_HYBRID",
    "SELECT_MEASUREMENT",
    "SELECT_UNCERTAINTY_ADJUSTED",
    "SHOCK_OVERRIDE_APPLIED",
    "SMOOTHING_APPLIED",
    "SOURCE_RECOVERED",
    "Severity",
    "Stage",
    "TRADE_BLOCKED",
    "UNCERTAINTY_APPLIED",
]
