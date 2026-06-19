"""Sizing-input contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from .reason_codes import ReasonCode

RawVolSource = Literal[
    "measurement_vol",
    "forecast_vol",
    "hybrid_vol",
    "atr",
    "fallback",
]

PolicyId = Literal[
    "measurement",
    "forecast",
    "hybrid",
    "uncertainty_adjusted",
]


@dataclass(frozen=True)
class SizingPolicy:
    """Frozen sizing-policy declaration consumed by the pipeline.

    The four declared ``policy_id`` values map to the sizing policy taxonomy:

    - ``measurement`` (A): ``raw_vol = measurement_vol``
    - ``forecast`` (B): ``raw_vol = sqrt(forecast_var_per_session_equiv)``
    - ``hybrid`` (C): ``raw_vol = max(measurement_vol, forecast_vol)``
    - ``uncertainty_adjusted`` (D): hybrid + uncertainty multiplier
    """

    policy_id: PolicyId
    policy_version: str
    policy_floor: float
    emergency_floor: float
    cap_vol: float
    uncertainty_k: float
    uncertainty_max_extra: float
    smoothing_half_life: int
    smoothing_max_daily_decrease: float
    sizing_unit: str = "per_session_vol"

    def __post_init__(self) -> None:
        if self.policy_floor < 0.0:
            raise ValueError("policy_floor must be non-negative")
        if self.emergency_floor < 0.0:
            raise ValueError("emergency_floor must be non-negative")
        if self.cap_vol <= 0.0:
            raise ValueError("cap_vol must be positive")
        if self.cap_vol < self.policy_floor:
            raise ValueError("cap_vol must be >= policy_floor")
        if self.cap_vol < self.emergency_floor:
            raise ValueError("cap_vol must be >= emergency_floor")
        if self.uncertainty_k < 0.0:
            raise ValueError("uncertainty_k must be non-negative")
        if self.uncertainty_max_extra < 0.0:
            raise ValueError("uncertainty_max_extra must be non-negative")
        if self.smoothing_half_life <= 0:
            raise ValueError("smoothing_half_life must be positive")
        if not 0.0 < self.smoothing_max_daily_decrease <= 1.0:
            raise ValueError("smoothing_max_daily_decrease must be in (0, 1]")


@dataclass(frozen=True)
class MeasurementInput:
    """Snapshot of the measurement track read at sizing time."""

    measurement_id: str
    measurement_vol: float
    source_valid_from: datetime
    valid_until: datetime
    is_quality_ok: bool = True
    is_stale: bool = False


@dataclass(frozen=True)
class ForecastInput:
    """Snapshot of the forecast row consumed at sizing time."""

    forecast_id: str
    forecast_vol: float
    forecast_var_per_session_equiv: float
    forecast_available_ts: datetime
    valid_until: datetime
    estimator_uncertainty_percentile: float = 0.0
    forecast_uncertainty_percentile: float = 0.0
    is_quality_ok: bool = True
    is_stale: bool = False


@dataclass(frozen=True)
class SizeVolInput:
    """Frozen sizing-input row emitted by :func:`compute_size_vol`."""

    symbol: str
    sizing_decision_ts: datetime
    forecast_id: str | None
    forecast_available_ts: datetime | None
    source_valid_from: datetime
    measurement_id: str | None
    policy_id: PolicyId
    policy_version: str
    raw_vol: float
    raw_vol_source: RawVolSource
    source_measurement_vol: float | None
    source_forecast_var: float | None
    source_forecast_vol: float | None
    previous_size_vol: float | None
    size_vol: float
    sizing_unit: str
    uncertainty_multiplier: float
    floor_applied: bool
    emergency_floor_applied: bool
    cap_applied: bool
    clip_applied: bool
    smoothing_applied: bool
    shock_override_applied: bool
    regime_labels: frozenset[str]
    reason_codes: tuple[ReasonCode, ...]
    valid_until: datetime
    diagnostics: dict[str, float] = field(default_factory=dict)


__all__ = [
    "ForecastInput",
    "MeasurementInput",
    "PolicyId",
    "RawVolSource",
    "SizeVolInput",
    "SizingPolicy",
]
