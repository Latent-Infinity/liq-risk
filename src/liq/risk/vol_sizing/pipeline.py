"""Ordered sizing transform.

The seven-step pipeline runs in a fixed order because the transformations
do not commute: ``smooth(max(a, b)) != max(smooth(a), smooth(b))``. The
order is:

1. SELECT raw source: pick raw_vol_candidate per policy.
2. CONSERVATIVE COMBINATION: if hybrid, raw_vol = max(measurement, forecast).
3. FALLBACK: if a quality flag fires, raw_vol = fallback.
4. UNCERTAINTY MULTIPLIER: adjusted_vol = raw_vol * multiplier (>= 1.0).
5. FLOORS & CAPS: bounded_vol = clip(max(adjusted, floor), cap).
6. SMOOTHING / HYSTERESIS: size_vol = smooth(bounded, previous_size_vol).
7. REASON CODES: emit an enum code per transform that fired.

Every step that changes the sizing value adds a :class:`ReasonCode` to the
returned ``SizeVolInput.reason_codes`` stack. Quality-fallback events are
flagged ``conservative`` severity so the audit can distinguish them from
informational transformations.
"""

from __future__ import annotations

import math
from datetime import datetime

from .contracts import (
    ForecastInput,
    MeasurementInput,
    PolicyId,
    RawVolSource,
    SizeVolInput,
    SizingPolicy,
)
from .reason_codes import (
    CAP_APPLIED,
    CONSERVATIVE_COMBINATION_APPLIED,
    DATA_QUALITY_FALLBACK,
    EMERGENCY_FLOOR_APPLIED,
    FLOOR_APPLIED,
    HYSTERESIS_APPLIED,
    SELECT_FORECAST,
    SELECT_HYBRID,
    SELECT_MEASUREMENT,
    SELECT_UNCERTAINTY_ADJUSTED,
    SMOOTHING_APPLIED,
    UNCERTAINTY_APPLIED,
    ReasonCode,
)
from .shock_override import (
    DecisionKind,
    ShockOverrideConfig,
    apply_shock_override,
)


def _select_raw_vol(
    policy_id: PolicyId,
    *,
    measurement: MeasurementInput | None,
    forecast: ForecastInput | None,
) -> tuple[float, RawVolSource, ReasonCode]:
    if policy_id == "measurement":
        if measurement is None:
            raise ValueError("policy 'measurement' requires a measurement input")
        return float(measurement.measurement_vol), "measurement_vol", SELECT_MEASUREMENT
    if policy_id == "forecast":
        if forecast is None:
            raise ValueError("policy 'forecast' requires a forecast input")
        return float(forecast.forecast_vol), "forecast_vol", SELECT_FORECAST
    if policy_id == "hybrid":
        if measurement is None or forecast is None:
            raise ValueError("policy 'hybrid' requires measurement AND forecast inputs")
        # CONSERVATIVE COMBINATION is the canonical max(measurement, forecast).
        return (
            float(max(measurement.measurement_vol, forecast.forecast_vol)),
            "hybrid_vol",
            SELECT_HYBRID,
        )
    if policy_id == "uncertainty_adjusted":
        if measurement is None or forecast is None:
            raise ValueError(
                "policy 'uncertainty_adjusted' requires measurement AND forecast inputs"
            )
        return (
            float(max(measurement.measurement_vol, forecast.forecast_vol)),
            "hybrid_vol",
            SELECT_UNCERTAINTY_ADJUSTED,
        )
    raise ValueError(f"unknown policy_id {policy_id!r}")


def _uncertainty_multiplier(
    forecast: ForecastInput | None, policy: SizingPolicy
) -> float:
    if forecast is None:
        return 1.0
    rank = max(
        float(forecast.estimator_uncertainty_percentile),
        float(forecast.forecast_uncertainty_percentile),
    )
    rank = min(max(rank, 0.0), 1.0)
    extra = min(policy.uncertainty_k * rank, policy.uncertainty_max_extra)
    return float(1.0 + extra)


def _smoothing_alpha(half_life: int) -> float:
    """EWMA alpha for the requested half-life in sessions."""

    return float(1.0 - math.exp(-math.log(2.0) / half_life))


def _apply_smoothing(
    bounded_vol: float,
    previous_size_vol: float | None,
    policy: SizingPolicy,
) -> tuple[float, bool, bool]:
    """Smooth + hysteresis. Returns (size_vol, smoothing_applied, hysteresis_applied)."""

    if previous_size_vol is None or previous_size_vol <= 0.0:
        return bounded_vol, False, False
    alpha = _smoothing_alpha(policy.smoothing_half_life)
    smoothed = float(alpha * bounded_vol + (1.0 - alpha) * previous_size_vol)
    smoothing_applied = not math.isclose(smoothed, bounded_vol, rel_tol=1.0e-12)
    floor_decrease = float(previous_size_vol * (1.0 - policy.smoothing_max_daily_decrease))
    hysteresis_applied = False
    if smoothed < floor_decrease:
        smoothed = floor_decrease
        hysteresis_applied = True
    return smoothed, smoothing_applied, hysteresis_applied


def compute_size_vol(
    *,
    symbol: str,
    sizing_decision_ts: datetime,
    measurement: MeasurementInput | None,
    forecast: ForecastInput | None,
    policy: SizingPolicy,
    previous_size_vol: float | None,
    regime_labels: frozenset[str] = frozenset(),
    shock_config: ShockOverrideConfig | None = None,
    shock_labels: frozenset[str] = frozenset(),
    decision_kind: DecisionKind = "eod",
) -> SizeVolInput:
    """Run the ordered sizing transform.

    All seven steps execute in order. The returned :class:`SizeVolInput`
    carries the full reason-code stack: one code per pipeline step that
    changed the running vol.

    When ``shock_config`` is provided, the §9.6 shock-override is evaluated
    *after* smoothing: if the override fires, ``size_vol`` is replaced by
    ``max(size_vol_smoothed, shock_floor_multiplier * raw_vol)`` and the
    ``SHOCK_OVERRIDE_APPLIED`` reason code is appended.
    """

    reason_stack: list[ReasonCode] = []

    # 1. SELECT raw source (2. CONSERVATIVE COMBINATION folded into hybrid policies).
    raw_vol, raw_vol_source, select_code = _select_raw_vol(
        policy.policy_id, measurement=measurement, forecast=forecast
    )
    reason_stack.append(select_code)
    if select_code in (SELECT_HYBRID, SELECT_UNCERTAINTY_ADJUSTED):
        reason_stack.append(CONSERVATIVE_COMBINATION_APPLIED)

    # 3. FALLBACK: if either input is flagged, route to the conservative fallback.
    has_fallback = False
    fallback_components: list[float] = []
    if measurement is not None:
        if (not measurement.is_quality_ok) or measurement.is_stale:
            has_fallback = True
        else:
            fallback_components.append(float(measurement.measurement_vol))
    if forecast is not None:
        if (not forecast.is_quality_ok) or forecast.is_stale:
            has_fallback = True
        else:
            fallback_components.append(float(forecast.forecast_vol))
    if has_fallback:
        # Conservative fallback: the most-conservative trusted estimate available,
        # else the previous size_vol, else the policy_floor. Monotonic-conservative
        # Never reduce conservatism silently.
        conservative_candidate = (
            max(fallback_components) if fallback_components else float("nan")
        )
        if math.isfinite(conservative_candidate):
            raw_vol = max(raw_vol, conservative_candidate)
        elif previous_size_vol is not None and previous_size_vol > 0.0:
            raw_vol = max(raw_vol, float(previous_size_vol))
        else:
            raw_vol = max(raw_vol, policy.policy_floor)
        raw_vol_source = "fallback"
        reason_stack.append(DATA_QUALITY_FALLBACK)

    # 4. UNCERTAINTY MULTIPLIER: applies only to policy D, otherwise 1.0.
    if policy.policy_id == "uncertainty_adjusted":
        uncertainty_multiplier = _uncertainty_multiplier(forecast, policy)
    else:
        uncertainty_multiplier = 1.0
    adjusted_vol = raw_vol * uncertainty_multiplier
    if uncertainty_multiplier > 1.0 + 1.0e-12:
        reason_stack.append(UNCERTAINTY_APPLIED)

    # 5. FLOORS & CAPS: floor_vol = max(policy_floor, EMERGENCY_FLOOR); cap last.
    pre_floor = adjusted_vol
    floor_vol = max(policy.policy_floor, policy.emergency_floor)
    floored = max(pre_floor, floor_vol)
    floor_applied = (
        floored > pre_floor + 1.0e-12
        and floor_vol == policy.policy_floor
        and policy.policy_floor >= policy.emergency_floor
    )
    emergency_floor_applied = (
        floored > pre_floor + 1.0e-12
        and floor_vol == policy.emergency_floor
        and policy.emergency_floor > policy.policy_floor
    )
    # The EMERGENCY_FLOOR invariant: size_vol_t >= EMERGENCY_FLOOR at this step.
    floored = max(floored, policy.emergency_floor)
    if floored > pre_floor + 1.0e-12:
        if emergency_floor_applied:
            reason_stack.append(EMERGENCY_FLOOR_APPLIED)
        else:
            reason_stack.append(FLOOR_APPLIED)
            floor_applied = True
    pre_cap = floored
    bounded_vol = min(pre_cap, policy.cap_vol)
    cap_applied = bounded_vol < pre_cap - 1.0e-12
    if cap_applied:
        reason_stack.append(CAP_APPLIED)

    # 6. SMOOTHING / HYSTERESIS.
    size_vol, smoothing_applied, hysteresis_applied = _apply_smoothing(
        bounded_vol, previous_size_vol, policy
    )
    if smoothing_applied:
        reason_stack.append(SMOOTHING_APPLIED)
    if hysteresis_applied:
        reason_stack.append(HYSTERESIS_APPLIED)
    # Reassert the EMERGENCY_FLOOR invariant after smoothing.
    if size_vol < policy.emergency_floor:
        size_vol = policy.emergency_floor
        if EMERGENCY_FLOOR_APPLIED not in reason_stack:
            reason_stack.append(EMERGENCY_FLOOR_APPLIED)
            emergency_floor_applied = True

    # 6.5 SHOCK OVERRIDE (§9.6) — optional, applies after smoothing.
    shock_override_applied = False
    if shock_config is not None:
        shock_result = apply_shock_override(
            raw_vol=raw_vol,
            size_vol_smoothed=size_vol,
            previous_size_vol=previous_size_vol,
            labels=shock_labels,
            decision_kind=decision_kind,
            config=shock_config,
        )
        size_vol = shock_result.size_vol
        if shock_result.triggered and shock_result.reason_code is not None:
            shock_override_applied = True
            reason_stack.append(shock_result.reason_code)
        # Re-enforce EMERGENCY_FLOOR after override.
        if size_vol < policy.emergency_floor:
            size_vol = policy.emergency_floor
            if EMERGENCY_FLOOR_APPLIED not in reason_stack:
                reason_stack.append(EMERGENCY_FLOOR_APPLIED)
                emergency_floor_applied = True

    source_valid_from = (
        measurement.source_valid_from
        if measurement is not None
        else forecast.forecast_available_ts if forecast is not None else sizing_decision_ts
    )
    valid_until_candidates = [
        v
        for v in (
            measurement.valid_until if measurement is not None else None,
            forecast.valid_until if forecast is not None else None,
        )
        if v is not None
    ]
    valid_until = min(valid_until_candidates) if valid_until_candidates else sizing_decision_ts

    clip_applied = floor_applied or emergency_floor_applied or cap_applied

    return SizeVolInput(
        symbol=symbol,
        sizing_decision_ts=sizing_decision_ts,
        forecast_id=forecast.forecast_id if forecast is not None else None,
        forecast_available_ts=(
            forecast.forecast_available_ts if forecast is not None else None
        ),
        source_valid_from=source_valid_from,
        measurement_id=measurement.measurement_id if measurement is not None else None,
        policy_id=policy.policy_id,
        policy_version=policy.policy_version,
        raw_vol=raw_vol,
        raw_vol_source=raw_vol_source,
        source_measurement_vol=(
            measurement.measurement_vol if measurement is not None else None
        ),
        source_forecast_var=(
            forecast.forecast_var_per_session_equiv if forecast is not None else None
        ),
        source_forecast_vol=forecast.forecast_vol if forecast is not None else None,
        previous_size_vol=previous_size_vol,
        size_vol=size_vol,
        sizing_unit=policy.sizing_unit,
        uncertainty_multiplier=uncertainty_multiplier,
        floor_applied=floor_applied,
        emergency_floor_applied=emergency_floor_applied,
        cap_applied=cap_applied,
        clip_applied=clip_applied,
        smoothing_applied=smoothing_applied or hysteresis_applied,
        shock_override_applied=shock_override_applied,
        regime_labels=regime_labels,
        reason_codes=tuple(reason_stack),
        valid_until=valid_until,
        diagnostics={
            "pre_floor_vol": float(pre_floor),
            "pre_cap_vol": float(pre_cap),
            "bounded_vol": float(bounded_vol),
            "smoothing_alpha": float(
                _smoothing_alpha(policy.smoothing_half_life)
                if previous_size_vol is not None and previous_size_vol > 0.0
                else 1.0
            ),
        },
    )


__all__ = ["compute_size_vol"]
