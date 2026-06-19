from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from liq.risk.vol_sizing import (
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
    ForecastInput,
    MeasurementInput,
    SizingPolicy,
    compute_size_vol,
)

_NOW = datetime(2026, 6, 19, 14, 30, tzinfo=UTC)
_VALID_UNTIL = _NOW + timedelta(hours=4)


def _policy(
    *,
    policy_id: str = "hybrid",
    policy_floor: float = 0.005,
    emergency_floor: float = 0.003,
    cap_vol: float = 0.5,
    uncertainty_k: float = 0.25,
    uncertainty_max_extra: float = 0.50,
    smoothing_half_life: int = 5,
    smoothing_max_daily_decrease: float = 0.10,
) -> SizingPolicy:
    return SizingPolicy(
        policy_id=policy_id,
        policy_version="v1",
        policy_floor=policy_floor,
        emergency_floor=emergency_floor,
        cap_vol=cap_vol,
        uncertainty_k=uncertainty_k,
        uncertainty_max_extra=uncertainty_max_extra,
        smoothing_half_life=smoothing_half_life,
        smoothing_max_daily_decrease=smoothing_max_daily_decrease,
    )


def _measurement(vol: float = 0.02, *, ok: bool = True, stale: bool = False) -> MeasurementInput:
    return MeasurementInput(
        measurement_id="m_a",
        measurement_vol=vol,
        source_valid_from=_NOW,
        valid_until=_VALID_UNTIL,
        is_quality_ok=ok,
        is_stale=stale,
    )


def _forecast(
    vol: float = 0.025,
    *,
    estimator_unc: float = 0.0,
    forecast_unc: float = 0.0,
    ok: bool = True,
    stale: bool = False,
) -> ForecastInput:
    return ForecastInput(
        forecast_id="f_a",
        forecast_vol=vol,
        forecast_var_per_session_equiv=vol * vol,
        forecast_available_ts=_NOW,
        valid_until=_VALID_UNTIL,
        estimator_uncertainty_percentile=estimator_unc,
        forecast_uncertainty_percentile=forecast_unc,
        is_quality_ok=ok,
        is_stale=stale,
    )


def test_measurement_policy_picks_measurement_only_and_emits_select_code() -> None:
    row = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(),
        forecast=_forecast(),
        policy=_policy(policy_id="measurement"),
        previous_size_vol=None,
    )

    assert row.raw_vol_source == "measurement_vol"
    assert row.raw_vol == pytest.approx(0.02)
    assert SELECT_MEASUREMENT in row.reason_codes


def test_forecast_policy_picks_forecast_only() -> None:
    row = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(),
        forecast=_forecast(vol=0.025),
        policy=_policy(policy_id="forecast"),
        previous_size_vol=None,
    )

    assert row.raw_vol_source == "forecast_vol"
    assert row.raw_vol == pytest.approx(0.025)
    assert SELECT_FORECAST in row.reason_codes


def test_hybrid_policy_takes_conservative_max() -> None:
    row = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(vol=0.02),
        forecast=_forecast(vol=0.04),
        policy=_policy(policy_id="hybrid"),
        previous_size_vol=None,
    )

    assert row.raw_vol == pytest.approx(0.04)
    assert SELECT_HYBRID in row.reason_codes
    assert CONSERVATIVE_COMBINATION_APPLIED in row.reason_codes


def test_uncertainty_policy_multiplier_is_at_least_one_and_bounded() -> None:
    policy = _policy(policy_id="uncertainty_adjusted", uncertainty_k=0.25, uncertainty_max_extra=0.5)
    pre = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(vol=0.02),
        forecast=_forecast(vol=0.03, estimator_unc=0.0, forecast_unc=0.0),
        policy=policy,
        previous_size_vol=None,
    )
    post = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(vol=0.02),
        forecast=_forecast(vol=0.03, estimator_unc=1.0, forecast_unc=1.0),
        policy=policy,
        previous_size_vol=None,
    )

    assert pre.uncertainty_multiplier == pytest.approx(1.0)
    assert post.uncertainty_multiplier == pytest.approx(1.0 + min(0.25, 0.5))
    assert post.size_vol > pre.size_vol
    assert SELECT_UNCERTAINTY_ADJUSTED in post.reason_codes
    assert UNCERTAINTY_APPLIED in post.reason_codes


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    raw_vol=st.floats(min_value=1e-6, max_value=0.1, allow_nan=False, allow_infinity=False),
    rank=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
def test_uncertainty_multiplier_post_value_geq_pre_value(raw_vol: float, rank: float) -> None:
    policy = _policy(policy_id="uncertainty_adjusted", cap_vol=10.0)
    measurement = _measurement(vol=raw_vol)
    forecast = _forecast(vol=raw_vol, estimator_unc=rank, forecast_unc=rank)
    pre = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=measurement,
        forecast=_forecast(vol=raw_vol, estimator_unc=0.0, forecast_unc=0.0),
        policy=policy,
        previous_size_vol=None,
    )
    post = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=measurement,
        forecast=forecast,
        policy=policy,
        previous_size_vol=None,
    )

    assert post.size_vol >= pre.size_vol - 1.0e-12
    assert post.uncertainty_multiplier >= 1.0 - 1.0e-12


def test_policy_floor_clamps_low_raw_vol_and_emits_floor_code() -> None:
    policy = _policy(policy_floor=0.01, emergency_floor=0.005)
    row = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(vol=0.001),
        forecast=_forecast(vol=0.001),
        policy=policy,
        previous_size_vol=None,
    )

    assert row.size_vol == pytest.approx(0.01)
    assert row.floor_applied is True
    assert FLOOR_APPLIED in row.reason_codes


def test_emergency_floor_invariant_holds_even_when_policy_floor_lower() -> None:
    policy = _policy(policy_floor=0.001, emergency_floor=0.01)
    row = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(vol=0.0005),
        forecast=_forecast(vol=0.0005),
        policy=policy,
        previous_size_vol=None,
    )

    assert row.size_vol >= policy.emergency_floor
    assert row.emergency_floor_applied is True
    assert EMERGENCY_FLOOR_APPLIED in row.reason_codes


def test_cap_vol_clamps_high_raw_vol_and_emits_cap_code() -> None:
    policy = _policy(cap_vol=0.05)
    row = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(vol=0.20),
        forecast=_forecast(vol=0.10),
        policy=policy,
        previous_size_vol=None,
    )

    assert row.size_vol == pytest.approx(0.05)
    assert row.cap_applied is True
    assert CAP_APPLIED in row.reason_codes


def test_smoothing_carries_previous_size_vol_and_emits_smoothing_code() -> None:
    policy = _policy(smoothing_half_life=2, smoothing_max_daily_decrease=0.99)
    row = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(vol=0.05),
        forecast=_forecast(vol=0.05),
        policy=policy,
        previous_size_vol=0.02,
    )

    assert 0.02 < row.size_vol < 0.05
    assert row.smoothing_applied is True
    assert SMOOTHING_APPLIED in row.reason_codes


def test_hysteresis_caps_per_step_decline_and_emits_hysteresis_code() -> None:
    policy = _policy(smoothing_half_life=1, smoothing_max_daily_decrease=0.05)
    # bounded_vol drops to 0.005 but per-step max decrease floor = 0.05 * (1 - 0.05) = 0.0475.
    row = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(vol=0.005),
        forecast=_forecast(vol=0.005),
        policy=policy,
        previous_size_vol=0.05,
    )

    assert row.size_vol == pytest.approx(0.05 * (1.0 - 0.05))
    assert HYSTERESIS_APPLIED in row.reason_codes


def test_data_quality_fallback_uses_conservative_path() -> None:
    policy = _policy()
    row = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(vol=0.02, ok=False),
        forecast=_forecast(vol=0.03),
        policy=policy,
        previous_size_vol=None,
    )

    assert row.raw_vol_source == "fallback"
    assert DATA_QUALITY_FALLBACK in row.reason_codes
    assert row.size_vol >= 0.03 - 1.0e-12


def test_pipeline_order_floor_then_smooth_differs_from_smooth_then_floor() -> None:
    """The non-commutativity is between the (non-linear) FLOOR/CAP
    step and the SMOOTHING step. The pipeline runs FLOOR (5) then SMOOTH (6);
    reversing the order would yield a different size_vol.

    Concrete example: raw_vol = 0.001, policy_floor = 0.02, previous = 0.01.

    - Pipeline (floor THEN smooth): bounded = max(0.001, 0.02) = 0.02,
      size_vol = alpha * 0.02 + (1 - alpha) * 0.01.
    - Hypothetical reversed (smooth THEN floor): smoothed =
      alpha * 0.001 + (1 - alpha) * 0.01 < 0.01, then floored = 0.02.

    The two paths produce different size_vol values whenever the pre-floor
    raw_vol is below the floor and the previous size is also below the
    floor, so the order is load-bearing.
    """

    raw = 0.001
    previous = 0.01
    policy = _policy(policy_floor=0.02, emergency_floor=0.005)
    row = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(vol=raw),
        forecast=_forecast(vol=raw),
        policy=policy,
        previous_size_vol=previous,
    )
    pipeline_size_vol = row.size_vol

    alpha = 1.0 - math.exp(-math.log(2.0) / policy.smoothing_half_life)
    reversed_size_vol = max(alpha * raw + (1.0 - alpha) * previous, policy.policy_floor)

    assert not math.isclose(pipeline_size_vol, reversed_size_vol, rel_tol=1e-6)
    assert pipeline_size_vol < reversed_size_vol  # pipeline can sit below floor after smooth


@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    meas=st.floats(min_value=1e-5, max_value=0.5, allow_nan=False),
    fcst=st.floats(min_value=1e-5, max_value=0.5, allow_nan=False),
)
def test_size_vol_invariant_always_satisfies_emergency_floor(meas: float, fcst: float) -> None:
    policy = _policy(policy_floor=0.005, emergency_floor=0.003)
    row = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(vol=meas),
        forecast=_forecast(vol=fcst),
        policy=policy,
        previous_size_vol=None,
    )

    assert row.size_vol >= policy.emergency_floor - 1.0e-12


def test_reason_codes_emitted_for_every_changing_step() -> None:
    """If raw_vol enters near zero and bounces through floor, uncertainty, and smoothing,
    each changing step must emit its reason code.
    """

    policy = _policy(
        policy_id="uncertainty_adjusted",
        policy_floor=0.02,
        emergency_floor=0.01,
        smoothing_half_life=3,
        smoothing_max_daily_decrease=0.30,
    )
    row = compute_size_vol(
        symbol="AAPL",
        sizing_decision_ts=_NOW,
        measurement=_measurement(vol=0.001),
        forecast=_forecast(vol=0.001, estimator_unc=0.8, forecast_unc=0.8),
        policy=policy,
        previous_size_vol=0.10,
    )

    codes = {c.code for c in row.reason_codes}
    assert "SELECT_UNCERTAINTY_ADJUSTED" in codes
    assert "UNCERTAINTY_APPLIED" in codes
    assert "FLOOR_APPLIED" in codes
    # Smoothing applies because previous_size_vol carries; hysteresis may also apply.
    assert "SMOOTHING_APPLIED" in codes


def test_pipeline_rejects_unknown_policy_id() -> None:
    with pytest.raises(ValueError, match="unknown policy_id"):
        compute_size_vol(
            symbol="AAPL",
            sizing_decision_ts=_NOW,
            measurement=_measurement(),
            forecast=_forecast(),
            policy=_policy(policy_id="bad"),
            previous_size_vol=None,
        )


def test_pipeline_rejects_missing_inputs_for_policy() -> None:
    with pytest.raises(ValueError, match="requires a measurement"):
        compute_size_vol(
            symbol="AAPL",
            sizing_decision_ts=_NOW,
            measurement=None,
            forecast=_forecast(),
            policy=_policy(policy_id="measurement"),
            previous_size_vol=None,
        )
    with pytest.raises(ValueError, match="requires a forecast"):
        compute_size_vol(
            symbol="AAPL",
            sizing_decision_ts=_NOW,
            measurement=_measurement(),
            forecast=None,
            policy=_policy(policy_id="forecast"),
            previous_size_vol=None,
        )


def test_sizing_policy_rejects_misordered_thresholds() -> None:
    with pytest.raises(ValueError, match="cap_vol must be"):
        SizingPolicy(
            policy_id="hybrid",
            policy_version="v1",
            policy_floor=0.05,
            emergency_floor=0.001,
            cap_vol=0.001,
            uncertainty_k=0.25,
            uncertainty_max_extra=0.5,
            smoothing_half_life=5,
            smoothing_max_daily_decrease=0.1,
        )
