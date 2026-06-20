from __future__ import annotations

import pytest

from liq.risk.vol_sizing.reason_codes import SHOCK_OVERRIDE_APPLIED
from liq.risk.vol_sizing.shock_override import (
    EOD_ONLY_LABELS,
    INTRADAY_SAFE_LABELS,
    ShockOverrideConfig,
    ShockOverrideResult,
    apply_shock_override,
)


def _config(
    *,
    shock_ratio_threshold: float = 1.5,
    shock_floor_multiplier: float = 1.0,
) -> ShockOverrideConfig:
    return ShockOverrideConfig(
        shock_ratio_threshold=shock_ratio_threshold,
        shock_floor_multiplier=shock_floor_multiplier,
    )


def test_no_trigger_when_ratio_below_threshold_and_no_label() -> None:
    result = apply_shock_override(
        raw_vol=0.02,
        size_vol_smoothed=0.018,
        previous_size_vol=0.015,
        labels=frozenset(),
        decision_kind="eod",
        config=_config(shock_ratio_threshold=1.5),
    )

    assert isinstance(result, ShockOverrideResult)
    assert result.triggered is False
    assert result.size_vol == pytest.approx(0.018)
    assert result.reason_code is None


def test_ratio_trigger_overrides_smoothing_to_floor_times_raw() -> None:
    # raw_vol / previous = 0.06 / 0.02 = 3.0 >> 1.5 → trigger
    result = apply_shock_override(
        raw_vol=0.06,
        size_vol_smoothed=0.025,
        previous_size_vol=0.02,
        labels=frozenset(),
        decision_kind="eod",
        config=_config(shock_ratio_threshold=1.5, shock_floor_multiplier=1.0),
    )

    assert result.triggered is True
    assert result.size_vol == pytest.approx(0.06)
    assert result.reason_code == SHOCK_OVERRIDE_APPLIED
    assert result.trigger_source == "ratio"


def test_shock_floor_multiplier_scales_raw_when_floor_below_one() -> None:
    result = apply_shock_override(
        raw_vol=0.08,
        size_vol_smoothed=0.03,
        previous_size_vol=0.02,
        labels=frozenset(),
        decision_kind="eod",
        config=_config(shock_floor_multiplier=0.75),
    )

    assert result.triggered is True
    # max(0.03, 0.75 * 0.08) = max(0.03, 0.06) = 0.06
    assert result.size_vol == pytest.approx(0.06)


def test_smoothed_value_wins_when_already_above_shock_floor() -> None:
    # raw_vol high → triggers, but smoothed already conservative
    result = apply_shock_override(
        raw_vol=0.05,
        size_vol_smoothed=0.07,
        previous_size_vol=0.02,
        labels=frozenset(),
        decision_kind="eod",
        config=_config(),
    )

    assert result.triggered is True
    assert result.size_vol == pytest.approx(0.07)


def test_intraday_safe_label_triggers_override_for_intraday_decisions() -> None:
    result = apply_shock_override(
        raw_vol=0.02,
        size_vol_smoothed=0.018,
        previous_size_vol=0.02,
        labels=frozenset({"INTRADAY_SHOCK_TRIGGERED"}),
        decision_kind="intraday",
        config=_config(),
    )

    assert result.triggered is True
    assert result.trigger_source == "label"
    assert result.firing_labels == frozenset({"INTRADAY_SHOCK_TRIGGERED"})


def test_eod_label_triggers_override_for_eod_decisions() -> None:
    result = apply_shock_override(
        raw_vol=0.02,
        size_vol_smoothed=0.018,
        previous_size_vol=0.02,
        labels=frozenset({"JUMP_DAY"}),
        decision_kind="eod",
        config=_config(),
    )

    assert result.triggered is True
    assert result.firing_labels == frozenset({"JUMP_DAY"})


def test_session_complete_label_is_illegal_for_intraday_decision_kind() -> None:
    """Hard boundary #13: JUMP_DAY/GAP_DOMINATED_VOL/HIGH_ESTIMATOR_DISAGREEMENT
    cannot fire the sizer before close (they re-introduce same-session leakage).
    """

    with pytest.raises(ValueError, match="session-complete label"):
        apply_shock_override(
            raw_vol=0.02,
            size_vol_smoothed=0.02,
            previous_size_vol=0.02,
            labels=frozenset({"JUMP_DAY"}),
            decision_kind="intraday",
            config=_config(),
        )


def test_session_complete_label_in_mixed_set_still_rejected() -> None:
    with pytest.raises(ValueError, match="GAP_DOMINATED_VOL"):
        apply_shock_override(
            raw_vol=0.02,
            size_vol_smoothed=0.02,
            previous_size_vol=0.02,
            labels=frozenset({"INTRADAY_SHOCK_TRIGGERED", "GAP_DOMINATED_VOL"}),
            decision_kind="intraday",
            config=_config(),
        )


def test_intraday_label_during_eod_decision_is_allowed_but_fires_normally() -> None:
    """Intraday-safe labels remain valid when the decision is EOD; they just
    don't add information beyond the EOD labels. They should still trigger
    if their availability constraint is satisfied (which is always true at EOD).
    """

    result = apply_shock_override(
        raw_vol=0.02,
        size_vol_smoothed=0.018,
        previous_size_vol=0.02,
        labels=frozenset({"INTRADAY_SHOCK_TRIGGERED"}),
        decision_kind="eod",
        config=_config(),
    )

    assert result.triggered is True


def test_unknown_label_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown shock label"):
        apply_shock_override(
            raw_vol=0.02,
            size_vol_smoothed=0.02,
            previous_size_vol=0.02,
            labels=frozenset({"NOT_A_REAL_LABEL"}),
            decision_kind="eod",
            config=_config(),
        )


def test_label_sets_are_disjoint() -> None:
    """Sanity property: a label is either intraday-safe or session-complete,
    never both. The discriminator depends on the partition holding.
    """

    assert INTRADAY_SAFE_LABELS.isdisjoint(EOD_ONLY_LABELS)
    assert "JUMP_DAY" in EOD_ONLY_LABELS
    assert "GAP_DOMINATED_VOL" in EOD_ONLY_LABELS
    assert "INTRADAY_SHOCK_TRIGGERED" in INTRADAY_SAFE_LABELS


def test_no_trigger_returns_previous_smoothed_unmodified() -> None:
    smoothed = 0.05
    result = apply_shock_override(
        raw_vol=0.05,
        size_vol_smoothed=smoothed,
        previous_size_vol=0.05,
        labels=frozenset(),
        decision_kind="eod",
        config=_config(),
    )

    assert result.size_vol == smoothed
    assert result.triggered is False
    assert result.firing_labels == frozenset()


def test_zero_previous_size_vol_falls_back_to_label_only_trigger() -> None:
    """At first session, previous_size_vol is None/0. Ratio trigger is undefined
    but label-based trigger must still fire.
    """

    result = apply_shock_override(
        raw_vol=0.05,
        size_vol_smoothed=0.04,
        previous_size_vol=None,
        labels=frozenset({"JUMP_DAY"}),
        decision_kind="eod",
        config=_config(),
    )

    assert result.triggered is True
    assert result.trigger_source == "label"


def test_negative_or_zero_raw_vol_rejected() -> None:
    with pytest.raises(ValueError, match="raw_vol must be positive"):
        apply_shock_override(
            raw_vol=0.0,
            size_vol_smoothed=0.02,
            previous_size_vol=0.02,
            labels=frozenset(),
            decision_kind="eod",
            config=_config(),
        )


def test_config_rejects_subunit_ratio_threshold() -> None:
    with pytest.raises(ValueError, match="shock_ratio_threshold"):
        ShockOverrideConfig(shock_ratio_threshold=0.5, shock_floor_multiplier=1.0)


def test_config_rejects_negative_floor_multiplier() -> None:
    with pytest.raises(ValueError, match="shock_floor_multiplier"):
        ShockOverrideConfig(shock_ratio_threshold=1.5, shock_floor_multiplier=-0.1)
