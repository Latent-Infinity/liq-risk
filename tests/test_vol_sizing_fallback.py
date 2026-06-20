from __future__ import annotations

import pytest

from liq.risk.vol_sizing.fallback import (
    FallbackChain,
    FallbackConfig,
    FallbackDecision,
    FallbackObservation,
    FallbackState,
    advance_fallback,
)
from liq.risk.vol_sizing.reason_codes import (
    ATR_FALLBACK,
    FALLBACK_HOLD,
    FALLBACK_RECOVERY_DECAY,
    SOURCE_RECOVERED,
    TRADE_BLOCKED,
)


def _config(release_n: int = 5, max_decrease: float = 0.10) -> FallbackConfig:
    return FallbackConfig(
        fallback_release_n=release_n,
        smoothing_max_daily_decrease=max_decrease,
    )


def _obs(
    *,
    primary_ok: bool = True,
    approved_baseline_ok: bool = True,
    ewma_ok: bool = True,
    measurement_ok: bool = True,
    atr_ok: bool = True,
    proposed_raw_vol: float = 0.02,
) -> FallbackObservation:
    return FallbackObservation(
        primary_ok=primary_ok,
        approved_baseline_ok=approved_baseline_ok,
        ewma_ok=ewma_ok,
        measurement_ok=measurement_ok,
        atr_ok=atr_ok,
        proposed_raw_vol=proposed_raw_vol,
    )


def test_initial_state_with_primary_available_selects_primary_tier() -> None:
    state = FallbackState.initial()
    decision = advance_fallback(state=state, obs=_obs(), config=_config())

    assert isinstance(decision, FallbackDecision)
    assert decision.next_state.current_tier == FallbackChain.PRIMARY
    assert decision.raw_vol == pytest.approx(0.02)
    assert decision.reason_codes == ()


def test_when_primary_fails_routes_to_approved_baseline() -> None:
    state = FallbackState.initial()
    decision = advance_fallback(
        state=state,
        obs=_obs(primary_ok=False),
        config=_config(),
    )

    assert decision.next_state.current_tier == FallbackChain.APPROVED_BASELINE


def test_monotonic_conservatism_holds_size_vol_above_last_trusted() -> None:
    """While the primary is invalid, size_vol >= last_trusted_size_vol."""

    state = FallbackState(
        current_tier=FallbackChain.APPROVED_BASELINE,
        consecutive_valid_observations=0,
        last_trusted_size_vol=0.05,
        in_recovery=False,
    )
    decision = advance_fallback(
        state=state,
        obs=_obs(primary_ok=False, proposed_raw_vol=0.02),
        config=_config(),
    )

    # raw_vol returned must be >= last_trusted_size_vol (0.05): the HOLD.
    assert decision.raw_vol >= 0.05 - 1.0e-12
    assert FALLBACK_HOLD in decision.reason_codes


def test_release_valve_requires_n_consecutive_valid_observations() -> None:
    """After source recovers, release valve opens only after N consecutive
    valid observations.
    """

    config = _config(release_n=3)
    state = FallbackState(
        current_tier=FallbackChain.EWMA,
        consecutive_valid_observations=0,
        last_trusted_size_vol=0.05,
        in_recovery=False,
    )
    # Step 1: primary recovers but only 1 consecutive valid; still HOLD.
    step1 = advance_fallback(state=state, obs=_obs(), config=config)
    assert step1.next_state.consecutive_valid_observations == 1
    assert FALLBACK_HOLD in step1.reason_codes
    assert SOURCE_RECOVERED not in step1.reason_codes

    # Step 2
    step2 = advance_fallback(state=step1.next_state, obs=_obs(), config=config)
    assert step2.next_state.consecutive_valid_observations == 2
    assert FALLBACK_HOLD in step2.reason_codes
    assert SOURCE_RECOVERED not in step2.reason_codes

    # Step 3: the Nth valid observation opens the release valve
    step3 = advance_fallback(state=step2.next_state, obs=_obs(), config=config)
    assert step3.next_state.consecutive_valid_observations == 3
    assert SOURCE_RECOVERED in step3.reason_codes
    assert step3.next_state.in_recovery is True


def test_recovery_decay_bounded_by_smoothing_max_daily_decrease() -> None:
    """Once in recovery, raw_vol may decay but bounded by per-step max decrease."""

    config = _config(max_decrease=0.10)
    state = FallbackState(
        current_tier=FallbackChain.PRIMARY,
        consecutive_valid_observations=10,
        last_trusted_size_vol=0.10,
        in_recovery=True,
    )
    decision = advance_fallback(
        state=state,
        obs=_obs(proposed_raw_vol=0.05),
        config=config,
    )

    # last_trusted=0.10, max_decrease=0.10, minimum allowed = 0.09. Even though
    # proposed_raw_vol=0.05, the decay floor protects against a >10% drop.
    assert decision.raw_vol >= 0.10 * (1.0 - 0.10) - 1.0e-12
    assert FALLBACK_RECOVERY_DECAY in decision.reason_codes


def test_consecutive_valid_resets_when_source_fails_again() -> None:
    state = FallbackState(
        current_tier=FallbackChain.EWMA,
        consecutive_valid_observations=2,
        last_trusted_size_vol=0.05,
        in_recovery=False,
    )
    decision = advance_fallback(
        state=state,
        obs=_obs(primary_ok=False),
        config=_config(),
    )

    assert decision.next_state.consecutive_valid_observations == 0


def test_atr_fallback_routes_when_all_upstream_tiers_invalid() -> None:
    state = FallbackState.initial()
    decision = advance_fallback(
        state=state,
        obs=_obs(
            primary_ok=False,
            approved_baseline_ok=False,
            ewma_ok=False,
            measurement_ok=False,
            atr_ok=True,
        ),
        config=_config(),
    )

    assert decision.next_state.current_tier == FallbackChain.ATR
    assert ATR_FALLBACK in decision.reason_codes


def test_trade_blocked_when_every_tier_invalid() -> None:
    state = FallbackState.initial()
    decision = advance_fallback(
        state=state,
        obs=_obs(
            primary_ok=False,
            approved_baseline_ok=False,
            ewma_ok=False,
            measurement_ok=False,
            atr_ok=False,
        ),
        config=_config(),
    )

    assert decision.next_state.current_tier == FallbackChain.TRADE_BLOCK
    assert TRADE_BLOCKED in decision.reason_codes


def test_trade_block_has_zero_size_vol() -> None:
    state = FallbackState.initial()
    decision = advance_fallback(
        state=state,
        obs=_obs(
            primary_ok=False,
            approved_baseline_ok=False,
            ewma_ok=False,
            measurement_ok=False,
            atr_ok=False,
        ),
        config=_config(),
    )

    assert decision.raw_vol == 0.0


def test_tier_ordering_matches_research_plan() -> None:
    """The 6-tier ordering: primary, baseline, ewma, measurement, atr, block.
    """

    assert FallbackChain.PRIMARY.value < FallbackChain.APPROVED_BASELINE.value
    assert FallbackChain.APPROVED_BASELINE.value < FallbackChain.EWMA.value
    assert FallbackChain.EWMA.value < FallbackChain.MEASUREMENT_ONLY.value
    assert FallbackChain.MEASUREMENT_ONLY.value < FallbackChain.ATR.value
    assert FallbackChain.ATR.value < FallbackChain.TRADE_BLOCK.value


def test_fallback_config_rejects_non_positive_release_n() -> None:
    with pytest.raises(ValueError, match="fallback_release_n"):
        FallbackConfig(fallback_release_n=0, smoothing_max_daily_decrease=0.1)


def test_fallback_config_rejects_invalid_decrease_bound() -> None:
    with pytest.raises(ValueError, match="smoothing_max_daily_decrease"):
        FallbackConfig(fallback_release_n=5, smoothing_max_daily_decrease=1.5)


def test_fallback_config_rejects_invalid_staleness_ratio() -> None:
    with pytest.raises(ValueError, match="max_fallback_staleness_ratio"):
        FallbackConfig(
            fallback_release_n=5,
            smoothing_max_daily_decrease=0.1,
            max_fallback_staleness_ratio=1.5,
        )


def test_initial_state_has_no_last_trusted_size_vol() -> None:
    state = FallbackState.initial()
    assert state.current_tier == FallbackChain.PRIMARY
    assert state.consecutive_valid_observations == 0
    assert state.last_trusted_size_vol is None
    assert state.in_recovery is False


def test_last_trusted_size_vol_advances_on_successful_observation() -> None:
    state = FallbackState.initial()
    decision = advance_fallback(state=state, obs=_obs(proposed_raw_vol=0.04), config=_config())

    assert decision.next_state.last_trusted_size_vol == pytest.approx(0.04)
