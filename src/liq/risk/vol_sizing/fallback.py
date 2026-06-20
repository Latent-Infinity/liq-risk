"""Global fallback hierarchy with release valve.

The chain in priority order:

    1 PRIMARY              - model forecast
    2 APPROVED_BASELINE    - HAR-RV or other vetted baseline
    3 EWMA                 - RiskMetrics-style decayed estimator
    4 MEASUREMENT_ONLY     - measurement track with conservative floor
    5 ATR                  - legacy ATR-based sizing
    6 TRADE_BLOCK          - no-size / block trade

Monotonic-conservatism (HOLD): while the primary source is invalid,
``size_vol_t >= last_trusted_size_vol``. The system is not trapped high
forever: after the source recovers and passes ``fallback_release_n``
consecutive valid observations, the **release valve** opens and
``size_vol_t`` may DECAY per the approved smoothing policy, bounded by
``smoothing_max_daily_decrease``.

This module is pure: the state machine is encoded in :class:`FallbackState`
and the transition is computed by the pure :func:`advance_fallback`. The
caller is responsible for persisting the returned ``next_state``.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum

from .reason_codes import (
    ATR_FALLBACK,
    FALLBACK_HOLD,
    FALLBACK_RECOVERY_DECAY,
    SOURCE_RECOVERED,
    TRADE_BLOCKED,
    ReasonCode,
)


class FallbackChain(IntEnum):
    """The six-tier fallback hierarchy in priority order."""

    PRIMARY = 1
    APPROVED_BASELINE = 2
    EWMA = 3
    MEASUREMENT_ONLY = 4
    ATR = 5
    TRADE_BLOCK = 6


@dataclass(frozen=True)
class FallbackConfig:
    """Frozen fallback policy."""

    fallback_release_n: int
    smoothing_max_daily_decrease: float
    max_fallback_staleness_ratio: float = 0.20

    def __post_init__(self) -> None:
        if self.fallback_release_n <= 0:
            raise ValueError("fallback_release_n must be positive")
        if not 0.0 < self.smoothing_max_daily_decrease <= 1.0:
            raise ValueError("smoothing_max_daily_decrease must be in (0, 1]")
        if not 0.0 <= self.max_fallback_staleness_ratio <= 1.0:
            raise ValueError("max_fallback_staleness_ratio must be in [0, 1]")


@dataclass(frozen=True)
class FallbackObservation:
    """Per-decision input signaling tier availability + the proposed raw_vol.

    All boolean fields are TRUE when the tier passed its quality + freshness
    gates for this decision. ``proposed_raw_vol`` is the raw_vol the active
    tier would emit before the HOLD/DECAY filter is applied.
    """

    primary_ok: bool
    approved_baseline_ok: bool
    ewma_ok: bool
    measurement_ok: bool
    atr_ok: bool
    proposed_raw_vol: float


@dataclass(frozen=True)
class FallbackState:
    """Operator-persisted fallback state - pure data, no behavior."""

    current_tier: FallbackChain
    consecutive_valid_observations: int
    last_trusted_size_vol: float | None
    in_recovery: bool

    @classmethod
    def initial(cls) -> FallbackState:
        return cls(
            current_tier=FallbackChain.PRIMARY,
            consecutive_valid_observations=0,
            last_trusted_size_vol=None,
            in_recovery=False,
        )


@dataclass(frozen=True)
class FallbackDecision:
    """Outcome of :func:`advance_fallback`."""

    raw_vol: float
    next_state: FallbackState
    reason_codes: tuple[ReasonCode, ...]


def _pick_active_tier(obs: FallbackObservation) -> FallbackChain:
    """Return the highest-priority tier whose source is available."""

    if obs.primary_ok:
        return FallbackChain.PRIMARY
    if obs.approved_baseline_ok:
        return FallbackChain.APPROVED_BASELINE
    if obs.ewma_ok:
        return FallbackChain.EWMA
    if obs.measurement_ok:
        return FallbackChain.MEASUREMENT_ONLY
    if obs.atr_ok:
        return FallbackChain.ATR
    return FallbackChain.TRADE_BLOCK


def advance_fallback(
    *,
    state: FallbackState,
    obs: FallbackObservation,
    config: FallbackConfig,
) -> FallbackDecision:
    """Run one tick of the fallback state machine.

    Computes the next state, the post-HOLD/post-DECAY ``raw_vol`` the
    pipeline should consume, and the reason-code stack that documents
    every tier transition or HOLD/DECAY event.
    """

    if obs.proposed_raw_vol < 0.0:
        raise ValueError("proposed_raw_vol must be non-negative")
    reasons: list[ReasonCode] = []
    next_tier = _pick_active_tier(obs)
    primary_invalid = next_tier != FallbackChain.PRIMARY
    consecutive = (
        state.consecutive_valid_observations + 1
        if obs.primary_ok
        else 0
    )

    # Trade-block tier emits a single TRADE_BLOCKED code and zero raw_vol.
    if next_tier == FallbackChain.TRADE_BLOCK:
        reasons.append(TRADE_BLOCKED)
        next_state = FallbackState(
            current_tier=FallbackChain.TRADE_BLOCK,
            consecutive_valid_observations=consecutive,
            last_trusted_size_vol=state.last_trusted_size_vol,
            in_recovery=False,
        )
        return FallbackDecision(raw_vol=0.0, next_state=next_state, reason_codes=tuple(reasons))

    if next_tier == FallbackChain.ATR:
        reasons.append(ATR_FALLBACK)

    raw_vol = float(obs.proposed_raw_vol)
    in_recovery = state.in_recovery
    just_recovered = False

    if primary_invalid:
        # HOLD: monotonic-conservatism. size_vol >= last_trusted.
        if state.last_trusted_size_vol is not None:
            raw_vol = max(raw_vol, float(state.last_trusted_size_vol))
        reasons.append(FALLBACK_HOLD)
        in_recovery = False
    elif state.last_trusted_size_vol is not None and not in_recovery:
        # Primary recovered. Count consecutive valid; HOLD until release valve opens.
        if consecutive >= config.fallback_release_n:
            in_recovery = True
            just_recovered = True
            reasons.append(SOURCE_RECOVERED)
        else:
            raw_vol = max(raw_vol, float(state.last_trusted_size_vol))
            reasons.append(FALLBACK_HOLD)

    if in_recovery and state.last_trusted_size_vol is not None:
        # DECAY: bounded by max-daily-decrease.
        decay_floor = float(state.last_trusted_size_vol) * (
            1.0 - config.smoothing_max_daily_decrease
        )
        raw_vol = max(raw_vol, decay_floor)
        if not just_recovered:
            reasons.append(FALLBACK_RECOVERY_DECAY)
        elif raw_vol < state.last_trusted_size_vol - 1.0e-12:
            reasons.append(FALLBACK_RECOVERY_DECAY)

    # Persist last_trusted only when primary delivers; never on a fallback tier.
    new_last_trusted = (
        raw_vol if next_tier == FallbackChain.PRIMARY else state.last_trusted_size_vol
    )
    next_state = FallbackState(
        current_tier=next_tier,
        consecutive_valid_observations=consecutive,
        last_trusted_size_vol=new_last_trusted,
        in_recovery=in_recovery,
    )
    return FallbackDecision(raw_vol=raw_vol, next_state=next_state, reason_codes=tuple(reasons))


__all__ = [
    "FallbackChain",
    "FallbackConfig",
    "FallbackDecision",
    "FallbackObservation",
    "FallbackState",
    "advance_fallback",
]
