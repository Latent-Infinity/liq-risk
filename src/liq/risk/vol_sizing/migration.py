"""ATR-migration ledger plus stage advance and rollback.

Stages 0 through 4 mark progressive rollout of `size_vol_t` in production:

    0  ATR production, size_vol_t shadow-computed
    1  compare ATR vs size_vol_t, historical plus shadow
    2  size_vol_t reporting only (no capital)
    3  limited capital / bucket rollout
    4  full replacement if promotion gates pass

The migration is **sleeve-by-sleeve**: the reversal sleeve enters first;
other sleeves can only advance past Stage 0 after the reversal sleeve
reaches Stage 3.

The ledger is append-only and records every stage transition and rollback
with timestamp and reason. The :func:`propose_next_stage` evaluator is pure:
it reads the current ledger state and the latest parallel-run observation,
then proposes the next stage **or** signals a rollback when any of the
following deltas exceed registry thresholds:

    - vol_target_error_delta vs ATR baseline
    - turnover_delta vs ATR baseline
    - leverage_spike_delta vs ATR baseline
    - operational_failures count
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum

REVERSAL_SLEEVE: str = "reversal"
MIN_PARALLEL_SESSIONS = 20


class MigrationStage(IntEnum):
    STAGE_0 = 0
    STAGE_1 = 1
    STAGE_2 = 2
    STAGE_3 = 3
    STAGE_4 = 4


@dataclass(frozen=True)
class RollbackTriggerConfig:
    """Registry-tracked rollback thresholds."""

    vol_target_error_delta: float
    turnover_delta: float
    leverage_spike_delta: float

    def __post_init__(self) -> None:
        for name, value in (
            ("vol_target_error_delta", self.vol_target_error_delta),
            ("turnover_delta", self.turnover_delta),
            ("leverage_spike_delta", self.leverage_spike_delta),
        ):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class MigrationObservation:
    """Per-decision audit row consumed by :func:`propose_next_stage`."""

    sleeve: str
    parallel_sessions: int
    vol_target_error_delta: float
    turnover_delta: float
    leverage_spike_delta: float
    operational_failures: int

    def __post_init__(self) -> None:
        if not self.sleeve:
            raise ValueError("sleeve must be a non-empty name")
        if self.parallel_sessions < 0:
            raise ValueError("parallel_sessions must be non-negative")
        if self.operational_failures < 0:
            raise ValueError("operational_failures must be non-negative")


@dataclass(frozen=True)
class StageTransition:
    """Single row in the migration ledger."""

    sleeve: str
    from_stage: MigrationStage
    to_stage: MigrationStage
    at: datetime
    frozen_on: datetime
    reason: str


@dataclass(frozen=True)
class MigrationDecision:
    """Outcome of :func:`propose_next_stage`."""

    sleeve: str
    proposed_stage: MigrationStage
    should_rollback: bool
    rollback_reasons: tuple[str, ...]
    gating_reasons: tuple[str, ...]


@dataclass
class MigrationLedger:
    """Append-only migration record (operator-owned)."""

    _stage_by_sleeve: dict[str, MigrationStage] = field(default_factory=dict)
    _transitions: list[StageTransition] = field(default_factory=list)
    _rollbacks: list[StageTransition] = field(default_factory=list)

    def current_stage(self, sleeve: str) -> MigrationStage:
        return self._stage_by_sleeve.get(sleeve, MigrationStage.STAGE_0)

    def transitions(self) -> tuple[StageTransition, ...]:
        return tuple(self._transitions)

    def rollbacks(self) -> tuple[StageTransition, ...]:
        return tuple(self._rollbacks)

    def advance(
        self,
        *,
        sleeve: str,
        to_stage: MigrationStage,
        at: datetime,
        reason: str,
    ) -> StageTransition:
        current = self.current_stage(sleeve)
        if to_stage.value != current.value + 1:
            raise ValueError(
                "advance() is forward only and refuses non-adjacent stages: "
                f"{current.name} -> {to_stage.name}"
            )
        transition = StageTransition(
            sleeve=sleeve,
            from_stage=current,
            to_stage=to_stage,
            at=at,
            frozen_on=at,
            reason=reason,
        )
        self._stage_by_sleeve[sleeve] = to_stage
        self._transitions.append(transition)
        return transition

    def rollback(
        self,
        *,
        sleeve: str,
        to_stage: MigrationStage,
        at: datetime,
        reason: str,
    ) -> StageTransition:
        current = self.current_stage(sleeve)
        if to_stage.value >= current.value:
            raise ValueError(
                f"rollback() must go backward: current={current.name}, "
                f"requested={to_stage.name}"
            )
        transition = StageTransition(
            sleeve=sleeve,
            from_stage=current,
            to_stage=to_stage,
            at=at,
            frozen_on=at,
            reason=reason,
        )
        self._stage_by_sleeve[sleeve] = to_stage
        self._rollbacks.append(transition)
        return transition


def _evaluate_rollback(
    obs: MigrationObservation, config: RollbackTriggerConfig
) -> tuple[str, ...]:
    breaches: list[str] = []
    if obs.vol_target_error_delta > config.vol_target_error_delta:
        breaches.append("vol_target_error_delta")
    if obs.turnover_delta > config.turnover_delta:
        breaches.append("turnover_delta")
    if obs.leverage_spike_delta > config.leverage_spike_delta:
        breaches.append("leverage_spike_delta")
    if obs.operational_failures > 0:
        breaches.append("operational_failures")
    return tuple(breaches)


def _gating_reasons(
    *,
    sleeve: str,
    current: MigrationStage,
    obs: MigrationObservation,
    ledger: MigrationLedger,
) -> tuple[str, ...]:
    reasons: list[str] = []
    if current == MigrationStage.STAGE_4:
        reasons.append("already_at_stage_4")
        return tuple(reasons)
    if obs.parallel_sessions < MIN_PARALLEL_SESSIONS:
        reasons.append("insufficient_parallel_sessions")
    if sleeve != REVERSAL_SLEEVE and current == MigrationStage.STAGE_0:
        reversal_stage = ledger.current_stage(REVERSAL_SLEEVE)
        if reversal_stage < MigrationStage.STAGE_3:
            reasons.append("reversal_sleeve_not_at_stage_3")
    return tuple(reasons)


def propose_next_stage(
    *,
    sleeve: str,
    ledger: MigrationLedger,
    obs: MigrationObservation,
    config: RollbackTriggerConfig,
) -> MigrationDecision:
    """Read the ledger + the latest observation; propose the next stage."""

    current = ledger.current_stage(sleeve)
    rollback_reasons = _evaluate_rollback(obs, config)
    gating_reasons = _gating_reasons(
        sleeve=sleeve, current=current, obs=obs, ledger=ledger
    )
    should_rollback = len(rollback_reasons) > 0
    proposed = current
    if not should_rollback and not gating_reasons and current < MigrationStage.STAGE_4:
        proposed = MigrationStage(current.value + 1)
    return MigrationDecision(
        sleeve=sleeve,
        proposed_stage=proposed,
        should_rollback=should_rollback,
        rollback_reasons=rollback_reasons,
        gating_reasons=gating_reasons,
    )


__all__ = [
    "MigrationDecision",
    "MigrationLedger",
    "MigrationObservation",
    "MigrationStage",
    "MIN_PARALLEL_SESSIONS",
    "REVERSAL_SLEEVE",
    "RollbackTriggerConfig",
    "StageTransition",
    "propose_next_stage",
]
