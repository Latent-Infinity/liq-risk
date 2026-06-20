from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from liq.risk.vol_sizing.migration import (
    MIN_PARALLEL_SESSIONS,
    REVERSAL_SLEEVE,
    MigrationDecision,
    MigrationLedger,
    MigrationObservation,
    MigrationStage,
    RollbackTriggerConfig,
    propose_next_stage,
)

_NOW = datetime(2026, 6, 19, 14, 30, tzinfo=UTC)


def _config(
    *,
    vol_target_error_delta: float = 0.05,
    turnover_delta: float = 0.10,
    leverage_spike_delta: float = 0.05,
) -> RollbackTriggerConfig:
    return RollbackTriggerConfig(
        vol_target_error_delta=vol_target_error_delta,
        turnover_delta=turnover_delta,
        leverage_spike_delta=leverage_spike_delta,
    )


def _obs(
    *,
    sleeve: str = REVERSAL_SLEEVE,
    parallel_sessions: int = 30,
    vol_target_error_delta: float = 0.0,
    turnover_delta: float = 0.0,
    leverage_spike_delta: float = 0.0,
    operational_failures: int = 0,
) -> MigrationObservation:
    return MigrationObservation(
        sleeve=sleeve,
        parallel_sessions=parallel_sessions,
        vol_target_error_delta=vol_target_error_delta,
        turnover_delta=turnover_delta,
        leverage_spike_delta=leverage_spike_delta,
        operational_failures=operational_failures,
    )


def test_stage_ordering_matches_research_plan() -> None:
    assert MigrationStage.STAGE_0 < MigrationStage.STAGE_1
    assert MigrationStage.STAGE_1 < MigrationStage.STAGE_2
    assert MigrationStage.STAGE_2 < MigrationStage.STAGE_3
    assert MigrationStage.STAGE_3 < MigrationStage.STAGE_4


def test_initial_stage_is_zero_for_every_sleeve() -> None:
    ledger = MigrationLedger()
    assert ledger.current_stage("reversal") == MigrationStage.STAGE_0
    assert ledger.current_stage("trend") == MigrationStage.STAGE_0


def test_advancing_one_stage_records_a_ledger_transition() -> None:
    ledger = MigrationLedger()
    ledger.advance(
        sleeve="reversal",
        to_stage=MigrationStage.STAGE_1,
        at=_NOW,
        reason="parallel-audit-clean",
    )

    transitions = ledger.transitions()
    assert len(transitions) == 1
    assert transitions[0].sleeve == "reversal"
    assert transitions[0].from_stage == MigrationStage.STAGE_0
    assert transitions[0].to_stage == MigrationStage.STAGE_1
    assert transitions[0].at == _NOW
    assert transitions[0].frozen_on == _NOW
    assert ledger.current_stage("reversal") == MigrationStage.STAGE_1


def test_advance_rejects_skipping_stages() -> None:
    ledger = MigrationLedger()
    with pytest.raises(ValueError, match="non-adjacent"):
        ledger.advance(
            sleeve="reversal",
            to_stage=MigrationStage.STAGE_2,
            at=_NOW,
            reason="skipping",
        )


def test_advance_rejects_backward_transitions_via_advance() -> None:
    ledger = MigrationLedger()
    ledger.advance(sleeve="reversal", to_stage=MigrationStage.STAGE_1, at=_NOW, reason="up")
    with pytest.raises(ValueError, match="forward only"):
        ledger.advance(
            sleeve="reversal",
            to_stage=MigrationStage.STAGE_0,
            at=_NOW + timedelta(days=1),
            reason="down",
        )


def test_rollback_records_a_transition_back_to_earlier_stage() -> None:
    ledger = MigrationLedger()
    ledger.advance(sleeve="reversal", to_stage=MigrationStage.STAGE_1, at=_NOW, reason="up")
    ledger.advance(
        sleeve="reversal",
        to_stage=MigrationStage.STAGE_2,
        at=_NOW + timedelta(days=1),
        reason="continued",
    )

    ledger.rollback(
        sleeve="reversal",
        to_stage=MigrationStage.STAGE_0,
        at=_NOW + timedelta(days=2),
        reason="vol-target-error-breach",
    )

    rollbacks = ledger.rollbacks()
    assert len(rollbacks) == 1
    assert rollbacks[0].to_stage == MigrationStage.STAGE_0
    assert rollbacks[0].frozen_on == _NOW + timedelta(days=2)
    assert ledger.current_stage("reversal") == MigrationStage.STAGE_0


def test_propose_advances_with_clean_parallel_run() -> None:
    ledger = MigrationLedger()
    decision = propose_next_stage(
        sleeve="reversal",
        ledger=ledger,
        obs=_obs(parallel_sessions=30),
        config=_config(),
    )

    assert isinstance(decision, MigrationDecision)
    assert decision.proposed_stage == MigrationStage.STAGE_1
    assert decision.should_rollback is False
    assert decision.rollback_reasons == ()


def test_propose_blocks_advance_with_insufficient_parallel_sessions() -> None:
    ledger = MigrationLedger()
    decision = propose_next_stage(
        sleeve="reversal",
        ledger=ledger,
        obs=_obs(parallel_sessions=5),
        config=_config(),
    )

    # Should remain at Stage 0 because the parallel-session floor is not met.
    assert decision.proposed_stage == MigrationStage.STAGE_0
    assert MIN_PARALLEL_SESSIONS == 20


def test_rollback_triggers_when_vol_target_error_breaches() -> None:
    ledger = MigrationLedger()
    ledger.advance(sleeve="reversal", to_stage=MigrationStage.STAGE_1, at=_NOW, reason="up")
    decision = propose_next_stage(
        sleeve="reversal",
        ledger=ledger,
        obs=_obs(parallel_sessions=30, vol_target_error_delta=0.08),
        config=_config(vol_target_error_delta=0.05),
    )

    assert decision.should_rollback is True
    assert "vol_target_error_delta" in set(decision.rollback_reasons)


def test_rollback_triggers_on_turnover_breach() -> None:
    ledger = MigrationLedger()
    ledger.advance(sleeve="reversal", to_stage=MigrationStage.STAGE_1, at=_NOW, reason="up")
    decision = propose_next_stage(
        sleeve="reversal",
        ledger=ledger,
        obs=_obs(parallel_sessions=30, turnover_delta=0.20),
        config=_config(turnover_delta=0.10),
    )

    assert decision.should_rollback is True
    assert "turnover_delta" in decision.rollback_reasons


def test_rollback_triggers_on_leverage_spike_breach() -> None:
    ledger = MigrationLedger()
    ledger.advance(sleeve="reversal", to_stage=MigrationStage.STAGE_1, at=_NOW, reason="up")
    decision = propose_next_stage(
        sleeve="reversal",
        ledger=ledger,
        obs=_obs(parallel_sessions=30, leverage_spike_delta=0.10),
        config=_config(leverage_spike_delta=0.05),
    )

    assert decision.should_rollback is True
    assert "leverage_spike_delta" in decision.rollback_reasons


def test_rollback_triggers_on_operational_failures() -> None:
    ledger = MigrationLedger()
    ledger.advance(sleeve="reversal", to_stage=MigrationStage.STAGE_1, at=_NOW, reason="up")
    decision = propose_next_stage(
        sleeve="reversal",
        ledger=ledger,
        obs=_obs(parallel_sessions=30, operational_failures=3),
        config=_config(),
    )

    assert decision.should_rollback is True
    assert "operational_failures" in decision.rollback_reasons


def test_non_reversal_sleeve_cannot_advance_past_stage_0_until_reversal_at_stage_3() -> None:
    """Sleeve-by-sleeve invariant: 'reversal' sleeve enters first; other
    sleeves can only advance past Stage 0 after reversal reaches Stage 3.
    """

    ledger = MigrationLedger()
    # Reversal at Stage 1; other sleeves still blocked.
    ledger.advance(sleeve="reversal", to_stage=MigrationStage.STAGE_1, at=_NOW, reason="up")
    decision = propose_next_stage(
        sleeve="trend",
        ledger=ledger,
        obs=_obs(sleeve="trend", parallel_sessions=30),
        config=_config(),
    )

    assert decision.proposed_stage == MigrationStage.STAGE_0
    assert "reversal_sleeve_not_at_stage_3" in decision.gating_reasons


def test_non_reversal_sleeve_can_advance_once_reversal_at_stage_3() -> None:
    ledger = MigrationLedger()
    for stage in (
        MigrationStage.STAGE_1,
        MigrationStage.STAGE_2,
        MigrationStage.STAGE_3,
    ):
        ledger.advance(sleeve="reversal", to_stage=stage, at=_NOW, reason="up")
    decision = propose_next_stage(
        sleeve="trend",
        ledger=ledger,
        obs=_obs(sleeve="trend", parallel_sessions=30),
        config=_config(),
    )

    assert decision.proposed_stage == MigrationStage.STAGE_1
    assert "reversal_sleeve_not_at_stage_3" not in decision.gating_reasons


def test_stage_4_cannot_advance_further() -> None:
    ledger = MigrationLedger()
    for stage in (
        MigrationStage.STAGE_1,
        MigrationStage.STAGE_2,
        MigrationStage.STAGE_3,
        MigrationStage.STAGE_4,
    ):
        ledger.advance(sleeve="reversal", to_stage=stage, at=_NOW, reason="up")

    decision = propose_next_stage(
        sleeve="reversal",
        ledger=ledger,
        obs=_obs(parallel_sessions=200),
        config=_config(),
    )

    assert decision.proposed_stage == MigrationStage.STAGE_4
    assert "already_at_stage_4" in decision.gating_reasons


def test_rollback_trigger_config_rejects_non_positive_thresholds() -> None:
    with pytest.raises(ValueError, match="vol_target_error_delta"):
        RollbackTriggerConfig(
            vol_target_error_delta=0.0,
            turnover_delta=0.10,
            leverage_spike_delta=0.05,
        )


def test_observation_rejects_unknown_sleeve_name() -> None:
    """Sleeve names should be drawn from a small documented set; passing an
    empty string is rejected.
    """

    with pytest.raises(ValueError, match="sleeve"):
        MigrationObservation(
            sleeve="",
            parallel_sessions=10,
            vol_target_error_delta=0.0,
            turnover_delta=0.0,
            leverage_spike_delta=0.0,
            operational_failures=0,
        )
