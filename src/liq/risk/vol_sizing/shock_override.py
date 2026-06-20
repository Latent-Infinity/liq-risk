"""Shock override (research plan §9.6).

Smoothing risks slow response to shocks. The override bypasses smoothing
either partially or fully when the raw vol jump is large enough OR a
qualifying shock label fires:

    size_vol_t = max(size_vol_t_smoothed, shock_floor_multiplier * raw_vol_t)

The trigger condition is:

    raw_vol_t / size_vol_{t-1} > shock_ratio_threshold
    OR a qualifying shock label is set

**Hard boundary #13 (§9.6).** Session-complete labels — ``JUMP_DAY``,
``GAP_DOMINATED_VOL``, ``HIGH_ESTIMATOR_DISAGREEMENT`` — are only legal
when ``decision_kind == 'eod'``. Calling the override with one of these
labels while ``decision_kind == 'intraday'`` raises ``ValueError``: at
intraday tau, those labels are not yet available, and acting on them would
reintroduce the exact same-session leakage the feature layer prevents.

This module is intentionally pure — no I/O, no logging, no mutation of
external state. It returns a frozen :class:`ShockOverrideResult` carrying
the decision, the size_vol, the trigger source, and the firing labels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .reason_codes import SHOCK_OVERRIDE_APPLIED, ReasonCode

DecisionKind = Literal["intraday", "eod"]
TriggerSource = Literal["ratio", "label", "ratio_and_label"]

INTRADAY_SAFE_LABELS: frozenset[str] = frozenset(
    {
        "INTRADAY_SHOCK_TRIGGERED",
        "PARTIAL_RANGE_SPIKE",
        "PARTIAL_RV_SPIKE",
        "OPEN_GAP_KNOWN",
    }
)
EOD_ONLY_LABELS: frozenset[str] = frozenset(
    {
        "JUMP_DAY",
        "GAP_DOMINATED_VOL",
        "HIGH_ESTIMATOR_DISAGREEMENT",
    }
)
ALL_SHOCK_LABELS: frozenset[str] = INTRADAY_SAFE_LABELS | EOD_ONLY_LABELS


@dataclass(frozen=True)
class ShockOverrideConfig:
    """Frozen shock-override parameters consumed by :func:`apply_shock_override`."""

    shock_ratio_threshold: float
    shock_floor_multiplier: float

    def __post_init__(self) -> None:
        if self.shock_ratio_threshold <= 1.0:
            raise ValueError("shock_ratio_threshold must be > 1.0")
        if self.shock_floor_multiplier < 0.0:
            raise ValueError("shock_floor_multiplier must be non-negative")


@dataclass(frozen=True)
class ShockOverrideResult:
    """Outcome of the shock-override check."""

    triggered: bool
    size_vol: float
    trigger_source: TriggerSource | None
    firing_labels: frozenset[str]
    reason_code: ReasonCode | None


def _validate_labels(labels: frozenset[str], decision_kind: DecisionKind) -> None:
    unknown = labels - ALL_SHOCK_LABELS
    if unknown:
        raise ValueError(f"unknown shock label(s): {sorted(unknown)}")
    if decision_kind == "intraday":
        illegal = labels & EOD_ONLY_LABELS
        if illegal:
            raise ValueError(
                "session-complete label(s) "
                f"{sorted(illegal)} are illegal for intraday decisions "
                "(hard boundary #13: §9.6)"
            )


def _ratio_triggered(
    raw_vol: float, previous_size_vol: float | None, threshold: float
) -> bool:
    if previous_size_vol is None or previous_size_vol <= 0.0:
        return False
    return raw_vol / previous_size_vol > threshold


def apply_shock_override(
    *,
    raw_vol: float,
    size_vol_smoothed: float,
    previous_size_vol: float | None,
    labels: frozenset[str],
    decision_kind: DecisionKind,
    config: ShockOverrideConfig,
) -> ShockOverrideResult:
    """Decide whether the shock override fires; return the post-override size_vol.

    The function is pure: it does not depend on any state besides its
    arguments. When the override does not fire, ``size_vol_smoothed`` is
    returned unmodified and the result carries ``triggered=False`` with no
    reason code.
    """

    if raw_vol <= 0.0:
        raise ValueError("raw_vol must be positive")
    _validate_labels(labels, decision_kind)

    ratio_fires = _ratio_triggered(raw_vol, previous_size_vol, config.shock_ratio_threshold)
    label_fires = bool(labels)
    if not (ratio_fires or label_fires):
        return ShockOverrideResult(
            triggered=False,
            size_vol=float(size_vol_smoothed),
            trigger_source=None,
            firing_labels=frozenset(),
            reason_code=None,
        )

    if ratio_fires and label_fires:
        trigger_source: TriggerSource = "ratio_and_label"
    elif ratio_fires:
        trigger_source = "ratio"
    else:
        trigger_source = "label"

    override_size = max(
        float(size_vol_smoothed),
        config.shock_floor_multiplier * float(raw_vol),
    )
    return ShockOverrideResult(
        triggered=True,
        size_vol=float(override_size),
        trigger_source=trigger_source,
        firing_labels=labels,
        reason_code=SHOCK_OVERRIDE_APPLIED,
    )


__all__ = [
    "ALL_SHOCK_LABELS",
    "DecisionKind",
    "EOD_ONLY_LABELS",
    "INTRADAY_SAFE_LABELS",
    "ShockOverrideConfig",
    "ShockOverrideResult",
    "TriggerSource",
    "apply_shock_override",
]
