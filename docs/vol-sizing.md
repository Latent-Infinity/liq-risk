# Volatility sizing input (`size_vol_t`)

The sizing-layer half of the volatility-forecast workstream. The library
takes a forecast row (or its absence), a measurement row (or its
absence), an operator-owned policy, and the previous `size_vol`, and
returns one frozen `SizeVolInput` row carrying the post-transform
`size_vol`, the full reason-code stack, and the validity window.

This document anchors the four production surfaces:

1. The ordered seven-step pipeline.
2. The shock override applied after smoothing.
3. The six-tier fallback hierarchy with monotonic-conservatism + release valve.
4. The ATR migration ledger that records stage advances and rollbacks.

## Ordered pipeline

The transform is **non-commutative**: floor and cap interact with
smoothing such that re-ordering changes the result, so the seven steps
run in a fixed order:

```text
1. SELECT raw source         (per policy A/B/C/D)
2. CONSERVATIVE COMBINATION  raw_vol = max(measurement, forecast) for hybrid
3. FALLBACK                  most-conservative path on quality-flag failure
4. UNCERTAINTY MULTIPLIER    adjusted = raw * multiplier (>= 1.0)
5. FLOORS & CAPS             clip(max(adjusted, floor), cap)
6. SMOOTHING / HYSTERESIS    EWMA + per-step max-decrease floor
7. REASON CODES              one enum per step that changed size_vol
```

The four policies in §9.2 are `measurement`, `forecast`, `hybrid`, and
`uncertainty_adjusted`. The production default is
`uncertainty_adjusted` (decisions registry §11): hybrid raw vol followed
by a multiplier `1.0 + clip(uncertainty_k · rank, 0, uncertainty_max_extra)`.

The four input dataclasses + the result are:

- `MeasurementInput` — measurement-track snapshot
  (`measurement_id`, `measurement_vol`, `source_valid_from`,
  `valid_until`, plus quality flags `is_quality_ok` / `is_stale`).
- `ForecastInput` — forecast-track snapshot
  (`forecast_id`, `forecast_vol`, `forecast_var_per_session_equiv`,
  `forecast_available_ts`, `valid_until`, the two uncertainty percentiles,
  and the same quality flags).
- `SizingPolicy` — frozen policy (`policy_id`, `policy_version`,
  `policy_floor`, `emergency_floor`, `cap_vol`, `uncertainty_k`,
  `uncertainty_max_extra`, `smoothing_half_life`,
  `smoothing_max_daily_decrease`). Validates `cap_vol >= policy_floor`
  and `cap_vol >= emergency_floor` at construction.
- `SizeVolInput` — frozen result with the post-transform `size_vol`, the
  full reason-code stack, the `clip_applied` / `smoothing_applied` /
  `shock_override_applied` audit flags, and the validity window.

`compute_size_vol(*, symbol, sizing_decision_ts, measurement, forecast,
policy, previous_size_vol, regime_labels=..., shock_config=None,
shock_labels=..., decision_kind='eod')` is the entry point.

## Shock override

Smoothing risks slow response to shocks. The override bypasses smoothing
either partially or fully:

```text
If raw_vol / size_vol_{t-1} > shock_ratio_threshold
   OR a qualifying shock label is set:
  size_vol = max(size_vol_smoothed, shock_floor_multiplier * raw_vol)
```

`ShockOverrideConfig` carries the two thresholds. The pure
`apply_shock_override(*, raw_vol, size_vol_smoothed, previous_size_vol,
labels, decision_kind, config)` returns a frozen result with
`triggered`, `size_vol`, `trigger_source`
(`ratio` / `label` / `ratio_and_label`), `firing_labels`, and the
optional `SHOCK_OVERRIDE_APPLIED` reason code.

**Hard boundary #13.** Session-complete labels (`JUMP_DAY`,
`GAP_DOMINATED_VOL`, `HIGH_ESTIMATOR_DISAGREEMENT`) are illegal when
`decision_kind == 'intraday'`. The intraday-safe labels accepted at
intraday tau are `INTRADAY_SHOCK_TRIGGERED`, `PARTIAL_RANGE_SPIKE`,
`PARTIAL_RV_SPIKE`, `OPEN_GAP_KNOWN`. Calling the override with a
session-complete label while `decision_kind='intraday'` raises
`ValueError`.

## Fallback hierarchy

The chain in priority order:

```text
1 PRIMARY            (model forecast)
2 APPROVED_BASELINE  (HAR-RV)
3 EWMA               (RiskMetrics-style decayed)
4 MEASUREMENT_ONLY   (measurement track w/ conservative floor)
5 ATR                (legacy ATR-based sizing)
6 TRADE_BLOCK        (no-size / block)
```

`FallbackChain` is the `IntEnum`; `FallbackObservation` carries one
boolean per tier plus the proposed raw_vol;
`FallbackState(current_tier, consecutive_valid_observations,
last_trusted_size_vol, in_recovery)` is the operator-persisted record;
`advance_fallback(*, state, obs, config)` is the pure transition
function that returns a new `FallbackState`, the post-HOLD/post-DECAY
`raw_vol`, and the reason-code stack.

**Monotonic-conservatism HOLD.** While the primary source is invalid,
`size_vol >= last_trusted_size_vol`. The HOLD prevents silently reduced
conservatism on a stale or quality-flagged source.

**Release valve.** After the primary recovers and passes
`fallback_release_n=5` consecutive valid observations, `SOURCE_RECOVERED`
fires and decay may proceed bounded by `smoothing_max_daily_decrease`.

## ATR migration

Stages 0 → 4 mark progressive rollout of `size_vol_t` in production:

```text
0  ATR production, size_vol_t shadow-computed
1  compare ATR vs size_vol_t (history + shadow)
2  size_vol_t reporting only (no capital)
3  limited capital / bucket rollout
4  full replacement
```

The migration is **sleeve-by-sleeve**: the reversal sleeve enters first;
other sleeves can only advance past Stage 0 after reversal reaches
Stage 3.

`MigrationStage` is the `IntEnum` (`STAGE_0` ... `STAGE_4`).
`MigrationLedger` is append-only, recording every `StageTransition`
(`sleeve`, `from_stage`, `to_stage`, `at`, `reason`). `advance()` is
forward-only and refuses non-adjacent stages; `rollback()` records a
backward transition with its own reason. `propose_next_stage(*, sleeve,
ledger, obs, config)` reads the latest `MigrationObservation` and
returns a `MigrationDecision` carrying the proposed stage, the
`should_rollback` flag, and the per-condition rollback reasons (the
breaches that fired) and gating reasons (sleeve-by-sleeve, insufficient
parallel sessions, already at Stage 4).

The rollback triggers fire when any of the panel deltas exceed their
registry thresholds:

- `rollback_vol_target_error_delta` (default 0.05)
- `rollback_turnover_delta` (default 0.10)
- `rollback_leverage_spike_delta` (default 0.05)
- any `operational_failures > 0`

## Reason codes

Every step that changed `size_vol` appends a `ReasonCode` to the result's
`reason_codes` tuple. Each code carries a `stage`
(`select` / `combination` / `fallback` / `uncertainty` / `floor` / `cap`
/ `smoothing` / `shock`) and a `severity`
(`ok` / `info` / `warn` / `conservative`). The catalog:

- pipeline: `SELECT_MEASUREMENT`, `SELECT_FORECAST`, `SELECT_HYBRID`,
  `SELECT_UNCERTAINTY_ADJUSTED`, `CONSERVATIVE_COMBINATION_APPLIED`,
  `DATA_QUALITY_FALLBACK`, `UNCERTAINTY_APPLIED`, `FLOOR_APPLIED`,
  `EMERGENCY_FLOOR_APPLIED`, `CAP_APPLIED`, `SMOOTHING_APPLIED`,
  `HYSTERESIS_APPLIED`
- shock: `SHOCK_OVERRIDE_APPLIED`
- fallback: `FALLBACK_HOLD`, `FALLBACK_RECOVERY_DECAY`,
  `SOURCE_RECOVERED`, `ATR_FALLBACK`, `TRADE_BLOCKED`

The `EMERGENCY_FLOOR` invariant `size_vol_t >= EMERGENCY_FLOOR` is
asserted both before and after smoothing AND after any shock override.

## Parameter registry

Frozen values consumed by this library (full table lives in the
workspace parameter registry):

| Parameter | Value |
| --- | --- |
| `policy_floor` | 0.005 (0.5% daily vol) |
| `EMERGENCY_FLOOR` | 0.003 (absolute fail-safe) |
| `cap_vol` | 0.50 (50% daily vol upper bound) |
| `uncertainty_k` | 0.25 |
| `uncertainty_max_extra` | 0.50 |
| `smoothing_half_life` | 5 sessions |
| `smoothing_max_daily_decrease` | 0.10 |
| `shock_ratio_threshold` | 1.5 |
| `shock_floor_multiplier` | 1.0 |
| `fallback_release_n` | 5 observations |
| `max_fallback_staleness_ratio` | 0.20 |
| `rollback_vol_target_error_delta` | 0.05 |
| `rollback_turnover_delta` | 0.10 |
| `rollback_leverage_spike_delta` | 0.05 |
