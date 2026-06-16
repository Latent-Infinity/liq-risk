# ATR retirement — non-inferiority writeup (non-blocking)

This document is the head-to-head benchmark behind the canonical
risk-variance plan's downstream-sizing claim. It is **non-blocking**:
the actual retirement of ATR-based sizing in `liq-risk` is owned by
a separate downstream-migration plan and is not shipped here. What
ships here is the evidence.

| Field | Value |
| --- | --- |
| Canonical estimator | `gk_yang_zhang` (decisions registry §5 — MCS_B winner under Option B) |
| Comparator | `natr_t = ATR_t / close_t` per research plan §8 bridge |
| Bundle | `liq-experiments/fixtures/real/` (30 symbols × 753 sessions, TradeStation 1m source) |
| Manifest sha256 | `2113c5f405fca787c615dfc64806ff9937522137f72b05fffba493c0af5db48f` |
| Source artifact | `liq-experiments/results/sizing/sizing_comparison.json` |
| Source script | `liq-experiments/scripts/eval_sizing.py` |

## 1. Why the comparison is fair

Both estimators are smoothed with the **same effective memory profile**
(Wilder smoothing with `window=21`) so the comparison isolates
estimator information content from smoothing-window choice. Without
matched memory, a more responsive estimator would naturally produce
more turnover and the comparison would degenerate into a "knob-vs-knob"
test. Matched memory removes that confound.

## 2. Headline numbers

Default parameterization: `window=21`, `target_vol=0.15`, `max_clip=3.0`,
`cost_bps=1.0`. Per-symbol metrics averaged across the 30-symbol panel.

| Metric | risk_var (`gk_yang_zhang`) | natr (ATR / close) | Ratio (candidate / baseline) | Non-inferior? |
| --- | --- | --- | --- | --- |
| `vol_target_error` | 0.0081 | 0.0355 | **0.229** (77% better) | yes (margin 2%) |
| `turnover` | 0.0063 | 0.0068 | **0.922** (8% better) | yes (margin 5%) |
| `clipping_freq` | 0.000 | 0.000 | 1.000 (tie) | yes |
| `cost_drag` | linear in turnover | linear in turnover | **0.922** | improvement |
| `drawdown` | 0.248 | 0.183 | 1.358 (36% worse) | not in gate |

## 3. Gate 4 verdict

**Gate 4: PASSED.** Per research plan §10.1 thresholds:

- Non-inferiority bounds met on `vol_target_error`, `turnover`, `clipping_freq`.
- Materially improves three metrics: `vol_target_error` (77%), `turnover` (8%), `cost_drag` (8%).
- "NOT selected on Sharpe alone" criterion satisfied — the headline
  improvement is on vol-target-tracking accuracy.

## 4. Tradeoffs surfaced

- **Drawdown is materially worse** for `risk_var` (35.8% degradation
  at default parameters). Drawdown is not part of the Gate 4
  non-inferiority check, but the operator should be aware of the
  vol-tracking-vs-tail-risk tradeoff before any consumer migration.
- The vol-target-error improvement is large (4.4x better tracking).
  Combined with worse drawdown, the net effect is a sizer that
  produces a more *consistently* sized position but with deeper
  occasional drawdowns when realized return shocks are large relative
  to the smoothed vol estimate.
- Parameter sensitivity not yet swept here; a follow-up sweep over
  `window`, `target_vol`, `max_clip`, and `cost_bps` belongs in the
  downstream-migration plan that owns the actual retirement.

## 5. Migration path for `liq-risk` consumers

The current production sizers in `liq-risk` consume ATR via:

- `src/liq/risk/engine.py::VolatilitySizer` (the default sizer)
- `src/liq/risk/engine.py` ATR-based stops + targets (lines ~449, ~493)

A consumer migration would:

1. Replace ATR with `natr_t = ATR_t / close_t` as an intermediate, so
   the sizing math is already unit-less (no consumer code change in
   shape, only a normalization scaler).
2. Add a `risk_var_t` consumer path that reads the canonical
   `gk_yang_zhang` rolling estimate from `liq-features`.
3. Run the same head-to-head on the consumer's universe (which may
   differ from `ai-chip-liquid-30-riskvar`).
4. Cut over after a parallel-run period where both sizers compute
   positions and a delta report is logged.

The downstream-migration plan is the right home for those steps. This
doc is the evidence that the cutover is justified.

## 6. Reproduction

```bash
cd liq-experiments
uv run python scripts/eval_sizing.py --estimator risk_var --output json
```

Two runs against the committed bundle produce identical JSON (the
harness has no time-dependent fields). The `gate_4_passed` field in
the output is the authoritative Gate 4 verdict.
