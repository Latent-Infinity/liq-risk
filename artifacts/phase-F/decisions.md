# Phase F — liq-risk documentation polish

| Field | Value |
| --- | --- |
| Plan | `liq-features-canonical-risk-variance-impl-plan.md` v1.2 |
| Work package | F — Documentation Consolidation & Polish |
| Recorded on | 2026-06-15 |
| Repo | `liq-risk` |

## Capability delivered

- `README.md`: ATR-retirement pointer added (callout linking to
  `docs/atr-retirement.md` + the non-blocking nature of the writeup).
- `docs/atr-retirement.md` final-reviewed unchanged from Phase 5.
  Numbers and migration path remain authoritative.

## What did not change

- No production code change in this phase.
- The ATR-based sizer path in `src/liq/risk/engine.py::VolatilitySizer`
  and the ATR-based stops/targets remain in place — the retirement is
  owned by a separate downstream-migration plan.

## Acceptance

- Documentation consolidation list from impl plan §Phase F satisfied
  for liq-risk.
