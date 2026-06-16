# Phase 5 — liq-risk decisions

| Field | Value |
| --- | --- |
| Plan | `liq-features-canonical-risk-variance-impl-plan.md` v1.2 |
| Work package | 5 — Non-blocking ATR-retirement writeup |
| Frozen on | 2026-06-15 |
| Repo | `liq-risk` |
| Production change | **None** — separate downstream-migration plan owns the actual retirement |

## Capability delivered

- `docs/atr-retirement.md` — head-to-head writeup with Gate 4 PASSED
  numbers and the consumer migration path for `src/liq/risk/engine.py::VolatilitySizer`.

## Why this is non-blocking

The impl plan explicitly carves out the ATR retirement: "Read-only
consumption — no production code change. The plan's deliverable here
is a doc + a benchmark." The ATR-based stops/targets in `engine.py`
(lines ~449, ~493) and the ATR sizer path remain unchanged. Consumer
migration plans should reference `docs/atr-retirement.md` for the
non-inferiority evidence.
