# AGENT_STATE.md — liq-risk

State tracked per implementation plan that touches this repo.

## Plan: `liq-features-canonical-risk-variance-impl-plan` (v1.2)

| Work package | Capability | Status | Evidence |
| --- | --- | --- | --- |
| 5 | Non-blocking ATR-retirement writeup: `docs/atr-retirement.md` documents the canonical-risk-variance vs ATR head-to-head (Gate 4 PASSED) and the consumer migration path. **No production code change** — the actual retirement of `src/liq/risk/engine.py::VolatilitySizer`'s ATR path is owned by a separate downstream-migration plan | ready-for-review | `artifacts/phase-5/` |
| F | README.md ATR-retirement pointer added; docs/atr-retirement.md final-reviewed unchanged | ready-for-review | `artifacts/phase-F/` |
