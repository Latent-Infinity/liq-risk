"""Doc-structure tests for the consolidated vol-sizing doc."""

from __future__ import annotations

from pathlib import Path

_DOC_PATH = Path(__file__).parent.parent / "docs" / "vol-sizing.md"
_README_PATH = Path(__file__).parent.parent / "README.md"
_REQUIRED_HEADERS = (
    "# Volatility sizing input",
    "## Ordered pipeline",
    "## Shock override",
    "## Fallback hierarchy",
    "## ATR migration",
    "## Reason codes",
    "## Parameter registry",
)
_REQUIRED_SYMBOLS = (
    "SizeVolInput",
    "SizingPolicy",
    "MeasurementInput",
    "ForecastInput",
    "compute_size_vol",
    "ShockOverrideConfig",
    "apply_shock_override",
    "FallbackChain",
    "advance_fallback",
    "MigrationLedger",
    "MigrationStage",
    "propose_next_stage",
    "EMERGENCY_FLOOR",
    "SHOCK_OVERRIDE_APPLIED",
    "FALLBACK_HOLD",
    "TRADE_BLOCKED",
)


def _read_doc() -> str:
    assert _DOC_PATH.exists(), f"missing consolidated doc at {_DOC_PATH}"
    return _DOC_PATH.read_text(encoding="utf-8")


def test_consolidated_doc_exists() -> None:
    assert _DOC_PATH.exists()


def test_consolidated_doc_carries_every_required_section_header() -> None:
    body = _read_doc()
    for header in _REQUIRED_HEADERS:
        assert header in body, f"missing section header: {header!r}"


def test_consolidated_doc_references_every_load_bearing_symbol() -> None:
    body = _read_doc()
    for symbol in _REQUIRED_SYMBOLS:
        assert symbol in body, f"missing symbol reference: {symbol!r}"


def test_readme_links_to_consolidated_doc() -> None:
    assert _README_PATH.exists()
    readme = _README_PATH.read_text(encoding="utf-8")
    assert "vol-sizing.md" in readme or "Volatility sizing input" in readme


def test_consolidated_doc_does_not_carry_phase_language() -> None:
    body = _read_doc()
    forbidden = ("F0", "F1H", "F2", "F2H", "F3", "F3H", "F4", "F4H", "F5", "F5H", "FF+1")
    for token in forbidden:
        assert f" {token} " not in body, f"phase language leaked: {token!r}"
        assert f"({token})" not in body, f"phase language leaked: {token!r}"
