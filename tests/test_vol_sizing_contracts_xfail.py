"""Sizing contract xfail tests for ``liq.risk.vol_sizing`` (future-scope).

Sizing contracts are first introduced in a later phase. The tests are
kept present so the boundary is enforced once the implementation
lands.
"""

import pytest


@pytest.mark.xfail(strict=True, reason="SizeVolInput contract required in later phase")
def test_size_vol_input_contract_present() -> None:
    from liq.risk.vol_sizing import SizeVolInput  # type: ignore[import-not-found]

    assert SizeVolInput is not None


@pytest.mark.xfail(strict=True, reason="SizeVolInput contract required in later phase")
def test_size_vol_input_fields_present() -> None:
    from liq.risk.vol_sizing import SizeVolInput

    annotations = SizeVolInput.__annotations__
    expected = {
        "symbol",
        "sizing_decision_ts",
        "raw_vol",
        "raw_vol_source",
        "size_vol",
    }
    assert expected.issubset(set(annotations.keys()))
