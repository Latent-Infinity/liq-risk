"""Sizing contract tests for ``liq.risk.vol_sizing``.

These started life as xfail-strict markers pinning the future SizeVolInput
contract. The contract now ships, so the tests pass directly; the file
name is preserved for git history.
"""


def test_size_vol_input_contract_present() -> None:
    from liq.risk.vol_sizing import SizeVolInput

    assert SizeVolInput is not None


def test_size_vol_input_fields_present() -> None:
    from liq.risk.vol_sizing import SizeVolInput

    annotations = SizeVolInput.__annotations__
    expected = {
        "symbol",
        "sizing_decision_ts",
        "raw_vol",
        "raw_vol_source",
        "size_vol",
        "reason_codes",
        "uncertainty_multiplier",
        "previous_size_vol",
        "valid_until",
    }
    assert expected.issubset(set(annotations.keys()))
