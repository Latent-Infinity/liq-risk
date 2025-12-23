from datetime import UTC, datetime
from decimal import Decimal

import pytest
from liq.core import Bar, PortfolioState
from liq.signals import Signal

from liq.risk.config import MarketState, RiskConfig
from liq.risk.sizers import CryptoFractionalSizer


def _ctx(price: Decimal = Decimal("50000")):
    now = datetime.now(UTC)
    bar = Bar(
        timestamp=now,
        symbol="BTC_USDT",
        open=price,
        high=price,
        low=price,
        close=price,
        volume=Decimal("1"),
    )
    market = MarketState(
        current_bars={"BTC_USDT": bar},
        volatility={"BTC_USDT": Decimal("1")},
        liquidity={"BTC_USDT": Decimal("1")},
        timestamp=now,
    )
    portfolio = PortfolioState(cash=Decimal("1000000"), positions={}, timestamp=now)
    sig = Signal(symbol="BTC_USDT", timestamp=now, direction="long", strength=1.0)
    return sig, portfolio, market


class TestCryptoFractionalSizerValidation:
    """Test parameter validation safeguards."""

    def test_rejects_zero_fraction(self) -> None:
        """Fraction of 0 should be rejected."""
        with pytest.raises(ValueError, match="fraction must be in range"):
            CryptoFractionalSizer(fraction=0)

    def test_rejects_negative_fraction(self) -> None:
        """Negative fraction should be rejected."""
        with pytest.raises(ValueError, match="fraction must be in range"):
            CryptoFractionalSizer(fraction=-0.1)

    def test_rejects_fraction_greater_than_one(self) -> None:
        """Fraction > 1 should be rejected."""
        with pytest.raises(ValueError, match="fraction must be in range"):
            CryptoFractionalSizer(fraction=1.5)

    def test_accepts_fraction_of_one(self) -> None:
        """Fraction of 1 (full equity) should be accepted."""
        sizer = CryptoFractionalSizer(fraction=1.0)
        assert sizer.fraction == 1.0

    def test_rejects_zero_min_qty(self) -> None:
        """min_qty of 0 should be rejected."""
        with pytest.raises(ValueError, match="min_qty must be positive"):
            CryptoFractionalSizer(min_qty=Decimal("0"))

    def test_rejects_negative_min_qty(self) -> None:
        """Negative min_qty should be rejected."""
        with pytest.raises(ValueError, match="min_qty must be positive"):
            CryptoFractionalSizer(min_qty=Decimal("-0.001"))

    def test_rejects_zero_step_qty(self) -> None:
        """step_qty of 0 should be rejected."""
        with pytest.raises(ValueError, match="step_qty must be positive"):
            CryptoFractionalSizer(step_qty=Decimal("0"))

    def test_rejects_negative_step_qty(self) -> None:
        """Negative step_qty should be rejected."""
        with pytest.raises(ValueError, match="step_qty must be positive"):
            CryptoFractionalSizer(step_qty=Decimal("-0.001"))

    def test_accepts_none_step_qty(self) -> None:
        """step_qty of None should be accepted (uses default rounding)."""
        sizer = CryptoFractionalSizer(step_qty=None)
        assert sizer._step_qty is None


class TestCryptoFractionalSizerBehavior:
    """Test sizing behavior."""

    def test_crypto_fractional_sizes_fractional_qty(self) -> None:
        """Should produce fractional quantities for crypto."""
        sig, portfolio, market = _ctx(price=Decimal("50000"))
        sizer = CryptoFractionalSizer(
            fraction=0.01, min_qty=Decimal("0.0001"), step_qty=Decimal("0.0001")
        )
        targets = sizer.size_positions([sig], portfolio, market, RiskConfig())
        assert targets
        assert targets[0].target_quantity > 0
        assert targets[0].target_quantity < 1  # fractional

    def test_crypto_fractional_respects_min_qty(self) -> None:
        """Orders below min_qty should be skipped."""
        sig, portfolio, market = _ctx(price=Decimal("50000"))
        sizer = CryptoFractionalSizer(
            fraction=0.000001, min_qty=Decimal("0.001"), step_qty=Decimal("0.0001")
        )
        targets = sizer.size_positions([sig], portfolio, market, RiskConfig())
        assert targets == []

    def test_crypto_fractional_step_size(self) -> None:
        """Quantities should be rounded to step size."""
        sig, portfolio, market = _ctx(price=Decimal("10000"))
        sizer = CryptoFractionalSizer(
            fraction=0.01, min_qty=Decimal("0.0001"), step_qty=Decimal("0.001")
        )
        targets = sizer.size_positions([sig], portfolio, market, RiskConfig())
        assert targets
        qty = targets[0].target_quantity
        assert qty % Decimal("0.001") == 0

    def test_skips_missing_bar_data(self) -> None:
        """Should skip signals without bar data."""
        now = datetime.now(UTC)
        market = MarketState(
            current_bars={},  # No bar data
            volatility={},
            liquidity={},
            timestamp=now,
        )
        portfolio = PortfolioState(cash=Decimal("100000"), positions={}, timestamp=now)
        sig = Signal(symbol="NO_DATA", timestamp=now, direction="long", strength=1.0)

        sizer = CryptoFractionalSizer()
        targets = sizer.size_positions([sig], portfolio, market, RiskConfig())
        assert targets == []

    def test_skips_flat_signals(self) -> None:
        """Should skip flat signals."""
        sig, portfolio, market = _ctx()
        flat_sig = Signal(
            symbol="BTC_USDT",
            timestamp=sig.timestamp,
            direction="flat",
            strength=1.0,
        )
        sizer = CryptoFractionalSizer()
        targets = sizer.size_positions([flat_sig], portfolio, market, RiskConfig())
        assert targets == []

    def test_handles_short_signals(self) -> None:
        """Should produce short positions for short signals."""
        now = datetime.now(UTC)
        bar = Bar(
            timestamp=now,
            symbol="BTC_USDT",
            open=Decimal("50000"),
            high=Decimal("50000"),
            low=Decimal("50000"),
            close=Decimal("50000"),
            volume=Decimal("1"),
        )
        market = MarketState(
            current_bars={"BTC_USDT": bar},
            volatility={"BTC_USDT": Decimal("1")},
            liquidity={"BTC_USDT": Decimal("1")},
            timestamp=now,
        )
        portfolio = PortfolioState(cash=Decimal("1000000"), positions={}, timestamp=now)
        sig = Signal(symbol="BTC_USDT", timestamp=now, direction="short", strength=1.0)

        sizer = CryptoFractionalSizer(fraction=0.01)
        targets = sizer.size_positions([sig], portfolio, market, RiskConfig())
        assert targets
        assert targets[0].direction == "short"
        assert targets[0].target_quantity < 0
