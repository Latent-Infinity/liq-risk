"""Tests for bounded update constraints (Phase 2 Step A).

OutputSpaceBoundConstraint: weight delta, turnover, trade count limits.
RiskSpaceBoundConstraint: sigma delta, cvar delta limits.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest
from liq.core import Bar, OrderRequest, OrderSide, OrderType, PortfolioState, Position

from liq.risk.bounded_update import (
    OutputSpaceBoundConfig,
    OutputSpaceBoundConstraint,
    RiskSpaceBoundConfig,
    RiskSpaceBoundConstraint,
)
from liq.risk.config import MarketState, RiskConfig
from liq.risk.var_model import EWMARiskModel


@pytest.fixture
def now() -> datetime:
    return datetime.now(UTC)


@pytest.fixture
def risk_config() -> RiskConfig:
    return RiskConfig()


def _make_portfolio(
    cash: Decimal,
    positions: dict[str, tuple[Decimal, Decimal, Decimal]],
    ts: datetime,
) -> PortfolioState:
    """Create portfolio. positions: symbol → (quantity, average_price, current_price)."""
    return PortfolioState(
        cash=cash,
        positions={
            sym: Position(
                symbol=sym,
                quantity=qty,
                average_price=avg,
                realized_pnl=Decimal("0"),
                current_price=cur,
                timestamp=ts,
            )
            for sym, (qty, avg, cur) in positions.items()
        },
        timestamp=ts,
    )


def _make_bar(symbol: str, close: Decimal, ts: datetime) -> Bar:
    return Bar(
        timestamp=ts,
        symbol=symbol,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=Decimal("1000000"),
    )


def _make_market(bars: dict[str, Decimal], ts: datetime) -> MarketState:
    return MarketState(
        current_bars={sym: _make_bar(sym, price, ts) for sym, price in bars.items()},
        volatility={sym: Decimal("2.0") for sym in bars},
        liquidity={sym: Decimal("50000000") for sym in bars},
        timestamp=ts,
    )


class TestOutputSpaceBoundConstraint:
    """Tests for output-space bounded update checks."""

    def test_name(self) -> None:
        config = OutputSpaceBoundConfig(
            delta_w_max=0.05, delta_turnover_max=0.20, delta_trades_max=5
        )
        constraint = OutputSpaceBoundConstraint(config)
        assert constraint.name == "OutputSpaceBoundConstraint"

    def test_within_all_bounds(self, now: datetime, risk_config: RiskConfig) -> None:
        """Orders within all thresholds pass through."""
        config = OutputSpaceBoundConfig(
            delta_w_max=0.10, delta_turnover_max=0.50, delta_trades_max=10
        )
        constraint = OutputSpaceBoundConstraint(config)

        # Portfolio: 50k cash + AAPL 100*500=50k = 100k equity
        portfolio = _make_portfolio(
            Decimal("50000"),
            {"AAPL": (Decimal("100"), Decimal("500"), Decimal("500"))},
            now,
        )
        # Buy 5k worth of GOOGL at $100 → weight delta = 5k/100k = 5%
        orders = [
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("50"),
                timestamp=now,
            )
        ]
        market = _make_market({"AAPL": Decimal("500"), "GOOGL": Decimal("100")}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        assert len(result.orders) == 1
        assert len(result.rejected) == 0

    def test_weight_delta_exceeds_bound(self, now: datetime, risk_config: RiskConfig) -> None:
        """Single order exceeding delta_w_max is rejected."""
        config = OutputSpaceBoundConfig(
            delta_w_max=0.02, delta_turnover_max=1.0, delta_trades_max=100
        )
        constraint = OutputSpaceBoundConstraint(config)

        portfolio = _make_portfolio(
            Decimal("50000"),
            {"AAPL": (Decimal("100"), Decimal("500"), Decimal("500"))},
            now,
        )
        # Buy 10k of GOOGL → weight delta = 10k/100k = 10% > 2%
        orders = [
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("100"),
                timestamp=now,
            )
        ]
        market = _make_market({"AAPL": Decimal("500"), "GOOGL": Decimal("100")}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        assert len(result.orders) == 0
        assert len(result.rejected) == 1
        assert "weight delta" in result.rejected[0].reason.lower()

    def test_turnover_exceeds_bound(self, now: datetime, risk_config: RiskConfig) -> None:
        """Total turnover exceeding delta_turnover_max rejects all orders."""
        config = OutputSpaceBoundConfig(
            delta_w_max=1.0, delta_turnover_max=0.05, delta_trades_max=100
        )
        constraint = OutputSpaceBoundConstraint(config)

        # 100k equity, buy 20k GOOGL and sell 15k AAPL → turnover = (20+15)/(2*100) = 17.5%
        portfolio = _make_portfolio(
            Decimal("50000"),
            {"AAPL": (Decimal("100"), Decimal("500"), Decimal("500"))},
            now,
        )
        orders = [
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("200"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("30"),
                timestamp=now,
            ),
        ]
        market = _make_market({"AAPL": Decimal("500"), "GOOGL": Decimal("100")}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        assert len(result.orders) == 0
        assert any("turnover" in r.reason.lower() for r in result.rejected)

    def test_trade_count_exceeds_bound(self, now: datetime, risk_config: RiskConfig) -> None:
        """Excess trades beyond delta_trades_max rejected."""
        config = OutputSpaceBoundConfig(delta_w_max=1.0, delta_turnover_max=1.0, delta_trades_max=2)
        constraint = OutputSpaceBoundConstraint(config)

        portfolio = _make_portfolio(Decimal("100000"), {}, now)
        orders = [
            OrderRequest(
                symbol=f"SYM{i}",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                timestamp=now,
            )
            for i in range(5)
        ]
        market = _make_market({f"SYM{i}": Decimal("10") for i in range(5)}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        assert len(result.orders) == 2
        assert len(result.rejected) == 3
        assert any("trade count" in r.reason.lower() for r in result.rejected)

    def test_empty_orders_pass(self, now: datetime, risk_config: RiskConfig) -> None:
        """Empty orders list passes trivially."""
        config = OutputSpaceBoundConfig(
            delta_w_max=0.05, delta_turnover_max=0.20, delta_trades_max=5
        )
        constraint = OutputSpaceBoundConstraint(config)

        portfolio = _make_portfolio(Decimal("100000"), {}, now)
        market = _make_market({}, now)

        result = constraint.apply([], portfolio, market, risk_config)
        assert result.orders == []
        assert result.rejected == []

    def test_single_asset_portfolio(self, now: datetime, risk_config: RiskConfig) -> None:
        """Single-asset portfolio with sell reduces weight correctly."""
        config = OutputSpaceBoundConfig(
            delta_w_max=0.10, delta_turnover_max=0.50, delta_trades_max=10
        )
        constraint = OutputSpaceBoundConstraint(config)

        # All equity in AAPL: weight = 1.0
        portfolio = _make_portfolio(
            Decimal("0"),
            {"AAPL": (Decimal("100"), Decimal("1000"), Decimal("1000"))},
            now,
        )
        # Sell 5 shares → reduces weight by 5k/100k = 5%
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("5"),
                timestamp=now,
            )
        ]
        market = _make_market({"AAPL": Decimal("1000")}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        assert len(result.orders) == 1
        assert len(result.rejected) == 0

    def test_classify_risk_buy_increases(self, now: datetime) -> None:
        """Buying increases risk when no existing position."""
        config = OutputSpaceBoundConfig(
            delta_w_max=0.05, delta_turnover_max=0.20, delta_trades_max=5
        )
        constraint = OutputSpaceBoundConstraint(config)

        portfolio = _make_portfolio(Decimal("100000"), {}, now)
        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
            timestamp=now,
        )
        assert constraint.classify_risk(order, portfolio) is True

    def test_classify_risk_sell_reduces(self, now: datetime) -> None:
        """Selling existing long reduces risk."""
        config = OutputSpaceBoundConfig(
            delta_w_max=0.05, delta_turnover_max=0.20, delta_trades_max=5
        )
        constraint = OutputSpaceBoundConstraint(config)

        portfolio = _make_portfolio(
            Decimal("50000"),
            {"AAPL": (Decimal("100"), Decimal("500"), Decimal("500"))},
            now,
        )
        order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
            timestamp=now,
        )
        assert constraint.classify_risk(order, portfolio) is False

    def test_zero_equity_portfolio(self, now: datetime, risk_config: RiskConfig) -> None:
        """Zero equity portfolio handles gracefully."""
        config = OutputSpaceBoundConfig(
            delta_w_max=0.05, delta_turnover_max=0.20, delta_trades_max=5
        )
        constraint = OutputSpaceBoundConstraint(config)

        portfolio = _make_portfolio(Decimal("0"), {}, now)
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                timestamp=now,
            )
        ]
        market = _make_market({"AAPL": Decimal("100")}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        # All orders rejected when equity is zero
        assert len(result.orders) == 0
        assert len(result.rejected) == 1


class TestRiskSpaceBoundConstraint:
    """Tests for risk-space bounded update checks."""

    def _make_returns(self, n_assets: int, n_obs: int = 100) -> np.ndarray:
        """Generate synthetic return history."""
        rng = np.random.default_rng(42)
        return rng.normal(0.0, 0.02, size=(n_obs, n_assets))

    def test_name(self) -> None:
        config = RiskSpaceBoundConfig(delta_sigma_max=0.01, delta_cvar_max=0.02)
        model = EWMARiskModel()
        returns = self._make_returns(2)
        weights = np.array([0.5, 0.5])
        constraint = RiskSpaceBoundConstraint(config, model, returns, weights)
        assert constraint.name == "RiskSpaceBoundConstraint"

    def test_within_risk_bounds(self, now: datetime, risk_config: RiskConfig) -> None:
        """Small weight changes stay within risk bounds."""
        returns = self._make_returns(2)
        current_weights = np.array([0.5, 0.5])

        config = RiskSpaceBoundConfig(delta_sigma_max=0.50, delta_cvar_max=0.50)
        model = EWMARiskModel()
        constraint = RiskSpaceBoundConstraint(config, model, returns, current_weights)

        # Portfolio: 50k in each of 2 assets, 100k total
        portfolio = _make_portfolio(
            Decimal("0"),
            {
                "AAPL": (Decimal("100"), Decimal("500"), Decimal("500")),
                "GOOGL": (Decimal("500"), Decimal("100"), Decimal("100")),
            },
            now,
        )
        # Small sell → tiny weight change
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                timestamp=now,
            )
        ]
        market = _make_market({"AAPL": Decimal("500"), "GOOGL": Decimal("100")}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        assert len(result.orders) == 1
        assert len(result.rejected) == 0

    def test_sigma_delta_exceeds_bound(self, now: datetime, risk_config: RiskConfig) -> None:
        """Large weight shift that moves sigma past threshold rejects all orders."""
        # Use returns with different volatilities to make sigma sensitive to weights
        rng = np.random.default_rng(42)
        returns = np.column_stack(
            [
                rng.normal(0.0, 0.01, size=100),  # Low vol asset
                rng.normal(0.0, 0.10, size=100),  # High vol asset
            ]
        )
        current_weights = np.array([0.9, 0.1])

        # Very tight sigma bound
        config = RiskSpaceBoundConfig(delta_sigma_max=0.001, delta_cvar_max=1.0)
        model = EWMARiskModel()
        constraint = RiskSpaceBoundConstraint(config, model, returns, current_weights)

        # Portfolio heavily in AAPL (low vol), want to shift big to GOOGL (high vol)
        portfolio = _make_portfolio(
            Decimal("10000"),
            {
                "AAPL": (Decimal("180"), Decimal("500"), Decimal("500")),
                "GOOGL": (Decimal("100"), Decimal("100"), Decimal("100")),
            },
            now,
        )
        # Big shift: sell 80% AAPL, buy matching GOOGL
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("160"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("800"),
                timestamp=now,
            ),
        ]
        market = _make_market({"AAPL": Decimal("500"), "GOOGL": Decimal("100")}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        # All rejected (portfolio-level)
        assert len(result.orders) == 0
        assert len(result.rejected) == 2
        assert any("sigma" in r.reason.lower() for r in result.rejected)

    def test_cvar_delta_exceeds_bound(self, now: datetime, risk_config: RiskConfig) -> None:
        """CVaR delta breach rejects all orders."""
        rng = np.random.default_rng(42)
        returns = np.column_stack(
            [
                rng.normal(0.0, 0.01, size=100),
                rng.normal(0.0, 0.10, size=100),
            ]
        )
        current_weights = np.array([0.9, 0.1])

        # Tight cvar bound, loose sigma
        config = RiskSpaceBoundConfig(delta_sigma_max=1.0, delta_cvar_max=0.001)
        model = EWMARiskModel()
        constraint = RiskSpaceBoundConstraint(config, model, returns, current_weights)

        portfolio = _make_portfolio(
            Decimal("10000"),
            {
                "AAPL": (Decimal("180"), Decimal("500"), Decimal("500")),
                "GOOGL": (Decimal("100"), Decimal("100"), Decimal("100")),
            },
            now,
        )
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("160"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("800"),
                timestamp=now,
            ),
        ]
        market = _make_market({"AAPL": Decimal("500"), "GOOGL": Decimal("100")}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        assert len(result.orders) == 0
        assert any("cvar" in r.reason.lower() for r in result.rejected)

    def test_empty_orders_pass(self, now: datetime, risk_config: RiskConfig) -> None:
        """Empty orders pass trivially."""
        returns = self._make_returns(2)
        weights = np.array([0.5, 0.5])
        config = RiskSpaceBoundConfig(delta_sigma_max=0.01, delta_cvar_max=0.02)
        model = EWMARiskModel()
        constraint = RiskSpaceBoundConstraint(config, model, returns, weights)

        portfolio = _make_portfolio(Decimal("100000"), {}, now)
        market = _make_market({}, now)

        result = constraint.apply([], portfolio, market, risk_config)
        assert result.orders == []
        assert result.rejected == []

    def test_classify_risk(self, now: datetime) -> None:
        """Risk classification delegates to position direction logic."""
        returns = self._make_returns(1)
        weights = np.array([1.0])
        config = RiskSpaceBoundConfig(delta_sigma_max=0.01, delta_cvar_max=0.02)
        model = EWMARiskModel()
        constraint = RiskSpaceBoundConstraint(config, model, returns, weights)

        portfolio = _make_portfolio(
            Decimal("50000"),
            {"AAPL": (Decimal("100"), Decimal("500"), Decimal("500"))},
            now,
        )
        buy_order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
            timestamp=now,
        )
        sell_order = OrderRequest(
            symbol="AAPL",
            side=OrderSide.SELL,
            order_type=OrderType.MARKET,
            quantity=Decimal("10"),
            timestamp=now,
        )
        # Buying into existing long → risk increasing
        assert constraint.classify_risk(buy_order, portfolio) is True
        # Selling existing long → risk reducing
        assert constraint.classify_risk(sell_order, portfolio) is False

    def test_symbol_mismatch_without_mapping_rejects_all(
        self, now: datetime, risk_config: RiskConfig
    ) -> None:
        """When symbol count doesn't match returns columns and no mapping, reject all."""
        # 3-column returns matrix but only 2 portfolio symbols
        returns = self._make_returns(3)
        current_weights = np.array([0.4, 0.3, 0.3])

        config = RiskSpaceBoundConfig(delta_sigma_max=1.0, delta_cvar_max=1.0)
        model = EWMARiskModel()
        constraint = RiskSpaceBoundConstraint(config, model, returns, current_weights)

        portfolio = _make_portfolio(
            Decimal("50000"),
            {
                "AAPL": (Decimal("100"), Decimal("500"), Decimal("500")),
                "GOOGL": (Decimal("500"), Decimal("100"), Decimal("100")),
            },
            now,
        )
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                timestamp=now,
            )
        ]
        market = _make_market({"AAPL": Decimal("500"), "GOOGL": Decimal("100")}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        assert len(result.orders) == 0
        assert len(result.rejected) == 1

    def test_symbol_mapping_resolves_correctly(
        self, now: datetime, risk_config: RiskConfig
    ) -> None:
        """With explicit symbols, orders map to the correct returns-matrix columns."""
        rng = np.random.default_rng(42)
        returns = np.column_stack(
            [
                rng.normal(0.0, 0.01, size=100),  # col 0 = AAPL (low vol)
                rng.normal(0.0, 0.01, size=100),  # col 1 = GOOGL (low vol)
                rng.normal(0.0, 0.01, size=100),  # col 2 = MSFT (low vol)
            ]
        )
        current_weights = np.array([0.4, 0.3, 0.3])
        symbols = ["AAPL", "GOOGL", "MSFT"]

        config = RiskSpaceBoundConfig(delta_sigma_max=1.0, delta_cvar_max=1.0)
        model = EWMARiskModel()
        constraint = RiskSpaceBoundConstraint(
            config, model, returns, current_weights, symbols=symbols
        )

        portfolio = _make_portfolio(
            Decimal("0"),
            {
                "AAPL": (Decimal("80"), Decimal("500"), Decimal("500")),
                "GOOGL": (Decimal("300"), Decimal("100"), Decimal("100")),
                "MSFT": (Decimal("100"), Decimal("300"), Decimal("300")),
            },
            now,
        )
        orders = [
            OrderRequest(
                symbol="MSFT",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                timestamp=now,
            )
        ]
        market = _make_market(
            {"AAPL": Decimal("500"), "GOOGL": Decimal("100"), "MSFT": Decimal("300")},
            now,
        )

        result = constraint.apply(orders, portfolio, market, risk_config)
        # Small order with loose bounds → should pass
        assert len(result.orders) == 1
        assert len(result.rejected) == 0

    def test_symbol_mapping_unknown_symbol_rejects(
        self, now: datetime, risk_config: RiskConfig
    ) -> None:
        """Order for a symbol not in the mapping rejects all orders."""
        returns = self._make_returns(2)
        current_weights = np.array([0.5, 0.5])
        symbols = ["AAPL", "GOOGL"]

        config = RiskSpaceBoundConfig(delta_sigma_max=1.0, delta_cvar_max=1.0)
        model = EWMARiskModel()
        constraint = RiskSpaceBoundConstraint(
            config, model, returns, current_weights, symbols=symbols
        )

        portfolio = _make_portfolio(
            Decimal("0"),
            {
                "AAPL": (Decimal("100"), Decimal("500"), Decimal("500")),
                "GOOGL": (Decimal("500"), Decimal("100"), Decimal("100")),
            },
            now,
        )
        # Order for MSFT which is not in symbols mapping
        orders = [
            OrderRequest(
                symbol="MSFT",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("10"),
                timestamp=now,
            )
        ]
        market = _make_market(
            {"AAPL": Decimal("500"), "GOOGL": Decimal("100"), "MSFT": Decimal("300")},
            now,
        )

        result = constraint.apply(orders, portfolio, market, risk_config)
        assert len(result.orders) == 0
        assert len(result.rejected) == 1

    def test_order_for_unmapped_position_rejects(
        self, now: datetime, risk_config: RiskConfig
    ) -> None:
        """Position in a symbol not in the mapping rejects all orders."""
        returns = self._make_returns(2)
        current_weights = np.array([0.5, 0.5])
        symbols = ["AAPL", "GOOGL"]

        config = RiskSpaceBoundConfig(delta_sigma_max=1.0, delta_cvar_max=1.0)
        model = EWMARiskModel()
        constraint = RiskSpaceBoundConstraint(
            config, model, returns, current_weights, symbols=symbols
        )

        # Portfolio has MSFT position which is not in symbols mapping
        portfolio = _make_portfolio(
            Decimal("0"),
            {
                "AAPL": (Decimal("100"), Decimal("500"), Decimal("500")),
                "MSFT": (Decimal("50"), Decimal("300"), Decimal("300")),
            },
            now,
        )
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("1"),
                timestamp=now,
            )
        ]
        market = _make_market({"AAPL": Decimal("500"), "MSFT": Decimal("300")}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        assert len(result.orders) == 0
        assert len(result.rejected) == 1

    def test_mdd_delta_exceeds_bound(self, now: datetime, risk_config: RiskConfig) -> None:
        """MDD pred delta exceeding bound rejects all orders."""
        rng = np.random.default_rng(42)
        returns = np.column_stack(
            [
                rng.normal(0.0, 0.01, size=100),  # Low vol
                rng.normal(0.0, 0.10, size=100),  # High vol
            ]
        )
        current_weights = np.array([0.9, 0.1])

        # Tight mdd bound, loose sigma and cvar
        config = RiskSpaceBoundConfig(
            delta_sigma_max=1.0, delta_cvar_max=1.0, delta_mdd_pred_max=0.001
        )
        model = EWMARiskModel()
        constraint = RiskSpaceBoundConstraint(config, model, returns, current_weights)

        portfolio = _make_portfolio(
            Decimal("10000"),
            {
                "AAPL": (Decimal("180"), Decimal("500"), Decimal("500")),
                "GOOGL": (Decimal("100"), Decimal("100"), Decimal("100")),
            },
            now,
        )
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("160"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("800"),
                timestamp=now,
            ),
        ]
        market = _make_market({"AAPL": Decimal("500"), "GOOGL": Decimal("100")}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        assert len(result.orders) == 0
        assert len(result.rejected) == 2
        assert any("mdd" in r.reason.lower() for r in result.rejected)

    def test_mdd_not_checked_when_none(self, now: datetime, risk_config: RiskConfig) -> None:
        """When delta_mdd_pred_max is None (default), MDD is not checked."""
        rng = np.random.default_rng(42)
        returns = np.column_stack(
            [
                rng.normal(0.0, 0.01, size=100),  # Low vol
                rng.normal(0.0, 0.10, size=100),  # High vol
            ]
        )
        current_weights = np.array([0.9, 0.1])

        # No mdd bound (default None), loose sigma and cvar
        config = RiskSpaceBoundConfig(delta_sigma_max=1.0, delta_cvar_max=1.0)
        model = EWMARiskModel()
        constraint = RiskSpaceBoundConstraint(config, model, returns, current_weights)

        portfolio = _make_portfolio(
            Decimal("10000"),
            {
                "AAPL": (Decimal("180"), Decimal("500"), Decimal("500")),
                "GOOGL": (Decimal("100"), Decimal("100"), Decimal("100")),
            },
            now,
        )
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("160"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("800"),
                timestamp=now,
            ),
        ]
        market = _make_market({"AAPL": Decimal("500"), "GOOGL": Decimal("100")}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        # With loose sigma/cvar bounds and no MDD bound, orders should pass
        assert len(result.orders) == 2
        assert len(result.rejected) == 0

    def test_mdd_and_sigma_both_violated(self, now: datetime, risk_config: RiskConfig) -> None:
        """When both sigma and MDD bounds are tight, both appear in violation reason."""
        rng = np.random.default_rng(42)
        returns = np.column_stack(
            [
                rng.normal(0.0, 0.01, size=100),  # Low vol
                rng.normal(0.0, 0.10, size=100),  # High vol
            ]
        )
        current_weights = np.array([0.9, 0.1])

        # Tight both sigma and mdd, loose cvar
        config = RiskSpaceBoundConfig(
            delta_sigma_max=0.001, delta_cvar_max=1.0, delta_mdd_pred_max=0.001
        )
        model = EWMARiskModel()
        constraint = RiskSpaceBoundConstraint(config, model, returns, current_weights)

        portfolio = _make_portfolio(
            Decimal("10000"),
            {
                "AAPL": (Decimal("180"), Decimal("500"), Decimal("500")),
                "GOOGL": (Decimal("100"), Decimal("100"), Decimal("100")),
            },
            now,
        )
        orders = [
            OrderRequest(
                symbol="AAPL",
                side=OrderSide.SELL,
                order_type=OrderType.MARKET,
                quantity=Decimal("160"),
                timestamp=now,
            ),
            OrderRequest(
                symbol="GOOGL",
                side=OrderSide.BUY,
                order_type=OrderType.MARKET,
                quantity=Decimal("800"),
                timestamp=now,
            ),
        ]
        market = _make_market({"AAPL": Decimal("500"), "GOOGL": Decimal("100")}, now)

        result = constraint.apply(orders, portfolio, market, risk_config)
        assert len(result.orders) == 0
        assert len(result.rejected) == 2
        # Both sigma and mdd should be mentioned
        reason = result.rejected[0].reason.lower()
        assert "sigma" in reason
        assert "mdd" in reason
