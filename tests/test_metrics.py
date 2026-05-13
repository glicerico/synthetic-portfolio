"""Tests for portfolio_eval.metrics."""

import numpy as np
import pandas as pd
import pytest

from portfolio_eval.metrics import (
    annualized_return,
    annualized_volatility,
    average_turnover,
    max_drawdown,
    sharpe_ratio,
    total_return,
    transaction_cost_drag,
    TRADING_DAYS_PER_YEAR,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def flat_returns():
    """Daily returns of exactly 0 → total return = 0, vol = 0."""
    return pd.Series(np.zeros(252))


@pytest.fixture
def constant_positive_returns():
    """Positive-mean daily returns with small noise → high positive Sharpe."""
    rng = np.random.RandomState(42)
    return pd.Series(0.0004 + rng.normal(0, 0.0001, 252))


@pytest.fixture
def drawdown_returns():
    """
    Returns that create a known drawdown:
    +10% for 100 days, then -20% drop, then recovery.
    """
    up = np.full(100, 0.001)       # slow climb
    down = np.full(10, -0.02)      # sharp drop
    recover = np.full(142, 0.001)  # slow recovery
    return pd.Series(np.concatenate([up, down, recover]))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_zero_returns(self, flat_returns):
        assert sharpe_ratio(flat_returns) == 0.0

    def test_positive_returns_positive_sharpe(self, constant_positive_returns):
        sr = sharpe_ratio(constant_positive_returns)
        assert sr > 0

    def test_constant_returns_infinite_like_sharpe(self):
        """Constant non-zero returns → zero vol → Sharpe should handle it."""
        # With truly constant returns std=0, we return 0 to avoid inf
        ret = pd.Series(np.full(100, 0.001))
        assert sharpe_ratio(ret) == 0.0  # vol=0 guard

    def test_negative_mean_negative_sharpe(self):
        ret = pd.Series(np.random.RandomState(0).normal(-0.001, 0.01, 500))
        assert sharpe_ratio(ret) < 0

    def test_sharpe_with_risk_free(self, constant_positive_returns):
        sr_no_rf = sharpe_ratio(constant_positive_returns, 0.0)
        sr_rf = sharpe_ratio(constant_positive_returns, 0.05)
        assert sr_rf < sr_no_rf


class TestTotalReturn:
    def test_zero(self, flat_returns):
        assert total_return(flat_returns) == pytest.approx(0.0)

    def test_positive(self, constant_positive_returns):
        tr = total_return(constant_positive_returns)
        assert tr > 0

    def test_compounding(self):
        ret = pd.Series([0.10, -0.10])
        # (1.1)(0.9) - 1 = -0.01
        assert total_return(ret) == pytest.approx(-0.01)


class TestAnnualizedReturn:
    def test_one_year_constant(self, constant_positive_returns):
        ar = annualized_return(constant_positive_returns)
        tr = total_return(constant_positive_returns)
        # For 1 year of data, annualized == total
        assert ar == pytest.approx(tr, rel=1e-6)


class TestAnnualizedVolatility:
    def test_zero_vol(self, flat_returns):
        assert annualized_volatility(flat_returns) == 0.0

    def test_positive_vol(self):
        ret = pd.Series(np.random.RandomState(0).normal(0, 0.01, 252))
        vol = annualized_volatility(ret)
        assert vol > 0
        # Should be roughly 0.01 * sqrt(252) ≈ 0.159
        assert 0.10 < vol < 0.25


class TestMaxDrawdown:
    def test_no_drawdown(self, constant_positive_returns):
        """Monotonically increasing → drawdown is 0."""
        dd = max_drawdown(constant_positive_returns)
        assert dd == pytest.approx(0.0)

    def test_known_drawdown(self, drawdown_returns):
        dd = max_drawdown(drawdown_returns)
        assert dd > 0
        # Expect roughly 18-20% drawdown from the down period
        assert dd > 0.10

    def test_single_day_drop(self):
        ret = pd.Series([0.0, 0.0, -0.5, 0.0])
        dd = max_drawdown(ret)
        assert dd == pytest.approx(0.5)


class TestTurnover:
    def test_no_changes(self):
        wc = pd.DataFrame(np.zeros((5, 3)))
        assert average_turnover(wc) == 0.0

    def test_full_rotation(self):
        """Going from [1,0,0] to [0,1,0]: turnover = 1.0."""
        wc = pd.DataFrame([[-1, 1, 0]])
        assert average_turnover(wc) == pytest.approx(1.0)

    def test_empty(self):
        assert average_turnover(pd.DataFrame()) == 0.0


class TestTransactionCostDrag:
    def test_zero_trading(self):
        wc = pd.DataFrame(np.zeros((5, 3)))
        assert transaction_cost_drag(wc, 30) == 0.0

    def test_known_cost(self):
        # Trade 1.0 unit total, 30bps one-way → 0.003
        wc = pd.DataFrame([[0.5, -0.5, 0.0]])
        drag = transaction_cost_drag(wc, 30)
        assert drag == pytest.approx(1.0 * 30 / 10_000)
