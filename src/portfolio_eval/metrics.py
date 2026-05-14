"""
Performance and risk metrics for portfolio evaluation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


def annualized_return(daily_returns: pd.Series) -> float:
    """Compound annualized return."""
    total = (1 + daily_returns).prod()
    n_years = len(daily_returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return 0.0
    return float(total ** (1 / n_years) - 1)


def annualized_volatility(daily_returns: pd.Series) -> float:
    """Annualized standard deviation of daily returns."""
    return float(daily_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """
    Annualized Sharpe ratio.

    Parameters
    ----------
    daily_returns : Series of daily portfolio returns.
    risk_free_rate : annualized risk-free rate (default 0).
    """
    vol = annualized_volatility(daily_returns)
    if vol < 1e-10:
        return 0.0
    ann_ret = annualized_return(daily_returns)
    return float((ann_ret - risk_free_rate) / vol)


def information_ratio(active_returns: pd.Series) -> float:
    """Annualized Information Ratio based on active returns vs baseline."""
    vol = annualized_volatility(active_returns)
    if vol < 1e-10:
        return 0.0
    # Mean of active returns * 252 is typically used for IR, but let's use annualized_return to be consistent
    # Note: If active return is very negative, this formulation is fine.
    ann_ret = annualized_return(active_returns)
    return float(ann_ret / vol)


def total_return(daily_returns: pd.Series) -> float:
    """Cumulative total return."""
    return float((1 + daily_returns).prod() - 1)


def max_drawdown(daily_returns: pd.Series) -> float:
    """Maximum drawdown (returned as a positive number)."""
    cum = (1 + daily_returns).cumprod()
    running_max = cum.cummax()
    drawdowns = (cum - running_max) / running_max
    return float(-drawdowns.min())


def average_turnover(weight_changes: pd.DataFrame) -> float:
    """
    Average single-period turnover.

    Parameters
    ----------
    weight_changes : DataFrame
        Absolute weight changes at each rebalance, shape (n_rebalances, n_assets).
    """
    if weight_changes.empty:
        return 0.0
    # Turnover = sum of |Δw_i| / 2 per rebalance (buy-side turnover)
    per_period = weight_changes.abs().sum(axis=1) / 2
    return float(per_period.mean())


def transaction_cost_drag(
    weight_changes: pd.DataFrame,
    cost_bps: float,
) -> float:
    """
    Total transaction cost drag over the evaluation period (as a fraction).

    Parameters
    ----------
    weight_changes : DataFrame of absolute weight changes.
    cost_bps : one-way transaction cost in basis points.
    """
    cost_frac = cost_bps / 10_000
    total_traded = weight_changes.abs().sum().sum()  # sum over all rebalances & assets
    return float(total_traded * cost_frac)
