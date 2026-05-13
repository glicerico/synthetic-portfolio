"""
Built-in baseline strategies.

These are importable from the library and also serve as reference
implementations for the strategy interface.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd


# ===================================================================
# Equal Weight
# ===================================================================

class EqualWeight:
    """1/N allocation, rebalanced monthly."""

    @staticmethod
    def fit(train_returns, train_features, val_returns, val_features, metadata):
        return {"n_assets": len(metadata["asset_ids"])}

    @staticmethod
    def generate_weights(date, history_returns, history_features, state, metadata):
        n = state["n_assets"]
        assets = metadata["asset_ids"]
        return pd.Series(1.0 / n, index=assets)


# ===================================================================
# Sector-Balanced Equal Weight
# ===================================================================

class SectorBalancedEqualWeight:
    """Equal weight within each sector, equal allocation across sectors."""

    @staticmethod
    def fit(train_returns, train_features, val_returns, val_features, metadata):
        sector_labels = metadata["sector_labels"]
        sectors = sorted(set(sector_labels.values()))
        n_sectors = len(sectors)
        sector_assets = {}
        for s in sectors:
            sector_assets[s] = [a for a, sec in sector_labels.items() if sec == s]
        return {"sector_assets": sector_assets, "n_sectors": n_sectors}

    @staticmethod
    def generate_weights(date, history_returns, history_features, state, metadata):
        assets = metadata["asset_ids"]
        weights = pd.Series(0.0, index=assets)
        sa = state["sector_assets"]
        n_sectors = state["n_sectors"]
        for s, s_assets in sa.items():
            w = 1.0 / (n_sectors * len(s_assets))
            for a in s_assets:
                weights[a] = w
        return weights


# ===================================================================
# Momentum
# ===================================================================

class Momentum:
    """
    Long-only momentum: overweight assets with strong past 63-day returns.
    Weights are proportional to rank (higher momentum = higher weight),
    capped at 20%.
    """

    @staticmethod
    def fit(train_returns, train_features, val_returns, val_features, metadata):
        return {}

    @staticmethod
    def generate_weights(date, history_returns, history_features, state, metadata):
        assets = metadata["asset_ids"]
        n = len(assets)
        max_w = metadata["constraints"]["max_weight_per_asset"]

        if len(history_returns) < 63:
            return pd.Series(1.0 / n, index=assets)

        mom = history_returns[assets].iloc[-63:].sum()
        # Rank-based weights (shift so all positive)
        ranks = mom.rank()
        raw = ranks / ranks.sum()
        # Cap and renormalise
        weights = raw.clip(upper=max_w)
        weights = weights / weights.sum()
        return weights


# ===================================================================
# Low Volatility / Inverse Volatility
# ===================================================================

class LowVolatility:
    """
    Inverse-volatility weighting: lower-vol assets get higher weights.
    Uses 63-day rolling volatility.
    """

    @staticmethod
    def fit(train_returns, train_features, val_returns, val_features, metadata):
        return {}

    @staticmethod
    def generate_weights(date, history_returns, history_features, state, metadata):
        assets = metadata["asset_ids"]
        n = len(assets)
        max_w = metadata["constraints"]["max_weight_per_asset"]

        if len(history_returns) < 63:
            return pd.Series(1.0 / n, index=assets)

        vol = history_returns[assets].iloc[-63:].std()
        vol = vol.replace(0, np.nan).fillna(vol.median())
        inv_vol = 1.0 / vol
        raw = inv_vol / inv_vol.sum()
        weights = raw.clip(upper=max_w)
        weights = weights / weights.sum()
        return weights


# ===================================================================
# Mean Reversion
# ===================================================================

class MeanReversion:
    """
    Contrarian: overweight assets with negative 5-day returns.
    """

    @staticmethod
    def fit(train_returns, train_features, val_returns, val_features, metadata):
        return {}

    @staticmethod
    def generate_weights(date, history_returns, history_features, state, metadata):
        assets = metadata["asset_ids"]
        n = len(assets)
        max_w = metadata["constraints"]["max_weight_per_asset"]

        if len(history_returns) < 5:
            return pd.Series(1.0 / n, index=assets)

        rev = -history_returns[assets].iloc[-5:].sum()
        # Shift so all positive, then normalise
        shifted = rev - rev.min() + 1e-8
        raw = shifted / shifted.sum()
        weights = raw.clip(upper=max_w)
        weights = weights / weights.sum()
        return weights


# ===================================================================
# Validation-Selected Heuristic Blend
# ===================================================================

class ValidationSelectedBlend:
    """
    Evaluate multiple simple strategies on validation data and pick the
    blend that maximises validation Sharpe.

    Blends: equal-weight, momentum, low-vol, mean-reversion.
    """

    STRATEGIES = {
        "equal_weight": EqualWeight,
        "momentum": Momentum,
        "low_vol": LowVolatility,
        "mean_reversion": MeanReversion,
    }

    @staticmethod
    def fit(train_returns, train_features, val_returns, val_features, metadata):
        from portfolio_eval.metrics import sharpe_ratio

        assets = metadata["asset_ids"]
        n = len(assets)
        all_ret = pd.concat([train_returns, val_returns])
        val_dates = val_returns.index

        # Simulate each baseline on validation period
        scores = {}
        for name, cls in ValidationSelectedBlend.STRATEGIES.items():
            state = cls.fit(train_returns, train_features,
                            val_returns, val_features, metadata)
            daily = []
            for date in val_dates:
                hist = all_ret.loc[:date]
                w = cls.generate_weights(date, hist, {}, state, metadata)
                w = w.reindex(assets, fill_value=0.0)
                day_ret = val_returns.loc[date]
                daily.append((w * day_ret).sum())
            scores[name] = sharpe_ratio(pd.Series(daily, index=val_dates))

        # Pick top 2 and blend 50/50
        sorted_strats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top2 = [sorted_strats[0][0], sorted_strats[1][0]]
        return {"blend": top2, "scores": scores}

    @staticmethod
    def generate_weights(date, history_returns, history_features, state, metadata):
        assets = metadata["asset_ids"]
        blend = state["blend"]
        combined = pd.Series(0.0, index=assets)
        for name in blend:
            cls = ValidationSelectedBlend.STRATEGIES[name]
            s = cls.fit.__func__(None, pd.DataFrame(), {}, pd.DataFrame(), {}, metadata) if False else {}
            w = cls.generate_weights(date, history_returns, history_features, {}, metadata)
            w = w.reindex(assets, fill_value=0.0)
            combined += w / len(blend)
        # Cap and renormalise
        max_w = metadata["constraints"]["max_weight_per_asset"]
        combined = combined.clip(upper=max_w)
        combined = combined / combined.sum()
        return combined
