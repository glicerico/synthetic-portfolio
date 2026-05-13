"""
Low-volatility (inverse-volatility) strategy example.

Lower-volatility assets receive higher weights.
"""

import numpy as np
import pandas as pd


def fit(train_returns, train_features, validation_returns, validation_features, metadata):
    """No fitting needed."""
    return {}


def generate_weights(date, history_returns, history_features, state, metadata):
    """Inverse-volatility weighting using 63-day rolling vol."""
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
