"""
Mean-reversion strategy example.

Overweights assets with negative 5-day trailing returns (contrarian).
"""

import pandas as pd


def fit(train_returns, train_features, validation_returns, validation_features, metadata):
    """No fitting needed."""
    return {}


def generate_weights(date, history_returns, history_features, state, metadata):
    """Contrarian weights: higher weight for recent losers."""
    assets = metadata["asset_ids"]
    n = len(assets)
    max_w = metadata["constraints"]["max_weight_per_asset"]

    if len(history_returns) < 5:
        return pd.Series(1.0 / n, index=assets)

    # Negative of 5-day return = reversion signal
    rev = -history_returns[assets].iloc[-5:].sum()
    shifted = rev - rev.min() + 1e-8
    raw = shifted / shifted.sum()
    weights = raw.clip(upper=max_w)
    weights = weights / weights.sum()
    return weights
