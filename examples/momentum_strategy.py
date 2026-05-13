"""
Momentum strategy example.

Overweights assets with strong 63-day trailing returns.
"""

import pandas as pd


def fit(train_returns, train_features, validation_returns, validation_features, metadata):
    """No fitting needed."""
    return {}


def generate_weights(date, history_returns, history_features, state, metadata):
    """Rank-based momentum weights using 63-day trailing return."""
    assets = metadata["asset_ids"]
    n = len(assets)
    max_w = metadata["constraints"]["max_weight_per_asset"]

    if len(history_returns) < 63:
        return pd.Series(1.0 / n, index=assets)

    mom = history_returns[assets].iloc[-63:].sum()
    ranks = mom.rank()
    raw = ranks / ranks.sum()
    weights = raw.clip(upper=max_w)
    weights = weights / weights.sum()
    return weights
