"""
Equal-weight strategy example.

Allocates 1/N to each asset every rebalance.
"""

import pandas as pd


def fit(train_returns, train_features, validation_returns, validation_features, metadata):
    """Nothing to fit — just record number of assets."""
    return {"n_assets": len(metadata["asset_ids"])}


def generate_weights(date, history_returns, history_features, state, metadata):
    """Return equal weights across all assets."""
    n = state["n_assets"]
    assets = metadata["asset_ids"]
    return pd.Series(1.0 / n, index=assets)
