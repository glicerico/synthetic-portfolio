"""
Strategy interface definition.

Every submitted strategy must implement two functions:
  - fit(train_returns, train_features, validation_returns, validation_features, metadata) -> state
  - generate_weights(date, history_returns, history_features, state, metadata) -> weights

See the docstrings below for the exact contract.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd


# ---------------------------------------------------------------------------
# Abstract contract (for documentation; strategies are plain modules)
# ---------------------------------------------------------------------------

def fit(
    train_returns: pd.DataFrame,
    train_features: Dict[str, pd.DataFrame],
    validation_returns: pd.DataFrame,
    validation_features: Dict[str, pd.DataFrame],
    metadata: Dict[str, Any],
) -> Any:
    """
    Fit strategy state using only train and validation data.

    Parameters
    ----------
    train_returns : DataFrame
        Daily returns, shape (T_train, n_assets).
    train_features : dict[str, DataFrame]
        Feature name -> DataFrame of same shape as train_returns.
    validation_returns : DataFrame
        Daily returns, shape (T_val, n_assets).
    validation_features : dict[str, DataFrame]
        Feature name -> DataFrame of same shape as validation_returns.
    metadata : dict
        Contains asset_ids, sector_labels, rebalance_rule, transaction_cost_bps,
        constraints, etc.

    Returns
    -------
    state : any serializable object
        Will be passed to generate_weights on each rebalance date.
    """
    raise NotImplementedError


def generate_weights(
    date: pd.Timestamp,
    history_returns: pd.DataFrame,
    history_features: Dict[str, pd.DataFrame],
    state: Any,
    metadata: Dict[str, Any],
) -> pd.Series:
    """
    Generate portfolio weights for a given rebalance date.

    Called by the evaluator on each monthly rebalance date.
    Inputs contain ONLY data available up to (and including) *date*.

    Parameters
    ----------
    date : pd.Timestamp
        The rebalance date.
    history_returns : DataFrame
        Daily returns from the start of the PUBLIC data up to *date* (inclusive).
    history_features : dict[str, DataFrame]
        Features available up to *date*.
    state : object
        State returned by fit().
    metadata : dict
        Same metadata dict passed to fit().

    Returns
    -------
    weights : pd.Series or dict
        Asset weights (keyed by asset_id).  Must be long-only, fully invested,
        max 20 % per asset, and contain no NaNs.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Loader utility
# ---------------------------------------------------------------------------

def load_strategy(path: str | Path):
    """
    Dynamically import a strategy module from *path* and return it.

    The module must expose ``fit`` and ``generate_weights`` at module level.
    """
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Strategy file not found: {path}")

    spec = importlib.util.spec_from_file_location("submitted_strategy", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["submitted_strategy"] = mod
    spec.loader.exec_module(mod)

    if not hasattr(mod, "fit"):
        raise AttributeError(f"Strategy {path.name} is missing 'fit' function")
    if not hasattr(mod, "generate_weights"):
        raise AttributeError(
            f"Strategy {path.name} is missing 'generate_weights' function"
        )
    return mod
