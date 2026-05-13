"""
Validation-selected heuristic blend strategy example.

Evaluates multiple simple strategies on validation data, picks the best two,
and blends them 50/50.
"""

import pandas as pd

# Import the individual strategy modules
import importlib.util
from pathlib import Path


def _load_peer(name):
    """Load a sibling strategy module."""
    here = Path(__file__).parent
    spec = importlib.util.spec_from_file_location(name, here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit(train_returns, train_features, validation_returns, validation_features, metadata):
    """
    Simulate each sub-strategy on validation data, pick the top two by
    Sharpe ratio, blend them.
    """
    from portfolio_eval.metrics import sharpe_ratio

    strategies = {
        "equal_weight_strategy": _load_peer("equal_weight_strategy"),
        "momentum_strategy": _load_peer("momentum_strategy"),
        "low_vol_strategy": _load_peer("low_vol_strategy"),
        "mean_reversion_strategy": _load_peer("mean_reversion_strategy"),
    }

    assets = metadata["asset_ids"]
    all_ret = pd.concat([train_returns, validation_returns])
    val_dates = validation_returns.index

    scores = {}
    for name, mod in strategies.items():
        s = mod.fit(train_returns, train_features,
                    validation_returns, validation_features, metadata)
        daily = []
        for date in val_dates:
            hist = all_ret.loc[:date]
            w = mod.generate_weights(date, hist, {}, s, metadata)
            w = w.reindex(assets, fill_value=0.0)
            daily.append((w * validation_returns.loc[date]).sum())
        scores[name] = sharpe_ratio(pd.Series(daily, index=val_dates))

    sorted_strats = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    top2 = [sorted_strats[0][0], sorted_strats[1][0]]

    # Store sub-strategy states
    sub_states = {}
    for name in top2:
        mod = strategies[name]
        sub_states[name] = mod.fit(train_returns, train_features,
                                   validation_returns, validation_features,
                                   metadata)

    return {"blend": top2, "scores": scores, "sub_states": sub_states,
            "strategy_modules": strategies}


def generate_weights(date, history_returns, history_features, state, metadata):
    """Blend top-2 strategies 50/50."""
    assets = metadata["asset_ids"]
    blend = state["blend"]
    strategies = state["strategy_modules"]
    combined = pd.Series(0.0, index=assets)

    for name in blend:
        mod = strategies[name]
        w = mod.generate_weights(date, history_returns, history_features,
                                 state["sub_states"][name], metadata)
        w = w.reindex(assets, fill_value=0.0)
        combined += w / len(blend)

    max_w = metadata["constraints"]["max_weight_per_asset"]
    combined = combined.clip(upper=max_w)
    combined = combined / combined.sum()
    return combined
