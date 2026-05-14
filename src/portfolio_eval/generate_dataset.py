"""
Synthetic dataset generator.

Generates daily returns for 20 assets (5 sectors × 4 assets) over ~5 years
of business days using a factor model with hidden regimes.

Data-Generating Process (DGP)
-----------------------------
1. Hidden regime sequence: momentum, low-vol, mean-reversion, noisy/stress.
   Regimes switch according to a Markov chain.
2. Common market factor drives broad co-movement.
3. Sector factors add within-sector correlation.
4. Asset-specific idiosyncratic noise.
5. Regime-dependent factor loadings modulate signal strength.
6. Features are computed from the generated returns + noise columns.

CLI
---
    python -m portfolio_eval.generate_dataset \
        --config configs/dataset_medium.yaml \
        --public-out data/public_medium \
        --hidden-out data/hidden_medium
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_SECTORS = 5
ASSETS_PER_SECTOR = 4
N_ASSETS = N_SECTORS * ASSETS_PER_SECTOR
REGIME_NAMES = ["momentum", "low_vol", "mean_reversion", "noisy"]

SECTOR_NAMES = [
    "Technology",
    "Healthcare",
    "Financials",
    "Energy",
    "Consumer",
]


def _default_config() -> Dict[str, Any]:
    """Return sensible defaults (overridden by YAML config)."""
    return dict(
        seed=42,
        n_years=5,
        # Regime transition matrix (row = from, col = to)
        regime_transition=[[0.96, 0.02, 0.01, 0.01],
                           [0.02, 0.95, 0.02, 0.01],
                           [0.01, 0.02, 0.95, 0.02],
                           [0.02, 0.01, 0.02, 0.95]],
        regime_start=0,
        # Market factor params per regime  (mean, vol)
        market_factor={
            "momentum":       {"mean": 0.0004, "vol": 0.010},
            "low_vol":        {"mean": 0.0003, "vol": 0.006},
            "mean_reversion": {"mean": 0.0001, "vol": 0.012},
            "noisy":          {"mean": -0.0001, "vol": 0.020},
        },
        # Sector factor vol per regime
        sector_factor_vol={
            "momentum": 0.005,
            "low_vol": 0.003,
            "mean_reversion": 0.006,
            "noisy": 0.010,
        },
        # Idiosyncratic vol per regime
        idio_vol={
            "momentum": 0.008,
            "low_vol": 0.005,
            "mean_reversion": 0.009,
            "noisy": 0.015,
        },
        # Momentum signal strength per regime (added to next-day return)
        momentum_signal={
            "momentum": 0.0003,
            "low_vol": 0.00005,
            "mean_reversion": -0.0002,
            "noisy": 0.0,
        },
        # Mean-reversion signal strength per regime
        reversion_signal={
            "momentum": -0.00005,
            "low_vol": 0.00005,
            "mean_reversion": 0.0003,
            "noisy": 0.0,
        },
        # Low-vol premium per regime (alpha for low-vol assets)
        lowvol_premium={
            "momentum": 0.00005,
            "low_vol": 0.0003,
            "mean_reversion": 0.00005,
            "noisy": 0.0,
        },
        transaction_cost_bps=30,
        rebalance_rule="monthly",
    )


# ---------------------------------------------------------------------------
# Regime simulation
# ---------------------------------------------------------------------------

def _simulate_regimes(rng: np.random.Generator, n_days: int,
                      transition: np.ndarray, start: int) -> np.ndarray:
    """Simulate hidden Markov regime sequence."""
    regimes = np.empty(n_days, dtype=int)
    regimes[0] = start
    for t in range(1, n_days):
        regimes[t] = rng.choice(len(transition), p=transition[regimes[t - 1]])
    return regimes


# ---------------------------------------------------------------------------
# Return generation
# ---------------------------------------------------------------------------

def _generate_returns(
    rng: np.random.Generator,
    dates: pd.DatetimeIndex,
    regimes: np.ndarray,
    cfg: Dict[str, Any],
) -> pd.DataFrame:
    """Generate daily returns for all assets."""
    n_days = len(dates)
    asset_ids = [f"A{i:02d}" for i in range(N_ASSETS)]
    sector_map = {}
    for s in range(N_SECTORS):
        for a in range(ASSETS_PER_SECTOR):
            asset_ids_idx = s * ASSETS_PER_SECTOR + a
            sector_map[asset_ids[asset_ids_idx]] = SECTOR_NAMES[s]

    # Pre-compute per-asset idiosyncratic vol multiplier (some assets are
    # naturally more volatile)
    asset_vol_mult = 0.7 + 0.6 * rng.random(N_ASSETS)  # [0.7, 1.3]

    returns = np.zeros((n_days, N_ASSETS))

    for t in range(n_days):
        regime = REGIME_NAMES[regimes[t]]

        # Market factor
        mf_mean = cfg["market_factor"][regime]["mean"]
        mf_vol = cfg["market_factor"][regime]["vol"]
        market = rng.normal(mf_mean, mf_vol)

        # Sector factors
        sf_vol = cfg["sector_factor_vol"][regime]
        sector_factors = rng.normal(0, sf_vol, N_SECTORS)

        # Idiosyncratic
        idio_base = cfg["idio_vol"][regime]

        for i in range(N_ASSETS):
            s = i // ASSETS_PER_SECTOR
            idio = rng.normal(0, idio_base * asset_vol_mult[i])
            returns[t, i] = market + sector_factors[s] + idio

    # Overlay momentum signal: assets with positive past-21d return get a
    # regime-dependent boost.
    cum = np.zeros(N_ASSETS)
    for t in range(n_days):
        regime = REGIME_NAMES[regimes[t]]
        mom_strength = cfg["momentum_signal"][regime]
        rev_strength = cfg["reversion_signal"][regime]
        lowvol_prem = cfg["lowvol_premium"][regime]

        if t >= 21:
            past_21 = returns[t - 21:t].sum(axis=0)
            # Momentum: boost assets with positive past returns
            signal = np.sign(past_21) * mom_strength
            returns[t] += signal

        if t >= 5:
            past_5 = returns[t - 5:t].sum(axis=0)
            # Mean reversion: negative of short-term return
            returns[t] -= np.sign(past_5) * rev_strength

        # Low-vol premium: assets with below-median vol get a tiny boost
        if t >= 63:
            rolling_vol = returns[t - 63:t].std(axis=0)
            median_vol = np.median(rolling_vol)
            lowvol_mask = (rolling_vol < median_vol).astype(float)
            returns[t] += lowvol_mask * lowvol_prem

    df = pd.DataFrame(returns, index=dates, columns=asset_ids)
    return df, sector_map, asset_vol_mult


# ---------------------------------------------------------------------------
# Feature generation
# ---------------------------------------------------------------------------

def _generate_features(
    returns: pd.DataFrame,
    rng: np.random.Generator,
    regimes: np.ndarray,
    sector_map: dict,
) -> Dict[str, pd.DataFrame]:
    """Compute features from returns. Some are predictive, some are noise."""
    features: Dict[str, pd.DataFrame] = {}

    # Momentum features
    features["mom_21d"] = returns.rolling(21).sum()
    features["mom_63d"] = returns.rolling(63).sum()
    features["mom_126d"] = returns.rolling(126).sum()

    # Volatility features
    features["vol_21d"] = returns.rolling(21).std()
    features["vol_63d"] = returns.rolling(63).std()

    # Reversal
    features["rev_5d"] = -returns.rolling(5).sum()

    # Sector momentum (63-day mean return of sector peers)
    sectors = list(set(sector_map.values()))
    sector_groups = {s: [a for a, sec in sector_map.items() if sec == s]
                     for s in sectors}
    sector_mom = pd.DataFrame(index=returns.index, columns=returns.columns,
                              dtype=float)
    for s, assets in sector_groups.items():
        sect_ret = returns[assets].rolling(63).sum().mean(axis=1)
        for a in assets:
            sector_mom[a] = sect_ret
    features["sector_mom_63d"] = sector_mom

    # Drawdown (63-day)
    cum = (1 + returns).cumprod()
    rolling_max = cum.rolling(63, min_periods=1).max()
    features["drawdown_63d"] = (cum - rolling_max) / rolling_max

    # Macro regime proxy — noisy version of true regime label
    regime_series = pd.Series(regimes, index=returns.index, dtype=float)
    noise = pd.Series(rng.normal(0, 0.8, len(returns)), index=returns.index)
    proxy = regime_series + noise
    macro_proxy = pd.DataFrame(
        np.tile(proxy.values[:, None], (1, len(returns.columns))),
        index=returns.index,
        columns=returns.columns,
    )
    features["macro_regime_proxy"] = macro_proxy

    # Pure noise distractors
    for k in range(1, 4):
        features[f"noise_{k}"] = pd.DataFrame(
            rng.normal(0, 1, returns.shape),
            index=returns.index,
            columns=returns.columns,
        )

    # Forward-fill NaNs created by rolling windows (for usability)
    for name in features:
        features[name] = features[name].bfill().ffill()

    return features


# ---------------------------------------------------------------------------
# Public API / CLI
# ---------------------------------------------------------------------------

def generate_dataset(config_path: str) -> Dict[str, Any]:
    """
    Generate a full synthetic dataset from a YAML config.

    Returns a dict with keys: returns, features, regimes, sector_map, metadata.
    """
    cfg = _default_config()
    with open(config_path) as f:
        overrides = yaml.safe_load(f) or {}
    # Deep merge one level
    for key, val in overrides.items():
        if isinstance(val, dict) and isinstance(cfg.get(key), dict):
            cfg[key].update(val)
        else:
            cfg[key] = val

    rng = np.random.default_rng(cfg["seed"])

    # Business days
    start = pd.Timestamp("2018-01-02")
    n_bdays = int(cfg["n_years"] * 252)
    dates = pd.bdate_range(start, periods=n_bdays)

    # Regimes
    transition = np.array(cfg["regime_transition"], dtype=float)
    regimes = _simulate_regimes(rng, n_bdays, transition, cfg["regime_start"])

    # Returns
    returns, sector_map, asset_vol_mult = _generate_returns(
        rng, dates, regimes, cfg,
    )

    # Features
    features = _generate_features(returns, rng, regimes, sector_map)

    # Metadata
    metadata = {
        "asset_ids": list(returns.columns),
        "sector_labels": sector_map,
        "rebalance_rule": cfg.get("rebalance_rule", "monthly"),
        "transaction_cost_bps": cfg.get("transaction_cost_bps", 30),
        "constraints": {
            "long_only": True,
            "fully_invested": True,
            "max_weight_per_asset": 0.20,
            "no_nans": True,
            "monthly_rebalance_only": True,
        },
    }

    hidden_metadata = {
        "regimes": regimes.tolist(),
        "regime_names": REGIME_NAMES,
        "dgp_config": cfg,
    }

    return {
        "returns": returns,
        "features": features,
        "regimes": regimes,
        "sector_map": sector_map,
        "metadata": metadata,
        "hidden_metadata": hidden_metadata,
        "dates": dates,
    }


def save_split_dataset(dataset: Dict[str, Any], public_out: str, hidden_out: str) -> None:
    """Persist the dataset split into public and hidden packages."""
    returns = dataset["returns"]
    features = dataset["features"]
    metadata = dataset["metadata"]
    hidden_meta = dataset["hidden_metadata"]
    regimes = dataset["regimes"]
    dates = returns.index

    n = len(returns)
    train_end = int(n * 0.6)
    val_end = int(n * 0.8)

    slices = {
        "train": slice(0, train_end),
        "validation": slice(train_end, val_end),
        "test": slice(val_end, n),
    }

    pub_dir = Path(public_out)
    hid_dir = Path(hidden_out)

    pub_dir.mkdir(parents=True, exist_ok=True)
    hid_dir.mkdir(parents=True, exist_ok=True)

    for split_name, sl in slices.items():
        target_dir = pub_dir if split_name in ["train", "validation"] else hid_dir
        
        sdir = target_dir / split_name
        sdir.mkdir(exist_ok=True)
        returns.iloc[sl].to_parquet(sdir / "returns.parquet")

        feat_dir = sdir / "features"
        feat_dir.mkdir(exist_ok=True)
        for fname, fdf in features.items():
            fdf.iloc[sl].to_parquet(feat_dir / f"{fname}.parquet")

    # Public metadata
    pub_meta = dict(metadata)
    pub_meta["train_dates"] = {
        "start": str(dates[0].date()),
        "end": str(dates[train_end - 1].date()),
    }
    pub_meta["validation_dates"] = {
        "start": str(dates[train_end].date()),
        "end": str(dates[val_end - 1].date()),
    }
    with open(pub_dir / "metadata_public.json", "w") as f:
        json.dump(pub_meta, f, indent=2)

    instructions = (
        "# Portfolio-Eval Challenge\n\n"
        "Your task is to build a portfolio strategy that maximizes the risk-adjusted "
        "return (Sharpe Ratio) while adhering to the specified constraints.\n\n"
        "This public package contains `train` and `validation` data, along with `metadata_public.json`.\n"
        "Hidden test data is held back for evaluation.\n\n"
        "## Required Submission Format\n\n"
        "You must submit a Python module containing exactly these two functions:\n\n"
        "```python\n"
        "def fit(train_returns, train_features, validation_returns, validation_features, metadata):\n"
        "    \"\"\"\n"
        "    Fit strategy state using only train/validation data.\n"
        "    Return a serializable state object.\n"
        "    \"\"\"\n"
        "    # ... your logic ...\n"
        "    return state\n\n"
        "def generate_weights(date, history_returns, history_features, state, metadata):\n"
        "    \"\"\"\n"
        "    Called on each monthly rebalance date.\n"
        "    Inputs contain only data available up to the decision date.\n"
        "    Return asset weights as a pd.Series or dict mapping asset_ids to floats.\n"
        "    \"\"\"\n"
        "    # ... your logic ...\n"
        "    return weights\n"
        "```\n"
    )
    with open(pub_dir / "instructions.md", "w") as f:
        f.write(instructions)

    # Hidden metadata
    hid = dict(hidden_meta)
    hid["test_dates"] = {
        "start": str(dates[val_end].date()),
        "end": str(dates[-1].date()),
    }
    hid["test_regimes"] = regimes[val_end:].tolist()
    with open(hid_dir / "metadata_hidden.json", "w") as f:
        json.dump(hid, f, indent=2, default=str)
        
    if "dgp_config" in hid:
        with open(hid_dir / "dgp_config.json", "w") as f:
            json.dump(hid["dgp_config"], f, indent=2, default=str)
    np.save(hid_dir / "true_regimes.npy", regimes[val_end:])

    print(f"Dataset generated and split successfully.")
    print(f"Public data saved to: {pub_dir}")
    print(f"Hidden data saved to: {hid_dir}")
    print(f"  Train:      {dates[0].date()} – {dates[train_end-1].date()}  ({train_end} days)")
    print(f"  Validation: {dates[train_end].date()} – {dates[val_end-1].date()}  ({val_end - train_end} days)")
    print(f"  Test:       {dates[val_end].date()} – {dates[-1].date()}  ({n - val_end} days)")


def main(config_path: str, public_out: str, hidden_out: str) -> None:
    ds = generate_dataset(config_path)
    save_split_dataset(ds, public_out, hidden_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--public-out", required=True)
    parser.add_argument("--hidden-out", required=True)
    args = parser.parse_args()
    main(args.config, args.public_out, args.hidden_out)
