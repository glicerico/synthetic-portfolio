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

    import shutil
    shutil.copy("AGENT_INSTRUCTIONS.md", pub_dir / "instructions.md")

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

    # --- Pre-evaluate baselines ---
    from portfolio_eval.evaluator import evaluate
    baselines = {}
    examples_dir = Path("examples")
    if examples_dir.exists():
        print(f"\nPre-evaluating baseline strategies in {examples_dir}...")
        for script in examples_dir.glob("*_strategy.py"):
            try:
                res = evaluate(public_out, hidden_out, str(script), verbose=False)
                baselines[script.stem] = res
                print(f"  {script.stem:30s} : Sharpe = {res['annualized_sharpe']:.4f}")
            except Exception as e:
                print(f"  {script.stem:30s} : FAILED ({e})")
                
    with open(hid_dir / "baselines.json", "w") as f:
        json.dump(baselines, f, indent=2, default=str)

    print(f"\nDataset generated and split successfully.")
    print(f"Public data saved to: {pub_dir}")
    print(f"Hidden data saved to: {hid_dir}")
    print(f"  Train:      {dates[0].date()} – {dates[train_end-1].date()}  ({train_end} days)")
    print(f"  Validation: {dates[train_end].date()} – {dates[val_end-1].date()}  ({val_end - train_end} days)")
    print(f"  Test:       {dates[val_end].date()} – {dates[-1].date()}  ({n - val_end} days)")




# ---------------------------------------------------------------------------
# Quality gate
# ---------------------------------------------------------------------------

def validate_dataset(dataset: Dict[str, Any], cfg: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate the generated dataset against quality_gate criteria in the config.

    Returns (passed: bool, messages: list[str]).

    Quality gate keys supported (all optional):
      test_regime_min:  {regime_idx: min_fraction}  -- test must have >= this fraction
      test_regime_max:  {regime_idx: max_fraction}  -- test must have <= this fraction
      oracle_min_information_ratio: float  -- regime-oracle IR vs EW must exceed this
      oracle_max_information_ratio: float  -- regime-oracle IR vs EW must not exceed this
    """
    gate = cfg.get("quality_gate")
    if not gate:
        return True, ["No quality_gate defined — skipping validation."]

    returns = dataset["returns"]
    regimes = dataset["regimes"]
    n = len(returns)
    val_end = int(n * 0.8)
    test_regimes = regimes[val_end:]
    test_returns = returns.iloc[val_end:]
    n_test = len(test_regimes)

    messages: list[str] = []
    passed = True

    # --- Structural: regime composition of test period ---
    regime_counts = {i: int(np.sum(test_regimes == i)) for i in range(4)}
    regime_fracs  = {i: regime_counts[i] / n_test for i in range(4)}

    for regime_idx_str, min_frac in gate.get("test_regime_min", {}).items():
        regime_idx = int(regime_idx_str)
        actual = regime_fracs[regime_idx]
        if actual < min_frac:
            messages.append(
                f"FAIL test_regime_min: regime {regime_idx} ({REGIME_NAMES[regime_idx]}) "
                f"= {actual:.2%} < required {min_frac:.2%}"
            )
            passed = False
        else:
            messages.append(
                f"OK   test_regime_min: regime {regime_idx} ({REGIME_NAMES[regime_idx]}) "
                f"= {actual:.2%} >= {min_frac:.2%}"
            )

    for regime_idx_str, max_frac in gate.get("test_regime_max", {}).items():
        regime_idx = int(regime_idx_str)
        actual = regime_fracs[regime_idx]
        if actual > max_frac:
            messages.append(
                f"FAIL test_regime_max: regime {regime_idx} ({REGIME_NAMES[regime_idx]}) "
                f"= {actual:.2%} > limit {max_frac:.2%}"
            )
            passed = False
        else:
            messages.append(
                f"OK   test_regime_max: regime {regime_idx} ({REGIME_NAMES[regime_idx]}) "
                f"= {actual:.2%} <= {max_frac:.2%}"
            )

    # --- Functional: oracle information ratio ---
    min_ir = gate.get("oracle_min_information_ratio")
    max_ir = gate.get("oracle_max_information_ratio")

    if min_ir is not None or max_ir is not None:
        # Oracle: perfect regime-aware weights each day
        assets = list(returns.columns)
        n_assets = len(assets)
        max_w = 1.0 / n_assets  # upper bound per oracle signal

        oracle_daily = []
        ew_daily = []
        for t, date in enumerate(test_returns.index):
            regime = REGIME_NAMES[test_regimes[t]]
            day_ret = test_returns.iloc[t].values
            ew_ret  = day_ret.mean()
            ew_daily.append(ew_ret)

            # Build regime-optimal signal
            if regime == "momentum":
                # Favor high past momentum (use 21-day prior)
                start_t = max(0, val_end + t - 21)
                hist_slice = returns.iloc[start_t: val_end + t]
                if len(hist_slice) >= 5:
                    signal = hist_slice.values.sum(axis=0)
                else:
                    signal = np.ones(n_assets)
            elif regime == "mean_reversion":
                start_t = max(0, val_end + t - 5)
                hist_slice = returns.iloc[start_t: val_end + t]
                if len(hist_slice) >= 2:
                    signal = -hist_slice.values.sum(axis=0)
                else:
                    signal = np.ones(n_assets)
            elif regime == "low_vol":
                start_t = max(0, val_end + t - 63)
                hist_slice = returns.iloc[start_t: val_end + t]
                if len(hist_slice) >= 10:
                    signal = 1.0 / (hist_slice.values.std(axis=0) + 1e-8)
                else:
                    signal = np.ones(n_assets)
            else:  # noisy
                signal = np.ones(n_assets)

            # Normalise to weights
            signal = signal - signal.min() + 1e-8
            w = signal / signal.sum()
            w = np.clip(w, 0, 0.20)
            w = w / w.sum()
            oracle_daily.append(np.dot(w, day_ret))

        oracle_s = np.array(oracle_daily)
        ew_s     = np.array(ew_daily)
        active   = oracle_s - ew_s
        ir = active.mean() / (active.std() + 1e-10) * np.sqrt(252)

        if min_ir is not None:
            if ir < min_ir:
                messages.append(
                    f"FAIL oracle_min_information_ratio: IR = {ir:.3f} < required {min_ir}"
                )
                passed = False
            else:
                messages.append(f"OK   oracle_min_information_ratio: IR = {ir:.3f} >= {min_ir}")

        if max_ir is not None:
            if ir > max_ir:
                messages.append(
                    f"FAIL oracle_max_information_ratio: IR = {ir:.3f} > limit {max_ir}"
                )
                passed = False
            else:
                messages.append(f"OK   oracle_max_information_ratio: IR = {ir:.3f} <= {max_ir}")

    return passed, messages


def main(config_path: str, public_out: str, hidden_out: str,
         max_retries: int = 20) -> None:
    """
    Generate a dataset and validate it against quality_gate criteria.
    If validation fails, automatically increment the seed and retry up to
    max_retries times.
    """
    with open(config_path) as f:
        raw_cfg = yaml.safe_load(f) or {}

    base_seed = raw_cfg.get("seed", 42)

    for attempt in range(max_retries + 1):
        seed = base_seed + attempt
        if attempt > 0:
            print(f"\n[Quality Gate] Retrying with seed={seed} (attempt {attempt}/{max_retries})...")
            raw_cfg["seed"] = seed
            # Write temp override (we re-read it in generate_dataset, so patch cfg instead)

        ds = generate_dataset(config_path)
        # Apply seed override for retries without re-writing the config file
        if attempt > 0:
            cfg_override = _default_config()
            for k, v in raw_cfg.items():
                if isinstance(v, dict) and isinstance(cfg_override.get(k), dict):
                    cfg_override[k].update(v)
                else:
                    cfg_override[k] = v
            cfg_override["seed"] = seed
            rng = np.random.default_rng(seed)
            n_bdays = int(cfg_override["n_years"] * 252)
            dates = pd.bdate_range(pd.Timestamp("2018-01-02"), periods=n_bdays)
            transition = np.array(cfg_override["regime_transition"], dtype=float)
            regimes = _simulate_regimes(rng, n_bdays, transition, cfg_override["regime_start"])
            returns, sector_map, _ = _generate_returns(rng, dates, regimes, cfg_override)
            features = _generate_features(returns, rng, regimes, sector_map)
            ds = {
                "returns": returns, "features": features, "regimes": regimes,
                "sector_map": sector_map, "metadata": ds["metadata"],
                "hidden_metadata": {
                    "regimes": regimes.tolist(),
                    "regime_names": REGIME_NAMES,
                    "dgp_config": cfg_override,
                },
                "dates": dates,
            }

        passed, messages = validate_dataset(ds, raw_cfg)

        print(f"\n[Quality Gate] Validation results (seed={seed}):")
        for msg in messages:
            print(f"  {msg}")

        if passed:
            print(f"[Quality Gate] PASSED with seed={seed}")
            if attempt > 0:
                print(f"[Quality Gate] NOTE: Original seed={base_seed} failed. "
                      f"Consider updating 'seed: {seed}' in {config_path}")
            save_split_dataset(ds, public_out, hidden_out)
            return
        else:
            print(f"[Quality Gate] FAILED — will retry with next seed.")

    print(f"\n[Quality Gate] WARNING: All {max_retries} retries exhausted. "
          f"Saving last attempt anyway. Review quality_gate thresholds in the config.")
    save_split_dataset(ds, public_out, hidden_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--public-out", required=True)
    parser.add_argument("--hidden-out", required=True)
    parser.add_argument("--max-retries", type=int, default=20,
                        help="Max seed increment retries if quality gate fails (default: 20)")
    args = parser.parse_args()
    main(args.config, args.public_out, args.hidden_out, args.max_retries)

