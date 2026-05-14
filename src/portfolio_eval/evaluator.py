"""
Evaluator — runs a submitted strategy on the hidden test split.

Responsibilities
-----------------
1. Load the submitted strategy module dynamically.
2. Call ``fit()`` with ONLY public train + validation data.
3. Walk through the test period month by month.
4. On each rebalance date call ``generate_weights()`` with only the history
   available up to that date (no lookahead).
5. Enforce constraints and compute net-of-cost returns.
6. Report metrics.

CLI
---
    python -m portfolio_eval.evaluator \
        --data data/benchmark_medium --strategy examples/equal_weight_strategy.py
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from portfolio_eval import metrics as M
from portfolio_eval.strategy_interface import load_strategy

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_split(split_dir: Path):
    returns = pd.read_parquet(split_dir / "returns.parquet")
    features: Dict[str, pd.DataFrame] = {}
    feat_dir = split_dir / "features"
    if feat_dir.exists():
        for p in sorted(feat_dir.glob("*.parquet")):
            features[p.stem] = pd.read_parquet(p)
    return returns, features


def assert_public_package_is_clean(public_dir: str | Path):
    """Ensure no hidden data exists in the public package."""
    pub = Path(public_dir)
    forbidden = [
        "test_returns", "test_features", "true_regimes", "dgp_config",
        "full_returns", "full_features", "metadata_hidden",
    ]
    # Also reject 'test' directory explicitly
    if (pub / "test").exists():
        raise AssertionError(f"Public package {pub} contains a 'test' directory!")
    
    for root, dirs, files in os.walk(pub):
        for name in dirs + files:
            for f in forbidden:
                if f in name:
                    raise AssertionError(f"Public package {pub} contains forbidden file/dir: {name}")

def _load_benchmark(public_dir: str, hidden_dir: str):
    pub = Path(public_dir)
    hid = Path(hidden_dir)
    train_ret, train_feat = _load_split(pub / "train")
    val_ret, val_feat = _load_split(pub / "validation")
    test_ret, test_feat = _load_split(hid / "test")
    with open(pub / "metadata_public.json") as f:
        metadata = json.load(f)
    return train_ret, train_feat, val_ret, val_feat, test_ret, test_feat, metadata


# ---------------------------------------------------------------------------
# Constraint checking
# ---------------------------------------------------------------------------

def check_constraints(
    weights: pd.Series,
    metadata: Dict[str, Any],
) -> List[str]:
    """Return list of constraint violation descriptions (empty = pass)."""
    violations: List[str] = []
    constraints = metadata.get("constraints", {})

    # No NaNs
    if weights.isna().any():
        violations.append("weights contain NaN")

    # Long-only
    if constraints.get("long_only", True):
        if (weights < -1e-9).any():
            negs = weights[weights < -1e-9]
            violations.append(
                f"negative weights (long-only violated): {dict(negs)}"
            )

    # Fully invested
    if constraints.get("fully_invested", True):
        total = weights.sum()
        if abs(total - 1.0) > 1e-6:
            violations.append(f"weights sum to {total:.6f}, expected 1.0")

    # Max per-asset
    max_w = constraints.get("max_weight_per_asset", 0.20)
    over = weights[weights > max_w + 1e-9]
    if len(over):
        violations.append(
            f"weights exceed {max_w:.0%} max: {dict(over)}"
        )

    return violations


# ---------------------------------------------------------------------------
# Monthly rebalance dates
# ---------------------------------------------------------------------------

def _monthly_rebalance_dates(dates: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """First business day of each month in *dates*."""
    rebal: List[pd.Timestamp] = []
    current_month = None
    for d in dates:
        ym = (d.year, d.month)
        if ym != current_month:
            rebal.append(d)
            current_month = ym
    return rebal


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(
    public_dir: str,
    hidden_dir: str,
    strategy_path: str,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run full evaluation pipeline. Returns a results dict."""
    assert_public_package_is_clean(public_dir)
    
    # Load data
    (train_ret, train_feat, val_ret, val_feat,
     test_ret, test_feat, metadata) = _load_benchmark(public_dir, hidden_dir)

    # Load strategy
    strategy = load_strategy(strategy_path)

    # --- FIT (public data only) ---
    t0 = time.perf_counter()
    state = strategy.fit(
        train_ret, train_feat, val_ret, val_feat, metadata,
    )
    fit_time = time.perf_counter() - t0

    # --- Prepare combined history for walk-forward ---
    # During test, the strategy can see public train+val data as history,
    # plus test data up to the decision date.
    all_returns = pd.concat([train_ret, val_ret, test_ret])
    all_features: Dict[str, pd.DataFrame] = {}
    for fname in train_feat:
        parts = [train_feat[fname]]
        if fname in val_feat:
            parts.append(val_feat[fname])
        if fname in test_feat:
            parts.append(test_feat[fname])
        all_features[fname] = pd.concat(parts)

    test_dates = test_ret.index
    rebalance_dates = _monthly_rebalance_dates(test_dates)
    assets = list(test_ret.columns)
    n_assets = len(assets)

    # Walk-forward simulation
    current_weights = pd.Series(1.0 / n_assets, index=assets)
    weight_records: List[pd.Series] = []
    weight_change_records: List[pd.Series] = []
    all_violations: List[Dict[str, Any]] = []

    t1 = time.perf_counter()

    for rdate in rebalance_dates:
        # Provide only history up to (and including) rdate
        hist_ret = all_returns.loc[:rdate]
        hist_feat = {k: v.loc[:rdate] for k, v in all_features.items()}

        new_weights_raw = strategy.generate_weights(
            rdate, hist_ret, hist_feat, state, metadata,
        )

        # Normalise to Series
        if isinstance(new_weights_raw, dict):
            new_weights = pd.Series(new_weights_raw, dtype=float)
        else:
            new_weights = new_weights_raw.copy().astype(float)

        # Reindex to ensure all assets present
        new_weights = new_weights.reindex(assets, fill_value=0.0)

        # Check constraints
        violations = check_constraints(new_weights, metadata)
        if violations:
            all_violations.append({"date": str(rdate.date()), "violations": violations})

        # Record weight change
        change = new_weights - current_weights
        weight_change_records.append(change)
        weight_records.append(new_weights.copy())
        current_weights = new_weights.copy()

    eval_time = time.perf_counter() - t1

    # --- Compute daily portfolio returns ---
    cost_bps = metadata.get("transaction_cost_bps", 30)
    cost_frac = cost_bps / 10_000

    # Build weight matrix aligned to test dates
    # Between rebalance dates, weights drift with returns.
    portfolio_daily_ret = []
    w = pd.Series(1.0 / n_assets, index=assets)  # initial equal weight
    rebal_idx = 0

    for i, date in enumerate(test_dates):
        # Check if we rebalance today
        if rebal_idx < len(rebalance_dates) and date == rebalance_dates[rebal_idx]:
            new_w = weight_records[rebal_idx]
            tc = (new_w - w).abs().sum() * cost_frac
            w = new_w.copy()
            rebal_idx += 1
        else:
            tc = 0.0

        day_ret = test_ret.loc[date]
        port_ret = (w * day_ret).sum() - tc
        portfolio_daily_ret.append(port_ret)

        # Drift weights with returns
        w = w * (1 + day_ret)
        w = w / w.sum()  # renormalise

    daily_returns = pd.Series(portfolio_daily_ret, index=test_dates, name="portfolio")

    # Weight changes DataFrame
    if weight_change_records:
        wc_df = pd.DataFrame(weight_change_records)
    else:
        wc_df = pd.DataFrame()

    ew_daily_returns = test_ret.mean(axis=1)
    active_daily_returns = daily_returns - ew_daily_returns

    # --- Metrics ---
    results = {
        "annualized_sharpe": M.sharpe_ratio(daily_returns),
        "baseline_ew_sharpe": M.sharpe_ratio(ew_daily_returns),
        "information_ratio_vs_ew": M.information_ratio(active_daily_returns),
        "total_return": M.total_return(daily_returns),
        "baseline_ew_total_return": M.total_return(ew_daily_returns),
        "annualized_return": M.annualized_return(daily_returns),
        "baseline_ew_annualized_return": M.annualized_return(ew_daily_returns),
        "annualized_volatility": M.annualized_volatility(daily_returns),
        "baseline_ew_annualized_volatility": M.annualized_volatility(ew_daily_returns),
        "max_drawdown": M.max_drawdown(daily_returns),
        "baseline_ew_max_drawdown": M.max_drawdown(ew_daily_returns),
        "average_turnover": M.average_turnover(wc_df),
        "transaction_cost_drag": M.transaction_cost_drag(wc_df, cost_bps),
        "constraint_violations": all_violations,
        "n_constraint_violations": len(all_violations),
        "fit_time_seconds": round(fit_time, 3),
        "eval_time_seconds": round(eval_time, 3),
        "total_time_seconds": round(fit_time + eval_time, 3),
        "n_rebalance_dates": len(rebalance_dates),
    }

    if verbose:
        print("\n" + "=" * 60)
        print("EVALUATION RESULTS (vs Equal-Weight Baseline)")
        print("=" * 60)
        
        display_keys = [
            ("annualized_sharpe", "baseline_ew_sharpe"),
            ("information_ratio_vs_ew", None),
            ("total_return", "baseline_ew_total_return"),
            ("annualized_return", "baseline_ew_annualized_return"),
            ("annualized_volatility", "baseline_ew_annualized_volatility"),
            ("max_drawdown", "baseline_ew_max_drawdown"),
            ("average_turnover", None),
            ("transaction_cost_drag", None),
            ("n_constraint_violations", None),
            ("fit_time_seconds", None),
            ("eval_time_seconds", None),
            ("total_time_seconds", None),
            ("n_rebalance_dates", None),
        ]

        print(f"{'Metric':<30} | {'Strategy':>12} | {'EW Baseline':>12}")
        print("-" * 60)
        for strat_k, base_k in display_keys:
            strat_v = results[strat_k]
            
            if isinstance(strat_v, float):
                strat_str = f"{strat_v:>12.4f}"
            else:
                strat_str = f"{strat_v!s:>12s}"
                
            if base_k:
                base_v = results[base_k]
                if isinstance(base_v, float):
                    base_str = f"{base_v:>12.4f}"
                else:
                    base_str = f"{base_v!s:>12s}"
            else:
                base_str = f"{'-':>12s}"
                
            print(f"{strat_k:<30} | {strat_str} | {base_str}")

        if all_violations:
            print(f"\n  Constraint violations ({len(all_violations)}):")
            for viol in all_violations[:5]:
                print(f"    {viol['date']}: {viol['violations']}")
            if len(all_violations) > 5:
                print(f"    ... and {len(all_violations) - 5} more")
        print("=" * 60)

        # Leaderboard
        import datetime
        import shutil

        baselines_path = Path(hidden_dir) / "baselines.json"
        if baselines_path.exists():
            with open(baselines_path) as f:
                baselines = json.load(f)
                
            # Create a timestamped strategy name
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            original_stem = Path(strategy_path).stem
            strategy_name = f"{original_stem}_{ts}"

            # Copy strategy to hidden_dir/submissions
            submissions_dir = Path(hidden_dir) / "submissions"
            submissions_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(strategy_path, submissions_dir / f"{strategy_name}.py")

            # Append current strategy to baselines.json
            clean_results = results.copy()
            clean_results["constraint_violations"] = [] # Don't bloat the JSON
            clean_results["submission_time"] = ts
            baselines[strategy_name] = clean_results
            
            # Save it back so the leaderboard is persistent for this dataset
            with open(baselines_path, "w") as f:
                json.dump(baselines, f, indent=2, default=str)
            
            print("\n" + "=" * 65)
            print("LEADERBOARD COMPARISON")
            print("=" * 65)
            print(f"{'Strategy Name':<40} | {'Sharpe':>8} | {'Total Ret':>9} | {'Max DD':>8}")
            print("-" * 65)
            
            # Sort all strategies by Sharpe
            sorted_baselines = sorted(baselines.items(), key=lambda x: x[1].get('annualized_sharpe', -999), reverse=True)
            
            for name, metrics in sorted_baselines:
                sharpe = metrics.get('annualized_sharpe', 0.0)
                tot_ret = metrics.get('total_return', 0.0)
                max_dd = metrics.get('max_drawdown', 0.0)
                
                display_name = name
                if name == strategy_name:
                    display_name = f">>> {name}"
                    
                # Truncate long names to 40 chars
                if len(display_name) > 40:
                    display_name = display_name[:37] + "..."

                print(f"{display_name:<40} | {sharpe:>8.4f} | {tot_ret:>9.4f} | {max_dd:>8.4f}")
            print("=" * 65)

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(public_dir: str, hidden_dir: str, strategy_path: str, out_path: str) -> None:
    results = evaluate(public_dir, hidden_dir, strategy_path)
    # Save results JSON
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--public-data", required=True)
    parser.add_argument("--hidden-data", required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    main(args.public_data, args.hidden_data, args.strategy, args.out)
