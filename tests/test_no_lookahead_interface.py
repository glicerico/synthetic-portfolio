"""
Tests that the evaluator enforces no-lookahead: generate_weights() must
only receive data up to and including the decision date.
Also tests the two-package workflow safety checks.
"""

import pandas as pd
import numpy as np
import pytest
import json
import tempfile
from pathlib import Path

from portfolio_eval.evaluator import evaluate, assert_public_package_is_clean


def _make_mini_benchmark(tmp_dir: Path, n_days: int = 300):
    """Create a tiny benchmark dataset for testing with public/hidden split."""
    rng = np.random.default_rng(123)
    dates = pd.bdate_range("2020-01-02", periods=n_days)
    assets = ["A00", "A01", "A02", "A03"]
    returns = pd.DataFrame(
        rng.normal(0, 0.01, (n_days, len(assets))),
        index=dates,
        columns=assets,
    )

    t_end = int(n_days * 0.6)
    v_end = int(n_days * 0.8)

    pub_dir = tmp_dir / "public"
    hid_dir = tmp_dir / "hidden"
    pub_dir.mkdir()
    hid_dir.mkdir()

    for split, sl in [("train", slice(0, t_end)),
                      ("validation", slice(t_end, v_end))]:
        sdir = pub_dir / split
        sdir.mkdir(parents=True, exist_ok=True)
        returns.iloc[sl].to_parquet(sdir / "returns.parquet")
        feat_dir = sdir / "features"
        feat_dir.mkdir(exist_ok=True)
        feat = pd.DataFrame(rng.normal(0, 1, (len(returns.iloc[sl]), len(assets))),
                            index=returns.iloc[sl].index, columns=assets)
        feat.to_parquet(feat_dir / "mom_21d.parquet")

    # Hidden test split
    sdir = hid_dir / "test"
    sdir.mkdir(parents=True, exist_ok=True)
    returns.iloc[slice(v_end, n_days)].to_parquet(sdir / "returns.parquet")
    feat_dir = sdir / "features"
    feat_dir.mkdir(exist_ok=True)
    feat = pd.DataFrame(rng.normal(0, 1, (n_days - v_end, len(assets))),
                        index=returns.iloc[slice(v_end, n_days)].index, columns=assets)
    feat.to_parquet(feat_dir / "mom_21d.parquet")

    metadata = {
        "asset_ids": assets,
        "sector_labels": {"A00": "Tech", "A01": "Tech",
                          "A02": "Fin", "A03": "Fin"},
        "rebalance_rule": "monthly",
        "transaction_cost_bps": 30,
        "constraints": {
            "long_only": True,
            "fully_invested": True,
            "max_weight_per_asset": 0.50,
            "no_nans": True,
            "monthly_rebalance_only": True,
        },
    }
    with open(pub_dir / "metadata_public.json", "w") as f:
        json.dump(metadata, f)
        
    with open(hid_dir / "metadata_hidden.json", "w") as f:
        json.dump(metadata, f)

    return returns, dates, t_end, v_end, pub_dir, hid_dir


def _write_spy_strategy(path: Path, test_start_date: str):
    code = f'''
import pandas as pd

TEST_START = pd.Timestamp("{test_start_date}")

def fit(train_returns, train_features, validation_returns, validation_features, metadata):
    return {{}}

def generate_weights(date, history_returns, history_features, state, metadata):
    max_hist_date = history_returns.index.max()
    assert max_hist_date <= date, (
        f"Lookahead detected! Decision date={{date}}, but history goes to {{max_hist_date}}"
    )
    for fname, fdf in history_features.items():
        max_feat_date = fdf.index.max()
        assert max_feat_date <= date, (
            f"Feature lookahead! Feature {{fname}}: decision={{date}}, max={{max_feat_date}}"
        )

    n = len(metadata["asset_ids"])
    return pd.Series(1.0 / n, index=metadata["asset_ids"])
'''
    path.write_text(code)


class TestNoLookahead:
    def test_evaluator_no_future_data(self, tmp_path):
        returns, dates, t_end, v_end, pub_dir, hid_dir = _make_mini_benchmark(tmp_path)
        test_start = dates[v_end]

        strategy_path = tmp_path / "spy_strategy.py"
        _write_spy_strategy(strategy_path, str(test_start.date()))

        results = evaluate(
            public_dir=str(pub_dir),
            hidden_dir=str(hid_dir),
            strategy_path=str(strategy_path),
            verbose=False,
        )

        assert "annualized_sharpe" in results
        assert results["n_constraint_violations"] == 0

    def test_strategy_receives_train_val_in_fit(self, tmp_path):
        returns, dates, t_end, v_end, pub_dir, hid_dir = _make_mini_benchmark(tmp_path)

        strategy_code = f'''
import pandas as pd

def fit(train_returns, train_features, validation_returns, validation_features, metadata):
    assert train_returns.index.max() < validation_returns.index.min(), \\
        "Train and validation overlap!"
    assert len(train_returns) > 0
    assert len(validation_returns) > 0
    return {{}}

def generate_weights(date, history_returns, history_features, state, metadata):
    n = len(metadata["asset_ids"])
    return pd.Series(1.0 / n, index=metadata["asset_ids"])
'''
        strat_path = tmp_path / "check_fit_strategy.py"
        strat_path.write_text(strategy_code)

        results = evaluate(
            public_dir=str(pub_dir),
            hidden_dir=str(hid_dir),
            strategy_path=str(strat_path),
            verbose=False,
        )
        assert "annualized_sharpe" in results

    def test_assert_public_package_is_clean(self, tmp_path):
        # Should pass on clean package
        returns, dates, t_end, v_end, pub_dir, hid_dir = _make_mini_benchmark(tmp_path)
        assert_public_package_is_clean(pub_dir)
        
        # Test 1: Should fail if 'test' dir exists
        (pub_dir / "test").mkdir()
        with pytest.raises(AssertionError, match="contains a 'test' directory"):
            assert_public_package_is_clean(pub_dir)
        (pub_dir / "test").rmdir()
        
        # Test 2: Should fail if forbidden file name exists
        forbidden_file = pub_dir / "dgp_config.json"
        forbidden_file.touch()
        with pytest.raises(AssertionError, match="contains forbidden file/dir: dgp_config.json"):
            assert_public_package_is_clean(pub_dir)
