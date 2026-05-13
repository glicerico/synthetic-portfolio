"""Tests for constraint checking in portfolio_eval.evaluator."""

import pandas as pd
import pytest

from portfolio_eval.evaluator import check_constraints


@pytest.fixture
def metadata():
    return {
        "asset_ids": ["A00", "A01", "A02", "A03", "A04"],
        "constraints": {
            "long_only": True,
            "fully_invested": True,
            "max_weight_per_asset": 0.20,
            "no_nans": True,
            "monthly_rebalance_only": True,
        },
    }


class TestConstraints:
    def test_valid_equal_weight(self, metadata):
        """5 assets at 20% each — all constraints satisfied."""
        w = pd.Series(0.20, index=metadata["asset_ids"])
        violations = check_constraints(w, metadata)
        assert violations == []

    def test_nan_violation(self, metadata):
        w = pd.Series([0.2, 0.2, 0.2, 0.2, float("nan")],
                       index=metadata["asset_ids"])
        violations = check_constraints(w, metadata)
        assert any("NaN" in v for v in violations)

    def test_negative_weight_violation(self, metadata):
        w = pd.Series([0.4, 0.3, 0.2, 0.2, -0.1],
                       index=metadata["asset_ids"])
        violations = check_constraints(w, metadata)
        assert any("long-only" in v for v in violations)

    def test_not_fully_invested(self, metadata):
        w = pd.Series([0.1, 0.1, 0.1, 0.1, 0.1],
                       index=metadata["asset_ids"])
        violations = check_constraints(w, metadata)
        assert any("sum" in v for v in violations)

    def test_over_max_weight(self, metadata):
        w = pd.Series([0.5, 0.2, 0.1, 0.1, 0.1],
                       index=metadata["asset_ids"])
        violations = check_constraints(w, metadata)
        assert any("exceed" in v for v in violations)

    def test_all_violations(self, metadata):
        """Weights that violate everything at once."""
        w = pd.Series([-0.1, 0.5, float("nan"), 0.3, 0.1],
                       index=metadata["asset_ids"])
        violations = check_constraints(w, metadata)
        # Should catch NaN, negative, over-max, and likely not summing to 1
        assert len(violations) >= 3

    def test_exact_max_weight_ok(self, metadata):
        """Exactly at the max weight boundary should pass."""
        w = pd.Series(0.20, index=metadata["asset_ids"])
        violations = check_constraints(w, metadata)
        assert violations == []
