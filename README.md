# Portfolio-Eval: Synthetic Portfolio Strategy Benchmark

A reproducible benchmark for evaluating research/agent systems on a hidden-test
portfolio strategy task.  The benchmark generates synthetic financial data with
hidden regimes, provides a strict no-lookahead evaluation harness, and reports
risk-adjusted performance metrics net of transaction costs.

> **⚠️ Disclaimer:** This benchmark uses purely synthetic data.  It is not real
> investment advice and should not be used for actual trading decisions.

---

## Quick Start

```bash
# Install (editable, with dev dependencies)
pip install -e ".[dev]"

# 1. Generate a synthetic dataset into public and hidden packages
python -m portfolio_eval generate_dataset \
    --config configs/dataset_medium.yaml \
    --public-out data/public_medium \
    --hidden-out data/hidden_medium

# 2. Evaluate a strategy
python -m portfolio_eval evaluator \
    --public-data data/public_medium \
    --hidden-data data/hidden_medium \
    --strategy examples/equal_weight_strategy.py \
    --out data/results.json
```

---

## Benchmark Purpose

The goal is to test whether a research agent or human analyst can discover a
portfolio strategy that:

1. **Outperforms equal-weight** on risk-adjusted, net-of-cost returns.
2. **Generalises** from public train/validation data to hidden test data.
3. **Obeys constraints** (long-only, fully invested, max 20% per asset, etc.).

Different dataset configs create different difficulty levels — from easy
(similar regimes in validation and test) to adversarial (noise traps and
turnover traps that punish overfitting).

---

## Detailed Documentation

For full details on the evaluation protocol and the data generation specifics, please refer to the following documents:
- [Evaluator Instructions](EVALUATOR_INSTRUCTIONS.md): Contains the complete guide on managing the two-package separation, running the evaluator CLI, and prompting external agents/systems.
- [Dataset Generation](DATASET_GENERATION.md): Contains specifics on the hidden Markov regimes, data generation features, predictive signals, and benchmark config variants.

---

## Strategy Interface

A submitted strategy is a Python file with two functions:

```python
def fit(train_returns, train_features, validation_returns, validation_features, metadata):
    """
    Fit strategy state using only train/validation data.
    Return a serializable state object.
    """
    ...
    return state

def generate_weights(date, history_returns, history_features, state, metadata):
    """
    Called on each monthly rebalance date.
    Inputs contain only data available up to the decision date.
    Return asset weights as a pd.Series or dict.
    """
    ...
    return weights
```

### Constraints

| Constraint            | Requirement                        |
|-----------------------|------------------------------------|
| Long-only             | All weights ≥ 0                    |
| Fully invested        | Weights sum to 1.0                 |
| Max per-asset weight  | No single weight > 20%             |
| No NaNs               | All weights must be numeric        |
| Monthly rebalance     | Weights change only monthly        |

---


## Baselines

| Strategy                  | File                                   | Description                        |
|---------------------------|----------------------------------------|------------------------------------|
| Equal Weight              | `examples/equal_weight_strategy.py`    | 1/N allocation                     |
| Momentum                  | `examples/momentum_strategy.py`        | Rank-based 63-day momentum         |
| Low Volatility            | `examples/low_vol_strategy.py`         | Inverse-volatility weighting       |
| Mean Reversion            | `examples/mean_reversion_strategy.py`  | 5-day contrarian                   |
| Validation-Selected Blend | `examples/validation_selected_strategy.py` | Best 2 from validation, 50/50 blend |

---



## Running Tests

```bash
pytest tests/ -v
```

Tests cover:
- Metric computations (Sharpe, max drawdown, turnover, transaction costs)
- Constraint checking (NaN, negative weights, over-max, not fully invested)
- No-lookahead enforcement (the evaluator never leaks future data)

---

## Project Structure

```
├── configs/
│   ├── dataset_easy.yaml
│   ├── dataset_medium.yaml
│   ├── dataset_hard.yaml
│   ├── dataset_noise_trap.yaml
│   └── dataset_turnover_trap.yaml
├── examples/
│   ├── equal_weight_strategy.py
│   ├── momentum_strategy.py
│   ├── low_vol_strategy.py
│   ├── mean_reversion_strategy.py
│   └── validation_selected_strategy.py
├── src/portfolio_eval/
│   ├── __init__.py
│   ├── __main__.py
│   ├── baselines.py
│   ├── evaluator.py
│   ├── generate_dataset.py
│   ├── metrics.py
│   ├── split_dataset.py
│   └── strategy_interface.py
├── tests/
│   ├── test_metrics.py
│   ├── test_constraints.py
│   └── test_no_lookahead_interface.py
├── pyproject.toml
└── README.md
```

---

## No-Lookahead Assumptions

The benchmark enforces strict temporal ordering:

1. `fit()` receives only train and validation data — never test data.
2. `generate_weights()` receives history only up to the decision date.
3. Features are computed from past returns only (no forward-looking features).
4. The evaluator walks forward through the test period day by day.

These guarantees are verified by the `test_no_lookahead_interface.py` test suite,
which uses a "spy" strategy that asserts it never sees future dates.

---

## Limitations

- **Synthetic data only** — the factor model is a simplification of real
  markets.  Results do not transfer to live trading.
- **No order-book effects** — transaction costs are a flat basis-point
  assumption, not market-impact-aware.
- **Monthly rebalance only** — higher-frequency strategies are not supported.
- **No short selling** — the long-only constraint restricts the strategy space.
- **No leverage** — weights must sum to 1.0.

---

## License

MIT
