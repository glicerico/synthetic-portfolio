# Evaluator Instructions

This document is intended for human evaluators who are managing the benchmark or scoring an external system/agent. It explains the exact process for maintaining the two-package separation.

## Public vs Hidden Data Workflow

To ensure fair evaluation and prevent data leakage, the benchmark uses a strict two-package workflow. **⚠️ WARNING: Hidden data must NEVER be placed in the agent workspace.**

1. **Benchmark Creator Generates and Splits Packages:** Generates the synthetic dataset with regimes and configurations using `generate_dataset` and outputs directly to separated directories:
   ```bash
   python -m portfolio_eval generate_dataset \
       --config configs/dataset_medium.yaml \
       --public-out data/public_medium \
       --hidden-out data/hidden_medium
   ```
   - **Public Package:** Contains `train` data, `validation` data, `metadata_public.json`, and `instructions.md`.
   - **Hidden Package:** Contains `test` data, `metadata_hidden.json`, `true_regimes.npy`, and `dgp_config.json`.

2. **Agent Evaluation:** The agent receives *only* the Public Package to develop their strategy. **Provide the agent with `instructions.md` inside that folder.**

3. **Independent Scoring:** An independent evaluator uses the `evaluator` command with both the public package and the hidden package to score the submitted strategy.

| Split       | Available to Strategy | Purpose                         |
|-------------|-----------------------|---------------------------------|
| **Train**   | ✅ Yes                | Fit parameters, learn patterns  |
| **Validation** | ✅ Yes             | Model selection, hyperparameters|
| **Test**    | ❌ No                 | Hidden evaluation only          |

The **evaluator** enforces the split: `fit()` receives only train + validation
data, and `generate_weights()` receives only history up to the rebalance date.
The evaluator will also abort if it detects any hidden files accidentally placed in the public package.

### What Metadata Is Public (`metadata_public.json`)

- Asset IDs and sector labels
- Train and validation date ranges
- Rebalance rule (monthly)
- Transaction cost (basis points)
- Constraint definitions

### What Metadata Is Hidden (Evaluator-only Artifacts)

- True regime labels (`true_regimes.npy`)
- DGP configuration (`dgp_config.json`)
- Test date range and test returns/features
- `metadata_hidden.json`

---

## Evaluation Report & Human Evaluator Instructions

As a human evaluator scoring a given system's performance, follow these steps:

1. **Setup the Challenge:** Generate the public and hidden data using the steps above. Provide ONLY the public data folder to the external system/agent.
2. **Prompt the Agent:** Ask the agent to read `instructions.md` in the public directory and create a strategy script conforming to the expected interface (`fit` and `generate_weights` functions). You may provide them with the base `README.md` to understand constraints and the strategy interface.
3. **Collect Submission:** Obtain the agent's strategy Python file (e.g., `submitted_strategy.py`).
4. **Run the Evaluator:**
   ```bash
   python -m portfolio_eval evaluator \
       --public-data data/public_medium \
       --hidden-data data/hidden_medium \
       --strategy path/to/submitted_strategy.py \
       --out data/results.json
   ```
5. **Safety Check:** The evaluator will automatically check if the `public` package is clean of any hidden data leaks before running.

The evaluator computes and reports:

- **Annualised Sharpe ratio**
- **Total return**
- **Annualised return**
- **Annualised volatility**
- **Maximum drawdown**
- **Average turnover**
- **Transaction cost drag**
- **Constraint violations** (count and details)
- **Runtime** (fit + evaluation)

Results are saved as a JSON file at the specified `--out` path.
