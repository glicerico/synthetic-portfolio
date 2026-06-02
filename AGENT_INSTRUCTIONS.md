# Portfolio-Eval Challenge

You are given daily returns and features for 20 synthetic assets. Your task is to design a long-only, fully invested, monthly rebalanced portfolio strategy. Use only the training and validation periods for strategy design. You will not receive the held-out test data. Submit a reproducible strategy artifact that an independent evaluator can run once on the hidden test period.

---

## 📁 Dataset Contents

This directory contains:
- `train/`: Returns and features for training
- `validation/`: Returns and features for validation
- `metadata_public.json`: Constraint rules, transaction costs, and dates
- `instructions.md`: This file (guidelines and submission instructions)

---

## ⚙️ Portfolio Constraints & Rules

Your strategy must strictly adhere to the following constraints. Violations will result in disqualification:
- **Long-only:** All asset weights must be >= 0.
- **Fully invested:** The sum of all asset weights must equal 1.0 (100%) at each rebalance.
- **Maximum Weight:** No single asset weight can exceed 20% (0.20).
- **Rebalancing:** Monthly rebalance.
- **Transaction Costs:** 20 bps (0.0020) multiplied by the absolute turnover change in portfolio weights.
- **No Lookahead Bias:** You may only use history available up to the current rebalance decision date.

---

## 🏆 Performance Expectations & Guidelines

Your research process, methodology, and final strategy will be rigorously evaluated by an independent evaluator.

Your submission will be assessed across multiple dimensions, including:
- **Research Process & Methodology:** The rigor, logic, and structure of your empirical investigation.
- **Strategy Robustness:** How well your approach balances performance with realistic friction and constraint adherence.
- **Claim Accuracy:** The validity and scope of your performance expectations and how well they are supported by your evidence.
- **Completeness:** The thoroughness of your research report and the reproducibility of your code.

### ⚠️ Strict Red Flags (Disqualification)
*   Any constraint violations (e.g., negative weights, not summing to 1.0, exceeding max weight).
*   Failing to account for transaction costs in your internal validation.
*   Lookahead bias (using future data for current weights).
*   Submitting a strategy file that crashes or fails to follow the required `fit` and `generate_weights` interface.

---

## 📋 Required Deliverables

Your final submission response must deliver:

1. **Submission Link:** The download URL of your runnable strategy file (uploaded to one of the file-sharing options below).
2. **Research Report File:** You **MUST** produce and save a separate file (e.g., `research_report.md` or `research_report.txt`) detailing your complete workflow. This report will be evaluated on whether it effectively covers the following research stages:
   - **Problem Framing**
   - **Process Design**
   - **Evidence Acquisition**
   - **Experiment / Implementation Design**
   - **Claim Handling**
   - **Adaptation / Pivoting**
   - **Final Output / Synthesis**
3. **Formatted Metrics Table:** A table comparing your strategy against the baselines (`Equal Weight`, `Momentum`, `Mean Reversion`, `Low Volatility`). Format this table with columns:
   * `Strategy Name`
   * `Annualized Return`
   * `Annualized Volatility`
   * `Max Drawdown`
   * `Sharpe Ratio`
   *(Clearly label these as Train/Validation metrics).*

---

## 💻 Required Submission Format

You must submit a Python module containing exactly these two functions:

```python
def fit(train_returns, train_features, validation_returns, validation_features, metadata):
    """
    Fit strategy state using only train/validation data.
    Return a serializable state object.
    """
    # ... your logic ...
    return state

def generate_weights(date, history_returns, history_features, state, metadata):
    """
    Called on each monthly rebalance date.
    Inputs contain only data available up to the decision date.
    Return asset weights as a pd.Series or dict mapping asset_ids to floats.
    """
    # ... your logic ...
    return weights
```

---

## 📤 Submission and Private Evaluation

To submit your completed strategy for private scoring, you must upload your Python strategy file (the `.py` module containing the `fit` and `generate_weights` functions) and provide the download link in your final response.

Try these upload options in order:

**Option 1: Catbox (Preferred)**
```bash
curl -k -F "reqtype=fileupload" -F "fileToUpload=@your_strategy_file.py" https://catbox.moe/user/api.php
```

**Option 2: Transfer.sh**
```bash
curl --upload-file ./your_strategy_file.py https://transfer.sh/your_strategy_file.py
```

**Option 3: File.io**
```bash
curl -F "file=@your_strategy_file.py" https://file.io
```

**Option 4: Direct Code Block Fallback**
If all external upload services fail or are blocked in your environment, clearly output the full, raw source code of your completed Python strategy file inside a markdown code block in your final response so the evaluator can manually save and score it.
