# Dataset Generation and Variants

## Dataset Generation

Synthetic returns are generated via a factor model:

```
r_i(t) = β_market * f_market(t) + β_sector * f_sector(t) + ε_i(t) + signal(t)
```

**Hidden regimes** (momentum, low-volatility, mean-reversion, noisy/stress)
control signal strengths via a Markov chain.

### Features

| Feature            | Description                         | Predictive? |
|--------------------|-------------------------------------|-------------|
| `mom_21d`          | 21-day momentum                     | In momentum regime |
| `mom_63d`          | 63-day momentum                     | In momentum regime |
| `mom_126d`         | 126-day momentum                    | In momentum regime |
| `vol_21d`          | 21-day volatility                   | In low-vol regime  |
| `vol_63d`          | 63-day volatility                   | In low-vol regime  |
| `rev_5d`           | 5-day reversal                      | In mean-reversion regime |
| `sector_mom_63d`   | Sector 63-day momentum              | Partially   |
| `drawdown_63d`     | 63-day drawdown                     | Partially   |
| `macro_regime_proxy` | Noisy regime indicator            | Weak signal |
| `noise_1/2/3`      | Pure random noise                   | ❌ Never    |

---

## Dataset Variants

| Config               | Difficulty  | Trap                                      |
|----------------------|-------------|-------------------------------------------|
| `dataset_easy.yaml`  | Easy        | None — validation ≈ test regimes          |
| `dataset_medium.yaml`| Medium      | Mild regime shift                         |
| `dataset_hard.yaml`  | Hard        | Validation favours momentum, test favours mean-reversion |
| `dataset_noise_trap.yaml` | Trap   | No robust improvement over equal-weight   |
| `dataset_turnover_trap.yaml` | Trap | Reversal works before costs, fails after  |
