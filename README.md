# quant-dl

[![CI](https://github.com/realRurichan/quant-dl/actions/workflows/ci.yml/badge.svg)](https://github.com/realRurichan/quant-dl/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Deep-learning equity timing research pipeline: an LSTM predicts next-day returns
from technical indicators, and the signal is validated with vectorized backtesting
on held-out data — with strict leakage controls and transaction-cost analysis.

> **Disclaimer**: research/educational project, not investment advice.

## Highlights

- **Leakage-safe by construction** — indicators use only data up to day *t*,
  labels are day *t+1* returns, splits are chronological (never shuffled), and the
  feature scaler is fit on the training split only.
- **Reusable, tested pipeline** — `data → features → models → train → backtest → pipeline`
  as independent modules with 26 pytest unit tests and CI on every push.
- **Honest evaluation** — held-out test period, buy & hold benchmark, and
  transaction-cost sensitivity; results below are reported as-is, including
  where the strategy loses.

## Results

Cross-sectional evaluation on 8 large-cap US stocks, 2018–2025 daily data
(test period = last 15% chronologically, 0.1% fees per trade, seed fixed).
Full reproduction: [`notebooks/05_cross_section.ipynb`](notebooks/05_cross_section.ipynb).

| ticker | strat return | strat ann. | strat Sharpe | strat max DD | win rate | trades | B&H return | B&H Sharpe | B&H max DD |
|:-------|:-------------|:-----------|-------------:|:-------------|:---------|-------:|:-----------|-----------:|:-----------|
| AAPL   | 3.8%         | 5.4%       | 0.38         | -17.0%       | 46.2%    | 13     | 29.0%      | 1.47       | -15.4%     |
| MSFT   | 6.4%         | 9.1%       | 0.60         | -9.7%        | 61.9%    | 21     | 14.5%      | 0.92       | -15.5%     |
| GOOGL  | 23.8%        | 35.1%      | 1.44         | -12.9%       | 66.7%    | 12     | 37.2%      | 1.50       | -22.1%     |
| AMZN   | 46.3%        | 71.0%      | 1.81         | -19.2%       | 72.7%    | 11     | 44.1%      | 1.71       | -19.5%     |
| NVDA   | 234.4%       | 448.1%     | 3.48         | -22.2%       | 80.0%    | 10     | 178.9%     | 2.63       | -27.0%     |
| META   | 3.6%         | 5.1%       | 0.32         | -23.1%       | 60.0%    | 10     | 68.1%      | 1.90       | -18.4%     |
| JPM    | 34.0%        | 51.0%      | 1.72         | -10.1%       | 83.3%    | 12     | 47.2%      | 2.09       | -10.1%     |
| XOM    | 7.0%         | 10.1%      | 0.79         | -10.8%       | 55.6%    | 9      | 9.7%       | 0.69       | -15.1%     |
| **median** | **15.4%** | **22.6%** | **1.12** | **-15.0%** | **64.3%** | —  | **40.6%**  | **1.61**   | **-17.0%** |

**Read of the results.** The strategy beats buy & hold on 4 of 8 names (MSFT,
AMZN, NVDA, XOM) with better drawdown on 6 of 8, but trails on median total
return because it sits flat ~50% of days — it is a timing overlay, not a
return maximizer. Single-name daily-return prediction has thin edge; the
pipeline is built so that cross-sectional training (many tickers at once) is
the natural next step.

**Transaction-cost sensitivity** (AAPL; the strategy trades ~1×/month):

| fee per trade | 0.00% | 0.05% | 0.10% | 0.20% |
|:--------------|:------|:------|:------|:------|
| total return  | 6.6%  | 5.2%  | 3.8%  | 1.2%  |
| Sharpe        | 0.59  | 0.48  | 0.38  | 0.18  |

## Architecture

```
yfinance ──▶ data/         download + parquet cache
                │
                ▼
           features/       RSI, MACD, MA ratios, volatility (t-day only)
                │          sliding windows [window × features] → next-day return
                ▼
           models/         LSTM regressor (PyTorch, MPS/CUDA/CPU)
                │
                ▼
           train.py        Adam + MSE, early stopping, best-weight restore
                │
                ▼
           backtest.py     long/flat signals → vectorbt → Sharpe / max DD /
                           win rate / cost sensitivity
                │
                ▼
           pipeline.py     one-call single-ticker experiment (used by nb 05)
```

## Reproduce

```bash
git clone https://github.com/realRurichan/quant-dl.git
cd quant-dl
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest                                   # 26 unit tests
jupyter notebook notebooks/              # run 01 → 05 in order
```

## Layout

- `src/quant_dl/` — the package: `data`, `features`, `models`, `train.py`, `backtest.py`, `pipeline.py`
- `notebooks/` — 01 data · 02 features · 03 train · 04 backtest · 05 cross-section
- `tests/` — unit tests per module, incl. no-look-ahead and early-stopping behavior
- `docs/superpowers/specs/` — design document

## Roadmap

- [ ] Cross-sectional training (all tickers in one model)
- [ ] Transformer encoder baseline vs LSTM
- [ ] Classification target (up/down) with precision/recall reporting
- [ ] Signal threshold tuning on the validation split
