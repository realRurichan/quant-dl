# Quant-DL Design Document

Date: 2026-08-26
Status: Approved (approach A)

## Goal

A deep-learning quantitative trading research project: predict US stock price
movements with PyTorch models and validate strategies with vectorized backtesting.

## Decisions

| Topic | Decision |
|-------|----------|
| Objective | Price prediction + backtest |
| Market | US stocks via yfinance (daily bars) |
| DL framework | PyTorch (MPS acceleration on Apple Silicon) |
| Backtest | vectorbt (vectorized, pandas-native) |
| Structure | Reusable package (`src/quant_dl/`) + notebooks for experiments |

## Architecture

```
quant-dl/
├── src/quant_dl/
│   ├── data/        # yfinance download + local parquet cache
│   ├── features/    # technical indicators, normalization, sliding-window dataset
│   ├── models/      # LSTM baseline (extensible to Transformer later)
│   ├── train.py     # training loop with MPS/CUDA/CPU device selection
│   └── backtest.py  # prediction signals -> vectorbt backtest + metrics
├── notebooks/       # 01_data -> 02_features -> 03_train -> 04_backtest
├── tests/           # pytest unit tests per module
└── docs/superpowers/specs/
```

## Data flow

1. `data.download(tickers, start, end)` -> raw OHLCV DataFrame, cached to `data/raw/*.parquet`
2. `features.build(df)` -> feature matrix (returns, RSI, MACD, moving averages, volatility)
3. `features.make_windows(df, window)` -> sliding-window `Dataset` (X: [window, n_features], y: next-day return direction or value)
4. `train.py` -> chronological train/val/test split, standardization fit on train only, Adam + early stopping
5. `backtest.py` -> model predictions -> long/flat signals -> vectorbt portfolio -> Sharpe, max drawdown, total return vs buy-and-hold

## Key correctness rules

- **No look-ahead**: all features use only data up to day t; labels use day t+1.
- **Chronological splits only** — never shuffle time series into train/test.
- **Scaler fit on train split only**, applied to val/test.
- Baseline first: LSTM must beat naive buy-and-hold comparison before any fancier model.

## Testing

- pytest unit tests per module (data caching, feature shapes, window alignment, model forward pass shapes, backtest signal conversion).
- End-to-end smoke run on a single ticker (e.g. AAPL) as final validation.
