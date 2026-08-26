# quant-dl

Deep learning for quantitative trading: US stock price prediction with PyTorch, validated with vectorbt backtesting.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Project layout

- `src/quant_dl/` — reusable package: `data`, `features`, `models`, `train.py`, `backtest.py`
- `notebooks/` — experiment walkthroughs: 01_data → 02_features → 03_train → 04_backtest
- `tests/` — pytest unit tests
- `docs/superpowers/specs/` — design documents

## Quick start

See `notebooks/` in order, or run the tests:

```bash
pytest
```
