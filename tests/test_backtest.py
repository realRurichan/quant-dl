"""Tests for quant_dl.backtest: prediction signals -> vectorbt portfolio stats."""

import numpy as np
import pandas as pd

from quant_dl.backtest import cost_sensitivity, run_backtest, signals_from_predictions


def _close(prices) -> pd.Series:
    idx = pd.date_range("2024-01-01", periods=len(prices), freq="D")
    return pd.Series(prices, index=idx, name="close")


def test_signals_long_when_prediction_above_threshold():
    close = _close([100, 101, 102, 103, 104])
    pred = np.array([0.01, -0.01, 0.02, -0.02, 0.03])
    entries, exits = signals_from_predictions(close, pred, threshold=0.0)
    assert entries.tolist() == [True, False, True, False, True]
    assert exits.tolist() == [False, True, False, True, False]
    assert entries.index.equals(close.index)


def test_run_backtest_profits_from_perfect_forecast_on_uptrend():
    prices = 100 * (1.01 ** np.arange(50))  # steady +1% daily
    close = _close(prices)
    pred = np.full(50, 0.01)  # always predict up -> always long
    entries, exits = signals_from_predictions(close, pred, threshold=0.0)
    stats = run_backtest(close, entries, exits)
    assert stats["total_return"] > 0.4  # ~63% buy-and-hold, minus fees
    assert set(stats) >= {"total_return", "sharpe_ratio", "max_drawdown", "n_trades"}


def test_run_backtest_loses_nothing_when_flat():
    prices = 100 * (1.01 ** np.arange(50))
    close = _close(prices)
    pred = np.full(50, -0.01)  # always predict down -> never enter
    entries, exits = signals_from_predictions(close, pred, threshold=0.0)
    stats = run_backtest(close, entries, exits)
    assert stats["total_return"] == 0.0
    assert stats["n_trades"] == 0


def test_run_backtest_reports_drawdown_as_negative():
    prices = np.concatenate([100 * (1.02 ** np.arange(25)),
                             100 * 1.02**24 * (0.98 ** np.arange(1, 26))])
    close = _close(prices)
    pred = np.full(50, 0.01)
    entries, exits = signals_from_predictions(close, pred, threshold=0.0)
    stats = run_backtest(close, entries, exits)
    assert stats["max_drawdown"] < 0


def test_run_backtest_reports_extended_metrics():
    prices = 100 * (1.01 ** np.arange(50))  # steady +1% daily uptrend
    close = _close(prices)
    pred = np.full(50, 0.01)
    entries, exits = signals_from_predictions(close, pred, threshold=0.0)
    stats = run_backtest(close, entries, exits)
    assert set(stats) >= {"total_return", "sharpe_ratio", "max_drawdown", "n_trades",
                          "annualized_return", "annualized_volatility", "win_rate"}
    # 50 trading days << 1 year, so annualized return exceeds total return
    assert stats["annualized_return"] > stats["total_return"]
    assert stats["annualized_volatility"] >= 0
    # every 1-day trade on a steady uptrend is a win
    assert stats["win_rate"] == 1.0


def test_cost_sensitivity_decreases_with_fees():
    prices = 100 * (1.005 ** np.arange(100))
    close = _close(prices)
    # alternate in/out daily to generate many trades
    pred = np.tile([0.01, -0.01], 50)
    entries, exits = signals_from_predictions(close, pred, threshold=0.0)
    table = cost_sensitivity(close, entries, exits, fee_levels=[0.0, 0.0005, 0.001, 0.002])
    assert list(table.columns) == ["fee", "total_return", "sharpe_ratio", "n_trades"]
    returns = table["total_return"].tolist()
    assert returns == sorted(returns, reverse=True)  # non-increasing in fees
