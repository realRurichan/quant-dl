"""Tests for quant_dl.features: indicators, sliding windows, chronological split."""

import numpy as np
import pandas as pd
import pytest
import torch

from quant_dl.features import (
    FEATURE_COLUMNS,
    WindowDataset,
    add_indicators,
    chronological_split,
    make_windows,
)


def _ohlcv(n: int = 100) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    close = 100 + np.cumsum(np.sin(np.arange(n) / 5.0))
    return pd.DataFrame(
        {
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1000 + np.arange(n),
        },
        index=idx,
    )


def test_add_indicators_adds_expected_columns():
    df = add_indicators(_ohlcv())
    for col in FEATURE_COLUMNS:
        assert col in df.columns


def test_rsi_is_bounded_between_0_and_100():
    df = add_indicators(_ohlcv())
    rsi = df["rsi_14"].dropna()
    assert len(rsi) > 0
    assert rsi.min() >= 0.0
    assert rsi.max() <= 100.0


def test_make_windows_shapes():
    df = add_indicators(_ohlcv()).dropna()
    window = 10
    X, y = make_windows(df, FEATURE_COLUMNS, window)
    assert X.shape == (len(df) - window, window, len(FEATURE_COLUMNS))
    assert y.shape == (len(df) - window,)


def test_make_windows_label_is_next_day_return():
    df = add_indicators(_ohlcv()).dropna()
    window = 10
    X, y = make_windows(df, FEATURE_COLUMNS, window)
    # y[i] must be the return of the day AFTER the last row of X[i]
    expected = df["return_1d"].iloc[window]
    assert y[0] == pytest.approx(expected)


def test_make_windows_uses_only_past_data():
    df = add_indicators(_ohlcv()).dropna()
    window = 10
    X, _ = make_windows(df, FEATURE_COLUMNS, window)
    last_row_in_first_window = X[0, -1, :]
    expected = df[FEATURE_COLUMNS].iloc[window - 1].to_numpy()
    np.testing.assert_allclose(last_row_in_first_window, expected)


def test_chronological_split_preserves_order_and_sizes():
    X = np.arange(100).reshape(100, 1)
    y = np.arange(100)
    (Xtr, ytr), (Xv, yv), (Xte, yte) = chronological_split(X, y, 0.7, 0.15)
    assert len(Xtr) == 70 and len(Xv) == 15 and len(Xte) == 15
    assert Xtr[-1, 0] == 69
    assert Xv[0, 0] == 70
    assert Xte[0, 0] == 85


def test_window_dataset_returns_float_tensors():
    X = np.random.randn(20, 5, 3)
    y = np.random.randn(20)
    ds = WindowDataset(X, y)
    xi, yi = ds[0]
    assert isinstance(xi, torch.Tensor) and xi.dtype == torch.float32
    assert isinstance(yi, torch.Tensor) and yi.dtype == torch.float32
    assert len(ds) == 20
