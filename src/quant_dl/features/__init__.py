"""Feature engineering: technical indicators, sliding windows, chronological splits."""

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

FEATURE_COLUMNS: List[str] = [
    "return_1d",
    "rsi_14",
    "macd",
    "macd_signal",
    "ma_ratio_10",
    "ma_ratio_20",
    "volatility_10",
]


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicator columns to an OHLCV DataFrame.

    All indicators at row t use only data up to and including day t (no look-ahead).
    Warmup rows contain NaN — drop them before building windows.
    """
    out = df.copy()
    close = out["close"]

    out["return_1d"] = close.pct_change()

    # RSI(14) with Wilder smoothing
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rs = gain / loss
    out["rsi_14"] = 100 - 100 / (1 + rs)

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_signal"] = out["macd"].ewm(span=9, adjust=False).mean()

    # Price relative to moving averages
    out["ma_ratio_10"] = close / close.rolling(10).mean() - 1
    out["ma_ratio_20"] = close / close.rolling(20).mean() - 1

    # 10-day volatility of daily returns
    out["volatility_10"] = out["return_1d"].rolling(10).std()

    return out


def make_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding-window samples.

    X[i] = feature rows [i, i+window); y[i] = return_1d at row i+window,
    i.e. the return realized the day AFTER the window ends (no look-ahead).

    Returns X of shape [n_samples, window, n_features] and y of shape [n_samples].
    """
    features = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df["return_1d"].to_numpy(dtype=np.float32)

    n_samples = len(df) - window
    X = np.stack([features[i : i + window] for i in range(n_samples)])
    y = targets[window : window + n_samples]
    return X, y


def chronological_split(X, y, train_frac: float = 0.7, val_frac: float = 0.15):
    """Split into (train, val, test) keeping time order. Never shuffle."""
    n = len(X)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return (
        (X[:n_train], y[:n_train]),
        (X[n_train : n_train + n_val], y[n_train : n_train + n_val]),
        (X[n_train + n_val :], y[n_train + n_val :]),
    )


class WindowDataset(Dataset):
    """Torch Dataset over sliding-window arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32))
        self.y = torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.y[i]
