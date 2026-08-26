"""End-to-end single-ticker experiment: features -> train -> predict -> backtest."""

import random
from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .backtest import cost_sensitivity, run_backtest, signals_from_predictions
from .features import (
    FEATURE_COLUMNS,
    WindowDataset,
    add_indicators,
    chronological_split,
    make_windows,
)
from .models import LSTMModel
from .train import get_device, train_model


def run_experiment(
    df: pd.DataFrame,
    window: int = 20,
    epochs: int = 100,
    patience: int = 10,
    lr: float = 1e-3,
    hidden_size: int = 64,
    num_layers: int = 2,
    fees: float = 0.001,
    seed: int = 42,
    device: torch.device = None,
) -> Dict:
    """Run the full pipeline on one OHLCV DataFrame.

    Chronological 70/15/15 split; scaler fit on train only; LSTM predicts
    next-day returns on the test period; long/flat signals backtested with
    vectorbt and compared against buy & hold.

    Returns {"strategy": stats, "buy_and_hold": stats, "cost_sensitivity": df}.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = device or get_device()

    feat = add_indicators(df).dropna()
    X, y = make_windows(feat, FEATURE_COLUMNS, window)
    (X_train, y_train), (X_val, y_val), (X_test, _) = chronological_split(X, y)

    mu = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    sigma = X_train.reshape(-1, X_train.shape[-1]).std(axis=0)
    X_train_s = (X_train - mu) / sigma
    X_val_s = (X_val - mu) / sigma
    X_test_s = (X_test - mu) / sigma

    model = LSTMModel(n_features=len(FEATURE_COLUMNS),
                      hidden_size=hidden_size, num_layers=num_layers)
    train_model(
        model,
        DataLoader(WindowDataset(X_train_s, y_train), batch_size=64, shuffle=True),
        DataLoader(WindowDataset(X_val_s, y_val), batch_size=256),
        epochs=epochs, lr=lr, patience=patience, device=device,
    )

    model.eval()
    with torch.no_grad():
        pred = model(torch.from_numpy(X_test_s).to(device)).cpu().numpy()

    test_close = feat["close"].iloc[-len(X_test):]
    entries, exits = signals_from_predictions(test_close, pred, threshold=0.0)

    all_true = pd.Series(True, index=test_close.index)
    all_false = pd.Series(False, index=test_close.index)

    return {
        "strategy": run_backtest(test_close, entries, exits, fees=fees),
        "buy_and_hold": run_backtest(test_close, all_true, all_false, fees=fees),
        "cost_sensitivity": cost_sensitivity(test_close, entries, exits),
    }
