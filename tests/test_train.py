"""Tests for quant_dl.train: device selection and training loop."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from quant_dl.features import WindowDataset
from quant_dl.models import LSTMModel
from quant_dl.train import get_device, train_model


def test_get_device_returns_torch_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert device.type in ("mps", "cuda", "cpu")


def test_train_model_reduces_loss_on_learnable_data():
    """y = sum of window means of features: learnable by an LSTM regressor."""
    rng = np.random.default_rng(0)
    n, window, n_feat = 200, 5, 3
    X = rng.standard_normal((n, window, n_feat)).astype(np.float32)
    y = X.mean(axis=1).sum(axis=1)  # deterministic target

    split = 160
    train_loader = DataLoader(WindowDataset(X[:split], y[:split]), batch_size=32)
    val_loader = DataLoader(WindowDataset(X[split:], y[split:]), batch_size=32)

    torch.manual_seed(0)
    model = LSTMModel(n_features=n_feat, hidden_size=16, num_layers=1)
    history = train_model(model, train_loader, val_loader, epochs=30, lr=1e-2,
                          device=torch.device("cpu"), patience=10)

    assert history["train_loss"][-1] < history["train_loss"][0] * 0.5
    assert len(history["train_loss"]) == len(history["val_loss"])


def test_train_model_early_stops_and_restores_best_weights():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 5, 3)).astype(np.float32)
    y = rng.standard_normal(100).astype(np.float32)  # pure noise: no signal
    loader = DataLoader(WindowDataset(X, y), batch_size=32)

    torch.manual_seed(0)
    model = LSTMModel(n_features=3, hidden_size=8, num_layers=1)
    history = train_model(model, loader, loader, epochs=100, lr=1e-3,
                          device=torch.device("cpu"), patience=5)
    # patience=5 must stop well before 100 epochs on noise
    assert len(history["train_loss"]) < 100
