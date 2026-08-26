"""Training loop with device selection, MSE loss, early stopping."""

import copy
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def get_device() -> torch.device:
    """Pick the best available device: MPS (Apple GPU) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _epoch_loss(model, loader, device, optimizer=None) -> float:
    training = optimizer is not None
    model.train(training)
    total, count = 0.0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = nn.functional.mse_loss(model(xb), yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(xb)
            count += len(xb)
    return total / count


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 100,
    lr: float = 1e-3,
    device: torch.device = None,
    patience: int = 10,
) -> Dict[str, List[float]]:
    """Train with Adam + MSE; early-stop on val loss and restore best weights.

    Returns a history dict with per-epoch "train_loss" and "val_loss".
    """
    device = device or get_device()
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for _ in range(epochs):
        train_loss = _epoch_loss(model, train_loader, device, optimizer)
        val_loss = _epoch_loss(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    return history
