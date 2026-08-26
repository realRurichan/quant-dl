"""Tests for quant_dl.models: LSTM regressor for next-day return."""

import torch

from quant_dl.models import LSTMModel


def test_lstm_forward_output_shape():
    model = LSTMModel(n_features=7, hidden_size=32, num_layers=2)
    x = torch.randn(16, 10, 7)  # [batch, window, features]
    out = model(x)
    assert out.shape == (16,)


def test_lstm_is_deterministic_in_eval_mode():
    torch.manual_seed(0)
    model = LSTMModel(n_features=3, hidden_size=8, num_layers=1, dropout=0.0)
    model.eval()
    x = torch.randn(4, 5, 3)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
    assert torch.equal(out1, out2)


def test_lstm_has_trainable_parameters():
    model = LSTMModel(n_features=7, hidden_size=32, num_layers=2)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params > 0
