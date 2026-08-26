"""Tests for quant_dl.pipeline: single-ticker end-to-end experiment."""

import numpy as np
import pandas as pd
import torch

from quant_dl.pipeline import run_experiment


def _synthetic_ohlcv(n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    return pd.DataFrame(
        {
            "open": close * (1 - 0.002),
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.integers(1_000, 10_000, n),
        },
        index=idx,
    )


def test_run_experiment_returns_strategy_and_benchmark_stats():
    result = run_experiment(_synthetic_ohlcv(), window=10, epochs=3, seed=0,
                            device=torch.device("cpu"))
    assert set(result) >= {"strategy", "buy_and_hold", "cost_sensitivity"}
    for key in ("total_return", "sharpe_ratio", "max_drawdown", "n_trades"):
        assert np.isfinite(result["strategy"][key])
        assert np.isfinite(result["buy_and_hold"][key])
    # buy & hold benchmark never trades more than once
    assert result["buy_and_hold"]["n_trades"] <= 1


def test_run_experiment_is_deterministic_with_seed():
    r1 = run_experiment(_synthetic_ohlcv(), window=10, epochs=3, seed=42,
                        device=torch.device("cpu"))
    r2 = run_experiment(_synthetic_ohlcv(), window=10, epochs=3, seed=42,
                        device=torch.device("cpu"))
    assert r1["strategy"]["total_return"] == r2["strategy"]["total_return"]
