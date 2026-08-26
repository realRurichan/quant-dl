"""Backtesting: prediction signals -> vectorbt portfolio -> performance stats."""

from typing import Dict, Tuple

import numpy as np
import pandas as pd
import vectorbt as vbt


def signals_from_predictions(
    close: pd.Series,
    predictions: np.ndarray,
    threshold: float = 0.0,
) -> Tuple[pd.Series, pd.Series]:
    """Convert return predictions to long/flat entry-exit signals.

    Long when predicted return > threshold, flat otherwise.
    Returns boolean (entries, exits) aligned with `close`'s index.
    """
    long_mask = pd.Series(predictions > threshold, index=close.index)
    return long_mask, ~long_mask


def run_backtest(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    fees: float = 0.001,
) -> Dict[str, float]:
    """Run a vectorbt long-only portfolio and return key performance stats."""
    pf = vbt.Portfolio.from_signals(
        close, entries, exits, fees=fees, freq="1D"
    )
    return {
        "total_return": float(pf.total_return()),
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "n_trades": int(pf.trades.count()),
    }
