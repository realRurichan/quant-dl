"""Backtesting: prediction signals -> vectorbt portfolio -> performance stats."""

from typing import Dict, List, Tuple

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
    n_trades = int(pf.trades.count())
    return {
        "total_return": float(pf.total_return()),
        "annualized_return": float(pf.annualized_return()),
        "annualized_volatility": float(pf.annualized_volatility()),
        "sharpe_ratio": float(pf.sharpe_ratio()),
        "max_drawdown": float(pf.max_drawdown()),
        "win_rate": float(pf.trades.win_rate()) if n_trades > 0 else 0.0,
        "n_trades": n_trades,
    }


def cost_sensitivity(
    close: pd.Series,
    entries: pd.Series,
    exits: pd.Series,
    fee_levels: List[float] = (0.0, 0.0005, 0.001, 0.002),
) -> pd.DataFrame:
    """Re-run the backtest at several fee levels to show cost drag.

    Returns a DataFrame with one row per fee level.
    """
    rows = []
    for fee in fee_levels:
        stats = run_backtest(close, entries, exits, fees=fee)
        rows.append({
            "fee": fee,
            "total_return": stats["total_return"],
            "sharpe_ratio": stats["sharpe_ratio"],
            "n_trades": stats["n_trades"],
        })
    return pd.DataFrame(rows)
