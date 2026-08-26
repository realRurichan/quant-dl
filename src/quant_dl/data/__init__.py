"""Data layer: download OHLCV from Yahoo Finance with a local parquet cache."""

from pathlib import Path
from typing import Callable, Union

import pandas as pd
import yfinance as yf

DEFAULT_CACHE_DIR = Path("data/raw")


def _default_fetcher(ticker: str, start: str, end: str) -> pd.DataFrame:
    return yf.download(ticker, start=start, end=end, auto_adjust=True)


def download(
    ticker: str,
    start: str,
    end: str,
    cache_dir: Union[str, Path] = DEFAULT_CACHE_DIR,
    fetcher: Callable[..., pd.DataFrame] = _default_fetcher,
    force: bool = False,
) -> pd.DataFrame:
    """Return OHLCV for `ticker` with lowercase columns, indexed by date.

    Results are cached to `<cache_dir>/<ticker>.parquet`; subsequent calls
    read the cache unless `force=True`.
    """
    cache_dir = Path(cache_dir)
    cache_path = cache_dir / f"{ticker}.parquet"

    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    df = fetcher(ticker, start, end)
    if df is None or len(df) == 0:
        raise ValueError(f"empty data for {ticker} ({start} -> {end})")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)[["open", "high", "low", "close", "volume"]]

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path)
    return df
