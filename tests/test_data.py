"""Tests for quant_dl.data: yfinance download with local parquet cache."""

from pathlib import Path

import pandas as pd
import pytest

from quant_dl.data import download


def _fake_df() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": range(100, 105),
            "High": range(101, 106),
            "Low": range(99, 104),
            "Close": range(100, 105),
            "Volume": range(1000, 1005),
        },
        index=idx,
    )


def test_download_normalizes_columns_to_lowercase(tmp_path: Path):
    df = download("FAKE", "2024-01-01", "2024-01-06", cache_dir=tmp_path,
                  fetcher=lambda *a, **k: _fake_df())
    assert list(df.columns) == ["open", "high", "low", "close", "volume"]


def test_download_caches_to_parquet(tmp_path: Path):
    download("FAKE", "2024-01-01", "2024-01-06", cache_dir=tmp_path,
             fetcher=lambda *a, **k: _fake_df())
    assert (tmp_path / "FAKE.parquet").exists()


def test_download_uses_cache_without_fetching(tmp_path: Path):
    download("FAKE", "2024-01-01", "2024-01-06", cache_dir=tmp_path,
             fetcher=lambda *a, **k: _fake_df())

    def exploding_fetcher(*a, **k):
        raise AssertionError("fetcher must not be called when cache exists")

    df = download("FAKE", "2024-01-01", "2024-01-06", cache_dir=tmp_path,
                  fetcher=exploding_fetcher)
    assert len(df) == 5


def test_download_refetches_when_forced(tmp_path: Path):
    download("FAKE", "2024-01-01", "2024-01-06", cache_dir=tmp_path,
             fetcher=lambda *a, **k: _fake_df())
    calls = {"n": 0}

    def counting_fetcher(*a, **k):
        calls["n"] += 1
        return _fake_df()

    download("FAKE", "2024-01-01", "2024-01-06", cache_dir=tmp_path,
             fetcher=counting_fetcher, force=True)
    assert calls["n"] == 1


def test_download_rejects_empty_result(tmp_path: Path):
    with pytest.raises(ValueError, match="empty"):
        download("FAKE", "2024-01-01", "2024-01-06", cache_dir=tmp_path,
                 fetcher=lambda *a, **k: pd.DataFrame())
