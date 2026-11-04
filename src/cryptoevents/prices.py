"""Hourly spot prices from Binance's public data dumps (data.binance.vision).

Monthly 1h kline zips are downloaded once, checked against the published
SHA-256 file and cached under cache/ (gitignored). The dataset is licensed
CC BY-NC-SA 4.0, so raw files are never committed.
"""
from __future__ import annotations

import hashlib
import io
import zipfile
from concurrent.futures import ThreadPoolExecutor

import pandas as pd

from .http import Client

BASE = "https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{y}-{m:02d}.zip"
COINS = ["BTC", "ETH", "BNB", "XRP", "ADA", "SOL", "DOGE", "LTC", "LINK", "TRX"]


def months(start: str, end: str) -> list[tuple[int, int]]:
    return [(p.year, p.month) for p in pd.period_range(start, end, freq="M")]


def verify_checksum(body: bytes, checksum_file: bytes) -> bool:
    expected = checksum_file.decode().split()[0].strip().lower()
    return hashlib.sha256(body).hexdigest() == expected


def parse_klines(body: bytes) -> pd.DataFrame:
    """Open time and close price from one kline zip.

    Binance switched spot timestamps from milliseconds to microseconds on
    2025-01-01, and a few files carry a header row; both are handled here."""
    with zipfile.ZipFile(io.BytesIO(body)) as zf:
        raw = zf.read(zf.namelist()[0])
    df = pd.read_csv(io.BytesIO(raw), header=None, usecols=[0, 4], names=["open_time", "close"])
    df = df[pd.to_numeric(df["open_time"], errors="coerce").notna()]
    t = df["open_time"].astype("int64")
    unit = "us" if t.max() > 10**14 else "ms"
    return pd.DataFrame({"time": pd.to_datetime(t, unit=unit),
                         "close": df["close"].astype(float).to_numpy()})


def fetch_symbol(client: Client, coin: str, start: str, end: str) -> pd.Series:
    sym = f"{coin}USDT"
    frames = []
    for y, m in months(start, end):
        url = BASE.format(s=sym, y=y, m=m)
        body = client.get(url, key=f"binance/{sym}/{y}-{m:02d}.zip", expect="zip")
        if body is None:  # not listed yet in that month
            continue
        check = client.get(url + ".CHECKSUM", key=f"binance/{sym}/{y}-{m:02d}.zip.CHECKSUM",
                           expect="text")
        if check is not None and not verify_checksum(body, check):
            raise ValueError(f"checksum mismatch for {url}")
        frames.append(parse_klines(body))
    if not frames:
        return pd.Series(dtype=float, name=coin)
    df = pd.concat(frames).drop_duplicates("time").set_index("time").sort_index()
    return df["close"].rename(coin)


def hourly_closes(client: Client, coins: list[str], start: str, end: str,
                  workers: int = 4) -> pd.DataFrame:
    """Wide frame of hourly closes, one column per coin, UTC hour index.
    A few coins download side by side; the client still spaces out requests."""
    with ThreadPoolExecutor(workers) as pool:
        series = list(pool.map(lambda c: fetch_symbol(client, c, start, end), coins))
    data = pd.concat(series, axis=1)
    full = pd.date_range(data.index.min(), data.index.max(), freq="h")
    return data.reindex(full)


def daily_closes(hourly: pd.DataFrame) -> pd.DataFrame:
    """Close of the 23:00 UTC bar, i.e. the price at midnight UTC."""
    return hourly.resample("D").last()
