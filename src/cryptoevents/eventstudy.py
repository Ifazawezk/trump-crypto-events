"""Abnormal returns and cumulative abnormal returns (CARs).

`car_table` fits the model in a rolling estimation window for every date at
once using cumulative sums, so placebo tests can draw thousands of random
dates without refitting anything.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DAILY_WINDOWS = {"pre[-5,-1]": (-5, -1), "[0,0]": (0, 0), "[0,+1]": (0, 1),
                 "[-1,+1]": (-1, 1), "[0,+3]": (0, 3)}
DAILY_EST, DAILY_MIN_OBS = (-250, -31), 120
HOURLY_WINDOWS = {"pre[-6h,-1h]": (-6, -1), "[0,+1h]": (0, 1), "[0,+6h]": (0, 6),
                  "[0,+24h]": (0, 24)}
HOURLY_EST, HOURLY_MIN_OBS = (-24 * 30, -24 * 2), 500
TARGETS = ["BTC", "ETH", "XRP", "SOL", "ADA", "DOGE"]


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices).diff()


def market_index(returns: pd.DataFrame, exclude: str | None = None, min_coins: int = 3) -> pd.Series:
    """Equal-weight average log return of the coins trading at that time."""
    cols = [c for c in returns.columns if c != exclude]
    idx = returns[cols].mean(axis=1, skipna=True)
    return idx.where(returns[cols].notna().sum(axis=1) >= min_coins)


def study_series(returns: pd.DataFrame, targets: list[str] = TARGETS) -> dict:
    """name -> (returns, benchmark or None, model).

    MKT is the equal-weight index under a constant-mean model: it measures the
    market-wide move. Each coin gets a market model against the index of the
    other coins (leave-one-out), so its CAR is the coin-specific part."""
    out = {"MKT": (market_index(returns), None, "mean")}
    for coin in targets:
        if coin in returns:
            out[coin] = (returns[coin], market_index(returns, exclude=coin), "market")
    return out


def _csum(a: np.ndarray) -> np.ndarray:
    return np.concatenate([[0.0], np.cumsum(a)])


def car_table(y: pd.Series, x: pd.Series | None, windows: dict, est: tuple[int, int],
              min_obs: int, model: str = "market") -> pd.DataFrame:
    """CAR for every date (as day 0) and every window.

    model="market": AR = r - (alpha + beta * r_m), OLS on the estimation window.
    model="mean":   AR = r - mean(r) over the estimation window.
    A CAR is NaN when the estimation window has fewer than `min_obs` usable
    points or any return inside the event window is missing."""
    T = len(y)
    yv = y.to_numpy(float)
    xv = np.zeros(T) if x is None or model == "mean" else x.reindex(y.index).to_numpy(float)
    valid = np.isfinite(yv) & np.isfinite(xv)
    y0, x0 = np.where(valid, yv, 0.0), np.where(valid, xv, 0.0)
    sums = {"n": _csum(valid.astype(float)), "y": _csum(y0), "x": _csum(x0),
            "xx": _csum(x0 * x0), "xy": _csum(x0 * y0)}
    pos = np.arange(T)

    def wsum(name, lo, hi):
        ok = (lo >= 0) & (hi < T)
        s = sums[name][np.clip(hi, -1, T - 1) + 1] - sums[name][np.clip(lo, 0, T)]
        return np.where(ok, s, np.nan)

    lo, hi = pos + est[0], pos + est[1]
    n, sy, sx = wsum("n", lo, hi), wsum("y", lo, hi), wsum("x", lo, hi)
    with np.errstate(invalid="ignore", divide="ignore"):
        if model == "market":
            den = n * wsum("xx", lo, hi) - sx ** 2
            beta = np.where(den > 0, (n * wsum("xy", lo, hi) - sx * sy) / den, np.nan)
        else:
            beta = np.zeros(T)
        alpha = (sy - beta * sx) / n
    alpha = np.where(n >= min_obs, alpha, np.nan)
    out = {}
    for name, (w0, w1) in windows.items():
        length = w1 - w0 + 1
        nw = wsum("n", pos + w0, pos + w1)
        car = wsum("y", pos + w0, pos + w1) - length * alpha - beta * wsum("x", pos + w0, pos + w1)
        out[name] = np.where(nw == length, car, np.nan)
    return pd.DataFrame(out, index=y.index)


def ar_path(y, x, est, min_obs, model, span: int = 10) -> pd.DataFrame:
    """Single-period ARs for relative days -span..+span, one column each."""
    windows = {k: (k, k) for k in range(-span, span + 1)}
    return car_table(y, x, windows, est, min_obs, model)
