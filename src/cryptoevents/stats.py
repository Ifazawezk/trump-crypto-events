"""Placebo tests, bootstrap intervals and Benjamini-Hochberg correction."""
from __future__ import annotations

import numpy as np
import pandas as pd


def bootstrap_mean_ci(values, n_boot: int = 10_000, level: float = 0.95,
                      rng: np.random.Generator | None = None) -> tuple[float, float]:
    """Percentile interval for the mean, resampling events with replacement."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = rng or np.random.default_rng(0)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    tail = (1 - level) / 2 * 100
    return tuple(np.percentile(means, [tail, 100 - tail]))


def benjamini_hochberg(pvals) -> np.ndarray:
    """BH adjusted p-values (q-values); NaNs stay NaN."""
    p = np.asarray(pvals, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return q
    ps = p[ok]
    order = np.argsort(ps)
    ranked = ps[order] * m / np.arange(1, m + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(ranked, 1.0)
    q[ok] = out
    return q


def placebo_pool(index: pd.DatetimeIndex, values: np.ndarray, event_times,
                 buffer: pd.Timedelta) -> np.ndarray:
    """Positions usable as placebo dates: finite CAR, not near a real event."""
    ok = np.isfinite(values)
    for t in event_times:
        ok &= ~((index >= t - buffer) & (index <= t + buffer))
    return np.flatnonzero(ok)


def candidates(index: pd.DatetimeIndex, pool: np.ndarray, t: pd.Timestamp, match: str,
               radius: pd.Timedelta | None) -> np.ndarray:
    """Placebo candidates for one event. match: none | weekday | weekday_hour.
    `radius` keeps draws within that distance of the event so the volatility
    regime is roughly the same."""
    times = index[pool]
    keep = np.ones(len(pool), bool)
    if match in ("weekday", "weekday_hour"):
        keep &= times.day_of_week == t.day_of_week
    if match == "weekday_hour":
        keep &= times.hour == t.hour
    if radius is not None:
        keep &= np.abs(times - t) <= radius
    return pool[keep]


def placebo_test(car: pd.Series, event_times, n_sets: int, rng: np.random.Generator,
                 match: str = "none", radius: pd.Timedelta | None = None,
                 buffer: pd.Timedelta = pd.Timedelta(days=10)) -> dict:
    """Compare the mean CAR of the real events with the mean CAR of `n_sets`
    random sets of dates of the same size. Two-sided p-value around the
    placebo mean."""
    values = car.to_numpy(float)
    pos = [car.index.get_loc(t) for t in event_times if t in car.index]
    obs = values[pos]
    keep = np.isfinite(obs)
    times = [t for t, k in zip([car.index[p] for p in pos], keep) if k]
    obs = obs[keep]
    if len(obs) == 0:
        return {"n": 0, "mean": np.nan, "p": np.nan, "null_sd": np.nan}
    pool = placebo_pool(car.index, values, event_times, buffer)
    draws = np.column_stack([rng.choice(candidates(car.index, pool, t, match, radius), n_sets)
                             for t in times])
    null = values[draws].mean(axis=1)
    centre = null.mean()
    p = (1 + np.sum(np.abs(null - centre) >= abs(obs.mean() - centre))) / (n_sets + 1)
    return {"n": len(obs), "mean": obs.mean(), "p": p, "null_sd": null.std(ddof=1),
            "null_lo": np.percentile(null, 2.5), "null_hi": np.percentile(null, 97.5)}


def single_event_test(car: pd.Series, t: pd.Timestamp, event_times, match: str,
                      radius: pd.Timedelta | None,
                      buffer: pd.Timedelta = pd.Timedelta(days=10)) -> dict:
    """Where one event's CAR sits among matched placebo dates."""
    values = car.to_numpy(float)
    obs = car.get(t, np.nan)
    if not np.isfinite(obs):
        return {"car": np.nan, "p": np.nan, "null_lo": np.nan, "null_hi": np.nan}
    pool = placebo_pool(car.index, values, event_times, buffer)
    null = values[candidates(car.index, pool, t, match, radius)]
    centre = null.mean()
    p = (1 + np.sum(np.abs(null - centre) >= abs(obs - centre))) / (len(null) + 1)
    return {"car": obs, "p": p, "null_lo": np.percentile(null, 2.5),
            "null_hi": np.percentile(null, 97.5)}
