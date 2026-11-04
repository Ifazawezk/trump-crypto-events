"""Run the price event study on the event list and return result tables."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import eventstudy as es
from .events import event_day, event_hour
from .prices import daily_closes
from .stats import benjamini_hochberg, bootstrap_mean_ci, placebo_test, single_event_test

YEAR = pd.Timedelta(days=365)


def prepare_events(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.copy()
    ev["day"] = [event_day(t, p) for t, p in zip(ev["timestamp"], ev["time_precision"])]
    ev["hour"] = [event_hour(t) if p == "minute" else pd.NaT
                  for t, p in zip(ev["timestamp"], ev["time_precision"])]
    return ev


def pooled(cars: dict, times, windows, n_sets, n_boot, rng, buffer, radius, match) -> pd.DataFrame:
    """Mean CAR across events per series and window, with a bootstrap CI and
    two placebo p-values (any date, and matched on weekday or weekday+hour
    within a year of each event)."""
    rows = []
    for name, table in cars.items():
        for w in windows:
            car = table[w]
            vals = np.array([car.get(t, np.nan) for t in times], float)
            lo, hi = bootstrap_mean_ci(vals, n_boot, rng=rng)
            free = placebo_test(car, times, n_sets, rng, "none", None, buffer)
            matched = placebo_test(car, times, n_sets, rng, match, radius, buffer)
            rows.append({"series": name, "window": w, "n": free["n"], "caar_pct": 100 * free["mean"],
                         "ci_lo_pct": 100 * lo, "ci_hi_pct": 100 * hi,
                         "placebo_sd_pct": 100 * matched["null_sd"],
                         "effect_z": (free["mean"] / matched["null_sd"]) if matched["null_sd"] else np.nan,
                         "p_placebo": free["p"], "p_matched": matched["p"]})
    df = pd.DataFrame(rows)
    df["q_placebo"] = benjamini_hochberg(df["p_placebo"])
    df["q_matched"] = benjamini_hochberg(df["p_matched"])
    return df


def per_event(cars: dict, ev: pd.DataFrame, time_col: str, windows, buffer, match) -> pd.DataFrame:
    times = list(ev[time_col])
    rows = []
    for name, table in cars.items():
        for w in windows:
            for eid, t in zip(ev["event_id"], times):
                r = single_event_test(table[w], t, times, match, YEAR, buffer)
                rows.append({"series": name, "window": w, "event_id": eid, "car_pct": 100 * r["car"],
                             "placebo_lo_pct": 100 * r["null_lo"], "placebo_hi_pct": 100 * r["null_hi"],
                             "p_matched": r["p"]})
    df = pd.DataFrame(rows)
    df["q_matched"] = benjamini_hochberg(df["p_matched"])
    return df


def path(series: dict, names, times, n_sets, rng, span=10) -> pd.DataFrame:
    """Cumulative average AR from day -span to +span, with a placebo band."""
    rows = []
    for name in names:
        y, x, model = series[name]
        ar = es.ar_path(y, x, es.DAILY_EST, es.DAILY_MIN_OBS, model, span)
        ar = ar.dropna()
        real = ar.loc[[t for t in times if t in ar.index]]
        idx = rng.integers(0, len(ar), size=(n_sets, len(real)))
        null = np.stack([ar.to_numpy()[i].mean(axis=0).cumsum() for i in idx])
        cum = real.mean().cumsum()
        for k, v in cum.items():
            j = int(k) + span
            rows.append({"series": name, "day": int(k), "car_pct": 100 * v,
                         "band_lo_pct": 100 * np.percentile(null[:, j], 2.5),
                         "band_hi_pct": 100 * np.percentile(null[:, j], 97.5)})
    return pd.DataFrame(rows)


def run(events: pd.DataFrame, hourly_prices: pd.DataFrame, n_sets: int = 2000,
        n_boot: int = 5000, seed: int = 7) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    ev = prepare_events(events)
    day_buf, hour_buf = pd.Timedelta(days=10), pd.Timedelta(days=10)

    rd = es.log_returns(daily_closes(hourly_prices))
    sd = es.study_series(rd)
    cars_d = {n: es.car_table(y, x, es.DAILY_WINDOWS, es.DAILY_EST, es.DAILY_MIN_OBS, m)
              for n, (y, x, m) in sd.items()}
    days = list(ev["day"])
    out = {"events": ev}
    out["daily"] = pooled(cars_d, days, es.DAILY_WINDOWS, n_sets, n_boot, rng, day_buf, YEAR, "weekday")
    clean = list(ev.loc[~ev["flagged"], "day"]) if "flagged" in ev else days
    out["daily_unflagged"] = pooled(cars_d, clean, es.DAILY_WINDOWS, n_sets, n_boot, rng,
                                    day_buf, YEAR, "weekday")
    out["daily_events"] = per_event(cars_d, ev, "day", es.DAILY_WINDOWS, day_buf, "weekday")
    out["path"] = path(sd, ["MKT", "BTC", "ETH"], days, n_sets, rng)

    timed = ev[ev["hour"].notna()]
    if len(timed):
        rh = es.log_returns(hourly_prices)
        sh = es.study_series(rh)
        cars_h = {n: es.car_table(y, x, es.HOURLY_WINDOWS, es.HOURLY_EST, es.HOURLY_MIN_OBS, m)
                  for n, (y, x, m) in sh.items()}
        out["hourly_events"] = per_event(cars_h, timed, "hour", es.HOURLY_WINDOWS, hour_buf,
                                         "weekday_hour")
    return out
