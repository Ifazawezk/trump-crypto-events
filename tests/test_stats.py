import numpy as np
import pandas as pd

from cryptoevents.stats import (benjamini_hochberg, bootstrap_mean_ci, candidates, placebo_test,
                                single_event_test)


def test_bh_known_values():
    q = benjamini_hochberg([0.01, 0.04, 0.03, 0.20])
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.20])


def test_bh_keeps_nan():
    q = benjamini_hochberg([0.01, np.nan])
    assert q[0] == 0.01 and np.isnan(q[1])


def test_bootstrap_ci_contains_mean():
    v = np.random.default_rng(0).normal(1.0, 0.5, 40)
    lo, hi = bootstrap_mean_ci(v, 2000)
    assert lo < v.mean() < hi and hi - lo < 0.5


def series(n=1500, seed=3):
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(np.random.default_rng(seed).normal(0, 0.03, n), index=idx)


def test_placebo_detects_a_real_shock():
    car = series()
    events = list(car.index[[300, 500, 700, 900, 1100, 1300]])
    car.loc[events] += 0.08
    res = placebo_test(car, events, 2000, np.random.default_rng(1), "weekday", pd.Timedelta(days=365))
    assert res["p"] < 0.01 and res["n"] == 6


def test_placebo_is_quiet_without_a_shock():
    car = series(seed=5)
    events = list(car.index[[300, 500, 700, 900, 1100, 1300]])
    res = placebo_test(car, events, 2000, np.random.default_rng(1))
    assert res["p"] > 0.05


def test_candidates_hold_weekday_and_hour():
    idx = pd.date_range("2024-01-01", periods=24 * 60, freq="h")
    t = idx[500]
    c = candidates(idx, np.arange(len(idx)), t, "weekday_hour", pd.Timedelta(days=20))
    assert len(c) > 0
    assert (idx[c].day_of_week == t.day_of_week).all() and (idx[c].hour == t.hour).all()
    assert (abs(idx[c] - t) <= pd.Timedelta(days=20)).all()


def test_single_event_excludes_nearby_dates():
    car = series()
    t = car.index[700]
    car.loc[t] = 0.5
    res = single_event_test(car, t, [t], "weekday", pd.Timedelta(days=365))
    assert res["p"] < 0.05 and res["null_hi"] < 0.5
