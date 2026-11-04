import numpy as np
import pandas as pd

from cryptoevents import study
from cryptoevents.prices import COINS


def fake_prices(days=700, seed=2):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-01", periods=days * 24, freq="h")
    common = rng.normal(0, 0.004, len(idx))
    data = {c: 100 * np.exp(np.cumsum(common + rng.normal(0, 0.003, len(idx)))) for c in COINS}
    return pd.DataFrame(data, index=idx)


def test_run_end_to_end_on_synthetic_prices():
    events = pd.DataFrame({
        "event_id": ["a", "b", "c"],
        "timestamp": ["2024-03-06T00:00:00-05:00", "2024-06-03T00:00:00-04:00", "2024-09-10T14:30:00+00:00"],
        "time_precision": ["day", "day", "minute"], "flagged": [False, True, False]})
    out = study.run(events, fake_prices(), n_sets=200, n_boot=200)
    daily = out["daily"]
    assert set(daily["series"]) >= {"MKT", "BTC", "ETH"}
    assert (daily["n"] == 3).all()
    assert daily[["p_placebo", "p_matched", "q_matched"]].le(1).all().all()
    assert (out["daily_unflagged"]["n"] == 2).all()
    assert set(out["hourly_events"]["event_id"]) == {"c"}
    assert out["path"]["day"].between(-10, 10).all()
