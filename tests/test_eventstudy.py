import numpy as np
import pandas as pd

from cryptoevents import eventstudy as es

WINDOWS = {"[0,0]": (0, 0), "[0,+1]": (0, 1), "pre[-2,-1]": (-2, -1)}
EST = (-60, -11)


def simulated(n=300, beta=1.5, seed=1):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    x = pd.Series(rng.normal(0, 0.02, n), index=idx)
    y = pd.Series(0.001 + beta * x + rng.normal(0, 0.01, n), index=idx)
    return y, x


def test_matches_explicit_ols():
    y, x = simulated()
    table = es.car_table(y, x, WINDOWS, EST, 30, "market")
    t = 200
    ey, ex = y.iloc[t - 60:t - 10], x.iloc[t - 60:t - 10]
    beta, alpha = np.polyfit(ex, ey, 1)
    ar = y - (alpha + beta * x)
    assert np.isclose(table["[0,+1]"].iloc[t], ar.iloc[t:t + 2].sum())
    assert np.isclose(table["pre[-2,-1]"].iloc[t], ar.iloc[t - 2:t].sum())


def test_constant_mean_model():
    y, _ = simulated()
    table = es.car_table(y, None, WINDOWS, EST, 30, "mean")
    t = 150
    mean = y.iloc[t - 60:t - 10].mean()
    assert np.isclose(table["[0,0]"].iloc[t], y.iloc[t] - mean)


def test_injected_shock_shows_up_and_market_part_is_removed():
    y, x = simulated()
    y2 = y.copy()
    y2.iloc[220] += 0.05            # coin-specific jump
    x2 = x.copy()
    x2.iloc[240] += 0.05            # market-wide jump; coin follows with beta
    y2.iloc[240] += 1.5 * 0.05
    table = es.car_table(y2, x2, WINDOWS, EST, 30, "market")
    assert table["[0,0]"].iloc[220] > 0.04
    assert abs(table["[0,0]"].iloc[240]) < 0.03


def test_short_history_gives_nan():
    y, x = simulated()
    y.iloc[:180] = np.nan
    table = es.car_table(y, x, WINDOWS, EST, 30, "market")
    assert np.isnan(table["[0,0]"].iloc[185])
    assert np.isfinite(table["[0,0]"].iloc[250])
    assert table["[0,0]"].iloc[:EST[1] * -1].isna().all()


def test_market_index_leaves_coin_out_and_needs_three_coins():
    r = pd.DataFrame({"BTC": [0.1, 0.1], "ETH": [0.2, 0.2], "XRP": [0.3, np.nan], "SOL": [0.4, np.nan]})
    assert np.isclose(es.market_index(r, exclude="BTC").iloc[0], 0.3)
    assert np.isnan(es.market_index(r, exclude="BTC").iloc[1])


def test_study_series_models():
    r = pd.DataFrame(np.zeros((5, 4)), columns=["BTC", "ETH", "XRP", "BNB"])
    s = es.study_series(r, ["BTC", "ETH"])
    assert s["MKT"][2] == "mean" and s["BTC"][2] == "market" and "XRP" not in s
