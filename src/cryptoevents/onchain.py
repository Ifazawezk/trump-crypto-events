"""Token flows of publicly attributed addresses, from Blockscout's free API.

Only addresses listed in data/wallets.csv with a cited source are read.
Counterparties are reduced to classes (zero address, another listed address,
contract, other) before anything is saved to results/, so no private
counterparty address is published. The data shows timing, not motive.
"""
from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd

from .http import Client
from .stats import benjamini_hochberg, placebo_pool

BLOCKSCOUT = "https://eth.blockscout.com/api/v2/addresses/{a}/token-transfers"
# Token contracts as published by their issuers. Everything else (mostly
# unsolicited airdrops) is ignored.
TOKENS = {
    "WLFI": "0xda5e1988097297dcdc1f90d4dfe7909e847cbef6",
    "USD1": "0x8d0d000ee44948fc98c9b98a4fa4921476f08b0d",
    "WETH": "0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2",
    "USDC": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
    "USDT": "0xdac17f958d2ee523a2206206994597c13d831ec7",
}
ZERO = "0x0000000000000000000000000000000000000000"
CONFIDENCE = {"high", "medium"}
_EVM = re.compile(r"0x[0-9a-fA-F]{40}")
_SOL = re.compile(r"[1-9A-HJ-NP-Za-km-z]{32,44}")


def load_wallets(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    problems = []
    for r in df.itertuples():
        pattern = _EVM if r.chain == "ethereum" else _SOL
        if not pattern.fullmatch(r.address):
            problems.append(f"{r.address}: bad address format")
        if r.confidence not in CONFIDENCE:
            problems.append(f"{r.address}: confidence must be one of {sorted(CONFIDENCE)}")
        if not str(r.source_url).startswith("https://"):
            problems.append(f"{r.address}: needs an https source")
    if problems:
        raise ValueError("; ".join(problems))
    return df


def fetch_transfers(client: Client, address: str, token: str, max_pages: int = 400) -> list[dict]:
    """All ERC-20 transfers of one token touching `address` (paged API)."""
    params = {"type": "ERC-20", "token": token}
    items = []
    for _ in range(max_pages):
        body = json.loads(client.get(BLOCKSCOUT.format(a=address), params=params, expect="json"))
        items += body.get("items", [])
        nxt = body.get("next_page_params")
        if not nxt:
            return items
        params = {"type": "ERC-20", "token": token, **nxt}
    raise RuntimeError(f"{address}/{token}: more than {max_pages} pages")


def normalize(items: list[dict], wallet: str, listed: set[str]) -> pd.DataFrame:
    """One row per transfer: time, token, amount, direction, counterparty class."""
    wallet = wallet.lower()
    rows = []
    for it in items:
        src, dst = it["from"]["hash"].lower(), it["to"]["hash"].lower()
        if wallet not in (src, dst) or src == dst:
            continue
        inflow = dst == wallet
        other = it["from"] if inflow else it["to"]
        addr = other["hash"].lower()
        if addr == ZERO:
            cls = "zero address (mint or burn)"
        elif addr in listed:
            cls = "another listed address"
        elif other.get("is_contract"):
            cls = "contract"
        else:
            cls = "other address"
        dec = int(it["total"].get("decimals") or it["token"].get("decimals") or 0)
        rows.append({"time": pd.Timestamp(it["timestamp"]).tz_convert("UTC").tz_localize(None),
                     "token": it["token"]["symbol"], "amount": int(it["total"]["value"]) / 10 ** dec,
                     "direction": "in" if inflow else "out", "counterparty": cls})
    return pd.DataFrame(rows, columns=["time", "token", "amount", "direction", "counterparty"])


def daily_flows(tx: pd.DataFrame) -> pd.DataFrame:
    """Daily inflow and outflow (token units) over the address's active span."""
    if tx.empty:
        return pd.DataFrame(columns=["inflow", "outflow"])
    day = tx["time"].dt.floor("D")
    wide = tx.assign(day=day).pivot_table(index="day", columns="direction", values="amount",
                                          aggfunc="sum", fill_value=0.0)
    wide = wide.reindex(columns=["in", "out"], fill_value=0.0)
    full = pd.date_range(day.min(), day.max(), freq="D")
    return wide.reindex(full, fill_value=0.0).rename(columns={"in": "inflow", "out": "outflow"})


def window_sum(series: pd.Series, t: pd.Timestamp, lo: int, hi: int) -> float:
    return float(series.loc[t + pd.Timedelta(days=lo): t + pd.Timedelta(days=hi)].sum())


def event_flows(daily: pd.DataFrame, event_days, pre=(-7, -1), post=(0, 6)) -> pd.DataFrame:
    """Inflow and outflow in the week before and the week from each event.
    Events outside the address's active span are skipped."""
    rows = []
    if daily.empty:
        return pd.DataFrame(rows)
    for eid, t in event_days:
        if not (daily.index.min() + pd.Timedelta(days=7) <= t <= daily.index.max() - pd.Timedelta(days=6)):
            continue
        rows.append({"event_id": eid,
                     "in_pre": window_sum(daily["inflow"], t, *pre),
                     "out_pre": window_sum(daily["outflow"], t, *pre),
                     "in_post": window_sum(daily["inflow"], t, *post),
                     "out_post": window_sum(daily["outflow"], t, *post)})
    return pd.DataFrame(rows)


def flow_placebo(daily: pd.DataFrame, days: list[pd.Timestamp], n_sets: int,
                 rng: np.random.Generator, post=(0, 6),
                 buffer: pd.Timedelta = pd.Timedelta(days=10)) -> dict:
    """Mean gross flow (in + out) in the post window across events, compared
    with the same statistic on random sets of same-weekday dates from the
    address's active span. As in the price tests, dates within `buffer` of
    any event are not drawn, so an event's own flows don't count as noise."""
    gross = (daily["inflow"] + daily["outflow"])
    roll = gross[::-1].rolling(post[1] - post[0] + 1, min_periods=1).sum()[::-1]
    roll = roll.iloc[: len(roll) - post[1]] if post[1] > 0 else roll
    # Same eligibility as event_flows: a full week of history before day 0.
    roll = roll.loc[roll.index.min() + pd.Timedelta(days=7):]
    obs = np.array([roll.get(t, np.nan) for t in days], float)
    obs = obs[np.isfinite(obs)]
    if len(obs) == 0 or len(roll) < 30:
        return {"n": 0, "mean": np.nan, "p": np.nan, "placebo_mean": np.nan}
    vals = roll.to_numpy(float)
    pool = placebo_pool(roll.index, vals, days, buffer)
    weekday = roll.index.day_of_week[pool]
    choices = [pool[weekday == t.day_of_week] for t in days if t in roll.index]
    if any(len(c) == 0 for c in choices):
        return {"n": 0, "mean": np.nan, "p": np.nan, "placebo_mean": np.nan}
    draws = np.column_stack([rng.choice(c, n_sets) for c in choices])
    null = vals[draws].mean(axis=1)
    p = (1 + np.sum(null >= obs.mean())) / (n_sets + 1)  # one-sided: unusually large flow
    return {"n": len(obs), "mean": obs.mean(), "p": p, "placebo_mean": null.mean(),
            "ratio": obs.mean() / null.mean() if null.mean() > 0 else np.nan}


def summarise(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    if "p" in df:
        df["q"] = benjamini_hochberg(df["p"])
    return df
