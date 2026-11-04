from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cryptoevents import onchain

DATA = Path(__file__).resolve().parents[1] / "data"
WALLET = "0x1111111111111111111111111111111111111111"
LISTED = "0x2222222222222222222222222222222222222222"


def item(src, dst, value, ts="2025-09-01T12:00:00.000000Z", contract=False):
    return {"from": {"hash": src, "is_contract": contract}, "to": {"hash": dst, "is_contract": contract},
            "timestamp": ts, "token": {"symbol": "WLFI", "decimals": "18"},
            "total": {"value": str(value * 10**18), "decimals": "18"}}


def test_committed_wallets_file_is_valid():
    w = onchain.load_wallets(DATA / "wallets.csv")
    assert set(w["confidence"]) <= onchain.CONFIDENCE
    assert w["source_url"].str.startswith("https://").all()


def test_wallet_without_source_is_rejected(tmp_path):
    p = tmp_path / "w.csv"
    p.write_text("address,chain,label,entity,source_url,source_name,confidence,analyzed,note\n"
                 f"{WALLET},ethereum,x,y,,s,guess,yes,\n")
    with pytest.raises(ValueError, match="confidence"):
        onchain.load_wallets(p)


def test_normalize_classes_and_no_addresses():
    items = [item(onchain.ZERO, WALLET, 5), item(WALLET, LISTED, 2),
             item(WALLET, "0x3333333333333333333333333333333333333333", 1, contract=True),
             item("0x4444444444444444444444444444444444444444", WALLET, 3)]
    tx = onchain.normalize(items, WALLET, {WALLET, LISTED})
    assert list(tx["direction"]) == ["in", "out", "out", "in"]
    assert list(tx["counterparty"]) == ["zero address (mint or burn)", "another listed address",
                                        "contract", "other address"]
    assert tx["amount"].sum() == 11
    flat = tx.astype(str).to_string()
    assert "0x" not in flat


def test_daily_flows_and_event_windows():
    tx = pd.DataFrame({"time": pd.to_datetime(["2025-01-01", "2025-01-10", "2025-01-12", "2025-01-30"]),
                       "token": "WLFI", "amount": [1.0, 2.0, 4.0, 8.0],
                       "direction": ["in", "in", "out", "out"], "counterparty": "other address"})
    daily = onchain.daily_flows(tx)
    assert len(daily) == 30 and daily["inflow"].sum() == 3 and daily["outflow"].sum() == 12
    w = onchain.event_flows(daily, [("e", pd.Timestamp("2025-01-11")), ("late", pd.Timestamp("2025-01-29"))])
    assert list(w["event_id"]) == ["e"]
    assert w.iloc[0][["in_pre", "out_pre", "in_post", "out_post"]].tolist() == [2.0, 0.0, 0.0, 4.0]


def test_flow_placebo_flags_big_flows():
    idx = pd.date_range("2025-01-01", periods=200, freq="D")
    daily = pd.DataFrame({"inflow": 1.0, "outflow": 1.0}, index=idx)
    days = [idx[50], idx[100], idx[150]]
    for d in days:
        daily.loc[d, "inflow"] = 100.0
    res = onchain.flow_placebo(daily, days, 1000, np.random.default_rng(0))
    assert res["n"] == 3 and res["p"] < 0.05 and res["ratio"] > 2


def test_flow_placebo_skips_events_in_first_week():
    idx = pd.date_range("2025-01-01", periods=200, freq="D")
    daily = pd.DataFrame({"inflow": 1.0, "outflow": 0.0}, index=idx)
    res = onchain.flow_placebo(daily, [idx[3], idx[100]], 500, np.random.default_rng(0))
    assert res["n"] == 1
