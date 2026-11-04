from pathlib import Path

import pandas as pd

from cryptoevents import events as ev

DATA = Path(__file__).resolve().parents[1] / "data"


def row(**kw):
    base = {"event_id": "e1", "timestamp": "2025-03-06T00:00:00-05:00", "time_precision": "day",
            "type": "executive order", "subtype": "executive order", "title": "t",
            "source_url": "https://example.org/x", "source_name": "s", "date_basis": "signing date",
            "origin": "test"}
    return {**base, **kw}


def test_committed_event_list_is_valid():
    events = pd.read_csv(DATA / "events.csv")
    assert ev.validate_events(events) == []
    assert len(events) >= 10
    assert set(events["type"]) <= ev.TYPES


def test_curated_rows_are_valid():
    assert ev.validate_events(pd.read_csv(DATA / "curated_events.csv")) == []


def test_validation_catches_problems():
    df = pd.DataFrame([row(type="tweetstorm"), row(event_id="e2", source_url="http://x"),
                       row(event_id="e3", timestamp="2025-03-06T00:00:00")])
    problems = " ".join(ev.validate_events(df))
    assert "unknown type" in problems and "https" in problems and "timezone" in problems


def test_event_day_for_day_precision_uses_eastern_date():
    assert ev.event_day("2025-03-06T00:00:00-05:00", "day") == pd.Timestamp("2025-03-06")


def test_event_day_for_posts_uses_utc_date():
    # 20:15 Eastern on 11 July is already 12 July in UTC.
    assert ev.event_day("2019-07-12T00:15:22+00:00", "minute") == pd.Timestamp("2019-07-12")
    assert ev.event_hour("2019-07-12T00:15:22+00:00") == pd.Timestamp("2019-07-12 00:00")


def test_thread_becomes_one_event():
    tweets = pd.DataFrame({
        "id": ["1", "2", "3"], "text": ["Bitcoin is bad", "Libra too", "bitcoin again"],
        "date": pd.to_datetime(["2019-07-12 00:15:22", "2019-07-12 00:15:23", "2019-08-01 10:00:00"], utc=True),
        "rule": [True, True, True]})
    out = ev.tweet_events(tweets)
    assert list(out["event_id"]) == ["2019-07-12-tweet-1", "2019-08-01-tweet-3"]
    assert ev.validate_events(out) == []


def test_fr_events_mapping():
    cands = pd.DataFrame([{"document_number": "2025-03992", "executive_order_number": 14233,
                           "title": "Strategic Bitcoin Reserve", "subtype": "Executive Order",
                           "signing_date": "2025-03-06", "html_url": "https://www.federalregister.gov/d/2025-03992",
                           "rule": True},
                          {"document_number": "x", "executive_order_number": None, "title": "AI",
                           "subtype": "Executive Order", "signing_date": "2025-01-23",
                           "html_url": "https://www.federalregister.gov/d/x", "rule": False}])
    out = ev.fr_events(cands)
    assert list(out["event_id"]) == ["2025-03-06-eo-14233"]
    assert out.iloc[0]["timestamp"] == "2025-03-06T00:00:00-05:00"
