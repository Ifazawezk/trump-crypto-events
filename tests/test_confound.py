import pandas as pd

from cryptoevents.confound import flag_events


def test_flags_overlap_and_news():
    events = pd.DataFrame({"event_id": ["a", "b", "c"],
                           "day": pd.to_datetime(["2025-03-06", "2025-03-07", "2025-07-30"])})
    news = pd.DataFrame({"date": ["2025-07-30"], "title": ["FOMC rate decision"]})
    out = flag_events(events, news)
    assert out.loc[0, "overlaps"] == "b" and out.loc[1, "overlaps"] == "a"
    assert "FOMC" in out.loc[2, "near_news"] and out.loc[2, "overlaps"] == ""
    assert out["flagged"].tolist() == [True, True, True]


def test_isolated_event_not_flagged():
    events = pd.DataFrame({"event_id": ["a"], "day": pd.to_datetime(["2025-05-01"])})
    news = pd.DataFrame({"date": ["2025-05-10"], "title": ["x"]})
    assert not flag_events(events, news)["flagged"].iloc[0]
