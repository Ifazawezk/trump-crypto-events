"""Flag events that sit close to each other or to other big news."""
from __future__ import annotations

import pandas as pd


def flag_events(events: pd.DataFrame, news: pd.DataFrame, overlap_days: int = 3,
                news_days: int = 2) -> pd.DataFrame:
    """Add `overlaps`, `near_news` and `flagged` columns.

    `events` needs `event_id` and `day` (day 0, a date). `news` needs `date`
    and `title`. An event is flagged when another event or a news item falls
    within the given number of days on either side."""
    out = events.copy()
    days = pd.to_datetime(out["day"])
    news_days_ = pd.to_datetime(news["date"])
    overlaps, near = [], []
    for i, d in enumerate(days):
        gap = (days - d).abs().dt.days
        others = out.loc[(gap <= overlap_days) & (out.index != out.index[i]), "event_id"]
        overlaps.append("; ".join(others))
        close = news.loc[(news_days_ - d).abs().dt.days <= news_days]
        near.append("; ".join(f"{r.date} {r.title}" for r in close.itertuples()))
    out["overlaps"] = overlaps
    out["near_news"] = near
    out["flagged"] = (out["overlaps"] != "") | (out["near_news"] != "")
    return out
