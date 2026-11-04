"""Build and check the cited event list.

Three inputs:
  * Federal Register API: presidential documents signed by Trump that match
    crypto search terms, then filtered with the keyword rule.
  * Trump Twitter Archive CSV (tweets 2009 to Jan 2021), filtered with the
    keyword rule. Tweets in one thread within an hour become one event.
  * data/curated_events.csv: pardons, bill signings, remarks and launches,
    each with a primary or public source.
"""
from __future__ import annotations

import io
import json
from zoneinfo import ZoneInfo

import pandas as pd

from .classify import document_rule, naive_match, rule_match
from .http import Client

ET = ZoneInfo("America/New_York")
TYPES = {"post", "executive order", "announcement", "launch"}
COLUMNS = ["event_id", "timestamp", "time_precision", "type", "subtype", "title",
           "source_url", "source_name", "date_basis", "origin"]

FR_API = "https://www.federalregister.gov/api/v1/documents.json"
FR_TERMS = ['"digital asset"', '"digital assets"', "cryptocurrency", "bitcoin",
            "blockchain", "stablecoin", '"central bank digital currency"', "crypto"]
FR_FIELDS = ["document_number", "executive_order_number", "title", "subtype",
             "signing_date", "html_url", "excerpts"]
TTA_CSV = "https://drive.usercontent.google.com/download?id=1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6&export=download"
TTA_TWEET = "https://www.thetrumparchive.com/tweets/{id}"


def fetch_fr_candidates(client: Client) -> pd.DataFrame:
    rows = {}
    for term in FR_TERMS:
        params = {"conditions[term]": term, "conditions[type][]": "PRESDOCU",
                  "conditions[president][]": "donald-trump", "per_page": 100,
                  "fields[]": FR_FIELDS}
        body = json.loads(client.get(FR_API, params=params, expect="json"))
        for r in body.get("results", []):
            row = rows.setdefault(r["document_number"], {**r, "search_terms": [], "excerpts": ""})
            row["search_terms"].append(term.strip('"'))
            row["excerpts"] += " | " + (r.get("excerpts") or "")
    # The API's full-text search returns excerpts around each match. The
    # full-text download pages sit behind an access check, so we only use
    # what the API itself returns.
    df = pd.DataFrame(rows.values())
    df["rule"] = [document_rule(t, x) for t, x in zip(df["title"], df["excerpts"])]
    return df.sort_values("signing_date").reset_index(drop=True)


def fr_events(cands: pd.DataFrame) -> pd.DataFrame:
    keep = cands[cands["rule"]]
    out = []
    for _, r in keep.iterrows():
        eo = r.get("executive_order_number")
        kind = "executive order" if r["subtype"] == "Executive Order" else "announcement"
        tag = f"eo-{int(eo)}" if pd.notna(eo) and eo else r["document_number"]
        out.append({
            "event_id": f"{r['signing_date']}-{tag}",
            "timestamp": midnight_et(r["signing_date"]), "time_precision": "day",
            "type": kind, "subtype": r["subtype"].lower(), "title": r["title"],
            "source_url": r["html_url"], "source_name": "Federal Register API",
            "date_basis": "signing date", "origin": "federal_register"})
    return pd.DataFrame(out, columns=COLUMNS)


def load_tweets(client: Client) -> pd.DataFrame:
    df = pd.read_csv(io.BytesIO(client.get(TTA_CSV, key="tta/tweets_01-08-2021.csv", expect="text")),
                     dtype={"id": str})
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df["is_retweet"] = df["isRetweet"].eq("t")
    df["naive"] = df["text"].map(naive_match)
    df["rule"] = df["text"].map(rule_match) & ~df["is_retweet"]
    return df


def tweet_events(tweets: pd.DataFrame, gap_minutes: int = 60) -> pd.DataFrame:
    hits = tweets[tweets["rule"]].sort_values("date")
    out, last = [], None
    for _, t in hits.iterrows():
        if last is not None and (t["date"] - last).total_seconds() <= gap_minutes * 60:
            last = t["date"]
            continue
        last = t["date"]
        out.append({
            "event_id": f"{t['date']:%Y-%m-%d}-tweet-{t['id']}",
            "timestamp": t["date"].isoformat(), "time_precision": "minute",
            "type": "post", "subtype": "tweet", "title": t["text"][:90].strip(),
            "source_url": TTA_TWEET.format(id=t["id"]), "source_name": "Trump Twitter Archive",
            "date_basis": "post time (UTC)", "origin": "tweet_archive"})
    return pd.DataFrame(out, columns=COLUMNS)


def midnight_et(date: str) -> str:
    return pd.Timestamp(date).tz_localize(ET).isoformat()


def build_events(parts: list[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat([p[COLUMNS] for p in parts], ignore_index=True)
    df = df.drop_duplicates("event_id")
    df["_t"] = pd.to_datetime(df["timestamp"], utc=True, format="ISO8601")
    return df.sort_values("_t").drop(columns="_t").reset_index(drop=True)


def validate_events(df: pd.DataFrame) -> list[str]:
    """Return a list of problems; empty means the list is usable."""
    problems = []
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    for _, r in df.iterrows():
        rid = r["event_id"]
        if r[COLUMNS[:8]].isna().any():
            problems.append(f"{rid}: empty field")
            continue
        if r["type"] not in TYPES:
            problems.append(f"{rid}: unknown type {r['type']!r}")
        if r["time_precision"] not in {"minute", "day"}:
            problems.append(f"{rid}: bad precision")
        if not str(r["source_url"]).startswith("https://"):
            problems.append(f"{rid}: source must be an https URL")
        ts = pd.Timestamp(r["timestamp"])
        if ts.tzinfo is None:
            problems.append(f"{rid}: timestamp has no timezone")
    if df["event_id"].duplicated().any():
        problems.append("duplicate event ids")
    return problems


def event_day(timestamp: str, precision: str) -> pd.Timestamp:
    """The UTC daily bar that counts as day 0.

    Day-precision rows carry midnight Eastern time, so day 0 is that calendar
    date. Minute-precision rows use the UTC date of the post itself."""
    ts = pd.Timestamp(timestamp)
    if precision == "day":
        return pd.Timestamp(ts.tz_convert(ET).date())
    return pd.Timestamp(ts.tz_convert("UTC").date())


def event_hour(timestamp: str) -> pd.Timestamp:
    return pd.Timestamp(timestamp).tz_convert("UTC").floor("h").tz_localize(None)
