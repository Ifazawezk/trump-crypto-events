"""Command line entry point.

    python -m cryptoevents events   # rebuild data/events.csv and classifier scores
    python -m cryptoevents run      # prices, on-chain flows, statistics, report
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import events as ev
from . import onchain, report, study
from .classify import precision
from .confound import flag_events
from .http import Client
from .prices import COINS, hourly_closes

ROOT = Path(__file__).resolve().parents[2]
DATA, RESULTS, DOCS, CACHE = ROOT / "data", ROOT / "results", ROOT / "docs", ROOT / "cache"
MIN_TRANSFERS = 20  # fewer than this and a flow test means nothing


def score(pred: pd.Series, truth: pd.Series, corpus: str, rule: str) -> dict:
    prec, tp, pos = precision(list(pred), list(truth))
    return {"corpus": corpus, "rule": rule, "labelled": len(truth), "predicted": pos,
            "correct": tp, "precision": round(prec, 3),
            "recall_in_labelled": round(tp / truth.sum(), 3) if truth.sum() else np.nan}


def cmd_events(client: Client):
    fr = ev.fetch_fr_candidates(client)
    tweets = ev.load_tweets(client)
    labels_fr = pd.read_csv(DATA / "labels" / "federal_register.csv", dtype={"document_number": str})
    labels_tw = pd.read_csv(DATA / "labels" / "tweets.csv", dtype={"id": str})
    f = fr.merge(labels_fr, on="document_number", how="left")
    t = tweets[tweets["naive"] | tweets["rule"]].merge(labels_tw, on="id", how="left")
    if f["label"].isna().any() or t["label"].isna().any():
        raise SystemExit("new candidates without a hand label; add them to data/labels first")
    scores = [score(pd.Series(True, index=f.index), f["label"].astype(bool), "federal register", "API search hit"),
              score(f["rule"], f["label"].astype(bool), "federal register", "keyword rule"),
              score(t["naive"], t["label"].astype(bool), "tweets", "substring baseline"),
              score(t["rule"], t["label"].astype(bool), "tweets", "keyword rule")]
    RESULTS.mkdir(exist_ok=True)
    pd.DataFrame(scores).to_csv(RESULTS / "classifier_precision.csv", index=False)
    curated = pd.read_csv(DATA / "curated_events.csv")
    table = ev.build_events([ev.fr_events(fr), ev.tweet_events(tweets), curated])
    problems = ev.validate_events(table)
    if problems:
        raise SystemExit("\n".join(problems))
    table.to_csv(DATA / "events.csv", index=False)
    print(f"{len(table)} events written; classifier scores in results/classifier_precision.csv")


def cmd_run(client: Client, n_sets: int, n_boot: int, end: str):
    events = pd.read_csv(DATA / "events.csv")
    problems = ev.validate_events(events)
    if problems:
        raise SystemExit("\n".join(problems))
    news = pd.read_csv(DATA / "confounders.csv")
    prepared = flag_events(study.prepare_events(events), news)
    hourly = hourly_closes(client, COINS, "2018-07", end)
    res = study.run(prepared, hourly, n_sets=n_sets, n_boot=n_boot)

    # On-chain flows of attributed addresses (Ethereum, via Blockscout).
    wallets = onchain.load_wallets(DATA / "wallets.csv")
    listed = {a.lower() for a in wallets["address"]}
    chain_client = Client(CACHE / "http", min_interval=0.5)
    rng = np.random.default_rng(11)
    day_pairs = list(zip(prepared["event_id"], prepared["day"]))
    flows, windows, summary, mix = {}, [], [], []
    for w in wallets[(wallets["analyzed"] == "yes") & (wallets["chain"] == "ethereum")].itertuples():
        for sym, token in onchain.TOKENS.items():
            tx = onchain.normalize(onchain.fetch_transfers(chain_client, w.address, token), w.address, listed)
            if tx.empty:
                continue
            daily = onchain.daily_flows(tx)
            test = {}
            if len(tx) >= MIN_TRANSFERS:
                flows[f"{w.label} / {sym}"] = daily
                ew = onchain.event_flows(daily, day_pairs)
                if not ew.empty:
                    windows.append(ew.assign(address_label=w.label, token=sym))
                test = onchain.flow_placebo(daily, [d for _, d in day_pairs], n_sets, rng)
            summary.append({"address_label": w.label, "confidence": w.confidence, "token": sym,
                            "transfers": len(tx), "first": tx["time"].min().date(),
                            "last": tx["time"].max().date(), **test})
            share = tx.groupby("counterparty")["amount"].sum() / tx["amount"].sum()
            mix += [{"address_label": w.label, "token": sym, "counterparty": k, "share_of_volume": v}
                    for k, v in share.items()]

    RESULTS.mkdir(exist_ok=True)
    for name in ["daily", "daily_unflagged", "daily_events", "path", "hourly_events"]:
        if name in res:
            res[name].round(4).to_csv(RESULTS / f"{name}.csv", index=False)
    prepared.drop(columns=["hour"]).to_csv(RESULTS / "events_flagged.csv", index=False)
    if windows:
        pd.concat(windows).round(2).to_csv(RESULTS / "onchain_event_windows.csv", index=False)
    onchain.summarise(summary).round(4).to_csv(RESULTS / "onchain_summary.csv", index=False)
    pd.DataFrame(mix).round(4).to_csv(RESULTS / "onchain_counterparties.csv", index=False)
    meta = {"run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "price_range_utc": [str(hourly.index.min()), str(hourly.index.max())],
            "coins": COINS, "events": len(events), "placebo_sets": n_sets, "bootstrap": n_boot}
    (RESULTS / "run_meta.json").write_text(json.dumps(meta, indent=2))
    build_report(res, prepared, flows, meta)
    print("results in results/, report in docs/index.html")


def build_report(res: dict, prepared: pd.DataFrame, flows: dict, meta: dict):
    img = DOCS / "img"
    img.mkdir(parents=True, exist_ok=True)
    report.chart_caar(res["daily"], "[0,+1]", img / "caar.png")
    report.chart_path(res["path"], img / "path.png")
    report.chart_events(res["daily_events"], "MKT", "[0,+1]", img / "events_mkt.png")
    has_flows = report.chart_flows(flows, list(prepared["day"]), img / "flows.png")
    read = lambda n: pd.read_csv(RESULTS / n)  # noqa: E731
    keep = ["series", "window", "n", "caar_pct", "ci_lo_pct", "ci_hi_pct", "effect_z", "p_matched", "q_matched"]
    sections = [
        ("Headline", report.table_html(res["daily"][res["daily"]["window"].isin(["pre[-5,-1]", "[0,+1]"])][keep], 3)
         + '<img src="img/caar.png" alt="Mean CAR by series with bootstrap intervals and placebo range">'),
        ("Path around events", '<img src="img/path.png" alt="Cumulative abnormal return from day -10 to +10">'),
        ("Each event", '<img src="img/events_mkt.png" alt="Market CAR per event against placebo ranges">'),
        ("Without flagged events", report.table_html(res["daily_unflagged"][res["daily_unflagged"]["window"] == "[0,+1]"][keep], 3)),
        ("Events and flags", report.table_html(prepared[["event_id", "type", "time_precision", "overlaps", "near_news"]])),
    ]
    if "hourly_events" in res:
        h = res["hourly_events"]
        sections.append(("Hourly, timed posts", report.table_html(h[h["series"].isin(["MKT", "BTC", "ETH"])], 3)))
    chain = read("onchain_summary.csv")
    sections.append(("On-chain flows of attributed addresses",
                     "<p>The data shows timing, not motive.</p>" + report.table_html(chain, 3)
                     + ('<img src="img/flows.png" alt="Weekly token flows with event markers">' if has_flows else "")))
    sections.append(("Keyword rule precision", report.table_html(read("classifier_precision.csv"), 3)))
    lead = ("<p>Did BTC, ETH and related coins move abnormally around Donald Trump's crypto-related "
            "posts and official actions, more than on random dates? A neutral measurement: no political "
            "view, no claim about intent, no investment advice.</p>"
            f'<p class="muted">Run {meta["run_utc"]}. Prices: Binance Vision (CC BY-NC-SA 4.0); '
            "charts and tables on this page are derived from it and shared under the same licence. "
            "Code: MIT.</p>")
    report.write_html(DOCS / "index.html", sections, "Trump crypto events", lead)


def main(argv=None):
    p = argparse.ArgumentParser(prog="cryptoevents")
    p.add_argument("command", choices=["events", "run"])
    p.add_argument("--placebo-sets", type=int, default=2000)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--end", default="2026-10", help="last month of price data (YYYY-MM)")
    a = p.parse_args(argv)
    client = Client(CACHE / "http", min_interval=0.15)
    if a.command == "events":
        cmd_events(client)
    else:
        cmd_run(client, a.placebo_sets, a.bootstrap, a.end)


if __name__ == "__main__":
    main()
