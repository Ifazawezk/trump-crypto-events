# trump-crypto-events

An event study of crypto prices around Donald Trump's crypto-related posts and official actions. It asks whether BTC, ETH and a few related coins showed abnormal returns around 14 dated events (one 2019 tweet thread, five executive orders, three pardons, two bill signings, a summit, a report and two token launches), and whether any effect is larger than what random dates produce.

This is a neutral measurement. It takes no political position, makes no claim about anyone's intent and is not investment advice.

I found no detectable effect. Over days 0 to +1 the equal-weight crypto market moved -0.44% on average (95% CI -2.27% to 1.49%), BTC -0.29% (-1.10% to 0.52%) and ETH -0.52% (-2.06% to 1.02%), all well inside the range random dates produce. The smallest corrected q-value across 35 coin and window tests is 0.096.

The report with charts is `docs/index.html`, ready for GitHub Pages.

## Results

Mean cumulative abnormal return (CAR) across events, in percent. The interval resamples events; p compares against 2,000 placebo sets matched on weekday within a year of each event; q is Benjamini-Hochberg across all 35 tests.

| Series | Window | Events | Mean CAR | 95% CI | p | q |
|---|---|---|---|---|---|---|
| Market (equal weight) | days -5 to -1 | 14 | 0.28 | -2.84 to 3.31 | 0.75 | 0.91 |
| Market (equal weight) | days 0 to +1 | 14 | -0.44 | -2.27 to 1.49 | 0.80 | 0.91 |
| BTC | days 0 to +1 | 14 | -0.29 | -1.10 to 0.52 | 0.83 | 0.91 |
| ETH | days 0 to +1 | 14 | -0.52 | -2.06 to 1.02 | 0.59 | 0.91 |
| XRP | days 0 to +1 | 14 | -2.14 | -3.67 to -0.72 | 0.08 | 0.23 |
| SOL | days 0 to +1 | 13 | 1.90 | -0.45 to 5.48 | 0.01 | 0.10 |
| ADA | days 0 to +1 | 14 | -2.55 | -4.76 to -0.68 | 0.01 | 0.10 |
| DOGE | days 0 to +1 | 13 | 0.60 | -0.92 to 2.35 | 0.54 | 0.91 |

The coin rows are coin-specific moves after removing the market. SOL and ADA have small raw p-values that do not survive the correction. Most of the SOL figure is one event: SOL rose 19.7% relative to the market over the two days of the $TRUMP launch, which runs on Solana. The ADA pre-event window shows +40% before the March 2025 reserve order. That rise followed a 2 March 2025 post naming ADA, which is not in the event list (see data sources).

Other checks, all in `results/`:

- Dropping the six events that overlap each other or fall within two days of an FOMC decision, the inauguration or a tariff order leaves 8 events. Market CAR over days 0 to +1 is then 1.05% (-1.49% to 3.48%), BTC -0.52% (-1.71% to 0.67%).
- The 2019 tweet thread is the only event with a known time, so it gets the hourly test. BTC moved 0.16% over the hour of the post and the next, and -2.78% over 24 hours. Matched placebo hours range from -4.17% to 4.27% for the 24-hour window (p 0.12).
- A plain substring search for crypto words finds 54 tweets, of which 2 are about crypto (3.7% precision; the rest are words like "coincidence" and "library", t.co links and @handles). The keyword rule used here finds those 2 and nothing else. On the 14 Federal Register search hits it keeps 3, all correct, and misses one (EO 14330 on 401(k) plans), which was added by hand. I checked every candidate by hand, so the counts cover all of them.

## On-chain flows

`data/wallets.csv` lists addresses with a public source and a confidence level: contracts published in World Liberty Financial's docs (high) and one wallet that Arkham Intelligence labelled as Trump's, as reported by CoinDesk (medium). Transfers of WLFI, USD1, WETH, USDC and USDT come from Blockscout. Counterparties are reduced to classes (contract, other address, zero address, another listed address) and no counterparty address is saved. The data shows timing, not motive.

| Address | Token | Transfers | Events in span | Mean flow in the week from day 0 | Same on random dates | p |
|---|---|---|---|---|---|---|
| WLFI CCIP lock and release pool | WLFI | 1,491 | 2 | 92.7M | 29.9M | 0.11 |
| WLFI token contract | USDT | 35 | 13 | 332 | 32 | 0.0015 (q 0.003) |
| WLFI staking contract | WLFI | 1,061 | 0 | | | |
| Arkham-labelled wallet | USDC | 99 | 0 | | | |

The staking contract has only been active since 1 October 2026, and the labelled wallet's Ethereum stablecoin transfers all fall in May to July 2024, so neither overlaps an event. The USDT row is small transfers (5,214 USDT in total) sent to the token contract address itself. Its q stays small, but it rests on 35 transfers and says when they happened, not why. The vester contract is listed but not read (about 125,000 transfers, too many to page through politely on a free API), and the $TRUMP mint on Solana is listed but not read because the official site publishes no wallet addresses.

## How to run

```
python -m pip install -e .[dev]
python -m pytest
python -m cryptoevents run       # first run downloads about 2,000 small files, then uses cache/
python -m cryptoevents events    # optional: rebuild data/events.csv from the sources
```

Python 3.11 or newer. Results go to `results/`, charts and the report to `docs/`. Downloads are cached in `cache/` (gitignored). The run finishes in seconds once the cache is warm. `--placebo-sets` and `--bootstrap` change the number of draws.

## Data sources and licences

All sources were read on 4 November 2026 with a plain client name and no key.

- Executive orders: the [Federal Register API](https://www.federalregister.gov/developers/documentation/api/v1), US government work in the public domain. The full-text download pages sit behind an access check, so the rule reads only the search excerpts the API returns.
- Tweets 2009 to 8 January 2021: the [Trump Twitter Archive](https://www.thetrumparchive.com/faq) CSV. Its FAQ says the data can be downloaded and that "You can credit the site however you'd like." The repo keeps tweet IDs, hand labels and the opening words of the one crypto tweet.
- Pardons, bill signings, the summit and the working group report: justice.gov, govinfo.gov and whitehouse.gov (US government works). The $TRUMP launch date comes from Wikipedia (CC BY-SA), which cites CNBC; the WLFI unlock date from World Liberty Financial's docs. Each row in `data/events.csv` has its URL.
- Recent posts are left out. Truth Social and X were not scraped, and I found no archive of Truth Social posts whose terms allow reuse, so posts after January 2021 are missing, including the 2 March 2025 post about a crypto reserve.
- Prices: hourly spot klines from [Binance Vision](https://data.binance.vision), checked against the published SHA-256 files. The [dataset terms](https://github.com/binance/binance-public-data) (version 1.0, updated 26 August 2026) license the data under CC BY-NC-SA 4.0 for non-commercial research. Raw files are not committed. The tables in `results/` and the charts in `docs/` are derived from that data, credited to Binance Vision and shared under CC BY-NC-SA 4.0.
- On-chain: the [Blockscout](https://eth.blockscout.com) Ethereum API, used without a key at two requests per second, below its default limit of 300 per minute.

The code is MIT licensed.

## Method

- Returns are log returns of UTC daily closes, and of hourly closes for timed events. Day 0 is the signing or announcement date in US Eastern time; for posts it is the UTC date of the post.
- The market proxy is an equal-weight index of ten large coins on Binance (BTC, ETH, BNB, XRP, ADA, SOL, DOGE, LTC, LINK, TRX). Equal weights stop BTC from dominating the benchmark, and market caps are not available without a key. Each coin gets a market model against the index of the other nine, fitted on days -250 to -31. The index itself uses a constant-mean model, so its CAR is the market-wide move.
- CARs cover days -5 to -1 (a leakage check), 0, 0 to +1, -1 to +1 and 0 to +3. Hourly windows run from 6 hours before to 24 hours after.
- Significance comes from placebo dates at least 10 days from any event: one test draws any date, the other holds the weekday (and hour, for hourly) fixed within a year of each event. Intervals bootstrap the events. Benjamini-Hochberg corrects across coins and windows, and separately across the per-event tests (up to 490).
- On-chain, the test statistic is gross flow (in plus out) over days 0 to +6, compared one-sided with same-weekday random dates in the address's active span, at least 10 days from any event as in the price tests. An event counts only after the address's first week.
- Events within 3 days of each other, or within 2 days of an item in `data/confounders.csv`, are flagged, and the pooled tests are rerun without them.

## Limits

There are 14 events, 13 of them dated only to the day, and many dates are signing dates rather than the moment the news broke (the Zhao pardon row uses the DOJ warrant date, which can precede public reports). Several events cluster with each other and with macro news. The coin list is today's large coins, which adds survivorship bias. It is a correlation study and cannot show cause.
