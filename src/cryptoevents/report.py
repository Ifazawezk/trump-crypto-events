"""Charts (PNG) and a single static HTML report for GitHub Pages."""
from __future__ import annotations

import html
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # files only, never a window
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, AQUA, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#1baf7a", "#b9b8b2", "#0b0b0b", "#52514e"


def _style(ax, title: str, ylabel: str):
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel, color=MUTED)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.axhline(0, color=MUTED, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=MUTED)


def chart_caar(daily: pd.DataFrame, window: str, out: Path):
    d = daily[daily["window"] == window].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    x = range(len(d))
    half = 1.96 * d["placebo_sd_pct"]
    ax.bar(x, 2 * half, bottom=-half, width=0.5, color=GREY, alpha=0.45, label="95% range of placebo dates")
    ax.errorbar(x, d["caar_pct"], yerr=[d["caar_pct"] - d["ci_lo_pct"], d["ci_hi_pct"] - d["caar_pct"]],
                fmt="o", color=BLUE, ecolor=BLUE, elinewidth=2, capsize=0, markersize=7,
                label="Mean CAR, 95% bootstrap CI")
    ax.set_xticks(list(x), d["series"])
    _style(ax, f"Mean CAR in window {window} across events", "percent")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_path(path: pd.DataFrame, out: Path):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    mkt = path[path["series"] == "MKT"]
    ax.fill_between(mkt["day"], mkt["band_lo_pct"], mkt["band_hi_pct"], color=GREY, alpha=0.4,
                    label="MKT, 95% band from random dates")
    for name, colour in zip(["MKT", "BTC", "ETH"], [BLUE, ORANGE, AQUA]):
        p = path[path["series"] == name]
        ax.plot(p["day"], p["car_pct"], color=colour, linewidth=2, label=name)
        ax.annotate(name, (p["day"].iloc[-1], p["car_pct"].iloc[-1]), xytext=(4, 0),
                    textcoords="offset points", color=INK, fontsize=8, va="center")
    ax.axvline(0, color=MUTED, linewidth=0.8, linestyle=":")
    _style(ax, "Cumulative average abnormal return around events", "percent")
    ax.set_xlabel("days from event", color=MUTED)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_events(per_event: pd.DataFrame, series: str, window: str, out: Path):
    d = per_event[(per_event["series"] == series) & (per_event["window"] == window)].dropna(subset=["car_pct"])
    d = d.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.32 * len(d) + 1.2))
    y = range(len(d))
    ax.hlines(y, d["placebo_lo_pct"], d["placebo_hi_pct"], color=GREY, linewidth=6, alpha=0.6,
              label="95% range of matched placebo dates")
    ax.plot(d["car_pct"], y, "o", color=BLUE, markersize=7, label=f"{series} CAR {window}")
    ax.set_yticks(list(y), d["event_id"], fontsize=7)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_title(f"{series} CAR {window} by event (percent)", loc="left", fontsize=11, color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def chart_flows(flows: dict[str, pd.DataFrame], event_days, out: Path):
    keys = [k for k, v in flows.items() if not v.empty]
    if not keys:
        return False
    fig, axes = plt.subplots(len(keys), 1, figsize=(7, 1.9 * len(keys)), squeeze=False)
    for ax, k in zip(axes[:, 0], keys):
        weekly = flows[k].resample("W").sum()
        gross = weekly["inflow"] + weekly["outflow"]
        ax.bar(weekly.index, gross, width=5, color=BLUE)
        for t in event_days:
            if weekly.index.min() <= t <= weekly.index.max():
                ax.axvline(t, color=ORANGE, linewidth=0.8, alpha=0.8)
        ax.set_title(k, loc="left", fontsize=9, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(labelsize=7, colors=MUTED)
    axes[0, 0].text(1, 1.15, "weekly gross flow, token units; orange lines are events",
                    transform=axes[0, 0].transAxes, ha="right", fontsize=7, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return True


def table_html(df: pd.DataFrame, digits: int = 2) -> str:
    def fmt(v):
        if isinstance(v, float):
            return "" if pd.isna(v) else (f"{v:.3f}" if abs(v) < 1 and v != 0 and digits > 2 else f"{v:,.{digits}f}")
        return html.escape(str(v))
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body = "".join("<tr>" + "".join(f"<td>{fmt(v)}</td>" for v in row) + "</tr>"
                   for row in df.itertuples(index=False))
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#e6e5e0;--card:#ffffff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#33322f;--card:#222220}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--muted:#c3c2b7;--line:#33322f;--card:#222220}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 system-ui,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:1.6rem;margin:.2em 0}h2{font-size:1.15rem;margin-top:2em}
p,li{color:var(--ink)}.muted{color:var(--muted);font-size:.9rem}
img{max-width:100%;height:auto;background:#fff;border-radius:6px}
.scroll{overflow-x:auto}table{border-collapse:collapse;font-size:.82rem;width:100%}
th,td{border-bottom:1px solid var(--line);padding:4px 6px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}
"""


def write_html(out: Path, sections: list[tuple[str, str]], title: str, lead: str):
    parts = [f"<h2>{html.escape(h)}</h2>{body}" for h, body in sections]
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head>"
            f"<body><main><h1>{html.escape(title)}</h1>{lead}{''.join(parts)}</main></body></html>")
    out.write_text(page, encoding="utf-8")
