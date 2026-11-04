import pandas as pd

from cryptoevents import report


def test_table_html_escapes_and_formats():
    out = report.table_html(pd.DataFrame({"a": ["<b>"], "b": [1.23456]}))
    assert "&lt;b&gt;" in out and "1.23" in out


def test_charts_write_files(tmp_path):
    daily = pd.DataFrame({"series": ["MKT", "BTC"], "window": "[0,+1]", "caar_pct": [0.5, -0.2],
                          "ci_lo_pct": [-1, -2], "ci_hi_pct": [2, 1], "placebo_sd_pct": [1, 1]})
    report.chart_caar(daily, "[0,+1]", tmp_path / "c.png")
    path = pd.DataFrame([{"series": s, "day": d, "car_pct": d / 10, "band_lo_pct": -1, "band_hi_pct": 1}
                         for s in ["MKT", "BTC", "ETH"] for d in range(-2, 3)])
    report.chart_path(path, tmp_path / "p.png")
    report.write_html(tmp_path / "i.html", [("A", "<p>x</p>")], "Title", "<p>lead</p>")
    assert (tmp_path / "c.png").stat().st_size > 1000 and (tmp_path / "p.png").exists()
    assert "<h2>A</h2>" in (tmp_path / "i.html").read_text(encoding="utf-8")
