import hashlib
import io
import zipfile

import pandas as pd

from cryptoevents.prices import daily_closes, months, parse_klines, verify_checksum


def kline_zip(rows, header=False):
    lines = (["open_time,open,high,low,close,volume"] if header else []) + [
        f"{t},1,1,1,{c},1" for t, c in rows]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("x.csv", "\n".join(lines))
    return buf.getvalue()


def test_millisecond_timestamps():
    df = parse_klines(kline_zip([(1562889600000, 11000.5)]))
    assert df["time"].iloc[0] == pd.Timestamp("2019-07-12")
    assert df["close"].iloc[0] == 11000.5


def test_microsecond_timestamps_from_2025_and_header():
    df = parse_klines(kline_zip([(1735689600000000, 94000.0)], header=True))
    assert df["time"].iloc[0] == pd.Timestamp("2025-01-01")
    assert len(df) == 1


def test_checksum():
    body = b"abc"
    good = f"{hashlib.sha256(body).hexdigest()}  BTCUSDT-1h-2019-07.zip".encode()
    assert verify_checksum(body, good)
    assert not verify_checksum(b"abd", good)


def test_months_inclusive():
    assert months("2024-11", "2025-02") == [(2024, 11), (2024, 12), (2025, 1), (2025, 2)]


def test_daily_close_is_last_hour():
    idx = pd.date_range("2025-01-01", periods=48, freq="h")
    hourly = pd.DataFrame({"BTC": range(48)}, index=idx, dtype=float)
    daily = daily_closes(hourly)
    assert list(daily["BTC"]) == [23.0, 47.0]
