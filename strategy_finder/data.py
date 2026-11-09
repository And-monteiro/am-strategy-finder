"""Closed UTC candles, immutable snapshots, and Binance public history."""
from __future__ import annotations

import csv
import hashlib
import io
import re
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HOUR = 3600
STEPS = {"1h": HOUR, "4h": 4 * HOUR, "1d": 24 * HOUR}
DAY = STEPS["1d"]


@dataclass(frozen=True)
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def validate(rows):
    result = {}
    for row in rows:
        if row.timestamp % HOUR or row.timestamp < 0:
            raise ValueError("Candles must start on a UTC hour.")
        if not all(np.isfinite([row.open, row.high, row.low, row.close, row.volume])):
            raise ValueError("Non-finite candle value.")
        if not 0 < row.low <= min(row.open, row.close) <= max(row.open, row.close) <= row.high or row.volume < 0:
            raise ValueError("Invalid OHLCV candle.")
        if row.timestamp in result and result[row.timestamp] != row:
            raise ValueError("Conflicting duplicate candle.")
        result[row.timestamp] = row
    return sorted(result.values(), key=lambda row: row.timestamp)


def read_csv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = csv.DictReader(handle)
        if not rows.fieldnames or not {"timestamp", "open", "high", "low", "close", "volume"} <= set(rows.fieldnames):
            raise ValueError("CSV needs timestamp,open,high,low,close,volume.")
        return validate(Candle(int(r["timestamp"]), *(float(r[k]) for k in ("open", "high", "low", "close", "volume"))) for r in rows)


def _get(url, deadline=None, missing_ok=False):
    remaining = deadline - time.time() if deadline else 30
    if remaining <= 0:
        raise TimeoutError("Data preparation exceeded the investigation deadline.")
    try:
        with urllib.request.urlopen(url, timeout=min(30, remaining)) as response:
            return response.read(30_000_001)
    except urllib.error.HTTPError as exc:
        if missing_ok and exc.code == 404:
            return None
        raise


def _archive(path, cache, deadline):
    location = cache / Path(path).name
    if not location.exists():
        url = "https://data.binance.vision/data/spot/" + path
        data = _get(url, deadline, missing_ok=True)
        if data is None:
            return None
        checksum = _get(url + ".CHECKSUM", deadline).split()[0].decode()
        if hashlib.sha256(data).hexdigest() != checksum:
            raise ValueError(f"Archive checksum mismatch for {path}")
        temporary = location.with_suffix(".tmp")
        temporary.write_bytes(data)
        temporary.replace(location)
    with zipfile.ZipFile(location) as archive:
        with archive.open(archive.namelist()[0]) as raw:
            return list(csv.reader(io.TextIOWrapper(raw, encoding="utf-8")))


def _binance_row(row):
    stamp = int(row[0])
    stamp //= 1_000_000 if stamp >= 10**15 else 1_000
    return Candle(stamp, *(float(x) for x in row[1:6]))


def binance(symbol, start, end, cache, deadline=None):
    import json

    if not re.fullmatch(r"[A-Z0-9]{3,25}", symbol):
        raise ValueError("Invalid Binance symbol.")
    cache.mkdir(parents=True, exist_ok=True)
    first = datetime.fromtimestamp(start, timezone.utc)
    month = datetime(first.year, first.month, 1, tzinfo=timezone.utc)
    rows = []
    while month.timestamp() < end:
        year = month.year + (month.month == 12)
        following = month.replace(year=year, month=month.month % 12 + 1)
        if following.timestamp() > end:
            break
        label = month.strftime("%Y-%m")
        file = f"{symbol}-1h-{label}.zip"
        parsed = _archive(f"monthly/klines/{symbol}/1h/{file}", cache, deadline)
        if parsed is not None:
            rows.extend(_binance_row(r) for r in parsed if r and r[0].isdigit())
        month = following
    latest = max((row.timestamp for row in rows), default=start - HOUR) + HOUR
    # Only use the REST API for the recent tail. Missing historical archive
    # months remain visible as coverage gaps, including pre-listing dates.
    recent = max(start, latest, int(end) - 62 * DAY)
    while recent < end:
        query = f"symbol={symbol}&interval=1h&limit=1000&startTime={recent*1000}&endTime={end*1000-1}"
        raw = json.loads(_get("https://data-api.binance.vision/api/v3/klines?" + query, deadline))
        if not raw:
            break
        parsed = [_binance_row(row) for row in raw]
        rows.extend(parsed)
        following = parsed[-1].timestamp + HOUR
        if following <= recent:
            raise ValueError("Binance pagination did not advance.")
        recent = following
    return validate(row for row in rows if start <= row.timestamp and row.timestamp + HOUR <= end)


def align(hourly, timeframe):
    step = STEPS[timeframe]
    maps = [{row.timestamp: row for row in leg} for leg in hourly]
    common = set.intersection(*(set(leg) for leg in maps))
    times = []
    values = []
    for start in sorted(t for t in common if t % step == 0):
        members = list(range(start, start + step, HOUR))
        if not all(t in common for t in members):
            continue
        candles = [[leg[t] for t in members] for leg in maps]
        values.append([[r[0].open, max(x.high for x in r), min(x.low for x in r),
                        r[-1].close, sum(x.volume for x in r)] for r in candles])
        times.append(start)
    if not times:
        raise ValueError(f"No complete synchronized {timeframe} candles.")
    return np.array(times, dtype=np.int64), np.array(values, dtype=np.float64)


def partition_hash(times, values):
    h = hashlib.sha256()
    h.update(np.asarray(times, dtype="<i8").tobytes())
    h.update(np.asarray(values, dtype="<f8").tobytes())
    return h.hexdigest()


def save_snapshot(path, times, values):
    import pyarrow as pa
    import pyarrow.parquet as pq

    cols = {"timestamp": times}
    for i in range(values.shape[1]):
        for j, field in enumerate(("open", "high", "low", "close", "volume")):
            cols[f"{field}_{i}"] = values[:, i, j]
    temporary = path.with_suffix(".tmp")
    pq.write_table(pa.table(cols), temporary)
    temporary.replace(path)


def load_snapshot(path):
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    fields = ("open", "high", "low", "close", "volume")
    count = (len(table.column_names) - 1) // len(fields)
    values = np.stack([np.column_stack([table[f"{name}_{i}"].to_numpy() for name in fields])
                       for i in range(count)], axis=1)
    return table["timestamp"].to_numpy(), values
