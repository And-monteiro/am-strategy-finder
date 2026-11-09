"""Local command line entry point. No exchange credentials or trading APIs."""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import urllib.error
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .backtest import Execution
from .data import DAY, HOUR, STEPS, align, binance, partition_hash, read_csv, save_snapshot
from .report import publish
from .research import Ledger, run


def duration(raw):
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([smh])", raw)
    if not match:
        raise argparse.ArgumentTypeError("Use a duration such as 8h, 15m, or 30s.")
    seconds = float(match[1]) * {"s": 1, "m": 60, "h": 3600}[match[2]]
    if not math.isfinite(seconds) or seconds < 5:
        raise argparse.ArgumentTypeError("Duration must be at least 5 seconds.")
    return seconds


def date(raw):
    return int(datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp())


def parse_asset(raw):
    asset = raw.upper()
    if "/" in asset and not asset.endswith("/EUR"):
        raise ValueError("Use --asset BTC --pair ETH for a synthetic ratio; research fills use EUR markets.")
    asset = asset.removesuffix("/EUR")
    if asset.endswith("EUR") and len(asset) > 3:
        asset = asset[:-3]
    if not re.fullmatch("[A-Z0-9]{2,15}", asset):
        raise ValueError("Invalid asset.")
    return asset + "EUR"


@contextmanager
def lock(root):
    with open(root / ".writer.lock", "a+b") as handle:
        handle.seek(0, 2)
        if not handle.tell():
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def prepare(args, root, started):
    assets = [parse_asset(args.asset)]
    if args.pair:
        assets.append(parse_asset(args.pair))
    if len(set(assets)) != len(assets):
        raise ValueError("Choose two different assets for rotation.")
    if args.csv and len(args.csv) != len(assets):
        raise ValueError("Pass one --csv per EUR asset, in asset order.")
    if not 1 <= args.variants <= 1000 or args.experiments < 0 or not 0 <= args.seed < 2**32:
        raise ValueError("Use 1–1000 variants, nonnegative experiments, and a 32-bit seed.")
    if not 0 < args.drawdown <= 1:
        raise ValueError("Drawdown cap must be a fraction in (0, 1].")
    if not args.offline and not os.environ.get("OPENAI_API_KEY"):
        raise ValueError("Set OPENAI_API_KEY or choose --offline.")
    if not args.offline and not args.model:
        raise ValueError("Set OPENAI_MODEL or pass --model for Structured Outputs.")
    execution = Execution(args.capital, args.allocation, args.risk, args.fee_bps, args.slippage_bps)
    now_hour = int(started) // HOUR * HOUR
    end = args.end if args.end is not None else now_hour
    if end > now_hour:
        raise ValueError("End date is in the future.")
    endpoint = datetime.fromtimestamp(end, timezone.utc)
    try:
        default_start = int(endpoint.replace(year=endpoint.year-5).timestamp())
    except ValueError:  # Feb 29
        default_start = int(endpoint.replace(year=endpoint.year-5, day=28).timestamp())
    start = args.start if args.start is not None else default_start
    if start >= end:
        raise ValueError("Start must precede end.")
    deadline = started + args.duration
    if args.csv:
        hourly = [[row for row in read_csv(path) if start <= row.timestamp and row.timestamp + HOUR <= end]
                  for path in args.csv]
        source = "local_eur_csv"
    else:
        hourly = [binance(asset, start, end, args.cache, deadline) for asset in assets]
        source = "binance_spot_eur"
    observed, _ = align(hourly, "1h")
    common_start, common_end = int(observed[0]), int(observed[-1])+HOUR
    cut = common_start + (common_end-common_start)*4//5
    cut = cut // DAY * DAY
    if not common_start < cut < common_end:
        raise ValueError("Not enough history for a UTC-day aligned holdout.")
    datasets = {}
    for frame in dict.fromkeys(args.timeframes):
        times, bars = align(hourly, frame)
        split = int(np.searchsorted(times, cut))
        if split < 3 or split == len(times):
            raise ValueError(f"Insufficient bars on either side of the {frame} split.")
        save_snapshot(root / f"development-{frame}.parquet", times[:split], bars[:split])
        save_snapshot(root / f"sealed-{frame}.parquet", times[split:], bars[split:])
        datasets[frame] = {
            "development_hash": partition_hash(times[:split], bars[:split]),
            "final_hash": partition_hash(times[split:], bars[split:]),
            "development_bars": split, "final_bars": len(times)-split,
            "first_bar": int(times[0]), "last_bar": int(times[-1]),
            "missing_bars": (int(times[-1])-int(times[0]))//STEPS[frame]+1-len(times),
        }
    return {"assets": assets, "source": source, "requested_start": start, "requested_end": end,
            "split": cut, "datasets": datasets, "duration": args.duration, "deadline": deadline,
            "execution": asdict(execution), "drawdown": args.drawdown, "seed": args.seed,
            "evidence": {"lower_quantile": .1, "max_winner_contribution": .5,
                         "cost_multiplier": 1.5, "positive_period_fraction": 2/3},
            "variants": args.variants, "experiments": args.experiments,
            "model": args.model, "offline": args.offline}


def memory_from(runs, root, config):
    prior = []
    for item in sorted(runs.glob("*/development-memory.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if item.parent == root:
            continue
        try:
            memory = json.loads(item.read_text(encoding="utf-8"))
            if memory["assets"] == config["assets"] and memory["last_development_timestamp"] <= config["split"]:
                prior.append(memory)
        except (OSError, ValueError, KeyError):
            continue
        if len(prior) == 3:
            break
    return prior


def parser():
    cli = argparse.ArgumentParser(prog="strategy-finder")
    actions = cli.add_subparsers(dest="action", required=True)
    investigate = actions.add_parser("investigate")
    investigate.add_argument("--asset", required=True)
    investigate.add_argument("--pair")
    investigate.add_argument("--csv", type=Path, action="append")
    investigate.add_argument("--timeframes", nargs="+", choices=tuple(STEPS), default=list(STEPS))
    investigate.add_argument("--start", type=date)
    investigate.add_argument("--end", type=date)
    investigate.add_argument("--duration", type=duration, default=8*3600)
    investigate.add_argument("--capital", type=float, default=1000)
    investigate.add_argument("--allocation", type=float, default=1)
    investigate.add_argument("--risk", type=float, default=.005)
    investigate.add_argument("--drawdown", type=float, default=.30)
    investigate.add_argument("--fee-bps", type=float, default=10)
    investigate.add_argument("--slippage-bps", type=float, default=5)
    investigate.add_argument("--variants", type=int, default=1000)
    investigate.add_argument("--experiments", type=int, default=0, help="0 means continue until the time limit")
    investigate.add_argument("--seed", type=int, default=19)
    investigate.add_argument("--offline", action="store_true")
    investigate.add_argument("--model", default=os.environ.get("OPENAI_MODEL"))
    investigate.add_argument("--runs", type=Path, default=Path("runs"))
    investigate.add_argument("--cache", type=Path, default=Path("cache/binance"))
    resume = actions.add_parser("resume")
    resume.add_argument("run", type=Path)
    resume.add_argument("--extend", type=duration)
    report = actions.add_parser("report")
    report.add_argument("run", type=Path)
    return cli


def main():
    args = parser().parse_args()
    started = time.time()
    try:
        if args.action == "investigate":
            root = args.runs.resolve() / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            root.mkdir(parents=True)
            print(f"Investigation directory: {root}", flush=True)
            with lock(root):
                ledger = Ledger(root / "research.sqlite")
                config = prepare(args, root, started)
                ledger.set("config", config)
                run(root, ledger, memory_from(root.parent, root, config))
                publish(root, ledger)
        else:
            root = args.run.resolve()
            if not (root / "research.sqlite").is_file():
                raise ValueError("No investigation database in that directory.")
            with lock(root):
                ledger = Ledger(root / "research.sqlite")
                config = ledger.get("config")
                if not config:
                    raise ValueError("Data preparation did not finish. Start again; cached archives will be reused.")
                if args.action == "resume":
                    if args.extend:
                        config["deadline"] = max(time.time(), config["deadline"]) + args.extend
                        ledger.set("config", config)
                    run(root, ledger, memory_from(root.parent, root, config))
                publish(root, ledger, summarize=args.action != "report")
        print(f"Report: {root / 'report.md'}")
        return 0
    except KeyboardInterrupt:
        print("Interrupted. Resume from the last saved candidate using the investigation directory.", file=sys.stderr)
        return 130
    except (ValueError, OSError, urllib.error.URLError, ImportError) as exc:
        print(f"strategy-finder: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
