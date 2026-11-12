"""Test a forecast-driven exit while preserving the baseline entry schedule."""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .backtest import Execution, daily_returns, forecast_paths, simulate
from .data import DAY, align, binance, partition_hash
from .research import write_json
from .strategy import signals
from .time_machine_replay import DATES, SEED, STEP, shortlist, summarize, timestamp

EXIT_EDGE = .0025


def forecast_exit_edge(paths, observed_closes, strategy, entry, slip, fee):
    """Expected net advantage of selling next open over keeping existing exits."""
    floor, ceiling = entry*(1-strategy.stop_pct), entry*(1+strategy.target_pct)
    horizon = paths.shape[1]-1
    result = paths[:, -1, 0].copy()
    live = np.ones(len(paths), dtype=bool)
    for j in range(horizon):
        opening, high, low, close = paths[:, j, :4].T
        if j and strategy.exit_policy == "signal":
            n = strategy.rule.lookback
            past = paths[:, max(0, j-1-n):j-1, 3]
            reference = past.max(axis=1) if past.shape[1] else np.zeros(len(paths))
            missing = n-past.shape[1]
            if missing:
                reference = np.maximum(reference, observed_closes[-missing:].max())
            signal = live & (paths[:, j-1, 3]/reference-1 < strategy.rule.exit)
            result[signal] = opening[signal]
            live &= ~signal
        stopped = live & (low <= floor)
        result[stopped] = np.minimum(opening[stopped], floor)
        live &= ~stopped
        reached = live & (high >= ceiling)
        result[reached] = ceiling
        live &= ~reached
    return float((paths[:, 0, 0]-result).mean()*(1-slip)*(1-fee))


def entries(result):
    records = {t["entry_timestamp"]: t for t in result["trades"]}
    if result["open_position"]:
        records[result["open_position"]["entry_timestamp"]] = None
    return records


def replay(source, entry_test, output, cache, smoke=False):
    config = json.loads((source/"BTC"/"config.json").read_text(encoding="utf-8"))
    execution = Execution(**config["execution"])
    horizon = json.loads(entry_test.read_text(encoding="utf-8"))["settings"]["horizon_bars"]
    slip, fee = execution.slippage_bps/10000, execution.fee_bps/10000
    strategies = shortlist(source)
    if smoke:
        strategies = dict(list(strategies.items())[:3])
    dates = DATES[:2] if smoke else DATES
    times, bars = align([binance("BTCEUR", timestamp("2023-08-01"), timestamp(dates[-1]), cache)], "4h")
    reports, decisions, month_deltas = [], [], {}
    for first, last in zip(dates, dates[1:]):
        start, end = np.searchsorted(times, [timestamp(first), timestamp(last)])
        if start >= end or times[start] != timestamp(first) or times[end-1]+STEP != timestamp(last):
            raise ValueError("The requested period has incomplete boundary candles.")
        active = times[start:end]
        ends = np.r_[np.flatnonzero((active[1:]+STEP-1)//DAY != (active[:-1]+STEP-1)//DAY), len(active)-1]
        months = np.array([datetime.fromtimestamp(int(t)+STEP-1, timezone.utc).strftime("%Y-%m")
                           for t in active[ends]])
        path_cache = {}
        for key, strategy in strategies.items():
            if strategy.rule.indicator != "breakout" or strategy.rule.direction != "above":
                raise ValueError("Exit forecast expects the frozen long breakout rules.")
            signal = signals(times, bars, strategy, {})
            baseline = simulate(times, bars, signal, strategy, execution, start, end)
            schedule = entries(baseline)
            forecasts = {}

            def exit_at_open(i, leg, entry_i, entry_price):
                remaining = horizon
                if strategy.max_hold:
                    remaining = min(remaining, strategy.max_hold-(times[i]-times[entry_i])//STEP)
                sample_key = (i, remaining)
                if sample_key not in path_cache:
                    path_cache[sample_key] = forecast_paths(bars, times, i-1, leg, remaining, STEP, True, SEED)
                sampled = path_cache[sample_key]
                if sampled is None:
                    return False
                anchor = bars[i-1, leg, 3]
                history = bars[max(0, i-strategy.rule.lookback):i, leg, 3]/anchor
                edge = forecast_exit_edge(sampled[0], history, strategy, entry_price/anchor, slip, fee)
                if edge > EXIT_EDGE:
                    forecasts[int(times[i])] = edge
                return edge > EXIT_EDGE

            managed = simulate(times, bars, signal, strategy, execution, start, end,
                               entry_filter=lambda i, leg: int(times[i]) in schedule,
                               early_exit=exit_at_open)
            managed_entries = entries(managed)
            if schedule.keys() != managed_entries.keys():
                raise ValueError("The forecast exit changed the baseline entry schedule.")
            for trade in managed["trades"]:
                if trade["reason"] != "forecast_exit":
                    continue
                other = schedule[trade["entry_timestamp"]]

                def net_return(t):
                    return t["pnl"]/(t["quantity"]*t["entry"]*(1+fee))

                decisions.append({"period": first, "strategy": key, "entry": trade["entry_timestamp"],
                                  "forecast_exit": trade["exit_timestamp"],
                                  "forecast_edge": forecasts[trade["exit_timestamp"]],
                                  "managed_trade_return": net_return(trade),
                                  "baseline_trade_return": net_return(other) if other else None,
                                  "baseline_exit": other["exit_timestamp"] if other else None,
                                  "baseline_exit_reason": other["reason"] if other else None})
            reports.append({"period": first, "strategy": key,
                            "baseline": summarize(active, baseline, execution),
                            "managed": summarize(active, managed, execution)})
            delta = (np.log1p(daily_returns(active, managed["equity"], execution.capital, STEP))
                     - np.log1p(daily_returns(active, baseline["equity"], execution.capital, STEP)))
            for month in np.unique(months):
                month_deltas.setdefault(month, []).append(float(delta[months == month].sum()))
        print(first, last, len(strategies), "strategies", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"results.json", {
        "settings": {"horizon_bars": horizon, "minimum_edge": EXIT_EDGE},
        "data": {"hash": partition_hash(times, bars), "periods": dates},
        "replays": reports, "forecast_exits": decisions,
        "monthly_mean_log_return_deltas": {k: float(np.mean(v)) for k, v in month_deltas.items()},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("entry_test", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("cache/binance"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    replay(args.source, args.entry_test, args.output, args.cache, args.smoke)
