"""Evaluate entry forecasts and matched BTC/EUR breakout replays."""
import argparse
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import median

import numpy as np

from .backtest import Execution, daily_returns, metrics, simulate, time_machine
from .data import STEPS, align, binance, partition_hash
from .research import load_data, write_json
from .strategy import Strategy, signals

STEP = STEPS["4h"]
SEED = 19
DATES = ("2024-01-01", "2024-07-01", "2025-01-01", "2025-07-01",
         "2026-01-01", "2026-07-01", "2026-11-01")


def timestamp(date):
    return int(datetime.fromisoformat(date).replace(tzinfo=timezone.utc).timestamp())


def outcome(times, bars, i, strategy, execution, end):
    """Label the bracket forecast, including same-bar stop precedence."""
    horizon = strategy.forecast_horizon
    expiry = i+horizon
    needs_open = strategy.exit_policy == "bracket"
    if expiry+needs_open > end:
        return None
    if times[expiry-1+needs_open]-times[i] != (horizon-1+needs_open)*STEP:
        return None
    slip, fee = execution.slippage_bps/10000, execution.fee_bps/10000
    entry = bars[i, 0, 0]*(1+slip)
    floor, ceiling = entry*(1-strategy.stop_pct), entry*(1+strategy.target_pct)
    selected = bars[i:expiry, 0]
    hits = np.flatnonzero((selected[:, 2] <= floor) | (selected[:, 1] >= ceiling))
    if len(hits):
        bar = selected[hits[0]]
        event = "stop" if bar[2] <= floor else "target"
        price = min(bar[0], floor) if event == "stop" else ceiling
    else:
        event = "neither"
        price = bars[expiry, 0, 0] if needs_open else selected[-1, 3]
    return event, float(price*(1-slip)*(1-fee)/(entry*(1+fee))-1)


def opportunities(times, bars, signal, strategy, execution, start, end):
    forecasts = {}
    for i in range(start+1, end):
        if signal[i-1] != 0 or times[i]-times[i-1] != STEP:
            continue
        score = time_machine(bars, times, i-1, 0, strategy, execution, SEED)
        if score is None:
            continue
        label = outcome(times, bars, i, strategy, execution, end)
        forecasts[i] = (score, label)
    return forecasts


def accepted(score, threshold):
    return score[0] >= threshold and score[3] > 0


def summarize(times, result, execution):
    m = metrics(times, result, execution.capital, STEP)
    annual = (1+m["net_return"])**(365.25/m["elapsed_days"])-1
    m.update(annual_net_return=annual,
             calmar=annual/m["max_drawdown"] if m["max_drawdown"] else None,
             entries=m["trades"]+int(result["open_position"] is not None))
    return m


def random_replays(run, count, baseline_count, n, rng, repeats):
    """Tune random admission on fill count only, before comparing P&L."""
    if count == 0:
        return [run(lambda i, leg: False)]
    replays = []
    for _ in range(repeats):
        draws = rng.random(n)
        lower, upper = 0., 1.
        rate = count/baseline_count if baseline_count else .5
        rate = min(.999, rate)
        closest = None
        for _ in range(6):
            m, result = run(lambda i, leg: draws[i] < rate)
            difference = m["entries"]-count
            if closest is None or abs(difference) < closest[0]:
                closest = (abs(difference), m, result)
            if not difference:
                break
            if difference > 0:
                upper = rate
            else:
                lower = rate
            rate = (upper+lower)/2
        replays.append(closest[1:])
    return replays


def shortlist(source):
    results = json.loads((source/"hourly-results.json").read_text(encoding="utf-8"))
    strategies = {}
    for row in results["holdout"]:
        if row["pick"] != "shortlist" or row["candidate"] in strategies:
            continue
        with closing(sqlite3.connect(source/"BTC"/row["arm"]/"research.sqlite")) as db:
            raw, = db.execute("SELECT strategy FROM candidates WHERE id=?", (row["candidate"],)).fetchone()
        strategies[row["candidate"]] = Strategy.from_dict(json.loads(raw))
    return dict(sorted(strategies.items()))


def calibration(data, strategies, execution):
    times, bars = data
    holds = []
    for strategy in strategies.values():
        signal = signals(times, bars, strategy, {})
        result = simulate(times, bars, signal, strategy, execution, 0, len(times))
        holds.extend((t["exit_timestamp"]-t["entry_timestamp"])//STEP+1 for t in result["trades"])
    horizon = int(np.clip(np.quantile(holds, .75, method="higher"), 1, 48))
    thresholds = (0., .1, .2, .3, .4, .5)
    results = {value: [] for value in thresholds}
    baseline_count = 0
    targets = []
    for strategy in strategies.values():
        strategy = with_horizon(strategy, horizon)
        signal = signals(times, bars, strategy, {})
        forecasts = opportunities(times, bars, signal, strategy, execution, 0, len(times))
        targets.extend(label[0] == "target" for _, label in forecasts.values() if label is not None)
        base = simulate(times, bars, signal, strategy, execution, 0, len(times))
        baseline_count += len(base["trades"])+int(base["open_position"] is not None)
        for threshold in thresholds:
            def gate(i, leg):
                return i in forecasts and accepted(forecasts[i][0], threshold)
            result = simulate(times, bars, signal, strategy, execution, 0, len(times), entry_filter=gate)
            results[threshold].append(summarize(times, result, execution))
    grid = [{"threshold": threshold, "median_return": median(m["net_return"] for m in values),
             "entry_retention": sum(m["entries"] for m in values)/baseline_count}
            for threshold, values in results.items()]
    eligible = [r for r in grid if r["entry_retention"] >= .25]
    best = max(eligible, key=lambda r: r["median_return"]) if eligible else grid[0]
    return {"horizon_bars": horizon, "threshold": best["threshold"], "grid": grid,
            "calibration_target_rate": sum(targets)/len(targets),
            "met_retention_floor": bool(eligible)}


def with_horizon(strategy, bars):
    return replace(strategy, forecast_bars=min(bars, strategy.max_hold or bars))


def experiment(source, output, cache, smoke=False):
    config = json.loads((source/"BTC"/"config.json").read_text(encoding="utf-8"))
    execution = Execution(**config["execution"])
    strategies = shortlist(source)
    if smoke:
        strategies = dict(list(strategies.items())[:3])
    old_times, old_bars, _ = load_data(source/"BTC", config)["4h"]
    settings = calibration((old_times, old_bars), strategies, execution)
    print("Calibration", settings, flush=True)
    dates = DATES[:2] if smoke else DATES
    times, bars = align([binance("BTCEUR", timestamp("2023-08-01"), timestamp(dates[-1]), cache)], "4h")
    forecasts_log, comparisons, month_deltas = [], [], {}
    rng = np.random.default_rng(SEED)
    for first, last in zip(dates, dates[1:]):
        start, end = np.searchsorted(times, [timestamp(first), timestamp(last)])
        if start >= end or times[start] != timestamp(first) or times[end-1]+STEP != timestamp(last):
            raise ValueError("The requested period has incomplete boundary candles.")
        period_times = times[start:end]
        months = np.array([datetime.fromtimestamp(int(t)+STEP-1, timezone.utc).strftime("%Y-%m")
                           for t in period_times])
        day_ends = np.r_[np.flatnonzero((period_times[1:]+STEP-1)//86400 !=
                                      (period_times[:-1]+STEP-1)//86400), len(period_times)-1]
        for key, original in strategies.items():
            strategy = with_horizon(original, settings["horizon_bars"])
            signal = signals(times, bars, strategy, {})
            forecasts = opportunities(times, bars, signal, strategy, execution, start, end)
            threshold = settings["threshold"]
            gate_calls = set()
            def gate(i, leg):
                gate_calls.add(i)
                return i in forecasts and accepted(forecasts[i][0], threshold)

            def run(entry_filter):
                result = simulate(times, bars, signal, strategy, execution, start, end, entry_filter=entry_filter)
                return summarize(period_times, result, execution), result

            baseline, baseline_result = run(None)
            gated, gated_result = run(gate)
            controls = random_replays(run, gated["entries"], baseline["entries"], len(times), rng, 5 if smoke else 30)
            comparisons.append({"period": first, "strategy": key, "horizon_bars": strategy.forecast_horizon,
                                "baseline": baseline, "gated": gated, "random": [m for m, _ in controls]})
            baseline_entries = {t["entry_timestamp"]: t["pnl"] for t in baseline_result["trades"]}
            if baseline_result["open_position"]:
                baseline_entries[baseline_result["open_position"]["entry_timestamp"]] = None
            gated_entries = {t["entry_timestamp"] for t in gated_result["trades"]}
            if gated_result["open_position"]:
                gated_entries.add(gated_result["open_position"]["entry_timestamp"])
            for i, (score, label) in forecasts.items():
                entry = int(times[i])
                forecasts_log.append({"period": first, "strategy": key, "timestamp": entry,
                                      "p_target": score[0], "p_stop": score[1], "p_neither": score[2],
                                      "expected_net_return": score[3], "label": label,
                                      "approved": accepted(score, threshold), "gate_called": i in gate_calls,
                                      "baseline_entered": entry in baseline_entries,
                                      "baseline_closed_trade_pnl": baseline_entries.get(entry),
                                      "gate_entered": entry in gated_entries})

            def logs(result):
                return np.log1p(daily_returns(period_times, result["equity"], execution.capital, STEP))

            gate_daily = logs(gated_result)
            deltas = np.stack((gate_daily-logs(baseline_result),
                               gate_daily-np.mean([logs(r) for _, r in controls], axis=0)))
            for month in np.unique(months):
                month_deltas.setdefault(month, []).append(deltas[:, months[day_ends] == month].sum(axis=1))
        print(first, last, len(strategies), "strategies", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/"results.json", {
        "settings": settings, "strategies": {k: v.to_dict() for k, v in strategies.items()},
        "data": {"hash": partition_hash(times, bars), "periods": dates},
        "forecasts": forecasts_log, "replays": comparisons,
        "monthly_mean_log_return_deltas": {k: np.mean(v, axis=0).tolist() for k,v in month_deltas.items()},
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("cache/binance"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    experiment(args.source, args.output, args.cache, args.smoke)
