"""Run four independent matched-search arms against sealed market holdouts."""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import os
import random
import shutil
import sqlite3
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from threading import Event

from . import planner
from .backtest import Execution
from .cli import lock
from .feedback import campaign_map, development_score
from . import hourly_benchmark, hourly_forecasts
from .data import STEPS
from .learning_benchmark import APIUnavailable, BudgetExhausted, budgeted_response
from .matched_planner_test import (HOURLY_OPTIONS, OPTIONS, choose_program, evaluate_chunk, freeze,
                                   identity, make_choices, recipe, record_behavior, run_round, setup_worker)
from .research import Ledger, assess, encode, experiment_report, load_data, write_json
from .strategy import Strategy, signals
from .windows_launcher import load_openai_settings

ARMS = ("random", "programmatic", "luna_no_feedback", "luna_feedback")
CRITERION = {"minimum_mean_return_lift": .005, "minimum_market_wins": 2,
             "maximum_mean_drawdown_lift": 0}


def utc(stamp):
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def paired_chunk(chunk, seed, deadline):
    results = []
    for start in range(0, len(chunk), 2):
        if time.time() >= deadline:
            break
        pair = evaluate_chunk(chunk[start:start+2], seed)
        if time.time() >= deadline:
            break
        results.extend(pair)
    return results


def evaluate_batch(ledger, pool, experiment, protocol, deadline, progress, halt):
    pending = [(r["id"], json.loads(r["strategy"])) for r in ledger.db.execute(
        "SELECT id,strategy FROM candidates WHERE experiment=? AND development IS NULL ORDER BY rowid",
        (experiment,))]
    if len(pending) % 2:
        raise ValueError("Pending trials must contain complete pairs.")
    chunk_size = 60
    if protocol.get("timeframe") == "1h":
        pairs = list(zip(pending[::2], pending[1::2]))
        pairs.sort(key=lambda pair: (pair[1][1]["stop_pct"], pair[1][1]["target_pct"]))
        pending = [candidate for pair in pairs for candidate in pair]
        chunk_size = 12
    chunks = iter(pending[i:i+chunk_size] for i in range(0, len(pending), chunk_size))
    futures = set()
    completed = 0

    def dispatch():
        if not halt.is_set() and time.time() < deadline and (chunk := next(chunks, None)):
            futures.add(pool.submit(paired_chunk, chunk, protocol["seed"], deadline))

    for _ in range(protocol["workers_per_arm"]*2):
        dispatch()
    while futures:
        finished, futures = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
        for future in finished:
            rows = future.result()
            if rows:
                with ledger.db:
                    ledger.db.executemany("UPDATE candidates SET development=? WHERE id=?", rows)
                completed += len(rows)
            dispatch()
        progress(completed)


def pick_from_batch(ledger, experiment):
    best, key = None, None
    for row in ledger.db.execute(
        "SELECT id,strategy,development FROM candidates WHERE experiment=? AND development IS NOT NULL ORDER BY rowid",
        (experiment,),
    ):
        metrics = json.loads(row["development"])
        score = development_score(metrics)
        if key is None or score > key:
            key = score
            best = {"id": row["id"], "experiment": experiment, "strategy": json.loads(row["strategy"]),
                    "development": metrics}
    return best


def select_action(root, arm, history, protocol, deadline, allowed):
    number = len(history)
    seen = {e["proposal"]["indicator"] for e in history}
    hourly = protocol.get("timeframe") == "1h"
    if hourly and number <= 5:
        followup_families = {e["proposal"]["indicator"] for e in history[1:]}
        novel = [a for a in allowed if a.split(":")[0] not in followup_families]
        if novel:
            allowed = novel
    elif number % 4 == 3:
        novel = [a for a in allowed if a.split(":")[0] not in seen]
        if novel:
            allowed = novel
    prior = [{"experiment": e["experiment"], "choice": e["proposal"]["choice"]} for e in history]
    common = {"assets": protocol["current_assets"], "round": number, "allowed_actions": allowed,
              "prior_choices": prior}
    if arm == "random":
        rng = random.Random(protocol["seed"]+number)
        return {"choice": rng.choice(allowed), "parent": None,
                "reason": "Seeded selection from the eligible menu.", "expectation": None}, {"model": "random"}
    evidence = [{"experiment": e["experiment"], "proposal": e["proposal"], "effects": e["matched"]}
                for e in history]
    if arm == "programmatic":
        if hourly:
            qualifying = [e for e in history if e["proposal"]["choice"] in allowed
                          and e["diversity"]["new_10"]]
            if qualifying:
                best = max(qualifying, key=lambda e: (e["diversity"]["new_10"], e["diversity"]["new_15"]))
                return {"choice": best["proposal"]["choice"], "parent": best["experiment"],
                        "reason": "Repeat the design with the most new qualifying trade histories.",
                        "expectation": "Fresh settings produce additional qualifying histories."}, {"model": "programmatic"}
        return choose_program({"experiments": evidence}, allowed), {"model": "programmatic"}
    prompt = ("Choose one eligible matched experiment. Both AI planners use the same local grids. "
              "Each action combines an entry family with one change, removing the signal exit or enabling Time Machine. "
              "State an observable expectation. Stay within the action menu.")
    if hourly:
        prompt += (" Data are hourly spot candles. Entry_confirmation adds a same-timeframe trend gate. "
                   "Time_machine_20, _30 and _40 are minimum target-hit probabilities within 24 hours. "
                   "Trades can remain open beyond the forecast horizon. "
                   "Seek novel trading behavior and strategies clearing 10% annualized net return, "
                   "30% drawdown and 0.8 Sharpe. Use evidence of qualified discoveries when provided.")
    payload = common
    if arm == "luna_feedback":
        compact = campaign_map(history)
        for entry, e in zip(compact, evidence):
            windows = e["effects"]["windows"]
            entry["positive_windows"] = sum((w["median_net_return_delta"] or 0) > 0 for w in windows)
            entry["windows"] = len(windows)
        payload = {**common, "campaign_map": compact, "latest_experiment": evidence[-1]}
        if hourly:
            payload["qualified_discoveries"] = [
                {"choice": e["proposal"]["choice"], **e["diversity"]}
                for e in history if "diversity" in e
            ]
        prompt += (" Use paired development evidence. Name a parent when refining. "
                   "Consider drawdown and cross-window stability.")
    else:
        prompt += " No backtest results are available. Use market and prior choices only. Set parent to null."
    schema = planner.obj({"choice": {"type": "string", "enum": allowed},
                          "parent": {"type": ["integer", "null"]},
                          "reason": planner.STRING, "expectation": planner.STRING})
    decision, provider = budgeted_response(root/"api-spending.sqlite",
                                           round(protocol["max_api_usd"]*1_000_000), protocol["model"],
                                           prompt, payload, schema, deadline)
    parents = {None, *(e["experiment"] for e in history)} if arm == "luna_feedback" else {None}
    if decision["choice"] not in allowed or decision["parent"] not in parents:
        raise ValueError("Planner chose an ineligible action or parent.")
    return decision, provider


def start_batch(root, arm, ledger, history, protocol, deadline):
    exhausted = set(ledger.get("exhausted_actions", []))
    timeframe = protocol.get("timeframe", "1d")
    menu = HOURLY_OPTIONS if timeframe == "1h" else OPTIONS
    for _ in menu:
        allowed = [a for a in menu if a not in exhausted]
        if not allowed or time.time() >= deadline:
            return None
        decision, provider = select_action(root, arm, history, protocol, deadline, allowed)
        spec = recipe(decision["choice"], decision["parent"], decision["reason"], decision["expectation"], timeframe)
        if time.time() >= deadline:
            return None
        try:
            choices = make_choices(ledger, spec, protocol["batch"], protocol["seed"]+len(history))
        except ValueError:
            exhausted.add(decision["choice"])
            ledger.set("exhausted_actions", sorted(exhausted))
            continue
        if time.time() >= deadline:
            return None
        with ledger.db:
            experiment = ledger.db.execute("INSERT INTO experiments(proposal,provider,variants) VALUES (?,?,?)",
                                          (encode(spec), encode(provider), encode(choices))).lastrowid
            ledger.db.executemany("INSERT INTO candidates(id,experiment,strategy) VALUES (?,?,?)",
                                  [(identity(s), experiment, encode(s)) for s in choices])
        return experiment
    return None


def settle_batch(ledger, history, experiment, protocol, clock, leader, total):
    count = ledger.db.execute("SELECT count(development) FROM candidates WHERE experiment=?",
                              (experiment,)).fetchone()[0]
    status = "complete" if count == protocol["batch"] else "partial"
    if not count:
        with ledger.db:
            ledger.db.execute("UPDATE experiments SET status='partial' WHERE id=?", (experiment,))
        return leader, total
    if count % 2:
        raise ValueError("Only complete pairs can enter a checkpoint.")
    new = pick_from_batch(ledger, experiment)
    if development_score(new["development"]) > development_score(leader["development"]):
        leader = new
    total += count
    item, = experiment_report(ledger, limit=1)
    item["status"] = status
    checkpoint = {"experiment": item, "leader": leader, "batch_leader": new,
                  "backtests": total, "complete": status=="complete",
                  "elapsed_seconds": min(protocol["seconds"], max(0, time.time()-clock["start"]))}
    with ledger.db:
        if protocol.get("timeframe") == "1h":
            item["diversity"] = hourly_benchmark.record_batch(ledger, experiment)
        ledger.db.execute("UPDATE experiments SET status=? WHERE id=?", (status, experiment))
        ledger.db.execute("INSERT INTO batch_checkpoints VALUES (?,?)", (experiment, encode(checkpoint)))
    history.append(checkpoint)
    return leader, total


def run_arm(root, setup, arm, config, protocol, clock, halt):
    directory = setup/arm
    if (directory/"endpoint.json").exists():
        return
    ledger = Ledger(directory/"research.sqlite")
    with closing(ledger.db):
        ledger.db.execute("CREATE TABLE IF NOT EXISTS batch_checkpoints (experiment INTEGER PRIMARY KEY, result TEXT NOT NULL)")
        history = [json.loads(r[0]) for r in ledger.db.execute("SELECT result FROM batch_checkpoints ORDER BY experiment")]
        shared = read(setup/"shared"/"rounds.json")
        leader = history[-1]["leader"] if history else read(setup/"shared"/"leader.json")
        total = history[-1]["backtests"] if history else 0
        deadline = clock["deadline"]
        last_report = 0
        consecutive_failures = 0
        stop_reason = "deadline"

        def progress(active=0, force=False):
            nonlocal last_report
            now = time.time()
            if force or now-last_report >= 60:
                write_json(directory/"progress.json", {"asset": config["assets"][0], "arm": arm,
                           "backtests": total+active, "completed_batches": sum(x["complete"] for x in history),
                           "updated": utc(now), "deadline": utc(deadline)})
                last_report = now

        progress(force=True)
        if time.time() < clock["start"]:
            time.sleep(max(0, clock["start"]-time.time()))
        if time.time() < deadline:
            forecast_seed = protocol["seed"] if protocol.get("timeframe") == "1h" else None
            with ProcessPoolExecutor(max_workers=protocol["workers_per_arm"], initializer=setup_worker,
                                     initargs=(setup, config, forecast_seed)) as pool:
                while not halt.is_set() and time.time() < deadline:
                    if protocol["max_batches"] and len(history) >= protocol["max_batches"]:
                        stop_reason = "batch_cap"
                        break
                    row = ledger.db.execute("SELECT id FROM experiments WHERE status='open' ORDER BY id LIMIT 1").fetchone()
                    if row:
                        experiment = row[0]
                    else:
                        try:
                            experiment = start_batch(root, arm, ledger, shared+[x["experiment"] for x in history],
                                                     protocol, deadline)
                        except (TimeoutError, ValueError, OSError) as exc:
                            ledger.event("proposal_failure", {"error": str(exc)})
                            if isinstance(exc, (APIUnavailable, BudgetExhausted)):
                                raise
                            consecutive_failures += 1
                            if consecutive_failures >= 10:
                                raise
                            time.sleep(min(2, max(0, deadline-time.time())))
                            continue
                    if experiment is None:
                        if time.time() < deadline:
                            stop_reason = "no_fresh_pairs"
                        break
                    consecutive_failures = 0
                    before = ledger.db.execute("SELECT count(development) FROM candidates WHERE experiment=?",
                                               (experiment,)).fetchone()[0]
                    evaluate_batch(ledger, pool, experiment, protocol, deadline,
                                   lambda active: progress(before+active), halt)
                    if halt.is_set():
                        return
                    leader, total = settle_batch(ledger, history, experiment, protocol, clock, leader, total)
                    progress(force=True)
                    print(config["assets"][0], arm, f"{total:,} follow-up backtests", flush=True)
        if halt.is_set():
            return
        row = ledger.db.execute("SELECT id FROM experiments WHERE status='open' ORDER BY id LIMIT 1").fetchone()
        if row:
            leader, total = settle_batch(ledger, history, row[0], protocol, clock, leader, total)
        if not total:
            raise ValueError(f"{arm} finished without any follow-up pairs.")
        write_json(directory/"checkpoints.json", history)
        shortlist = hourly_benchmark.freeze_shortlist(ledger) if protocol.get("timeframe") == "1h" else []
        write_json(directory/"endpoint.json", {"leader": leader, "backtests": total, "shortlist": shortlist,
                   "completed_batches": sum(x["complete"] for x in history), "finished": utc(time.time()),
                   "stop_reason": stop_reason})
        progress(force=True)
        print(config["assets"][0], arm, "frozen", total, "follow-up backtests", flush=True)


def prepare_shared(setup, config, protocol):
    directory = setup/"shared"
    directory.mkdir(exist_ok=True)
    shared_db = directory/"research.sqlite"
    if not (directory/"leader.json").exists():
        ledger = Ledger(shared_db)
        with closing(ledger.db), ProcessPoolExecutor(max_workers=min(os.cpu_count() or 1, 10),
                       initializer=setup_worker, initargs=(setup, config)) as pool:
            history = {"shared": read(directory/"rounds.json", [])}
            run_round(pool, setup, {"shared": ledger}, 0,
                      {"seed": protocol["seed"], "backtests_per_batch": protocol["batch"],
                       "timeframe": protocol.get("timeframe", "1d")}, history)
            if protocol.get("timeframe") == "1h":
                with ledger.db:
                    history["shared"][0]["diversity"] = hourly_benchmark.record_batch(ledger, 1)
                write_json(directory/"rounds.json", history["shared"])
            write_json(directory/"leader.json", freeze(ledger, 0))
    for arm in ARMS:
        target = setup/arm
        target.mkdir(exist_ok=True)
        if not (target/"research.sqlite").exists():
            temp = target/"research-copy.tmp"
            shutil.copyfile(shared_db, temp)
            temp.replace(target/"research.sqlite")


def discovery(root, setups, protocol):
    load_openai_settings(Path(".env"))
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Configure OPENAI_API_KEY before starting the benchmark.")
    with closing(sqlite3.connect(root/"api-spending.sqlite", timeout=30)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, microdollars INTEGER)")
        db.execute("CREATE TABLE IF NOT EXISTS api_block (id INTEGER PRIMARY KEY, reason TEXT NOT NULL)")
        db.execute("DELETE FROM api_block")
    for setup, config in setups:
        if all((setup/arm/"endpoint.json").exists() for arm in ARMS):
            continue
        write_json(root/"status.json", {"phase": "shared_start", "market": config["assets"][0], "pid": os.getpid()})
        prepare_shared(setup, config, protocol)
        if protocol.get("timeframe") == "1h":
            write_json(root/"status.json", {"phase": "forecast_grid", "market": config["assets"][0],
                                           "pid": os.getpid()})
            hourly_forecasts.prepare_grid(setup, config, protocol["seed"], min(os.cpu_count() or 1, 10))
        clock_path = setup/"clock.json"
        clock = read(clock_path)
        if not clock:
            start = time.time()+3
            clock = {"start": start, "deadline": start+protocol["seconds"]}
            write_json(clock_path, clock)
        write_json(root/"status.json", {"phase": "discovery", "market": config["assets"][0],
                   "deadline": utc(clock["deadline"]), "pid": os.getpid()})
        print(config["assets"][0], "four-arm discovery deadline", utc(clock["deadline"]), flush=True)
        rules = {**protocol, "current_assets": config["assets"]}
        halt = Event()
        with ThreadPoolExecutor(max_workers=len(ARMS)) as pool:
            futures = [pool.submit(run_arm, root, setup, arm, config, rules, clock, halt) for arm in ARMS]
            try:
                for future in as_completed(futures):
                    future.result()
            except BaseException:
                halt.set()
                raise
    write_json(root/"status.json", {"phase": "holdouts_ready", "pid": os.getpid()})


def evaluate(root, setups, protocol):
    if not all((setup/arm/"endpoint.json").is_file() for setup, _ in setups for arm in ARMS):
        raise ValueError("Every market and arm must freeze an endpoint before any holdout opens.")
    write_json(root/"status.json", {"phase": "evaluation", "pid": os.getpid()})
    rows = []
    for setup, config in setups:
        timeframe = protocol.get("timeframe", "1d")
        times, bars, warmup = load_data(setup, config, final=True)[timeframe]
        execution = Execution(**config["execution"])
        cache = {}

        def add(arm, kind, count, elapsed, candidate, complete):
            key = identity(candidate["strategy"])
            if key not in cache:
                strategy = Strategy.from_dict(candidate["strategy"])
                cache[key], trades = assess(times, bars, signals(times, bars, strategy, {}), strategy, execution,
                                       config["drawdown"], protocol["seed"], final=True, warmup=warmup,
                                       policy=config.get("evidence"))
                if timeframe == "1h":
                    record_behavior(cache[key], trades)
            result = cache[key]
            rows.append({"market": config["assets"][0], "arm": arm, "pick": kind,
                         "backtests": count, "elapsed_seconds": elapsed, "complete_batch": complete,
                         "net_return": result["net_return"], "drawdown": result["max_drawdown"],
                         "qualified": result["outcome"]=="qualified", "candidate": key})
            if timeframe == "1h":
                rows[-1].update({k: result[k] for k in ("annual_net_return", "sharpe", "trades",
                                                       "win_rate", "median_holding_hours", "held_over_24h")})
                rows[-1].update(qualified_10=hourly_benchmark.qualifies(result),
                                qualified_15=hourly_benchmark.qualifies(result, .15),
                                development={k: candidate["development"][k] for k in (
                                    "annual_net_return", "max_drawdown", "sharpe", "trades",
                                    "median_holding_hours", "held_over_24h")})

        add("shared", "shared", 0, 0, read(setup/"shared"/"leader.json"), True)
        for arm in ARMS:
            for batch in read(setup/arm/"checkpoints.json", []):
                for label in ("leader", "batch_leader"):
                    add(arm, label, batch["backtests"], batch["elapsed_seconds"], batch[label], batch["complete"])
            end = read(setup/arm/"endpoint.json")
            add(arm, "endpoint", end["backtests"], protocol["seconds"], end["leader"], False)
            for candidate in end.get("shortlist", []):
                add(arm, "shortlist", end["backtests"], None, candidate, True)
    if protocol.get("timeframe") == "1h":
        hourly_benchmark.write_summary(root, setups, ARMS, read, rows)
    else:
        write_results(root, rows, protocol, setups)
    write_json(root/"status.json", {"phase": "complete", "report": "summary.md", "finished": utc(time.time())})


def write_results(root, rows, protocol, setups):
    with (root/"checkpoint-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    markets = [c["assets"][0] for _, c in setups]
    endpoints = {(r["market"], r["arm"]):r for r in rows if r["pick"]=="endpoint"}
    lines = ["# Long learning benchmark", "",
             "Holdouts opened after every endpoint was frozen. Trial counts exclude the shared starting batch.",
             "", "| Market | Arm | Backtests | Net return | Max drawdown | Qualified |",
             "|---|---|---:|---:|---:|---|"]
    for market in markets:
        for arm in ARMS:
            r = endpoints[market,arm]
            lines.append(f"| {market} | {arm} | {r['backtests']:,} | {r['net_return']:.2%} | "
                         f"{r['drawdown']:.2%} | {'Yes' if r['qualified'] else 'No'} |")
    comparison = {}
    lines.extend(["", "| Luna with feedback compared with | Mean return lift | Markets won | Mean drawdown lift | Pass |",
                  "|---|---:|---:|---:|---|"])
    for control in ("random", "programmatic", "luna_no_feedback"):
        pairs = [(endpoints[m,"luna_feedback"], endpoints[m,control]) for m in markets]
        deltas = [a["net_return"]-b["net_return"] for a,b in pairs]
        net, wins = mean(deltas), sum(d>0 for d in deltas)
        dd = mean(a["drawdown"]-b["drawdown"] for a,b in pairs)
        rule = protocol["criterion"]
        passed = net>=rule["minimum_mean_return_lift"] and wins>=rule["minimum_market_wins"] and dd<=rule["maximum_mean_drawdown_lift"]
        comparison[control] = {"return_lift": net, "market_wins": wins, "drawdown_lift": dd, "passed": passed}
        lines.append(f"| {control} | {net*100:+.2f} pp | {wins}/{len(markets)} | {dd*100:+.2f} pp | {'Yes' if passed else 'No'} |")
    lines.extend(["", "## Equal-trial comparison", "",
                  "| Market | Largest common completed batch count | Random | Programmatic | Luna without feedback | Luna with feedback |",
                  "|---|---:|---:|---:|---:|---:|"])
    matched = []
    for market in markets:
        series = {arm: {r["backtests"]:r for r in rows if r["market"]==market and r["arm"]==arm
                         and r["pick"]=="leader" and r["complete_batch"]} for arm in ARMS}
        common = set.intersection(*(set(s) for s in series.values()))
        for n in sorted(common):
            for arm in ARMS:
                matched.extend(r for r in rows if r["market"] == market and r["arm"] == arm
                               and r["backtests"] == n and r["pick"] in ("leader", "batch_leader"))
        if common:
            n = max(common)
            values = " | ".join(f"{series[arm][n]['net_return']:.2%}" for arm in ARMS)
            lines.append(f"| {market} | {n:,} | {values} |")
    with (root/"equal-trial-results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(matched)
    passed = all(comparison[a]["passed"] for a in ("programmatic", "luna_no_feedback"))
    lines.extend(["", f"Primary feedback decision rule: **{'passed' if passed else 'not met'}**.", "",
                  "[Cumulative and new-candidate checkpoints](checkpoint-results.csv) · "
                  "[Equal-trial checkpoints](equal-trial-results.csv) · [Frozen protocol](protocol.json)", ""])
    write_json(root/"results.json", {"endpoint_comparisons": comparison, "passed": passed, "checkpoints": rows})
    (root/"summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def show_status(root, watch):
    while True:
        status = read(root/"status.json", {"phase": "not_started"})
        protocol = read(root/"protocol.json", {})
        print("\n" + datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"), flush=True)
        print("Phase:", status["phase"], "| Market:", status.get("market", "—"),
              "| Runner PID:", status.get("pid", "—"), flush=True)
        if status.get("deadline"):
            print("Current market deadline:",
                  datetime.fromisoformat(status["deadline"]).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
                  flush=True)
        for market in protocol.get("market_order", []):
            cells = []
            for arm in ARMS:
                directory = root/market/arm
                endpoint = read(directory/"endpoint.json")
                progress = read(directory/"progress.json")
                if endpoint:
                    cell = f"{endpoint['backtests']:,} frozen"
                elif progress:
                    cell = f"{progress['backtests']:,} ({progress['completed_batches']} batches)"
                else:
                    cell = "waiting"
                cells.append(f"{arm}: {cell}")
            print(f"{market} | " + " | ".join(cells), flush=True)
        if status["phase"] == "complete":
            print("Report:", root/"summary.md", flush=True)
        elif status["phase"] == "failed":
            print("Stopped:", status.get("error", "See runner logs."), flush=True)
        if not watch or status["phase"] in {"complete", "failed"}:
            return
        print("Updates every 30 seconds. Ctrl+C closes this monitor only.", flush=True)
        try:
            time.sleep(30)
        except KeyboardInterrupt:
            return


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("discover", "evaluate", "run", "status"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--markets", nargs="+")
    parser.add_argument("--timeframe", choices=("1d", "1h"), default="1d")
    parser.add_argument("--watch", action="store_true", help="Refresh the status display every 30 seconds.")
    parser.add_argument("--seconds", type=int, default=28800)
    parser.add_argument("--batch", type=int, default=10000)
    parser.add_argument("--workers-per-arm", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=0, help="Optional smoke-test batch cap.")
    parser.add_argument("--max-api-usd", type=float, default=12)
    args = parser.parse_args()
    if args.phase == "status":
        show_status(args.root.resolve(), args.watch)
        return
    if not args.markets:
        parser.error("Run, discover, and evaluate require --markets.")
    if (args.seconds < 5 or not 2 <= args.batch <= 10000 or args.batch % 2 or args.workers_per_arm < 1
            or args.max_batches < 0 or not 0 < args.max_api_usd <= 12):
        parser.error("Invalid duration, batch size, worker count, or spending limit.")
    root = args.root.resolve()
    setups = [(root/m, read(root/m/"config.json")) for m in args.markets]
    previous_end = 0
    for setup, config in setups:
        if not config or list(config["datasets"]) != [args.timeframe] or len(config["assets"]) != 1:
            raise ValueError(f"{setup} needs a prepared single-asset {args.timeframe} snapshot.")
        if config["split"] < previous_end:
            raise ValueError("Market holdouts must be non-overlapping and ordered by date.")
        previous_end = config["datasets"][args.timeframe]["last_bar"]+STEPS[args.timeframe]
        if not config["execution"].get("fixed_allocation") or config["execution"]["allocation"] != .1:
            raise ValueError("Every setup must use fixed 10% entry allocation.")
    protocol = {"markets": {p.name: {"assets": c["assets"], "split": c["split"], "datasets": c["datasets"],
                                   "execution": c["execution"], "evidence": c.get("evidence"),
                                   "drawdown": c["drawdown"]} for p,c in setups},
                "market_order": args.markets, "arms": ARMS, "seconds": args.seconds, "batch": args.batch,
                "workers_per_arm": args.workers_per_arm, "max_batches": args.max_batches, "seed": 19,
                "model": "gpt-6-luna", "max_api_usd": args.max_api_usd, "criterion": CRITERION}
    if args.timeframe != "1d":
        protocol["timeframe"] = args.timeframe
    protocol = json.loads(encode(protocol))
    with lock(root):
        path = root/"protocol.json"
        existing = read(path)
        if existing is not None and existing != protocol:
            raise ValueError("Arguments or data differ from the frozen protocol.")
        if existing is None:
            if args.phase == "evaluate":
                raise ValueError("Discovery must freeze a protocol first.")
            write_json(path, protocol)
        if os.name == "nt":
            # Request wakefulness only for the lifetime of this runner.
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
        try:
            if args.phase in ("run", "discover"):
                discovery(root, setups, protocol)
            if args.phase in ("run", "evaluate"):
                evaluate(root, setups, protocol)
        except Exception as exc:
            write_json(root/"status.json", {"phase": "failed", "error": str(exc), "updated": utc(time.time())})
            raise
        finally:
            if os.name == "nt":
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == "__main__":
    main()
