"""Compare Luna and a rules-based planner using identical local matched searches."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

from . import planner
from .backtest import Execution
from .feedback import campaign_map, development_score
from .hourly_forecasts import HORIZON, STOPS, TARGETS, load_lookup
from .learning_benchmark import budgeted_response
from .research import Ledger, assess, encode, experiment_report, load_data, matched_variants, write_json
from .strategy import Strategy, signals
from .windows_launcher import load_openai_settings

FAMILIES = {
    "return": ("above", [x/100 for x in range(1, 17)], -.01),
    "sma_distance": ("above", [x/100 for x in range(1, 17)], -.02),
    "zscore": ("below", [-x/10 for x in range(10, 26)], .20),
    "breakout": ("above", [x/1000 for x in range(0, 16)], -.04),
    "volatility": ("below", [x/1000 for x in range(8, 39, 2)], .07),
}
CHANGES = ("exit_policy", "time_machine")
OPTIONS = tuple(f"{family}:{change}" for family in FAMILIES for change in CHANGES)
HOURLY_OPTIONS = tuple(f"{family}:{change}" for family in FAMILIES
                       for change in ("exit_policy", "entry_confirmation", "time_machine_20",
                                      "time_machine_30", "time_machine_40"))
BRANCHES = ("programmatic", "luna")
_WORKER = {}


def identity(spec):
    return hashlib.sha256(encode(spec).encode()).hexdigest()


def recipe(action, parent=None, reason=None, expectation=None, timeframe="1d"):
    family, change = action.split(":")
    direction, entries, opposing = FAMILIES[family]
    tm = change.startswith("time_machine")
    spec = {
        "hypothesis": reason or f"Compare {change} within the {family} entry family.",
        "expected_change": expectation, "parent": parent, "choice": action,
        "matched_change": "time_machine" if tm else change, "timeframe": timeframe, "indicator": family,
        "direction": direction, "lookbacks": list(range(10, 126, 5)), "entries": entries,
        "exits": [] if tm or change == "entry_confirmation" else [opposing],
        "holding_periods": [5, 8, 12, 16, 20, 25] if tm else [0, 5, 10, 15, 20, 30, 45],
        "stops": [x/100 for x in range(2, 16)], "targets": [x/100 for x in range(4, 37, 2)],
        "exit_policies": ["signal", "bracket"] if change == "exit_policy" else ["bracket"],
        "time_machine_thresholds": [.20] if tm else [0], "filter_rule": None,
    }
    if timeframe == "1h":
        scale = 1 if family == "zscore" else .25
        spec.update(entries=[x*scale for x in entries], exits=[x*scale for x in spec["exits"]],
                    holding_periods=[0, 12, 24, 48, 96, 168],
                    stops=list(STOPS), targets=list(TARGETS))
        if tm:
            spec.update(forecast_bars=HORIZON, time_machine_thresholds=[int(change.split("_")[-1])/100])
        if change == "entry_confirmation":
            indicator = "sma_distance" if family != "sma_distance" else "return"
            spec["filter_rule"] = {"indicator": indicator, "lookback": 48, "entry": 0,
                                   "exit": None, "direction": "above"}
    return spec


def setup_worker(root, config, forecast_seed=None):
    _WORKER.update(data=load_data(root, config), execution=Execution(**config["execution"]),
                   features={}, forecasts={}, cap=config["drawdown"], evidence=config.get("evidence"))
    if forecast_seed is not None:
        _WORKER.update(forecast_lookup=load_lookup(root, config, forecast_seed),
                       stress_forecast_lookup=load_lookup(root, config, forecast_seed, stressed=True))


def evaluate_chunk(chunk, seed):
    results = []
    for candidate_id, spec in chunk:
        before = time.perf_counter()
        strategy = Strategy.from_dict(spec)
        times, bars, _ = _WORKER["data"][strategy.timeframe]
        signal = signals(times, bars, strategy, _WORKER["features"])
        metrics, result = assess(times, bars, signal, strategy, _WORKER["execution"], _WORKER["cap"], seed,
                            tm_cache=_WORKER["forecasts"], policy=_WORKER["evidence"], all_periods=True,
                            forecast_lookup=_WORKER.get("forecast_lookup"),
                            stress_forecast_lookup=_WORKER.get("stress_forecast_lookup"))
        if strategy.timeframe == "1h":
            record_behavior(metrics, result)
        metrics["seconds"] = time.perf_counter()-before
        results.append((encode(metrics), candidate_id))
    return results


def record_behavior(metrics, result):
    from statistics import median

    trades = result["trades"]
    behavior = [(t["entry_timestamp"], t["exit_timestamp"], t["leg"], t["entry"], t["exit"]) for t in trades]
    position = result["open_position"]
    if position:
        behavior.append((position["entry_timestamp"], None, position["leg"], position["entry"], None))
    hours = [(t["exit_timestamp"]-t["entry_timestamp"])/3600 for t in trades]
    metrics.update(behavior=hashlib.sha256(encode(behavior).encode()).hexdigest(),
                   annual_net_return=(1+metrics["net_return"])**(365.25/metrics["elapsed_days"])-1,
                   median_holding_hours=median(hours) if hours else None,
                   held_over_24h=sum(h > 24 for h in hours)/len(hours) if hours else None)


def make_choices(ledger, proposal, budget, seed):
    for factor in (1, 2, 4):
        generated = matched_variants(proposal, budget*factor, seed, [proposal["timeframe"]])
        ids = [identity(spec) for spec in generated]
        seen = set()
        for offset in range(0, len(ids), 500):
            chunk = ids[offset:offset+500]
            placeholders = ",".join("?" for _ in chunk)
            seen.update(row[0] for row in ledger.db.execute(
                f"SELECT id FROM candidates WHERE id IN ({placeholders})", chunk))
        chosen = []
        for offset in range(0, len(generated), 2):
            if ids[offset] in seen or ids[offset+1] in seen:
                continue
            chosen.extend(generated[offset:offset+2])
            if len(chosen) == budget:
                return chosen
    raise ValueError("The matched grid could not supply a complete fresh trial budget.")


def choose_program(context, options):
    def strength(exp):
        m = exp["effects"]
        windows = m["windows"]
        return (sum((w["median_net_return_delta"] or 0) > 0 for w in windows) / max(1, len(windows)),
                m["median_net_return_delta"] or 0,
                -(m["median_drawdown_delta"] or 0))
    eligible = [e for e in context["experiments"] if e["proposal"]["choice"] in options]
    favorable = [e for e in eligible if strength(e)[0] >= .5 and strength(e)[1] > 0]
    if favorable:
        previous = max(favorable, key=strength)
        return {"choice": previous["proposal"]["choice"], "parent": previous["experiment"],
                "reason": "Repeat the change with the strongest positive development-window evidence.",
                "expectation": "The paired improvement persists on a fresh sample of settings."}
    counts = {a: sum(e["proposal"]["choice"] == a for e in eligible) for a in options}
    return {"choice": min(options, key=counts.get), "parent": None,
            "reason": "Explore a least-tested action because no eligible change passed the evidence rule.",
            "expectation": "Positive median paired return differences in at least half the development windows."}


def choose_luna(root, context, options):
    schema = planner.obj({"choice": {"type": "string", "enum": options},
                          "parent": {"type": ["integer", "null"]},
                          "reason": planner.STRING, "expectation": planner.STRING})
    prompt = ("Select one allowed matched experiment using the development-only evidence. "
              "The local engine samples the same fixed parameter grids for both planners. "
              "Use the paired deltas, risk, and window stability. Name the parent if refining "
              "and state a specific observable result that would support the change. "
              "Neither planner sees any holdout. Never invent a result.")
    proposal, provider = budgeted_response(root/"api-spending.sqlite", 1_000_000, "gpt-6-luna",
                                          prompt, context, schema, time.time()+120)
    if proposal["choice"] not in options or proposal["parent"] not in {
        None, *(e["experiment"] for e in context["experiments"])
    }:
        raise ValueError("Luna returned an invalid action or parent.")
    return proposal, provider


def schedule(root, ledger, branch, round_, protocol, history):
    exp = ledger.db.execute("SELECT * FROM experiments WHERE id=?", (round_+1,)).fetchone()
    if exp is not None:
        return exp
    if round_ == 0:
        proposal = recipe("return:exit_policy", timeframe=protocol.get("timeframe", "1d"))
        provider = {"model": "shared-fixed-recipe"}
    else:
        used = {e["proposal"]["indicator"] for e in history}
        options = [a for a in OPTIONS if round_ != 3 or a.split(":")[0] not in used]
        context = {"assets": protocol["assets"], "round": round_, "allowed_actions": options,
                   "campaign_map": campaign_map(history),
                   "experiments": [{"experiment": e["experiment"], "proposal": e["proposal"],
                                    "effects": e["matched"], "funnel": e["funnel"]} for e in history],
                   "latest": {"proposal": history[-1]["proposal"], "effects": history[-1]["matched"]}}
        if branch == "luna":
            decision, provider = choose_luna(root, context, options)
        else:
            decision, provider = choose_program(context, options), {"model": "programmatic"}
        proposal = recipe(decision["choice"], decision["parent"], decision["reason"], decision["expectation"])
        write_json(root/branch/f"round-{round_}-input.json", context)
    budget = protocol["backtests_per_batch"]
    choices = make_choices(ledger, proposal, budget, protocol["seed"]+round_)
    with ledger.db:
        ledger.db.execute("INSERT INTO experiments(id,proposal,provider,variants) VALUES (?,?,?,?)",
                          (round_+1, encode(proposal), encode(provider), encode(choices)))
        ledger.db.executemany("INSERT INTO candidates(id,experiment,strategy) VALUES (?,?,?)",
                              [(identity(s), round_+1, encode(s)) for s in choices])
    return ledger.db.execute("SELECT * FROM experiments WHERE id=?", (round_+1,)).fetchone()


def run_round(pool, root, ledgers, round_, protocol, histories):
    pending = {}
    completed = {name: 0 for name in ledgers}
    for name, ledger in ledgers.items():
        if len(histories[name]) > round_:
            continue
        exp = schedule(root, ledger, name, round_, protocol, histories[name])
        print(name, "round", round_, json.loads(exp["proposal"])["choice"], flush=True)
        candidates = [(row["id"], json.loads(row["strategy"])) for row in ledger.db.execute(
            "SELECT id,strategy FROM candidates WHERE experiment=? AND development IS NULL ORDER BY rowid",
            (exp["id"],))]
        completed[name] = protocol["backtests_per_batch"]-len(candidates)
        for offset in range(0, len(candidates), 100):
            future = pool.submit(evaluate_chunk, candidates[offset:offset+100], protocol["seed"])
            pending[future] = name
    for future in as_completed(pending):
        name = pending[future]
        ledger = ledgers[name]
        rows = future.result()
        with ledger.db:
            ledger.db.executemany("UPDATE candidates SET development=? WHERE id=?", rows)
        completed[name] += len(rows)
        if completed[name] % 2000 < len(rows) or completed[name] == protocol["backtests_per_batch"]:
            print(name, "round", round_, completed[name], "/", protocol["backtests_per_batch"], flush=True)
    for name, ledger in ledgers.items():
        if len(histories[name]) > round_:
            continue
        with ledger.db:
            ledger.db.execute("UPDATE experiments SET status='complete' WHERE id=?", (round_+1,))
        item, = experiment_report(ledger, limit=1)
        assert item["matched"]["completed_pairs"]*2 == protocol["backtests_per_batch"]
        histories[name].append(item)
        write_json(root/name/"rounds.json", histories[name])
        write_json(root/name/"campaign-map.json", campaign_map(histories[name]))


def freeze(ledger, round_):
    best, best_key = None, None
    for row in ledger.db.execute("SELECT strategy,development,experiment FROM candidates "
                                 "WHERE experiment<=? AND development IS NOT NULL ORDER BY rowid", (round_+1,)):
        metrics = json.loads(row["development"])
        key = development_score(metrics)
        if best_key is None or key > best_key:
            best_key = key
            best = {"strategy": json.loads(row["strategy"]), "development": metrics, "experiment": row["experiment"]}
    return best


def open_history(root, name):
    directory = root/name
    path = directory/"rounds.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def discovery(root, setup, config, protocol, workers):
    for name in ("shared", *BRANCHES):
        (root/name).mkdir(exist_ok=True)
    with closing(sqlite3.connect(root/"api-spending.sqlite",timeout=30)) as db, db:
        db.execute("CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, microdollars INTEGER)")
        db.execute("CREATE TABLE IF NOT EXISTS api_block (id INTEGER PRIMARY KEY, reason TEXT NOT NULL)")
        db.execute("DELETE FROM api_block")
    picks_path = root/"picks.json"
    picks = json.loads(picks_path.read_text(encoding="utf-8")) if picks_path.exists() else {}
    with ProcessPoolExecutor(max_workers=workers, initializer=setup_worker, initargs=(setup, config)) as pool:
        shared_db = root/"shared"/"research.sqlite"
        shared = Ledger(shared_db)
        with closing(shared.db):
            history = {"shared": open_history(root, "shared")}
            run_round(pool, root, {"shared": shared}, 0, protocol, history)
            if "shared" not in picks:
                picks["shared"] = freeze(shared, 0)
                write_json(picks_path, picks)
        ledgers, histories = {}, {}
        try:
            for name in BRANCHES:
                destination = root/name/"research.sqlite"
                if not destination.exists():
                    shutil.copy2(shared_db, destination)
                if not (root/name/"rounds.json").exists():
                    write_json(root/name/"rounds.json", history["shared"])
                ledgers[name] = Ledger(destination)
                histories[name] = open_history(root, name)
            for round_ in range(1, protocol["followup_rounds"]+1):
                run_round(pool, root, ledgers, round_, protocol, histories)
                if round_ in protocol["checkpoints"]:
                    for name in BRANCHES:
                        key = f"{name}-{round_}"
                        if key not in picks:
                            picks[key] = freeze(ledgers[name], round_)
                    write_json(picks_path, picks)
            for ledger in ledgers.values():
                expected = (protocol["followup_rounds"]+1)*protocol["backtests_per_batch"]
                assert ledger.db.execute("SELECT count(development) FROM candidates").fetchone()[0] == expected
        finally:
            for ledger in ledgers.values():
                ledger.db.close()


def evaluation(root, setup, config, protocol):
    picks = json.loads((root/"picks.json").read_text(encoding="utf-8"))
    expected = {"shared", *(f"{arm}-{n}" for arm in BRANCHES for n in protocol["checkpoints"])}
    if set(picks) != expected:
        raise ValueError("Both branches must freeze their checkpoint picks before holdout evaluation.")
    budget = protocol["backtests_per_batch"]
    for arm in BRANCHES:
        with closing(Ledger(root/arm/"research.sqlite").db) as db:
            counts = db.execute("SELECT experiment,count(development),count(final) FROM candidates GROUP BY experiment").fetchall()
            if (len(counts) != protocol["followup_rounds"]+1
                    or any(row[1] != budget or row[2] for row in counts)):
                raise ValueError(f"{arm} development budget or separation check failed.")
    times, bars, cut = load_data(setup, config, final=True)["1d"]
    execution = Execution(**config["execution"])
    utc_date = lambda stamp: datetime.fromtimestamp(int(stamp), timezone.utc).date()
    lines = ["# Matched planner comparison", "",
             f"Market: {config['assets'][0]}. Two planners shared one starting batch, then ran "
             f"{protocol['followup_rounds']} follow-ups of {budget:,} unique development backtests each.",
             "Every entry used 10% of equity. Both planners used the same matched parameter sampler.",
             f"The sealed holdout ran from {utc_date(times[cut])} to {utc_date(times[-1]+86400)} (end exclusive).",
             "", "| Frozen pick | Development return | Holdout return | Holdout drawdown | Closed trades | Qualified |",
             "|---|---:|---:|---:|---:|---|"]
    for name, record in picks.items():
        spec = Strategy.from_dict(record["strategy"])
        result, _ = assess(times, bars, signals(times, bars, spec, {}), spec, execution,
                           config["drawdown"], protocol["seed"], final=True, warmup=cut,
                           policy=config.get("evidence"))
        record["holdout"] = result
        lines.append(f"| {name} | {record['development']['net_return']:.2%} | {result['net_return']:.2%} | "
                     f"{result['max_drawdown']:.2%} | {result['trades']} | {result['outcome']} |")
    n = protocol["followup_rounds"]
    luna, program, shared = [picks[k]["holdout"] for k in (f"luna-{n}", f"programmatic-{n}", "shared")]
    differences = {"luna_vs_program": luna["net_return"]-program["net_return"],
                   "luna_vs_start": luna["net_return"]-shared["net_return"],
                   "program_vs_start": program["net_return"]-shared["net_return"]}
    passed = all(luna["net_return"]-control["net_return"] >= protocol["minimum_lift"]
                 and luna["max_drawdown"] <= control["max_drawdown"] for control in (program, shared))
    lines += ["", f"Luna minus programmatic: **{100*differences['luna_vs_program']:+.2f} pp** net return.",
              f"Luna minus shared start: **{100*differences['luna_vs_start']:+.2f} pp** net return.",
              f"Programmatic minus shared start: **{100*differences['program_vs_start']:+.2f} pp** net return.",
              f"Predefined pilot criterion: {'passed' if passed else 'not met'}. Luna needed at least "
              f"{100*protocol['minimum_lift']:.2f} pp higher net return with no higher drawdown against both comparisons.",
              "", "One market-period. Results measure this pilot rather than generalization.", "",
              "## Development decisions", "",
              "| Planner | Follow-up | Choice | Parent experiment | Median paired net-return change | Windows with positive median change |",
              "|---|---:|---|---:|---:|---:|"]
    decisions = {arm: open_history(root, arm)[1:] for arm in BRANCHES}
    for arm, history in decisions.items():
        for exp in history:
            spec, effects = exp["proposal"], exp["matched"]
            positive = sum((w["median_net_return_delta"] or 0) > 0 for w in effects["windows"])
            parent = spec["parent"] if spec["parent"] is not None else "—"
            lines.append(f"| {arm} | {exp['experiment']-1} | {spec['choice']} | {parent} | "
                         f"{100*effects['median_net_return_delta']:+.2f} pp | {positive}/{len(effects['windows'])} |")
    usage = [e["provider"].get("usage", {}) for e in decisions["luna"]]
    api_cost = sum(.1*u.get("input_tokens",0)+.5*u.get("output_tokens",0)
                   -.09*(u.get("input_tokens_details") or {}).get("cached_tokens",0) for u in usage)/1e6
    lines.extend(["", f"Completed Luna proposals: {len(usage)}. Estimated successful-call cost: ${api_cost:.3f}.",
                  "", "[Luna development evidence](luna/rounds.json) · "
                  "[Programmatic development evidence](programmatic/rounds.json) · [Frozen protocol](protocol.json)", ""])
    write_json(root/"results.json", {"picks": picks, "differences": differences, "passed": passed,
                                   "luna_cost_estimated_usd": api_cost})
    (root/"summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("discover", "evaluate"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--asset", default="BNB")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--batch", type=int, default=10000)
    parser.add_argument("--rounds", type=int, choices=(3, 6), default=6)
    args = parser.parse_args()
    if not 2 <= args.batch <= 10000 or args.batch % 2:
        parser.error("Batch must be even and between 2 and 10,000.")
    root = args.root.resolve()
    setup = root/args.asset.upper()
    config = json.loads((setup/"config.json").read_text(encoding="utf-8"))
    if list(config["datasets"]) != ["1d"] or len(config["assets"]) != 1:
        raise ValueError("The pilot requires a single asset and daily candles.")
    if not config["execution"].get("fixed_allocation") or config["execution"]["allocation"] != .1:
        raise ValueError("Prepare the snapshot with a fixed 10% allocation.")
    protocol = {"assets": config["assets"], "data": config["datasets"], "execution": config["execution"],
                "seed": 19, "backtests_per_batch": args.batch, "followup_rounds": args.rounds,
                "checkpoints": sorted({3, args.rounds}), "minimum_lift": .005, "model": "gpt-6-luna",
                "spending_guard_usd": 1, "exploration_round": 3}
    frozen = root/"protocol.json"
    if frozen.exists():
        if json.loads(frozen.read_text(encoding="utf-8")) != protocol:
            raise ValueError("Requested settings differ from the frozen pilot protocol.")
    else:
        if args.phase != "discover":
            raise ValueError("Freeze the discovery protocol before evaluation.")
        write_json(frozen, protocol)
    if args.phase == "discover":
        load_openai_settings(Path(".env"))
        discovery(root, setup, config, protocol, args.workers)
    else:
        evaluation(root, setup, config, protocol)


if __name__ == "__main__":
    main()
