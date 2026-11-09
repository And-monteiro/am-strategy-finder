"""Checkpointed discovery, evidence assessment, and frozen final evaluation."""
from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import sqlite3
import time
import urllib.error
from dataclasses import replace

import numpy as np

from . import planner
from .backtest import EvaluationDeadline, Execution, daily_returns, metrics, simulate
from .data import DAY, STEPS, load_snapshot, partition_hash
from .feedback import RICH_INSTRUCTIONS, RICH_SCHEMA, requested_action, rich_context, validate_action
from .strategy import Rule, Strategy, signals


def encode(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(",", ":"))


def write_json(path, data):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


class Ledger:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY, proposal TEXT NOT NULL, provider TEXT NOT NULL,
                variants TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'open');
            CREATE TABLE IF NOT EXISTS candidates (
                id TEXT PRIMARY KEY, experiment INTEGER NOT NULL, strategy TEXT NOT NULL,
                development TEXT, final TEXT);
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY, phase TEXT NOT NULL, details TEXT NOT NULL);
        """)

    def get(self, key, default=None):
        row = self.db.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set(self, key, value):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, encode(value)))

    def event(self, phase, details):
        with self.db:
            self.db.execute("INSERT INTO events(phase,details) VALUES (?,?)", (phase, encode(details)))


def variants(proposal, count, seed, frames, rotation=False):
    if proposal["timeframe"] not in frames:
        raise ValueError("Planner chose a timeframe outside this investigation.")
    fields = ("lookbacks", "entries", "exits", "holding_periods", "stops", "targets", "time_machine_thresholds")
    policies = proposal.get("exit_policies", ["signal"])
    if not isinstance(policies, list) or not policies or not all(p in ("signal", "bracket") for p in policies):
        raise ValueError("exit_policies must contain signal, bracket, or both.")
    policies = list(dict.fromkeys(policies))
    choices = []
    for field in fields:
        values = proposal[field]
        if field == "exits" and values == [] and policies == ["bracket"] and not rotation:
            choices.append([None])
            continue
        if not isinstance(values, list) or not 1 <= len(values) <= 24:
            raise ValueError(f"{field} must contain 1–24 choices.")
        if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
            raise ValueError(f"{field} must contain finite numbers.")
        choices.append(list(dict.fromkeys(values)))
    grids = []
    size = 0
    for policy in policies:
        grid = list(choices)
        if policy == "bracket" and not rotation:
            grid[2] = [None]
        size += math.prod(map(len, grid))
        grids.append((size, policy, grid))
    rng = random.Random(seed)
    candidates = {}
    # Evaluate paired unfiltered controls first whenever a gated variant is drawn.
    for offset in rng.sample(range(size), min(size, max(10_000, count * 30))):
        lower = 0
        for upper, policy, grid in grids:
            if offset < upper:
                offset -= lower
                break
            lower = upper
        parameters = []
        for values in reversed(grid):
            offset, index = divmod(offset, len(values))
            parameters.append(values[index])
        n, entry, exit_, hold, stop, target, threshold = reversed(parameters)
        try:
            rule = Rule(proposal["indicator"], n, entry, exit_, proposal["direction"])
            extra = Rule(**proposal["filter_rule"]) if proposal.get("filter_rule") else None
            strategy = Strategy(proposal["timeframe"], rule, hold, stop, target, threshold, extra, policy)
        except (TypeError, ValueError):
            continue
        group = [strategy]
        if threshold:
            group.insert(0, replace(strategy, time_machine=0))
        group = [item.to_dict() for item in group]
        if len(candidates.keys() | {encode(item) for item in group}) > count:
            continue
        for item in group:
            candidates[encode(item)] = item
        if len(candidates) == count:
            break
    if not candidates:
        raise ValueError("No valid variants in proposal.")
    return list(candidates.values())


STAGES = ["tested", "profitable_after_costs", "drawdown_within_return_and_cap",
          "adequate_evidence", "consistent_across_periods"]


def interval(daily, seed, block=14, samples=256, quantile=.1):
    if len(daily) < 4 * block or not np.any(daily):
        return None
    rng = np.random.default_rng(seed)
    chunks = math.ceil(len(daily)/block)
    starts = rng.integers(0, len(daily) - block + 1, size=(samples, chunks))
    indexes = (starts[..., None] + np.arange(block)).reshape(samples, -1)[:, :len(daily)]
    distribution = np.expm1(np.log1p(daily)[indexes].sum(axis=1))
    return [float(x) for x in np.quantile(distribution, [quantile, 1-quantile])]


def windows(times, step, final):
    # Use calendar-duration windows and avoid mistaking a missing bar for an independent day.
    start = 0 if final else int(np.searchsorted(times, times[0] + 365*DAY))
    length = 90*DAY if final else 182*DAY
    while start < len(times):
        end = int(np.searchsorted(times, times[start] + length))
        if times[end-1] - times[start] + step >= length * .75:
            yield start, end
        start = end


def assess(times, bars, signal, strategy, execution, cap, seed, final=False,
           warmup=0, tm_cache=None, deadline=None, policy=None):
    step = STEPS[strategy.timeframe]
    whole = simulate(times, bars, signal, strategy, execution, warmup, len(times), seed, tm_cache, deadline)
    active = times[warmup:]
    result = metrics(active, whole, execution.capital, step)
    result["buy_and_hold"] = [float(bars[-1, leg, 3] / bars[warmup, leg, 0] - 1)
                              for leg in range(bars.shape[1])]
    daily = daily_returns(active, whole["equity"], execution.capital, step)
    holding_days = [(t["exit_timestamp"] - t["entry_timestamp"] + step) / DAY for t in whole["trades"]]
    block = max(14, math.ceil(np.median(holding_days))) if holding_days else 14
    # Keep runtime bounded when a small number of trades last many months.
    policy = policy or {}
    ci = interval(daily, seed, block, quantile=policy.get("lower_quantile", .1)) if result["net_return"] > 0 and result["trades"] else None
    result["net_return_interval_80"] = ci
    result["bootstrap_block_days"] = block
    stages = [True, result["net_return"] > 0]
    stages.append(result["max_drawdown"] <= min(cap, result["net_return"]))
    evidence = bool(ci and ci[0] > 0 and result["profit_concentration"] is not None
                    and result["profit_concentration"] <= policy.get("max_winner_contribution", .5))
    result["periods"] = []
    if all(stages):
        for a, b in windows(active, step, final):
            a, b = a+warmup, b+warmup
            values = simulate(times, bars, signal, strategy, execution, a, b, seed, tm_cache, deadline)
            measured = metrics(times[a:b], values, execution.capital, step)
            result["periods"].append({"start": int(times[a]), "end": int(times[b-1]+step),
                                      "return": measured["net_return"], "trades": measured["trades"]})
        multiplier = policy.get("cost_multiplier", 1.5)
        stressed = Execution(execution.capital, execution.allocation, execution.risk,
                             min(9999, execution.fee_bps*multiplier),
                             min(9999, execution.slippage_bps*multiplier))
        cost_test = simulate(times, bars, signal, strategy, stressed, warmup, len(times), seed, deadline=deadline)
        result["stress_return"] = cost_test["equity"][-1] / execution.capital - 1
        evidence &= result["stress_return"] > 0
    positive = sum(p["return"] > 0 for p in result["periods"])
    fraction = policy.get("positive_period_fraction", 2/3)
    consistent = len(result["periods"]) >= 2 and positive >= math.ceil(len(result["periods"])*fraction)
    stages.extend([evidence, consistent])
    result["funnel"] = dict(zip(STAGES, itertools.accumulate(stages, lambda a, b: a and b)))
    result["outcome"] = ("qualified" if all(stages) else
                         "insufficient_evidence" if all(stages[:3]) and not evidence else "rejected")
    return result, whole


def counts(rows):
    return {stage: sum(row["funnel"][stage] for row in rows) for stage in STAGES}


def load_data(root, config, final=False):
    data = {}
    for frame, info in config["datasets"].items():
        times, values = load_snapshot(root / f"development-{frame}.parquet")
        if partition_hash(times, values) != info["development_hash"]:
            raise ValueError(f"Development dataset changed for {frame}.")
        warmup = 0
        if final:
            other_t, other_v = load_snapshot(root / f"sealed-{frame}.parquet")
            if partition_hash(other_t, other_v) != info["final_hash"]:
                raise ValueError(f"Final dataset changed for {frame}.")
            # Retain all past observations, while opening the final portfolio in cash.
            warmup = len(times)
            times, values = np.r_[times, other_t], np.concatenate((values, other_v))
        data[frame] = (times, values, warmup)
    return data


def run(root, ledger, memory):
    config = ledger.get("config")
    deadline = config["deadline"]
    execution = Execution(**config["execution"])
    build = load_data(root, config)
    frames = list(build)
    feature_cache = {frame: {} for frame in frames}
    tm_cache = {frame: {} for frame in frames}
    final_ids = ledger.get("final_ids")
    reserved = sum(max(.05, result["seconds"]) * 1.5
                   for row in ledger.db.execute("SELECT development FROM candidates "
                                                "WHERE development IS NOT NULL AND final IS NULL")
                   if (result := json.loads(row["development"]))["outcome"] == "qualified")
    if final_ids is None:
        while time.time() < deadline - max(config["duration"]*.12, reserved):
            experiment = ledger.db.execute("SELECT * FROM experiments WHERE status='open' ORDER BY id LIMIT 1").fetchone()
            if experiment is None:
                count = ledger.db.execute("SELECT count(*) FROM experiments").fetchone()[0]
                if config["experiments"] and count >= config["experiments"]:
                    break
                try:
                    if config["offline"]:
                        proposal, provider = planner.offline(count, frames), {"model": "offline"}
                    else:
                        recent = experiment_report(ledger, limit=10)
                        history = development_history(ledger)
                        context = rich_context(recent, history)
                        action = requested_action(count, bool(history))
                        proposal, provider = planner.response(
                            config["model"], RICH_INSTRUCTIONS,
                            {"assets": config["assets"], "allowed_timeframes": frames,
                             "development_feedback": context, "requested_action": action,
                             "prior_development_memory": memory},
                            RICH_SCHEMA, min(deadline - max(config["duration"]*.12, reserved), time.time()+90),
                            "strategy_experiment")
                        validate_action(proposal, context, action)
                    choices = variants(proposal, config["variants"], config["seed"]+count, frames,
                                       rotation=len(config["assets"]) == 2)
                except (TimeoutError, ValueError, TypeError, KeyError, urllib.error.URLError) as exc:
                    ledger.event("proposal_failure", {"error": str(exc)})
                    failures = ledger.db.execute("SELECT count(*) FROM events WHERE phase='proposal_failure'").fetchone()[0]
                    if config["offline"] or failures >= 3:
                        break
                    continue
                with ledger.db:
                    cursor = ledger.db.execute("INSERT INTO experiments(proposal,provider,variants) VALUES (?,?,?)",
                                               (encode(proposal), encode(provider), encode(choices)))
                    exp_id = cursor.lastrowid
                    for choice in choices:
                        raw = encode(choice)
                        identity = hashlib.sha256(raw.encode()).hexdigest()
                        ledger.db.execute("INSERT OR IGNORE INTO candidates(id,experiment,strategy) VALUES (?,?,?)",
                                          (identity, exp_id, raw))
                experiment = ledger.db.execute("SELECT * FROM experiments WHERE id=?", (exp_id,)).fetchone()
            pending = ledger.db.execute("SELECT * FROM candidates WHERE experiment=? AND development IS NULL",
                                        (experiment["id"],)).fetchall()
            for candidate in pending:
                if time.time() >= deadline - max(config["duration"]*.12, reserved):
                    break
                before = time.monotonic()
                strategy = Strategy.from_dict(json.loads(candidate["strategy"]))
                frame = strategy.timeframe
                times, bars, _ = build[frame]
                signal = signals(times, bars, strategy, feature_cache[frame])
                try:
                    result, _ = assess(times, bars, signal, strategy, execution, config["drawdown"], config["seed"],
                                       tm_cache=tm_cache[frame],
                                       deadline=deadline - max(config["duration"]*.12, reserved),
                                       policy=config.get("evidence"))
                except EvaluationDeadline:
                    break
                result["seconds"] = time.monotonic() - before
                if result["outcome"] == "qualified":
                    reserved += max(.05, result["seconds"]) * 1.5
                with ledger.db:
                    ledger.db.execute("UPDATE candidates SET development=? WHERE id=?",
                                      (encode(result), candidate["id"]))
            outstanding = ledger.db.execute("SELECT 1 FROM candidates WHERE experiment=? AND development IS NULL",
                                           (experiment["id"],)).fetchone()
            if not outstanding:
                with ledger.db:
                    ledger.db.execute("UPDATE experiments SET status='complete' WHERE id=?", (experiment["id"],))
        final_ids = [row["id"] for row in ledger.db.execute("SELECT id,development FROM candidates WHERE development IS NOT NULL ORDER BY id")
                     if json.loads(row["development"])["outcome"] == "qualified"]
        ledger.set("final_ids", final_ids)
    if final_ids and time.time() < deadline:
        del build
        final_data = load_data(root, config, final=True)
        feature_cache = {frame: {} for frame in frames}
        tm_cache = {frame: {} for frame in frames}
        for identity in final_ids:
            if time.time() >= deadline:
                break
            row = ledger.db.execute("SELECT strategy,final FROM candidates WHERE id=?", (identity,)).fetchone()
            if row["final"] is not None:
                continue
            strategy = Strategy.from_dict(json.loads(row["strategy"]))
            frame = strategy.timeframe
            times, bars, warmup = final_data[frame]
            signal = signals(times, bars, strategy, feature_cache[frame])
            try:
                result, _ = assess(times, bars, signal, strategy, execution, config["drawdown"], config["seed"],
                                   final=True, warmup=warmup, tm_cache=tm_cache[frame],
                                   deadline=deadline - min(90, config["duration"] * .05),
                                   policy=config.get("evidence"))
            except EvaluationDeadline:
                break
            with ledger.db:
                ledger.db.execute("UPDATE candidates SET final=? WHERE id=?", (encode(result), identity))


def experiment_report(ledger, limit=-1):
    report = []
    for exp in reversed(ledger.db.execute("SELECT * FROM experiments ORDER BY id DESC LIMIT ?", (limit,)).fetchall()):
        records = ledger.db.execute("SELECT strategy,development FROM candidates WHERE experiment=?",
                                    (exp["id"],)).fetchall()
        tested = [{"strategy": json.loads(r["strategy"]), "metrics": json.loads(r["development"])}
                  for r in records if r["development"] is not None]
        tested.sort(key=lambda item: (item["metrics"]["outcome"] == "qualified",
                                      item["metrics"]["sharpe"] or -999), reverse=True)
        report.append({"experiment": exp["id"], "proposal": json.loads(exp["proposal"]),
                       "provider": json.loads(exp["provider"]), "status": exp["status"],
                       "generated": len(json.loads(exp["variants"])), "unique": len(records),
                       "funnel": counts([t["metrics"] for t in tested]), "top_parameters": tested[:3]})
    return report


def development_history(ledger):
    return [{"experiment": row["experiment"], "strategy": json.loads(row["strategy"]),
             "development": json.loads(row["development"])}
            for row in ledger.db.execute(
                "SELECT experiment,strategy,development FROM candidates "
                "WHERE development IS NOT NULL ORDER BY experiment,rowid")]
