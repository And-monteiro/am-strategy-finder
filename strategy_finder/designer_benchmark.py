"""Compare Luna and programmatic strategy design with a shared local sampler."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sqlite3
import sys
import time
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import ExitStack, closing
from dataclasses import replace
from pathlib import Path
from statistics import median

import numpy as np

from . import design_search, holdout_budget, planner
from .backtest import Execution, forecast_paths, simulate, time_machine
from .cli import lock
from .data import DAY, STEPS
from .learning_benchmark import budgeted_response
from .matched_planner_test import FAMILIES, identity, recipe, record_behavior
from .research import encode, load_data, write_json
from .smc import INDICATORS as SMC_FAMILIES
from .strategy import Rule, Strategy, signals
from .time_machine_exit_replay import forecast_exit_edge
from .time_machine_replay import summarize
from .windows_launcher import load_openai_settings

ARMS = ("programmatic", "luna")
FRAME = "4h"
STEP = STEPS[FRAME]
SEED = 19
TOOLS = ("plain", "confirmation", "entry_gate", "forecast_exit", "both", "all")
WORKER = {}
DESIGN_FAMILIES = (*FAMILIES, *SMC_FAMILIES)


def design(family, policy="bracket", confirmation="none", entry_gate=False, forecast_exit=False):
    return dict(family=family, policy=policy, confirmation=confirmation,
                entry_gate=entry_gate, forecast_exit=forecast_exit)


def warmup_design(family, tool):
    if tool in SMC_FAMILIES:
        return design(family, confirmation=tool)
    confirmation = ("return" if family == "sma_distance" else "sma_distance") if tool in ("confirmation", "all") else "none"
    return design(family, confirmation=confirmation, entry_gate=tool in ("entry_gate", "both", "all"),
                  forecast_exit=tool in ("forecast_exit", "both", "all"))


def warmup_pairs():
    tools = (*TOOLS[1:], *SMC_FAMILIES)
    return [(family, tool) for tool in tools for family in DESIGN_FAMILIES if family != tool]


def smc_grid(family, confirmation=False):
    values = {"lookback": [2] if family == "fvg" else [2, 3, 5, 8, 13, 21, 34],
              "entry": [0], "smc_recent": [1, 3, 6, 12] if confirmation else [1]}
    if family != "fvg":
        values["smc_wick_break"] = [False, True]
    if family in ("fvg", "order_block"):
        values.update(smc_trigger=["formation", "touch", "midpoint", "rejection"],
                      smc_max_age=[12, 24, 48, 96], smc_min_size=[0., .001, .003],
                      smc_wick_invalidation=[False, True], smc_first_retest=[False, True])
    if family == "order_block":
        values["smc_body"] = [False, True]
    return values


def grid(d):
    family = d["family"]
    r = recipe(f'{family if family in FAMILIES else "breakout"}:exit_policy', timeframe=FRAME)
    values = {"stop": r["stops"], "target": r["targets"], "hold": [0, 6, 12, 24, 48, 96, 168]}
    if family in SMC_FAMILIES:
        values.update(smc_grid(family))
        if d["policy"] == "signal":
            values["exit"] = [0]
    else:
        values.update(lookback=r["lookbacks"], entry=r["entries"])
        if d["policy"] == "signal":
            values["exit"] = [r["exits"][0]*factor for factor in (.8, 1., 1.5)]
    if d["confirmation"] in SMC_FAMILIES:
        values.update({f"confirm_{key}": choices for key, choices in smc_grid(d["confirmation"], True).items()})
    elif d["confirmation"] != "none":
        values["confirm_lookback"] = [12, 24, 48, 72, 120]
        values["confirm_threshold"] = {
            "return": [0., .005, .01], "sma_distance": [0., .0025, .005],
            "zscore": [-.5, 0., .5], "breakout": [-.01, -.005, 0.],
            "volatility": [.003, .005, .008],
        }[d["confirmation"]]
    if d["entry_gate"]:
        values.update(p_target=[0., .1, .2, .3, .4], entry_payoff=[-.0025, 0., .0025])
    if d["forecast_exit"]:
        values.update(exit_edge=[.0025, .005, .01], exit_every=[1, 2, 4, 6],
                      exit_min_age=[1, 3, 6, 12])
    return values


def sample(d, rng, parents=()):
    choices = grid(d)
    if parents and rng.random() < .55:
        center = rng.choice(parents)
        params = {}
        for key, values in choices.items():
            if key in center and center[key] in values and rng.random() < .8:
                i = values.index(center[key])
                params[key] = rng.choice(values[max(0, i-2):i+3])
            else:
                params[key] = rng.choice(values)
    else:
        params = {key: rng.choice(values) for key, values in choices.items()}
    return {"design": d, "parameters": canonical_parameters(params)}


def canonical_parameters(params):
    for prefix in ("smc_", "confirm_smc_"):
        if params.get(prefix+"trigger") == "formation":
            for field in ("wick_invalidation", "first_retest"):
                params.pop(prefix+field, None)
    return params


def apply_tool(control, d, rng):
    params = {k: v for k, v in control["parameters"].items() if k in grid(d)}
    params.update({k: rng.choice(v) for k, v in grid(d).items() if k not in params})
    return {"design": d, "parameters": canonical_parameters(params)}


def strategy_for(spec):
    d, p = spec["design"], spec["parameters"]
    family = d["family"]
    direction = FAMILIES[family][0] if family in FAMILIES else "above"
    rule = Rule(family, p["lookback"], p["entry"], p.get("exit"), direction,
                **{k: v for k, v in p.items() if k.startswith("smc_")})
    confirmation = d["confirmation"]
    gate = None
    if confirmation != "none":
        gate = Rule(confirmation, p["confirm_lookback"], p.get("confirm_threshold", 0), None,
                    "below" if confirmation == "volatility" else "above",
                    **{k.removeprefix("confirm_"): v for k, v in p.items() if k.startswith("confirm_smc_")})
    return Strategy(FRAME, rule, p["hold"], p["stop"], p["target"], filter_rule=gate, exit_policy=d["policy"])


def candidates(batch, budget, seed, seen, decision=None, parents=()):
    rng = random.Random(seed)
    records = []
    pairs = warmup_pairs() if batch == 2 else ()
    for attempt in range(budget*50):
        if len(records) >= budget:
            return records
        if batch == 1:
            slot = len(records) % (2*len(DESIGN_FAMILIES))
            family = DESIGN_FAMILIES[slot//2]
            group = [sample(design(family, ("bracket", "signal")[slot%2]), rng)]
        elif batch == 2:
            family, tool = pairs[(len(records)//2) % len(pairs)]
            base = sample(design(family), rng)
            group = [base, apply_tool(base, warmup_design(family, tool), rng)]
        else:
            group = [sample(decision["design"], rng, parents if decision["sampling"] == "refine" else ())]
        ids = [identity(s) for s in group]
        if any(i in seen for i in ids):
            continue
        seen.update(ids)
        records.extend(zip(ids, group))
    raise ValueError("The selected design could not supply enough fresh candidates.")


def setup_worker(setup, config, final):
    WORKER.clear()
    WORKER.update(data=load_data(setup, config, final=final)[FRAME],
                  execution=Execution(**config["execution"]), features={}, paths=OrderedDict(),
                  forecasts=OrderedDict())


def cached(cache, key, calculate, limit):
    if key not in cache:
        if len(cache) >= limit:
            cache.popitem(last=False)
        cache[key] = calculate()
    else:
        cache.move_to_end(key)
    return cache[key]


def evaluate(spec, horizon):
    strategy = strategy_for(spec)
    d, p = spec["design"], spec["parameters"]
    times, bars, start = WORKER["data"]
    execution = WORKER["execution"]
    signal = signals(times, bars, strategy, WORKER["features"])
    active_signal = signal[start:]
    blocked_signals = 0
    if strategy.filter_rule:
        ungated = signals(times, bars, replace(strategy, filter_rule=None), WORKER["features"])
        blocked_signals = int(np.count_nonzero((ungated[start:] == 0) & (active_signal != 0)))
    forecast_bars = min(horizon, strategy.max_hold or horizon)
    forecasting = replace(strategy, forecast_bars=forecast_bars)
    admitted, scored = 0, 0

    def entry(i, leg):
        nonlocal admitted, scored
        key = (i, forecast_bars, p["stop"], p["target"], d["policy"])
        value = cached(WORKER["forecasts"], key, lambda: time_machine(
            bars, times, i-1, leg, forecasting, execution, SEED), 50000)
        scored += 1
        allowed = bool(value and value[0] >= p["p_target"] and value[3] > p["entry_payoff"])
        admitted += allowed
        return allowed

    def exit_now(i, leg, entry_i, price):
        age = (times[i]-times[entry_i])//STEP
        if age < p["exit_min_age"] or (age-p["exit_min_age"]) % p["exit_every"]:
            return False
        remaining = min(horizon, strategy.max_hold-age) if strategy.max_hold else horizon
        sampled = cached(WORKER["paths"], (i, remaining), lambda: forecast_paths(
            bars, times, i-1, leg, remaining, STEP, True, SEED), 768)
        if sampled is None:
            return False
        anchor = bars[i-1, leg, 3]
        edge = forecast_exit_edge(sampled[0], None, strategy, price/anchor,
                                  execution.slippage_bps/10000, execution.fee_bps/10000)
        return edge > p["exit_edge"]

    result = simulate(times, bars, signal, strategy, execution, start, len(times),
                      entry_filter=entry if d["entry_gate"] else None,
                      early_exit=exit_now if d["forecast_exit"] else None)
    active = times[start:]
    m = summarize(active, result, execution)
    record_behavior(m, result)
    equity = np.r_[execution.capital, result["equity"]]
    cuts = np.searchsorted(active, np.linspace(active[0], active[-1]+STEP, 4))
    m["windows"] = [(float(equity[b]/equity[a]-1)) for a, b in zip(cuts, cuts[1:])]
    m["entry_forecasts"] = {"evaluated": scored, "accepted": admitted}
    m["signal_activity"] = {"entry_bars": int(np.count_nonzero(active_signal == 0)),
                            "confirmation_blocked": blocked_signals}
    m["exit_reasons"] = dict(Counter(t["reason"] for t in result["trades"]))
    return m


def evaluate_chunk(chunk, horizon):
    results = []
    for key, spec in chunk:
        started = time.perf_counter()
        m = evaluate(spec, horizon)
        m["evaluation_seconds"] = time.perf_counter()-started
        results.append((key, m))
    return results


def qualifies(m, floor=.10):
    return (m["annual_net_return"] >= floor and m["max_drawdown"] < .30
            and (m["sharpe"] or 0) > .8)


def rank(m):
    sharpe = m["sharpe"] if m["sharpe"] is not None else -10
    return (qualifies(m), sum(x > 0 for x in m["windows"]) >= 2,
            m["max_drawdown"] < .30, median(m["windows"]), sharpe,
            m["annual_net_return"], m["trades"])


def distribution(values):
    q = np.quantile(values, [0, .1, .5, .9, 1])
    return dict(zip(("min", "p10", "median", "p90", "max"), q.tolist()))


def batch_report(records, history):
    seen = {m["behavior"] for _, m in history}
    fresh = {}
    for spec, m in records:
        if m["behavior"] not in seen:
            fresh.setdefault(m["behavior"], (spec, m))
    params = {}
    for spec, _ in records:
        for name, value in spec["parameters"].items():
            params.setdefault(name, Counter())[value] += 1
    designs = {}
    stable = Counter(encode(s["design"]) for s, m in fresh.values()
                     if qualifies(m) and sum(v > 0 for v in m["windows"]) >= 2)
    for spec, m in records:
        designs.setdefault(encode(spec["design"]), []).append(m)
    return {
        "trials": len(records), "distinct": len({m["behavior"] for _, m in records}),
        "new_distinct": len(fresh), "new_10": sum(qualifies(m) for _, m in fresh.values()),
        "new_15": sum(qualifies(m, .15) for _, m in fresh.values()),
        "new_stable_10": sum(qualifies(m) and sum(v > 0 for v in m["windows"]) >= 2 for _, m in fresh.values()),
        "parameter_counts": {k: dict(sorted(v.items())) for k, v in params.items()},
        "metrics": {k: distribution([m[k] for _, m in records if m[k] is not None])
                    for k in ("annual_net_return", "max_drawdown", "sharpe", "trades", "exposure")
                    if any(m[k] is not None for _, m in records)},
        "designs": [{"design": json.loads(k), "trials": len(ms),
                     "stable_new": stable[k],
                     "qualifiers": sum(qualifies(m) for m in ms),
                     "median_return": median(m["annual_net_return"] for m in ms),
                     "median_drawdown": median(m["max_drawdown"] for m in ms),
                     "median_window_returns": [median(m["windows"][i] for m in ms) for i in range(3)]}
                    for k, ms in sorted(designs.items())],
    }


def campaign(reports):
    totals = {}
    for report in reports:
        for item in report["designs"]:
            totals.setdefault(encode(item["design"]), []).append(item)
    return [{"design": json.loads(key), "trials": sum(v["trials"] for v in vals),
             "stable_new": sum(v.get("stable_new", 0) for v in vals),
             "qualifiers": sum(v["qualifiers"] for v in vals),
             "median_return": median(v["median_return"] for v in vals),
             "median_drawdown": median(v["median_drawdown"] for v in vals),
             "median_window_returns": [median(v["median_window_returns"][i] for v in vals) for i in range(3)]}
            for key, vals in sorted(totals.items())]


def design_menu():
    return [design(f, policy, confirmation, gate, early)
            for f in DESIGN_FAMILIES for policy in ("signal", "bracket")
            for confirmation in ("none", *DESIGN_FAMILIES) for gate in (False, True)
            for early in (False, True) if not early or policy == "bracket"]


def programmatic(summary, batch, seed):
    observed = {encode(x["design"]): x for x in summary}
    menu = design_menu()
    def utility(x):
        return (x["qualifiers"]/x["trials"], median(x["median_window_returns"]),
                x["median_return"], -x["median_drawdown"])
    if batch % 4 == 0:
        underexplored = min(observed.get(encode(d), {}).get("trials", 0) for d in menu)
        options = [d for d in menu if observed.get(encode(d), {}).get("trials", 0) == underexplored]
        return {"design": random.Random(seed).choice(options), "sampling": "broad",
                "reason": "Explore a least-tested supported combination."}
    best = max(summary, key=utility)
    return {"design": best["design"], "sampling": "refine",
            "reason": "Refine the design with the strongest development qualification rate and window returns."}


def luna(root, config, batch, summary, reports, records, budget):
    schema = planner.obj({
        "design": planner.obj({
            "family": {"type": "string", "enum": list(DESIGN_FAMILIES)},
            "policy": {"type": "string", "enum": ["bracket", "signal", "bracket_forecast"]},
            "confirmation": {"type": "string", "enum": ["none", *DESIGN_FAMILIES]},
            "entry_gate": {"type": "boolean"},
        }),
        "sampling": {"type": "string", "enum": ["broad", "refine"]}, "reason": planner.STRING,
        "search_family": {"type": "string", "enum": ["none", *DESIGN_FAMILIES]},
    })
    prompt = ("Design the next experiment from the development-only results. You may combine the supported "
              "tools or revisit a design. Broad draws from the full parameter grid. Refine also samples "
              "near strong previously evaluated settings in that family. Use bracket_forecast for bracket "
              "exits plus optional early closure by the forecast. "
              "Look for repeatable improvement after costs and across windows. Explore alternatives when evidence is weak. "
              "Set search_family to a family to retrieve distinct good and poor trade histories before finalizing, "
              "or none to finalize now. The search is available once per decision.")
    payload = {"asset": config["assets"], "timeframe": FRAME, "batch": batch,
               "target": {"annual_net": [.10, .15], "max_drawdown_below": .30, "sharpe_above": .8},
               "design_history": summary, "recent": [{k: r[k] for k in ("designs", "metrics", "new_10", "new_15", "new_distinct")}
                                                     for r in reports[-5:]], "examples": freeze(records, 4)}
    decision, receipt = budgeted_response(root/"api-spending.sqlite", round(budget*1_000_000),
                                          "gpt-6-luna", prompt, payload, schema, time.time()+180)
    family = decision["search_family"]
    if family != "none":
        subset = [(s, m) for s, m in records if s["design"]["family"] == family and m["trades"] >= 3]
        weak = []
        seen = set()
        for spec, m in sorted(subset, key=lambda x: rank(x[1])):
            if m["behavior"] in seen:
                continue
            seen.add(m["behavior"])
            weak.append({"spec": spec, "development": m})
            if len(weak) == 4:
                break
        payload["search"] = {"family": family, "strong": freeze(subset, 4), "weak": weak,
                             "tested_with_three_closed_trades": len(subset)}
        decision, followup = budgeted_response(root/"api-spending.sqlite", round(budget*1_000_000),
                                               "gpt-6-luna", prompt+" Finalize now and set search_family to none.",
                                               payload, schema, time.time()+180)
        receipt = {"initial": receipt, "final": followup}
    decision["search_family"] = family
    d = decision["design"]
    d["forecast_exit"] = d["policy"] == "bracket_forecast"
    if d["forecast_exit"]:
        d["policy"] = "bracket"
    return decision, receipt


def open_db(path):
    db = sqlite3.connect(path)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS batches (n INTEGER PRIMARY KEY, decision TEXT, report TEXT);
        CREATE TABLE IF NOT EXISTS trials (id TEXT PRIMARY KEY, batch INTEGER, spec TEXT, metrics TEXT);
        CREATE INDEX IF NOT EXISTS trials_batch ON trials(batch);
    """)
    return db


def prior(db, through=None):
    sql = "SELECT spec,metrics FROM trials WHERE metrics IS NOT NULL"
    rows = db.execute(sql) if through is None else db.execute(sql+" AND batch<=?", (through,))
    return [(json.loads(s), json.loads(m)) for s, m in rows]


def refinement_parents(history, target):
    ranked = sorted(((s, m) for s, m in history if s["design"]["family"] == target["family"] and m["trades"]),
        key=lambda x: (x[0]["design"] == target, rank(x[1])), reverse=True)
    parents = {}
    for spec, m in ranked:
        parents.setdefault(m["behavior"], spec["parameters"])
        if len(parents) == 32:
            break
    return list(parents.values())


def planned_candidates(decision, history, seen, seed):
    rng = random.Random(seed)
    records = []
    for part in decision["allocations"]:
        target, count = part["design"], part["trials"]
        parents = refinement_parents(history, target) if part["sampling"] != "broad" else ()
        cells = design_search.comparison_cells(target, part.get("components", ()))
        remaining = count
        for _ in range(count*100):
            if not remaining:
                break
            source = sample(target, rng, parents)
            group = [apply_tool(source, cell, rng) for cell in cells]
            ids = [identity(s) for s in group]
            if any(key in seen for key in ids):
                continue
            seen.update(ids)
            records.extend(zip(ids, group))
            remaining -= len(group)
        if remaining:
            raise ValueError(f"Insufficient fresh settings for {part}.")
    return records


def record_comparisons(records, decision):
    offset = 0
    comparisons = []
    for part in decision["allocations"]:
        selected = records[offset:offset+part["trials"]]
        offset += part["trials"]
        if part["sampling"] != "matched":
            continue
        cells = design_search.comparison_cells(part["design"], part["components"])
        width = len(cells)
        groups = [[m for _, m in selected[i:i+width]] for i in range(0, len(selected), width)]
        activity = [{"trades": sum(g[i]["trades"] for g in groups),
                     "confirmation_blocked": sum(g[i]["signal_activity"]["confirmation_blocked"] for g in groups),
                     "signal_exits": sum(g[i]["exit_reasons"].get("signal", 0) for g in groups),
                     "gate_evaluations": sum(g[i]["entry_forecasts"]["evaluated"] for g in groups),
                     "gate_rejections": sum(g[i]["entry_forecasts"]["evaluated"]-g[i]["entry_forecasts"]["accepted"] for g in groups),
                     "forecast_exits": sum(g[i]["exit_reasons"].get("forecast_exit", 0) for g in groups)}
                    for i in range(width)]
        comparisons.append(design_search.paired_evidence(cells, part["components"], groups, activity))
    return comparisons


def run_batch(db, pool, batch, decision, budget, horizon, initial, seed, warmup_kind=None):
    row = db.execute("SELECT decision,report FROM batches WHERE n=?", (batch,)).fetchone()
    if row and row[1]:
        return json.loads(row[1])
    history = initial+[(json.loads(s), json.loads(m)) for s,m in db.execute(
        "SELECT spec,metrics FROM trials WHERE batch<? AND metrics IS NOT NULL", (batch,))]
    if row is None:
        seen = {i for i, in db.execute("SELECT id FROM trials")}
        seen.update(identity(s) for s, _ in initial)
        parents = ()
        if decision and "allocations" in decision:
            batch_candidates = planned_candidates(decision, history, seen, seed)
        else:
            if decision:
                family = decision["design"]["family"]
                same_family = [(s, m) for s, m in history if s["design"]["family"] == family]
                parents = [s["parameters"] for s, _ in sorted(
                    same_family, key=lambda item: rank(item[1]), reverse=True)[:32]]
            batch_candidates = candidates(warmup_kind or batch, budget, seed, seen, decision, parents)
        with db:
            db.execute("INSERT INTO batches VALUES (?,?,NULL)", (batch, encode(decision)))
            db.executemany("INSERT INTO trials VALUES (?,?,?,NULL)", [(i,batch,encode(s)) for i,s in batch_candidates])
    else:
        decision = json.loads(row[0])
    pending = [(i,json.loads(s)) for i,s in db.execute(
        "SELECT id,spec FROM trials WHERE batch=? AND metrics IS NULL ORDER BY rowid", (batch,))]
    chunks = [pending[i:i+24] for i in range(0,len(pending),24)]
    for evaluated in pool.map(evaluate_chunk, chunks, [horizon]*len(chunks)):
        with db:
            db.executemany("UPDATE trials SET metrics=? WHERE id=?", [(encode(m), i) for i,m in evaluated])
    records = [(json.loads(s), json.loads(m)) for s,m in db.execute(
        "SELECT spec,metrics FROM trials WHERE batch=? ORDER BY rowid", (batch,))]
    result = batch_report(records, history)
    result.update(batch=batch, decision=decision)
    if decision and "allocations" in decision:
        result["comparisons"] = record_comparisons(records, decision)
    if (warmup_kind or batch) == 2:
        pairs = list(zip(records[::2], records[1::2]))
        result["paired_tools"] = []
        for tool in (*TOOLS[1:], *SMC_FAMILIES):
            selected = [(a[1],b[1]) for a,b in pairs if b[0]["design"] == warmup_design(b[0]["design"]["family"], tool)]
            if selected:
                result["paired_tools"].append({"tool": tool, "pairs": len(selected),
                    "median_return_delta": median(b["net_return"]-a["net_return"] for a,b in selected),
                    "median_drawdown_delta": median(b["max_drawdown"]-a["max_drawdown"] for a,b in selected)})
    with db:
        db.execute("UPDATE batches SET report=? WHERE n=?", (encode(result),batch))
    return result


def freeze(records, count, diverse=False):
    winners = []
    seen = set()
    ranked = sorted(records, key=lambda x: rank(x[1]), reverse=True)
    if diverse:
        def spread(pool):
            families = {}
            for spec, m in pool:
                families.setdefault(spec["design"]["family"], []).append((spec, m))
            return [entries[i] for i in range(max(map(len, families.values()), default=0))
                    for entries in families.values() if i < len(entries)]
        ranked = (spread([(s, m) for s, m in ranked if qualifies(m) and m["trades"]])
                  + spread([(s, m) for s, m in ranked if not qualifies(m) and m["trades"]]))
    for spec, m in ranked:
        if m["behavior"] in seen or not m["trades"]:
            continue
        seen.add(m["behavior"])
        winners.append({"id": identity(spec), "spec": spec, "development": m})
        if len(winners) == count:
            break
    return winners


def status(root, phase, market=None, **details):
    write_json(root/"status.json", {"phase": phase, "market": market, "pid": os.getpid(), **details})


def update_progress(directory, history, batch_size):
    write_json(directory/"progress.json", {"backtests": len(history)*batch_size,
                                          "completed_batches": len(history)})


def shortlist_snapshot(records, count, comparison):
    primary = freeze(records, count, diverse=not comparison)
    return {"picks": primary, "diverse_picks": freeze(records, count, diverse=True) if comparison else []}


def estimate_selection(work, selections, config):
    counts = config["datasets"][FRAME]
    holdout_budget.add_work(work, (picks for s in selections for picks in s.values()),
                            counts["development_bars"], counts["final_bars"])


def discover(root, market, config, protocol, workers=None):
    setup = root/market
    budget, rounds = protocol["batch"], protocol["batches"]
    arms = protocol["arms"]
    comparison = "legacy" in arms
    warmup_count = protocol.get("warmup_batches", 2)
    seconds = protocol.get("wall_seconds")
    clock_path = setup/"budget-clock.json"
    clock = read(clock_path) if seconds and clock_path.exists() else {}
    if seconds and not clock:
        clock = {"deadline": time.time()+seconds, "round_seconds": [], "selection_seconds": 0.}
        write_json(clock_path, clock)
    directories = {arm: setup/arm for arm in ("shared", *arms)}
    for directory in directories.values():
        directory.mkdir(exist_ok=True)
    remembered = protocol["development_memory"].get(market, [])
    if all((directories[arm]/"endpoint.json").exists() for arm in arms):
        return
    status(setup, "shared_warmup", market)
    with (ExitStack() as stack, closing(open_db(directories["shared"]/"design.sqlite")) as shared,
          ProcessPoolExecutor(max_workers=workers or protocol["workers"], initializer=setup_worker,
                              initargs=(setup, config, False)) as pool):
        warmup = [run_batch(shared, pool, 1, None, budget, 11, [], SEED)]
        holds = [hours for hours, in shared.execute(
            "SELECT json_extract(metrics,'$.median_holding_hours') FROM trials WHERE batch=1") if hours is not None]
        horizon = min(48, max(1, math.ceil(float(np.quantile(holds, .75))/4)+1)) if holds else 11
        write_json(setup/"forecast-horizon.json", {"bars": horizon,
                   "method": "75th percentile of candidate median holding times in shared batch one, plus exit bar"})
        warmup.append(run_batch(shared, pool, 2, None, budget, horizon, [], SEED+1))
        for n in range(3, warmup_count+1):
            warmup.append(run_batch(shared, pool, n, None, budget, horizon, [], SEED+n-1,
                                    warmup_kind=1 if n % 2 else 2))
        write_json(directories["shared"]/"batch-history.json", warmup)
        print(market, "shared warm-up complete", horizon, "forecast bars", flush=True)
        shared_records = prior(shared)
        histories = {a: [] for a in arms}
        dbs = {a: stack.enter_context(closing(open_db(directories[a]/"design.sqlite"))) for a in arms}
        for directory in (directories[a] for a in arms):
            write_json(directory/"batch-history.json", warmup)
            update_progress(directory, warmup, budget)
        status(setup, "discovery", market)
        work = {}
        last_completed = warmup_count
        for batch in range(warmup_count+1, rounds+1):
            completed = [db.execute("SELECT report FROM batches WHERE n=?", (batch,)).fetchone()
                         for db in dbs.values()]
            already_started = any(completed)
            if seconds and not already_started:
                started_selection = time.perf_counter()
                current = [shortlist_snapshot(shared_records+prior(db), protocol["shortlist"], comparison)
                           for db in dbs.values()]
                preview = work.copy()
                estimate_selection(preview, current, config)
                clock["selection_seconds"] = max(clock["selection_seconds"], time.perf_counter()-started_selection)
                # Budget one additional list per arm/selection in case the next batch replaces every pick.
                buffer = len(arms)*protocol["shortlist"]*(2 if comparison else 1)*max(preview.values(), default=.12)
                checkpoint_count = (last_completed-warmup_count)//protocol["checkpoint_every"]
                reserve = holdout_budget.reserve_seconds(
                    preview, clock["selection_seconds"]*(checkpoint_count+2)+buffer)
                batch_estimate = 1.3*max(clock["round_seconds"][-5:], default=60.)
                remaining = clock["deadline"]-time.time()
                clock.setdefault("estimates", {})[str(batch)] = {
                    "unique_candidates": len(preview), "reserve_seconds": reserve,
                    "remaining_seconds": round(remaining), "next_round_seconds": round(batch_estimate),
                }
                write_json(clock_path, clock)
                status(setup, "discovery", market, holdout_candidates=len(preview), reserve_seconds=reserve,
                       remaining_seconds=max(0, round(remaining)), next_round_seconds=round(batch_estimate),
                       completed_batches=last_completed)
                if remaining < reserve+batch_estimate:
                    print(market, "discovery stopped to reserve", reserve, "seconds for holdout", flush=True)
                    break
            round_start = time.perf_counter()
            for arm in arms:
                db = dbs[arm]
                existing = db.execute("SELECT decision FROM batches WHERE n=?", (batch,)).fetchone()
                if existing:
                    decision = json.loads(existing[0])
                else:
                    history = (remembered if arm == "programmatic" else [])+warmup+histories[arm]
                    context = campaign(history)
                    if arm == "programmatic":
                        decision = design_search.plan(context, history, design_menu(), budget, SEED+batch)
                    elif arm == "legacy":
                        decision = programmatic(context, batch, SEED+batch)
                    else:
                        decision, receipt = luna(root, config, batch, context, history,
                                                 shared_records+prior(db), protocol["max_api_usd"])
                        decision["provider"] = receipt
                print(market, arm, batch, decision["design"], flush=True)
                report = run_batch(db, pool, batch, decision, budget, horizon, shared_records, SEED+batch)
                histories[arm].append(report)
                write_json(directories[arm]/"batch-history.json", warmup+histories[arm])
                update_progress(directories[arm], warmup+histories[arm], budget)
                print(market, arm, batch, "complete", report["new_distinct"], "new histories",
                      report["new_10"], "new 10% qualifiers", flush=True)
            last_completed = batch
            if seconds:
                if not already_started:
                    clock["round_seconds"].append(time.perf_counter()-round_start)
                    write_json(clock_path, clock)
                if (batch-warmup_count) % protocol["checkpoint_every"] == 0:
                    selections = [shortlist_snapshot(shared_records+prior(db, batch), protocol["shortlist"], comparison)
                                  for db in dbs.values()]
                    estimate_selection(work, selections, config)
        frozen, diverse = {}, {}
        for arm, db in dbs.items():
            records = shared_records+prior(db)
            frozen[arm] = freeze(records, protocol["shortlist"], diverse=not comparison)
            if comparison:
                diverse[arm] = freeze(records, protocol["shortlist"], diverse=True)
        common = min(len(picks) for picks in (*frozen.values(), *diverse.values()))
        for arm in arms:
            checkpoints = {}
            if seconds:
                for n in range(warmup_count+protocol["checkpoint_every"], last_completed, protocol["checkpoint_every"]):
                    snapshot = shortlist_snapshot(shared_records+prior(dbs[arm], n), common, comparison)
                    checkpoints[str(n)] = snapshot
            write_json(directories[arm]/"endpoint.json", {
                "picks": frozen[arm][:common], "diverse_picks": diverse.get(arm, [])[:common],
                "checkpoints": checkpoints,
                "backtests": last_completed*budget, "completed_batches": last_completed,
            })
        write_json(setup/"development-memory.json", {
            "assets": config["assets"], "frame": FRAME, "execution": config["execution"],
            "split": config["split"], "development_hash": config["datasets"][FRAME]["development_hash"],
            "reports": remembered+warmup+histories["programmatic"],
        })
    status(setup, "holdouts_ready", market)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def evaluate_holdouts(root, protocol):
    markets = protocol["market_order"]
    arms = protocol["arms"]
    if not all((root/market/arm/"endpoint.json").is_file() for market in markets for arm in arms):
        raise ValueError("Freeze all shortlists before opening any holdout.")
    rows, diverse_rows, checkpoints, histories, timing = [], [], [], {}, {}
    warmup_count = protocol.get("warmup_batches", 2)
    for market in markets:
        started = time.perf_counter()
        setup = root/market
        config = read(setup/"config.json")
        horizon = read(setup/"forecast-horizon.json")["bars"]
        setup_worker(setup, config, True)
        evaluated = {}
        for arm in arms:
            directory = setup/arm
            histories[f"{market}/{arm}"] = read(directory/"batch-history.json")
            endpoint = read(directory/"endpoint.json")
            for selection, destination in (("picks", rows), ("diverse_picks", diverse_rows)):
                for pick in endpoint.get(selection, []):
                    if pick["id"] not in evaluated:
                        evaluated[pick["id"]] = evaluate(pick["spec"], horizon)
                    destination.append({"market": market, "arm": arm, **pick, "holdout": evaluated[pick["id"]]})
            for batch, selections in endpoint.get("checkpoints", {}).items():
                for selection, picks in selections.items():
                    for pick in picks:
                        if pick["id"] not in evaluated:
                            evaluated[pick["id"]] = evaluate(pick["spec"], horizon)
                        checkpoints.append({"market": market, "arm": arm, "batch": int(batch),
                                            "selection": selection, **pick, "holdout": evaluated[pick["id"]]})
        timing[market] = {"unique_evaluations": len(evaluated), "seconds": time.perf_counter()-started}
    summary = ["# Strategy designer comparison", "",
               f"Trial counts per arm include the same {warmup_count} shared warm-up batches. "
               "Holdout picks are frozen using development data before any holdout is loaded.",
               "", "| Market | Arm | New 10% qualifiers after warm-up | New stable 10% qualifiers | "
               "Holdout 10% / picks | Holdout 15% / picks | Median holdout annualized net |",
               "|---|---|---:|---:|---:|---:|---:|"]
    for market in markets:
        for arm in arms:
            results = [r["holdout"] for r in rows if r["market"] == market and r["arm"] == arm]
            reports = histories[f"{market}/{arm}"][warmup_count:]
            typical = f"{median(m['annual_net_return'] for m in results):.2%}" if results else "—"
            summary.append(f"| {market} | {arm} | {sum(r['new_10'] for r in reports):,} | "
                           f"{sum(r['new_stable_10'] for r in reports):,} | "
                           f"{sum(qualifies(m) for m in results)}/{len(results)} | "
                           f"{sum(qualifies(m, .15) for m in results)}/{len(results)} | {typical} |")
    summary.extend(["", "## Distinct 10% qualifiers found after warm-up", "",
                    "| Market | Batch | Backtests per arm | " + " | ".join(arms) + " |",
                    "|---|---:|---:|" + "---:|"*len(arms)])
    for market in markets:
        completed = min(len(histories[f"{market}/{arm}"]) for arm in arms)
        for batch in range(warmup_count+1, completed+1):
            counts = [sum(r["new_10"] for r in histories[f"{market}/{arm}"][warmup_count:batch]) for arm in arms]
            summary.append(f"| {market} | {batch} | {batch*protocol['batch']:,} | "
                           + " | ".join(f"{count:,}" for count in counts) + " |")
    summary.extend(["", "The detailed results contain every batch's parameter frequencies, performance quantiles, "
                    "new trade histories, matched warm-up comparisons, and frozen holdout picks."])
    write_json(root/"results.json", {"batches": histories, "holdout": rows, "diverse_holdout": diverse_rows,
                                    "checkpoint_holdout": checkpoints, "holdout_timing": timing})
    (root/"summary.md").write_text("\n".join(summary)+"\n", encoding="utf-8")
    print("\n".join(summary), flush=True)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("root", type=Path)
    cli.add_argument("--markets", nargs="+", required=True)
    cli.add_argument("--batch", type=int, default=5000)
    cli.add_argument("--batches", type=int)
    cli.add_argument("--wall-seconds", type=int, help="Per-market target including warm-up, search, and holdout reserve.")
    cli.add_argument("--warmup-batches", type=int)
    cli.add_argument("--checkpoint-every", type=int, default=10, help="Adaptive batches between holdout checkpoints.")
    cli.add_argument("--workers", type=int, default=8)
    cli.add_argument("--parallel-markets", action="store_true",
                     help="Run programmatic markets concurrently within the total worker count.")
    cli.add_argument("--max-api-usd", type=float, default=12.)
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--programmatic-only", action="store_true")
    mode.add_argument("--compare-programmatic", action="store_true")
    cli.add_argument("--discovery-only", action="store_true", help="Stop after freezing development picks.")
    cli.add_argument("--memory", type=Path, action="append", default=[],
                     help="Reuse a compatible development-memory.json from an earlier investigation.")
    args = cli.parse_args()
    if args.parallel_markets and (not args.compare_programmatic or args.workers < len(args.markets)):
        cli.error("Parallel markets require programmatic comparison and at least one worker per market.")
    warmup_count = args.warmup_batches if args.warmup_batches is not None else (10 if args.wall_seconds else 2)
    rounds = args.batches if args.batches is not None else (100_000 if args.wall_seconds else 20)
    if args.wall_seconds is not None and (args.wall_seconds <= 0 or not args.compare_programmatic):
        cli.error("A positive wall-clock budget requires --compare-programmatic.")
    if not 2 <= warmup_count < rounds or args.checkpoint_every < 1:
        cli.error("Use at least two warm-up batches, one adaptive batch, and a positive checkpoint interval.")
    if args.compare_programmatic and args.memory:
        cli.error("The programmatic comparison starts from the shared warm-up.")
    if args.batch < 50 or args.batch % 50 or args.workers < 1 or args.max_api_usd <= 0:
        cli.error("Use a batch size divisible by 50, at least three batches, and positive worker and API limits.")
    root = args.root.resolve()
    configs = {market: read(root/market/"config.json") for market in args.markets}
    for config in configs.values():
        if (list(config["datasets"]) != [FRAME] or len(config["assets"]) != 1
                or config["execution"]["capital"] != 1000 or config["execution"]["allocation"] != 1
                or not config["execution"]["fixed_allocation"] or config["drawdown"] != .30):
            raise ValueError("Prepare single-asset 4h spot data with EUR 1,000, full allocation, and 30% drawdown cap.")
    memory = {}
    for path in args.memory:
        saved = read(path)
        compatible = [market for market, cfg in configs.items()
                      if saved["assets"] == cfg["assets"] and saved["execution"] == cfg["execution"]
                      and saved["frame"] == FRAME and saved["split"] <= cfg["split"]
                      and saved["development_hash"] != cfg["datasets"][FRAME]["development_hash"]]
        if len(compatible) != 1 or compatible[0] in memory:
            raise ValueError(f"{path} must match one market and sizing setup, with earlier development data.")
        memory[compatible[0]] = saved["reports"]
    arms = ("legacy", "programmatic") if args.compare_programmatic else (
        ("programmatic",) if args.programmatic_only else ARMS)
    protocol = {"market_order": args.markets, "markets": configs, "development_memory": memory,
                "arms": arms, "batch": args.batch,
                "families": list(DESIGN_FAMILIES),
                "batches": rounds, "workers": args.workers, "max_api_usd": args.max_api_usd,
                "warmup_batches": warmup_count, "wall_seconds": args.wall_seconds,
                "checkpoint_every": args.checkpoint_every,
                "shortlist": 10, "model": "gpt-6-luna" if "luna" in arms else None,
                "seed": SEED, "search_version": 2}
    protocol = json.loads(encode(protocol))
    if "luna" in arms:
        load_openai_settings(Path(".env"))
    with lock(root):
        path = root/"protocol.json"
        if path.exists() and read(path) != protocol:
            raise ValueError("The benchmark arguments differ from the frozen protocol.")
        write_json(path, protocol)
        write_json(root/"monitor-runtime.json", {"python": sys.executable})
        try:
            if "luna" in arms:
                with closing(sqlite3.connect(root/"api-spending.sqlite")) as db, db:
                    db.execute("CREATE TABLE IF NOT EXISTS requests (id INTEGER PRIMARY KEY, microdollars INTEGER)")
                    db.execute("CREATE TABLE IF NOT EXISTS api_block (id INTEGER PRIMARY KEY, reason TEXT NOT NULL)")
                    db.execute("DELETE FROM api_block")
            status(root, "discovery")
            if args.parallel_markets:
                count = len(configs)
                with ThreadPoolExecutor(max_workers=count) as markets:
                    futures = [markets.submit(discover, root, market, config, protocol,
                                              args.workers//count + (i < args.workers%count))
                               for i, (market, config) in enumerate(configs.items())]
                    for future in as_completed(futures):
                        future.result()
            else:
                for market, config in configs.items():
                    discover(root, market, config, protocol)
            if args.discovery_only:
                status(root, "holdouts_ready")
                return
            status(root, "evaluation")
            evaluate_holdouts(root, protocol)
            status(root, "complete", report="summary.md")
        except Exception as exc:
            status(root, "failed", error=str(exc))
            raise


if __name__ == "__main__":
    main()
