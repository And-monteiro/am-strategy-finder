"""Equal-budget strategy discovery benchmark. Evaluation is a separate command."""
from __future__ import annotations

import argparse
import csv
import json
import random
import threading
import time
import urllib.error
from statistics import mean, median
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from . import planner
from .backtest import Execution
from .cli import parser as cli_parser, prepare as prepare_data
from .feedback import (RICH_INSTRUCTIONS, RICH_SCHEMA, development_score, requested_action,
                       rich_context, validate_action)
from .research import assess, counts, encode, load_data, variants, write_json
from .strategy import Strategy, signals
from .windows_launcher import load_openai_settings

ARMS = ("random", "ai_no_feedback", "ai_last_five", "ai_rich", "memory_reset")
OUTPUT_LOCK = threading.Lock()


def log(*parts):
    with OUTPUT_LOCK:
        print(*parts, flush=True)


def prepare(root, assets, start, end, frames):
    for asset in assets:
        if not asset.isascii() or not asset.isalnum():
            raise ValueError("Benchmark assets must be ticker symbols.")
        setup = root / asset.upper()
        setup.mkdir(parents=True, exist_ok=True)
        if (setup / "config.json").exists():
            raise ValueError(f"Dataset already frozen at {setup}. Use a new benchmark directory.")
        args = cli_parser().parse_args(["investigate", "--offline", "--asset", asset,
                                        "--start", start, "--end", end,
                                        "--duration", "30m", "--timeframes", *frames])
        config = prepare_data(args, setup, time.time())
        write_json(setup / "config.json", config)


def random_proposal(rng, frames):
    indicator = rng.choice(planner.FAMILIES)
    direction = rng.choice(["above", "below"])
    low, high = {"zscore": (-3, 3), "return": (-.20, .30), "sma_distance": (-.15, .20),
                 "breakout": (-.20, .05), "volatility": (.005, .10)}[indicator]
    threshold = sorted(round(rng.uniform(low, high), 4) for _ in range(6))
    entries, exits = (threshold[3:], threshold[:3]) if direction == "above" else (threshold[:3], threshold[3:])
    gate = None
    if rng.random() < 1/3:
        gate_indicator = rng.choice(planner.FAMILIES)
        gate_entry, gate_exit = {
            "zscore": (.5, -.5), "return": (.05, -.01), "sma_distance": (.02, -.01),
            "breakout": (.001, -.02), "volatility": (.02, .01),
        }[gate_indicator]
        gate = {"indicator": gate_indicator, "lookback": rng.randint(5, 150),
                "entry": gate_entry, "exit": gate_exit, "direction": "above"}
    return {"hypothesis": "Seeded rule and parameter sampling.", "parent": None,
            "timeframe": rng.choice(frames), "indicator": indicator, "direction": direction,
            "lookbacks": rng.sample(range(5, 151), 5), "entries": entries, "exits": exits,
            "holding_periods": [0, 12, 24], "stops": [0, .03, .07, .12],
            "targets": [0, .06, .14, .24], "time_machine_thresholds": [0, .15],
            "exit_policies": ["signal", "bracket"], "filter_rule": gate}


def experiment_summary(experiment, proposal, evaluated, generated):
    leaders = sorted(evaluated, key=lambda c: development_score(c["development"]), reverse=True)[:3]
    return {"experiment": experiment, "proposal": proposal, "generated": generated,
            "unique": len(evaluated), "funnel": counts([c["development"] for c in evaluated]),
            "top_parameters": [{"strategy": c["strategy"], "metrics": c["development"]} for c in leaders]}


def _state(candidates, experiments, checkpoints, attempts, usage, errors, duplicates):
    return {"candidates": candidates, "experiments": experiments, "checkpoints": checkpoints,
            "attempts": attempts, "usage": usage, "errors": errors, "duplicates": duplicates}


def search(path, arm, seed, data, config, model, goals, budget, inherited=None):
    """Search only development arrays. This function never opens holdout files."""
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["status"] == "complete":
            return saved
    state = inherited or _state([], [], {}, 0, [], [], 0)
    candidates = list(state["candidates"])
    experiments = list(state["experiments"])
    checkpoints = dict(state["checkpoints"])
    inherited_count = len(candidates)
    used = {encode(c["strategy"]) for c in candidates}
    attempts = duplicates = 0
    usage, errors = [], []
    rng = random.Random(seed)
    execution = Execution(**config["execution"])
    feature = {frame: {} for frame in data}
    replay = {frame: {} for frame in data}
    start = time.monotonic()
    base = {"assets": config["assets"], "allowed_timeframes": list(data)}
    leader = max(candidates, key=lambda c: development_score(c["development"]), default=None)
    deadline = time.time() + 1800

    def save(status):
        result = _state(candidates, experiments, checkpoints, attempts, usage, errors, duplicates)
        result.update({"arm": arm, "seed": seed, "model": model, "status": status,
                       "seconds": time.monotonic()-start, "inherited_count": inherited_count})
        write_json(path, result)
        return result

    while len(candidates) < budget and attempts < 42 and time.time() < deadline:
        attempts += 1
        local_experiments = experiments if arm != "memory_reset" else experiments[len(state["experiments"]):]
        local_candidates = candidates if arm != "memory_reset" else candidates[inherited_count:]
        n = len(experiments) + 1
        try:
            if arm == "random":
                proposal = random_proposal(rng, list(data))
            else:
                payload = {**base, "proposal_number": n, "max_variants": 50}
                instructions, schema = planner.INSTRUCTIONS, planner.SCHEMA
                if arm == "ai_last_five":
                    payload.update(recent_development_results=experiments[-5:], prior_development_memory=[])
                elif arm in ("ai_rich", "memory_reset", "memory_retained"):
                    context = rich_context(local_experiments, local_candidates)
                    action = requested_action(attempts-1, bool(local_experiments))
                    payload.update(development_feedback=context, requested_action=action)
                    instructions, schema = RICH_INSTRUCTIONS, RICH_SCHEMA
                proposal, response = planner.response(model, instructions, payload, schema, deadline,
                                                       "learning_experiment")
                usage.append(response)
                if arm in ("ai_rich", "memory_reset", "memory_retained"):
                    validate_action(proposal, context, action)
            choices = variants(proposal, 50, rng.randrange(2**32), list(data))
        except (ValueError, KeyError, TypeError, TimeoutError, urllib.error.URLError) as exc:
            errors.append(str(exc))
            if isinstance(exc, urllib.error.HTTPError) and exc.code in (401, 403, 404):
                break
            continue

        evaluated = []
        for spec in choices:
            identity = encode(spec)
            if identity in used:
                duplicates += 1
                continue
            used.add(identity)
            strategy = Strategy.from_dict(spec)
            frame = strategy.timeframe
            times, bars, _ = data[frame]
            signal = signals(times, bars, strategy, feature[frame])
            result, _ = assess(times, bars, signal, strategy, execution, config["drawdown"], seed,
                               tm_cache=replay[frame], policy=config.get("evidence"))
            candidate = {"strategy": spec, "development": result, "experiment": n}
            candidates.append(candidate)
            evaluated.append(candidate)
            if leader is None or development_score(result) > development_score(leader["development"]):
                leader = candidate
            if len(candidates) in goals:
                new_leader = max(candidates[inherited_count:],
                                 key=lambda c: development_score(c["development"]))
                checkpoints[str(len(candidates))] = {"leader": leader, "new_leader": new_leader}
            if len(candidates) == budget:
                break
        experiments.append(experiment_summary(n, proposal, evaluated, len(choices)))
        save("running")
        log(path.parent.name, path.stem, len(candidates), "/", budget, "attempts", attempts)
    return save("complete" if len(candidates) == budget else "incomplete")


def discover(root, model, seeds, workers, arms, budget):
    jobs = []
    configs = {}
    for setup in sorted(root.iterdir()):
        if (setup / "config.json").is_file():
            config = json.loads((setup / "config.json").read_text(encoding="utf-8"))
            configs[setup] = (config, load_data(setup, config))
            jobs.extend((setup, seed, arm) for seed in seeds for arm in arms)
    goals = sorted({min(50, budget), budget//2, budget})
    if not jobs:
        raise ValueError("No benchmark datasets were prepared.")

    def job(setup, seed, arm):
        config, data = configs[setup]
        directory = setup / f"seed-{seed}"
        directory.mkdir(exist_ok=True)
        if arm in ("ai_rich", "memory_reset"):
            base = search(directory / "shared-checkpoint.json", "ai_rich", seed, data, config,
                          model, goals, budget//2)
            if base["status"] != "complete":
                raise ValueError("Shared checkpoint did not finish.")
            effective_arm = "memory_retained" if arm == "ai_rich" else arm
            return search(directory / f"{arm}.json", effective_arm, seed, data, config,
                          model, goals, budget, base)
        return search(directory / f"{arm}.json", arm, seed, data, config, model, goals, budget)

    # Build shared checkpoints before launching either continuation.
    with ThreadPoolExecutor(workers) as pool:
        if set(arms) & {"ai_rich", "memory_reset"}:
            def checkpoint(setup, seed):
                config, data = configs[setup]
                directory = setup / f"seed-{seed}"
                directory.mkdir(exist_ok=True)
                return search(directory / "shared-checkpoint.json", "ai_rich", seed,
                              data, config, model, goals, budget//2)
            shared = [pool.submit(checkpoint, setup, seed) for setup in configs for seed in seeds]
        else:
            shared = []
        standard = [pool.submit(job, *args) for args in jobs if args[2] not in ("ai_rich", "memory_reset")]
        for future in as_completed(shared):
            if future.result()["status"] != "complete":
                log("A shared checkpoint is incomplete.")
        continuations = [pool.submit(job, *args) for args in jobs if args[2] in ("ai_rich", "memory_reset")]
        for future in as_completed(standard + continuations):
            try:
                result = future.result()
                if result["status"] != "complete":
                    log("INCOMPLETE", result["arm"], result["seed"], result["errors"][-3:])
            except ValueError as exc:
                log("INCOMPLETE", str(exc))


def evaluate(root, seeds, arms, budget):
    """Read frozen checkpoint selections. Never call the model or write planner memory."""
    rows = []
    for setup in sorted(root.iterdir()):
        if not (setup / "config.json").is_file():
            continue
        for seed in seeds:
            for arm in arms:
                path = setup / f"seed-{seed}" / f"{arm}.json"
                if not path.is_file():
                    raise ValueError(f"Missing discovery result: {path}")
                result = json.loads(path.read_text(encoding="utf-8"))
                if result["status"] != "complete" or len(result["candidates"]) != budget:
                    raise ValueError(f"Discovery budget incomplete: {path}")
    for setup in sorted(root.iterdir()):
        if not (setup / "config.json").is_file():
            continue
        config = json.loads((setup / "config.json").read_text(encoding="utf-8"))
        data = load_data(setup, config, final=True)
        execution = Execution(**config["execution"])
        for seed in seeds:
            directory = setup / f"seed-{seed}"
            cache = {}
            for arm in arms:
                path = directory / f"{arm}.json"
                search = json.loads(path.read_text(encoding="utf-8"))
                for count, checkpoint in search["checkpoints"].items():
                    for label in ("leader", "new_leader") if search["inherited_count"] else ("leader",):
                        candidate = checkpoint[label]
                        spec = candidate["strategy"]
                        identity = encode(spec)
                        if identity not in cache:
                            strategy = Strategy.from_dict(spec)
                            times, bars, cut = data[strategy.timeframe]
                            signal = signals(times, bars, strategy, {})
                            cache[identity], _ = assess(times, bars, signal, strategy, execution,
                                                        config["drawdown"], search["seed"], final=True,
                                                        warmup=cut, policy=config.get("evidence"))
                        result = cache[identity]
                        rows.append({"setup": setup.name, "seed": search["seed"], "arm": path.stem,
                                     "trials": int(count), "pick": label, "status": search["status"],
                                     "strategy": spec, "development": candidate["development"],
                                     "holdout": result})
    write_json(root / "results.json", rows)
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["setup", "seed", "arm", "trials", "pick", "status",
                         "dev_return", "holdout_return", "holdout_drawdown", "holdout_outcome"])
        for r in rows:
            writer.writerow([r["setup"], r["seed"], r["arm"], r["trials"], r["pick"], r["status"],
                             r["development"]["net_return"], r["holdout"]["net_return"],
                             r["holdout"]["max_drawdown"], r["holdout"]["outcome"]])
    log("Evaluated", len(rows), "frozen picks.")
    write_summary(root, rows, budget)


def write_summary(root, rows, budget):
    def net(r):
        return r["holdout"]["net_return"]

    def qualified(r):
        return r["holdout"]["outcome"] == "qualified"

    primary = [r for r in rows if r["trials"] == budget and r["pick"] == "leader"]
    lines = ["# Strategy Finder learning benchmark", "",
             f"Primary checkpoint: {budget} unique candidate evaluations per search.",
             "Development ranking fixed before discovery. Holdout results were read after discovery.",
             "", "| Search | Repeats | Mean holdout return | Median holdout return | Mean holdout drawdown | Qualified |",
             "|---|---:|---:|---:|---:|---:|"]
    for arm in ARMS:
        selected = [r for r in primary if r["arm"] == arm]
        if selected:
            drawdown = mean(r["holdout"]["max_drawdown"] for r in selected)
            lines.append(f"| {arm} | {len(selected)} | {mean(map(net, selected)):.2%} | "
                         f"{median(map(net, selected)):.2%} | {drawdown:.2%} | {sum(map(qualified, selected))} |")
    lines.extend(["", "## Paired comparisons", "",
                  "| Comparison | Pick | Pairs | Mean difference | Wins / ties / losses |",
                  "|---|---|---:|---:|---:|"])
    for left, right, pick in [
        ("ai_no_feedback", "random", "leader"),
        ("ai_last_five", "ai_no_feedback", "leader"),
        ("ai_rich", "ai_last_five", "leader"),
        ("ai_rich", "ai_no_feedback", "leader"),
        ("ai_rich", "memory_reset", "new_leader"),
    ]:
        comparison = {(r["setup"], r["seed"], r["arm"]): net(r) for r in rows
                      if r["pick"] == pick and r["trials"] == budget}
        differences = [value - comparison[setup, seed, right]
                       for (setup, seed, arm), value in comparison.items()
                       if arm == left and (setup, seed, right) in comparison]
        if differences:
            wins = sum(d > 1e-10 for d in differences)
            losses = sum(d < -1e-10 for d in differences)
            lines.append(f"| {left} vs {right} | {pick} | {len(differences)} | "
                         f"{mean(differences)*100:+.2f} pp | {wins} / {len(differences)-wins-losses} / {losses} |")
    lines.extend(["", "Seeds within each asset share the same holdout. Treat them as repeated searches on two "
                  "markets rather than independent market samples.", "",
                  "[Checkpoint results](results.csv)", ""])
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "discover", "evaluate"))
    parser.add_argument("root", type=Path)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--seeds", nargs="+", type=int, default=[19, 43, 97])
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=list(ARMS))
    parser.add_argument("--budget", type=int, default=300)
    parser.add_argument("--assets", nargs="+")
    parser.add_argument("--start")
    parser.add_argument("--end")
    parser.add_argument("--timeframes", nargs="+", default=["1d"])
    args = parser.parse_args()
    if args.budget < 4 or args.budget % 2:
        parser.error("Budget must be an even number at least four.")
    if args.phase == "prepare":
        if not all((args.assets, args.start, args.end)):
            parser.error("Prepare requires --assets, --start, and --end.")
        prepare(args.root, args.assets, args.start, args.end, args.timeframes)
    elif args.phase == "discover":
        load_openai_settings(Path(".env"))
        discover(args.root, args.model, args.seeds, args.workers, args.arms, args.budget)
    else:
        evaluate(args.root, args.seeds, args.arms, args.budget)


if __name__ == "__main__":
    main()
