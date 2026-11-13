"""Auditable reports, ranked evidence, and deterministic Pine exports."""
import csv
import json
import time
import urllib.error
from dataclasses import replace

import numpy as np

from . import planner
from .backtest import Execution, metrics, simulate
from .data import STEPS
from .feedback import campaign_map, rich_context
from .research import STAGES, counts, development_history, experiment_report, load_data, write_json
from .smc import INDICATORS as SMC_INDICATORS
from .strategy import Strategy, pine, signals


def ranking(item):
    dev, final = item["development"], item["final"]
    stable = sum(p["return"] > 0 for p in dev["periods"]) / max(1, len(dev["periods"]))
    def score(m):
        return np.clip(m["sharpe"] if m["sharpe"] is not None else -3, -3, 5)
    return (.45*score(dev) + .40*score(final) + .15*stable,
            min(dev["net_return_interval_80"][0], final["net_return_interval_80"][0]),
            item["id"])


def shortlist(rows, maximum=10):
    leaders = []
    deferred = []
    groups = set()
    for item in rows:
        spec = item["strategy"]
        group = (spec["timeframe"], spec["rule"]["indicator"], spec["rule"]["direction"],
                 spec.get("exit_policy", "signal"))
        if group in groups:
            deferred.append(item)
        else:
            leaders.append(item)
            groups.add(group)
    return (leaders + deferred)[:maximum]


def _parquet(path, records):
    import pyarrow as pa
    import pyarrow.parquet as pq

    temporary = path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(records), temporary)
    temporary.replace(path)


def _calibration(times, bars, forecasts, strategy, execution, limit):
    """Score the proposed bracket itself, including offers that were declined."""
    buckets = {}
    horizon = strategy.max_hold
    step = STEPS[strategy.timeframe]
    for forecast in forecasts:
        i = int(np.searchsorted(times, forecast["timestamp"]))
        last = i+horizon-1
        if last >= limit or times[last]-times[i] != (horizon-1)*step:
            continue
        window = bars[i:last+1, forecast["leg"]]
        entry = window[0, 0]*(1+execution.slippage_bps/10000)
        hits_stop = window[:, 2] <= entry*(1-strategy.stop_pct)
        hits_target = window[:, 1] >= entry*(1+strategy.target_pct)
        hits = np.flatnonzero(hits_stop | hits_target)
        target_first = bool(len(hits) and not hits_stop[hits[0]])
        probability = forecast["p_target"]
        bucket = min(9, int(probability*10))
        values = buckets.setdefault(bucket, [])
        values.append((probability, target_first))
    return [{"bucket_start": k/10, "count": len(v),
             "mean_probability": float(np.mean([p for p, _ in v])),
             "observed_frequency": float(np.mean([y for _, y in v])),
             "brier_score": float(np.mean([(p-y)**2 for p, y in v]))}
            for k, v in sorted(buckets.items())]


def _details(item, strategy, root, config, datasets, execution):
    frame = strategy.timeframe
    step = STEPS[frame]
    times, bars, split = datasets[frame]
    signal = signals(times, bars, strategy, {})
    path = root / "strategies" / item["id"][:16]
    path.mkdir(parents=True, exist_ok=True)
    item["readable_rules"] = {
        "asset_a": config["assets"][0],
        "asset_b_or_cash": config["assets"][1] if len(config["assets"]) > 1 else "EUR cash",
        "signal_input": "asset A close divided by asset B close" if len(config["assets"]) > 1 else "asset close",
        "enter_a": f"{strategy.rule.indicator}({strategy.rule.lookback}) "
                   f"{'>' if strategy.rule.direction == 'above' else '<'} {strategy.rule.entry}",
        "exit_policy": strategy.exit_policy,
        "stop_pct": strategy.stop_pct,
        "take_profit_pct": strategy.target_pct,
        "max_holding_bars": strategy.max_hold,
        "execution": "Evaluate at candle close and submit at the next consecutive candle open.",
    }
    if strategy.rule.exit is not None:
        key = "enter_b_when_flat" if strategy.exit_policy == "bracket" else "switch_to_b_or_cash"
        comparison = "<" if strategy.rule.direction == "above" else ">"
        item["readable_rules"][key] = (
            f"{strategy.rule.indicator}({strategy.rule.lookback}) {comparison} {strategy.rule.exit}"
        )
    for label, a, b in (("development", 0, split), ("final", split, len(times)), ("full", 0, len(times))):
        active_times = times[a:b]
        result = simulate(times, bars, signal, strategy, execution, a, b, config["seed"])
        _parquet(path / f"{label}-equity.parquet",
                 [{"timestamp": int(t)+step, "equity_eur": v} for t, v in zip(active_times, result["equity"])])
        _parquet(path / f"{label}-trades.parquet", result["trades"])
        _parquet(path / f"{label}-forecasts.parquet", result["forecasts"])
        if result["open_position"]:
            write_json(path / f"{label}-open-position.json", result["open_position"])
        if label == "full":
            item["full"] = metrics(active_times, result, execution.capital, step)
        if strategy.time_machine and label != "full":
            control = simulate(times, bars, signal, replace(strategy, time_machine=0), execution,
                               a, b, config["seed"])
            item[label]["without_time_machine"] = metrics(active_times, control, execution.capital, step)
            item[label]["target_calibration"] = _calibration(times, bars, result["forecasts"],
                                                            strategy, execution, b)
    write_json(path / "strategy.json", item)
    allocation = execution.allocation_for_stop(strategy.stop_pct)
    script, status = pine(strategy, config["assets"], execution.capital, allocation, execution.fee_bps,
                          execution.slippage_bps, int(times[0]), int(times[-1])+step)
    if script:
        (path / "strategy.pine").write_text(script, encoding="utf-8")
    return status


def publish(root, ledger, summarize=True):
    config = ledger.get("config")
    frozen = ledger.get("final_ids")
    execution = Execution(**config["execution"])
    experiments = experiment_report(ledger)
    candidates = []
    for row in ledger.db.execute("SELECT * FROM candidates WHERE development IS NOT NULL ORDER BY id"):
        candidate = dict(row)
        for field in ("strategy", "development", "final"):
            if candidate[field] is not None:
                candidate[field] = json.loads(candidate[field])
        candidates.append(candidate)
    development = counts([r["development"] for r in candidates])
    finals = [r["final"] for r in candidates if r["final"] is not None]
    probabilities = root / "time-machine-probabilities.csv"
    fields = ["candidate", "experiment", "timeframe", "indicator", "threshold", "period",
              "population", "count", "p_target", "p_stop", "p_neither"]
    temporary = probabilities.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for candidate in candidates:
            spec = candidate["strategy"]
            if not spec["time_machine"]:
                continue
            for period in ("development", "final"):
                measured = candidate[period]
                if measured is None:
                    continue
                for population, summary in measured.get("time_machine", {}).items():
                    writer.writerow({"candidate": candidate["id"], "experiment": candidate["experiment"],
                                     "timeframe": spec["timeframe"], "indicator": spec["rule"]["indicator"],
                                     "threshold": spec["time_machine"], "period": period,
                                     "population": population, **summary})
    temporary.replace(probabilities)
    final = counts(finals)
    qualified = sorted([r for r in candidates if r["final"] and r["final"]["outcome"] == "qualified"],
                       key=ranking, reverse=True)
    overall = shortlist(qualified)
    tradingview = shortlist([r for r in qualified if len(config["assets"]) == 1
                             and not r["strategy"]["time_machine"]
                             and all(rule is None or rule["indicator"] not in SMC_INDICATORS
                                     for rule in (r["strategy"]["rule"], r["strategy"].get("filter_rule")))])
    artifacts = {r["id"]: r for r in overall + tradingview}
    if artifacts:
        datasets = load_data(root, config, final=True)
        for identity, item in artifacts.items():
            strategy = Strategy.from_dict(item["strategy"])
            times, bars, split = datasets[strategy.timeframe]
            allocation = execution.allocation_for_stop(strategy.stop_pct)
            _, status = pine(strategy, config["assets"], execution.capital, allocation, execution.fee_bps,
                             execution.slippage_bps, int(times[0]), int(times[-1])+STEPS[strategy.timeframe])
            item["pine_status"] = status
            item["artifacts"] = f"strategies/{identity[:16]}"
            _details(item, strategy, root, config, datasets, execution)
    failures = ledger.db.execute("SELECT count(*) FROM events WHERE phase='proposal_failure'").fetchone()[0]
    pending = len(frozen or [])-len(finals)
    status = ("development_incomplete" if frozen is None else
              "final_incomplete" if pending else
              "proposal_limit_or_error" if failures >= 3 else
              "complete")
    if status == "complete" and not candidates:
        status = "no_evaluations"
    report = {"status": status, "config": config, "development_funnel": development,
              "final_funnel": final, "final_pending": pending, "proposal_failures": failures,
              "experiments": experiments, "shortlist": overall, "tradingview_shortlist": tradingview,
              "insufficient_evidence": sum(c["development"]["outcome"] == "insufficient_evidence"
                                           for c in candidates),
              "event_log": [{"phase": row["phase"], "details": json.loads(row["details"])}
                            for row in ledger.db.execute("SELECT phase,details FROM events ORDER BY id")]}
    # This memory has no holdout summaries. Only import it into later runs
    # whose own development cutoff covers all of the remembered observations.
    feedback = rich_context(experiments, development_history(ledger))
    if config.get("matched"):
        feedback["campaign_experiments"] = campaign_map(experiments)
        feedback["latest_matched_experiment"] = (
            {"proposal": experiments[-1]["proposal"], "effects": experiments[-1].get("matched")}
            if experiments else None
        )
    write_json(root / "development-memory.json", {"assets": config["assets"],
        "last_development_timestamp": config["split"],
        "feedback": feedback})
    ai = ledger.get("ai_summary")
    if ai and ai["tested_final"] != len(finals):
        ai = None
    if (summarize and frozen is not None and not pending and not ai and not config["offline"]
            and time.time()+5 < config["deadline"]):
        try:
            payload = {k: report[k] for k in ("status", "development_funnel", "final_funnel")}
            payload["shortlist"] = [{"id": r["id"], "strategy": r["strategy"], "development": r["development"],
                                     "final": r["final"]} for r in overall]
            output, provider = planner.response(config["model"],
                "Explain the supplied backtest evidence and limitations. Do not add numeric claims or reorder candidates.",
                payload, planner.obj({"findings": planner.array(planner.STRING),
                                      "limitations": planner.array(planner.STRING)}),
                config["deadline"], "strategy_summary")
            ai = {"tested_final": len(finals), "text": output, "provider": provider}
            ledger.set("ai_summary", ai)
        except (ValueError, TimeoutError, urllib.error.URLError) as exc:
            ledger.event("summary_failure", {"error": str(exc)})
    report["ai_commentary"] = ai
    write_json(root / "report.json", report)
    lines = [f'# Strategy Finder | {" / ".join(config["assets"])}', "",
             f"Status: {status}. Final candidates awaiting evaluation: {pending}.", "",
             "| Qualification stage | Development | Final |", "|---|---:|---:|"]
    for stage in STAGES:
        lines.append(f"| {stage.replace('_', ' ')} | {development[stage]} | {final[stage]} |")
    lines.extend(["", "## Experiments", "", "| ID | Timeframe | Indicator | Generated | Unique | Tested | Profitable | Drawdown | Evidence | Consistent |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"])
    for exp in experiments:
        numbers = " | ".join(str(exp["funnel"][key]) for key in STAGES)
        lines.append(f'| {exp["experiment"]} | {exp["proposal"]["timeframe"]} | {exp["proposal"]["indicator"]} | {exp["generated"]} | {exp["unique"]} | {numbers} |')
    comparisons = [e for e in experiments if "matched" in e]
    if comparisons:
        lines.extend(["", "## Matched development comparisons", "",
                      "| Experiment | Rule changed | Complete pairs | Treatment wins / ties / losses | Median net-return change |",
                      "|---|---|---:|---:|---:|"])
        for exp in comparisons:
            m = exp["matched"]
            delta = m["median_net_return_delta"]
            shown = "—" if delta is None else f"{delta*100:+.2f} pp"
            lines.append(f'| {exp["experiment"]} | {m["change"]} | {m["completed_pairs"]}/{m["planned_pairs"]} '
                         f'| {m["treatment_wins"]} / {m["ties"]} / {m["treatment_losses"]} | {shown} |')
    for name, leaders in (("Overall shortlist", overall), ("TradingView Strategy Tester shortlist", tradingview)):
        lines.extend(["", f"## {name}", ""])
        if not leaders:
            lines.append("No strategies qualified for this shortlist.")
            continue
        lines.extend(["| Strategy | Development return | Final return | Full return | Final drawdown | Final win rate | Final Sharpe | Final trades | Pine |",
                      "|---|---:|---:|---:|---:|---:|---:|---:|---|"])
        for item in leaders:
            spec, build, out = item["strategy"], item["development"], item["final"]
            rate = "—" if out["win_rate"] is None else format(out["win_rate"], ".1%")
            sharpe = "—" if out["sharpe"] is None else format(out["sharpe"], ".2f")
            pine_status = item["pine_status"]
            if pine_status != "python_only_smc":
                pine_status = f'[{pine_status}]({item["artifacts"]}/strategy.pine)'
            lines.append(f'| [{spec["timeframe"]} {spec["rule"]["indicator"]}]({item["artifacts"]}/strategy.json) '
                         f'| {build["net_return"]:.1%} | {out["net_return"]:.1%} | {item["full"]["net_return"]:.1%} | {out["max_drawdown"]:.1%} '
                         f'| {rate} | {sharpe} | {out["trades"]} | {pine_status} |')
    if ai:
        lines.extend(["", "## AI commentary", ""] + ["- " + t for t in ai["text"]["findings"] + ai["text"]["limitations"]])
    lines.extend(["", "[Time Machine probabilities by candidate and period](time-machine-probabilities.csv). "
                  "Values are fractions. Runs made before this metric was added may have no saved probability rows.",
                  "", "Final performance contributes to selection. Forward paper evaluation supplies the next independent evidence.",
                  "The database preserves results for all tested variants, including unqualified candidates."])
    (root / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
