"""Development diversity and frozen shortlist for intraday four-arm benchmarks."""
import json

from .research import write_json


def qualifies(metrics, floor=.10):
    return (metrics["annual_net_return"] >= floor and metrics["max_drawdown"] < .30
            and (metrics["sharpe"] or 0) > .8)


def record_batch(ledger, experiment):
    ledger.db.execute("""
        CREATE TABLE IF NOT EXISTS hourly_behavior (
            signature TEXT PRIMARY KEY, first_experiment INTEGER, tier INTEGER,
            candidate TEXT, score REAL
        )
    """)
    rows = ledger.db.execute(
        "SELECT id,development FROM candidates WHERE experiment=? AND development IS NOT NULL",
        (experiment,),
    )
    proposed = {}
    for row in rows:
        metrics = json.loads(row["development"])
        tier = 2 if qualifies(metrics, .15) else int(qualifies(metrics))
        proposed[metrics["behavior"]] = (tier, row["id"], metrics["sharpe"] or 0)
    ledger.db.executemany(
        "INSERT OR IGNORE INTO hourly_behavior VALUES (?,?,?,?,?)",
        [(signature, experiment, *item) for signature, item in proposed.items()],
    )
    first = ledger.db.execute(
        "SELECT count(*),coalesce(sum(tier>=1),0),coalesce(sum(tier>=2),0) "
        "FROM hourly_behavior WHERE first_experiment=?", (experiment,)
    ).fetchone()
    cumulative = ledger.db.execute(
        "SELECT count(*),coalesce(sum(tier>=1),0),coalesce(sum(tier>=2),0) "
        "FROM hourly_behavior"
    ).fetchone()
    return dict(batch_distinct=len(proposed), new_distinct=first[0], new_10=first[1], new_15=first[2],
                cumulative_distinct=cumulative[0], cumulative_10=cumulative[1], cumulative_15=cumulative[2])


def freeze_shortlist(ledger):
    rows = ledger.db.execute("""
        SELECT c.id,c.strategy,c.development,c.experiment FROM hourly_behavior b
        JOIN candidates c ON c.id=b.candidate WHERE b.tier>=1
        ORDER BY b.tier DESC,b.score DESC,c.id LIMIT 10
    """)
    return [{"id": r["id"], "strategy": json.loads(r["strategy"]), "experiment": r["experiment"],
             "development": json.loads(r["development"])} for r in rows]


def write_summary(root, setups, arms, read, rows):
    development = []
    starting_counts = {}
    frames = ", ".join(sorted({frame for _, config in setups for frame in config["datasets"]}))
    allocation = setups[0][1]["execution"]["allocation"]
    lines = [f"# {frames} four-arm benchmark", "",
             f"Fixed position sizing uses {allocation:.0%} of available cash per entry.",
             "Returns are annualized and net of modeled fees and slippage. "
             "The return tiers also require drawdown below 30% and Sharpe above 0.8.",
             "", "| Market | Arm | Follow-up trials | New histories | New at 10% | New at 15% |",
             "|---|---|---:|---:|---:|---:|"]
    for setup, config in setups:
        market = config["assets"][0]
        shared, = read(setup/"shared"/"rounds.json")
        starting_counts[market] = shared["diversity"]["cumulative_10"]
        for arm in arms:
            batches = read(setup/arm/"checkpoints.json")
            for batch in batches:
                development.append({"market": market, "arm": arm, "backtests": batch["backtests"],
                                    "complete": batch["complete"],
                                    "action": batch["experiment"]["proposal"]["choice"],
                                    **batch["experiment"]["diversity"]})
            totals = [sum(batch["experiment"]["diversity"][key] for batch in batches)
                      for key in ("new_distinct", "new_10", "new_15")]
            lines.append(f"| {market} | {arm} | {batches[-1]['backtests']:,} | "
                         f"{totals[0]:,} | {totals[1]:,} | {totals[2]:,} |")
    lines.extend(["", "New discoveries exclude the common starting batch.", "",
                  "## New 10% qualifiers at equal trial counts", "",
                  "| Market | Follow-up trials per arm | Random | Programmatic | Luna without feedback | Luna with feedback |",
                  "|---|---:|---:|---:|---:|---:|"])
    for _, config in setups:
        market = config["assets"][0]
        series = {arm: {r["backtests"]: r["cumulative_10"] for r in development
                        if r["market"] == market and r["arm"] == arm and r["complete"]}
                  for arm in arms}
        common = sorted(set.intersection(*(set(v) for v in series.values())))
        for index, count in enumerate(common):
            if index in (0, 4, 9, 14) or count == common[-1]:
                values = " | ".join(f"{series[arm][count]-starting_counts[market]:,}" for arm in arms)
                lines.append(f"| {market} | {count:,} | {values} |")
    lines.extend(["",
                  "## Frozen shortlist on the holdout", "",
                  "| Market | Arm | Evaluated | Passed 10% | Passed 15% |",
                  "|---|---|---:|---:|---:|"])
    for _, config in setups:
        market = config["assets"][0]
        for arm in arms:
            shortlisted = [row for row in rows if row["market"] == market
                           and row["arm"] == arm and row["pick"] == "shortlist"]
            lines.append(f"| {market} | {arm} | {len(shortlisted)} | "
                         f"{sum(r['qualified_10'] for r in shortlisted)} | "
                         f"{sum(r['qualified_15'] for r in shortlisted)} |")
    lines.extend(["", "The JSON contains per-batch discoveries and holdout performance for each frozen pick."])
    write_json(root/"hourly-results.json", {"development": development, "holdout": rows})
    summary = "\n".join(lines)+"\n"
    (root/"summary.md").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
