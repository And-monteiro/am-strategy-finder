# AM Strategy Finder

A local research CLI for long-only spot crypto and two-asset rotation. The engine designs and iterates on experiment proposals with OpenAI, tests bounded sets of rule variations, freezes development survivors, and evaluates them on a final chronological holdout. It never sends orders.

## Start

On Windows, double-click `Start Strategy Finder.bat` after installing the project
into `.venv`. Press Enter to keep a prompt's default or type a replacement.
The launcher loads `OPENAI_API_KEY` and `OPENAI_MODEL` from the ignored `.env`
file beside it. Choose **quick** for one small AI experiment, **full** for
the eight-hour investigation, or **offline** for a trial without an API key.

Python 3.11+ and an OpenAI API key are needed for AI investigations. Set `OPENAI_API_KEY` and `OPENAI_MODEL` to a model available to your API project that supports Responses API Structured Outputs. The offline planner is useful for testing the workflow without credentials.

```text
python -m pip install -e .
strategy-finder investigate --asset BTC --timeframes 1h 4h 1d --duration 8h
strategy-finder investigate --asset BTC --pair ETH --duration 8h
strategy-finder investigate --asset BTC --offline --experiments 1 --variants 12 --timeframes 4h --duration 5m
strategy-finder report runs/<run-id>
strategy-finder resume runs/<run-id> --extend 30m
strategy-finder resume runs/<run-id> --continue-discovery --extend 30m
```

Docker Compose runs the same CLI. Supply the variables in the calling environment. The default mounts persist data and reports.

```text
docker compose run --build --rm finder investigate --asset BTC --duration 8h
```

By default the CLI requests the previous five years of 1h Binance spot candles, checks bulk archive checksums and fills the recent tail with the public API. It uses direct EUR markets (`BTC/EUR`, `ETH/EUR`) so initial cash, fills and equity share the same denomination. Arbitrary quote currencies and automatic FX conversion are outside this version. Listing dates and missing archives can shorten coverage. Each dataset records actual bars, missing intervals, and partition hashes. Local hourly EUR CSVs can be supplied with `--csv` once per asset, in asset order. The header is `timestamp,open,high,low,close,volume`, with UTC epoch **seconds** for candle opens. Supply `--start YYYY-MM-DD --end YYYY-MM-DD` to select a UTC range with an exclusive end date.

## Research policy

Use `--matched --variants 10000` to let one proposal direct up to 10,000 local
backtests. Each sampled parameter set gets a control and a treatment, so a
budget of 10,000 covers up to 5,000 pairs. The planner can compare exit policies
or the Time Machine filter. The engine samples the proposed grid using a
recorded seed. The next proposal receives the latest comparison and a compact
map of all development experiments, while individual results stay in SQLite.

The AI receives development-only feedback from the last ten experiments, the
strongest and weakest distinct candidates, and progress by strategy family.
An explicit continuation reuses the same dataset, candidate ledger, and
development feedback. Previously evaluated holdout results remain in the
evaluator and are reused for reporting. Continue with fresh data for an independent
test if decisions about what to search have already been influenced by the holdout.

The AI can propose signal exits, bracket exits, or both within an experiment.
Signal exits can close or rotate a holding when the opposing condition fires.
Bracket exits require a stop-loss and take-profit measured from the entry fill.
An optional holding limit can also close the position. For rotation with bracket
exits, the opposing signal can select asset B while the strategy is flat.
The standard discovery planner uses the holding limit of 1–48 bars as the
Time Machine horizon. The hourly benchmark can set a separate forecast horizon
and allow trades to stay open beyond it.

- The development/final boundary is the 80% point of observed common history, rounded down to a UTC day. The planner receives development metrics only. Final candidates and thresholds are frozen before opening the final snapshot. Separate Parquet files prevent accidental reads in the discovery code. OS users with access to the run directory can still read both snapshots.
- In the development stage, fixed rules run across the whole period and in six-month windows after the first year. Indicators can warm up from prior candles. Each scored window starts flat, first signals may fill from its second candle, and ending holdings are marked at the final close. Holdout consistency uses quarterly windows. There is no statistical fitting stage for the current rule language.
- The funnel tests positive net return, drawdown no greater than both net return and the configured ceiling (30% by default), evidence quality, and consistency. The evidence stage uses a positive lower bound of an 80% moving-block bootstrap interval, no single closed winner contributing over half the gross winning P&L, and profitability at 1.5× estimated costs. The block length is at least 14 days and extends for slower trading based on median holding time. At least two-thirds of a minimum of two available windows must be profitable. This bootstrap is a diagnostic and does not correct for adaptive search or every possible source of dependence.
- Signals use completed bars and fill at the next consecutive open. In a gap, the engine skips new signals and values positions at available observations. Stop takes precedence when both brackets are touched in the same bar. Stops can fill through their threshold. Reported drawdown is from close-marked equity. Defaults are €1,000, 10 bps fee and 5 bps adverse fill adjustment per side, with 0.5% nominal risk on stopped positions capped by cash. Adjust the research assumptions with the CLI flags. There is no liquidity or order-size model.
- Rules use fixed thresholds on supported price features, optional entry confirmation, optional brackets and maximum hold, and optional Time Machine gates. Rotation computes the signal from synchronized closes and moves between real holdings at the next eligible open. Time Machine samples historical contiguous blocks and uses target-before-stop frequency plus simulated net payoff. The proposal generator pairs filtered variants with unfiltered controls. Sampling assumes historical blocks are informative and estimates probabilities conditional on that model.
- The eight-hour default is a scheduling deadline. A long data request or report write can extend wall-clock time. Candidate results checkpoint to SQLite, and a final-stage resume uses the same frozen shortlist. Use `--experiments` for a short, repeatable smoke run. The research memory contains development summaries only and is imported only if their date ranges precede the current holdout.
- The overall shortlist gives one representative to each distinct indicator/direction/timeframe group before filling spare slots. Ranking uses development and final Sharpe, development consistency, and interval lower bounds. Final-period performance contributes to selection. Use future paper trading for independent evidence.

## Outputs

A run contains `report.md`, `report.json`, a SQLite ledger, and hashed candle snapshots. Shortlisted strategies include their exact rules, Pine source, equity curves, trade ledgers, and any Time Machine forecasts. Open positions are recorded separately from closed trades.

The TradingView shortlist includes qualifying single-asset strategies without the Time Machine gate. Use the generated Pine v6 file on the matching Binance EUR chart and timeframe. Matching data and execution settings are necessary for comparison. Percentage slippage and some stop/target semantics have no exact translation in the standard emulator, so affected scripts are labelled approximate. Rotation and Time Machine emit Pine diagnostics for the underlying rule, with their full evaluation retained in Python.

Run the regression checks with `python -m unittest discover -s tests -v`.

## Learning benchmark

Run these phases in order. Discovery compares seeded random proposals, AI
without feedback, AI with the last five experiments, and AI with richer
development feedback. The richer search also forks at half the trial budget
to compare retained and reset memory from the same candidate ledger.

```text
python -m strategy_finder.learning_benchmark prepare runs/learning-test --assets ETH SOL --start 2021-11-01 --end 2026-10-01 --timeframes 1d
python -m strategy_finder.learning_benchmark discover runs/learning-test --model gpt-6.1-sol
python -m strategy_finder.learning_benchmark evaluate runs/learning-test
```

Discovery reads development snapshots only and uses the local OpenAI settings.
Evaluation checks that every requested arm finished its unique-candidate budget
before opening holdout data. It tests the leaders frozen at the discovery
checkpoints and writes `summary.md`, `results.csv`, and `results.json`.
The tested strategy space and seeded random sampling policy are in the benchmark
module. Keep the same dates, arms, seeds, budget, and ranking for a comparison.

For the larger Luna comparison, use a fresh prepared directory and pass
`--model gpt-6-luna --budget 10000 --batch 250 --hours-per-search 2 --max-api-usd 12
--arms random ai_no_feedback ai_last_five ai_rich` to `discover`. The spending
guard is shared by all workers in that directory and prices in-flight requests
conservatively. Keep the same directory, model, seeds, budget, and batch size to
resume discovery. Pass the same budget, seeds, and arms to `evaluate`. The large
benchmark freezes picks at 300, 1,000, 3,000, and 10,000 unique evaluations.

Pass `--fixed-allocation 0.10` to benchmark `prepare` to size every new position
at 10% of equity, regardless of whether it has a stop or how far away the stop
is. Use the same sizing policy for every comparison arm. Trade timing and the
distance to exits can still produce different realized exposures and drawdowns.

The matched-planner comparison gives Luna and a rules-based planner the same
initial experiment, choices, matched sampler, and unique-candidate budgets.
Prepare a daily, single-asset snapshot at fixed 10% allocation, then run the
discovery and evaluation phases separately:

```text
python -m strategy_finder.learning_benchmark prepare runs/matched-test --assets BNB --start 2021-11-01 --end 2026-11-01 --timeframes 1d --fixed-allocation 0.10
python -m strategy_finder.matched_planner_test discover runs/matched-test --asset BNB
python -m strategy_finder.matched_planner_test evaluate runs/matched-test --asset BNB
```

Discovery checkpoints both branches in SQLite. Evaluation checks the trial
budgets and frozen picks before opening the holdout. It writes `summary.md` and
`results.json` beside the protocol and both decision histories.

The long benchmark runs four independent planners for eight hours per prepared
market, with three workers per arm by default. Prepare daily snapshots with
fixed 10% allocation and put the markets in non-overlapping holdout order.

```text
python -m strategy_finder.long_learning_benchmark run runs/long-learning-benchmark --markets DOT AVAX SUI
python -m strategy_finder.long_learning_benchmark status runs/long-learning-benchmark --watch
```

Discovery saves the cumulative leader and the latest batch leader in each
arm's SQLite ledger. The runner evaluates the sealed holdouts automatically
after all markets finish and writes `summary.md`, `checkpoint-results.csv`,
and `equal-trial-results.csv`. On Windows, run
`"Watch Learning Benchmark.bat" -Run runs/long-learning-benchmark`
to monitor that directory.

To run a small hourly comparison on one prepared market:

```text
python -m strategy_finder.learning_benchmark prepare runs/hourly-test --assets BTC --start 2024-01-01 --end 2026-11-01 --timeframes 1h --fixed-allocation 0.10
python -m strategy_finder.long_learning_benchmark run runs/hourly-test --markets BTC --timeframe 1h --batch 5000 --max-batches 1 --workers-per-arm 2 --seconds 3600
python -m strategy_finder.long_learning_benchmark status runs/hourly-test --watch
```

Use a separate directory for a longer test if the smoke run opened its
holdout. The hourly report counts distinct trade histories and tracks new
qualifiers at 10% and 15% annualized net return, below 30% drawdown and above
0.8 Sharpe. Its Time Machine actions use a 24-hour forecast at 20%, 30%, or
40% target-hit probability. The holding limit remains optional.

For the 4h spot comparison, prepare with `--timeframes 4h --fixed-allocation 1`
and run with `--timeframe 4h --allocation 1 --batch 5000 --max-batches 20`.
The Time Machine forecast still spans 24 hours.
Pass `-Run runs/4h-full-allocation` or `-Run runs/hourly-learning-20`
to the Windows monitor to open an earlier run.

The strategy designer benchmark compares Luna and the programmatic designer
after two shared warm-up batches. Prepare single-asset 4h snapshots with
`--fixed-allocation 1`, then run:

```text
python -m strategy_finder.designer_benchmark runs/designer-comparison --markets ETH SOL --batch 5000 --batches 20
```

Pass `-Run runs/designer-comparison` to the Windows monitor for this run. Each arm
records its batch decisions, parameter frequencies, performance distributions,
and distinct trade histories. Both arms on all markets freeze equally sized
shortlists before the runner reads any holdout.

The designer benchmark also supports Smart Money Concepts on single-asset
OHLC candles. Use BOS, CHoCH, fair value gaps, and order blocks as entries or
as confirmation for another rule. An event is available at its confirmation
bar's close and orders use the next eligible open.

- BOS and CHoCH use confirmed symmetric swings. Smaller pivot lengths track
  internal structure, and larger lengths track swing structure. The first
  break establishes direction. A break against that direction is CHoCH.
- Fair value gaps use the outer wicks of three completed candles.
- Order blocks use the last opposing candle between a confirmed pivot and its
  break, with body or full-range bounds.
- Zones support formation, touch, midpoint, or rejection triggers. The
  designer can vary zone size, age, invalidation, repeat retests, and how long
  a confirmation event stays eligible. Gaps in candles reset the state.

These definitions are inspired by
[LuxAlgo's SMC indicator](https://www.luxalgo.com/library/indicator/smart-money-concepts-smc/).
The SMC rules currently execute in Python. Reports mark their Pine export
status as `python_only_smc`.

The programmatic designer splits each adaptive batch between refinement,
under-tested combinations, and matched component tests. Its development
memory records component activity, paired effects, and whether those effects
repeat across windows. Use `--programmatic-only --discovery-only` to exercise
the search and freeze picks with the holdout sealed. Pass an earlier
`development-memory.json` with `--memory` to reuse compatible evidence.

Use `--compare-programmatic` to compare the legacy and revised programmatic
designers with the same warm-up, rule catalogue, and primary shortlist
ranking. The runner also evaluates a family-diverse shortlist for each arm.
Double-click `Watch Learning Benchmark.bat` to follow
`runs/programmatic-search-comparison`.

For a timed programmatic comparison, add `--wall-seconds 28800`.
The per-market clock includes the ten shared warm-up batches and the
adaptive discovery rounds. The runner estimates the holdout reserve from
unique checkpoint and endpoint candidates, measured development runtime,
and shortlist preparation time. It stops starting rounds when the next
round would use the reserve. Both arms run each round on the same worker
pool in sequence. After all markets freeze, every frozen holdout candidate
is scored even if evaluation takes longer than the estimate.
