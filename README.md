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
```

Docker Compose runs the same CLI. Supply the variables in the calling environment. The default mounts persist data and reports.

```text
docker compose run --build --rm finder investigate --asset BTC --duration 8h
```

By default the CLI requests the previous five years of 1h Binance spot candles, checks bulk archive checksums and fills the recent tail with the public API. It uses direct EUR markets (`BTC/EUR`, `ETH/EUR`) so initial cash, fills and equity share the same denomination. Arbitrary quote currencies and automatic FX conversion are outside this version. Listing dates and missing archives can shorten coverage. Each dataset records actual bars, missing intervals, and partition hashes. Local hourly EUR CSVs can be supplied with `--csv` once per asset, in asset order. The header is `timestamp,open,high,low,close,volume`, with UTC epoch **seconds** for candle opens. Supply `--start YYYY-MM-DD --end YYYY-MM-DD` to select a UTC range with an exclusive end date.

## Research policy

The AI can propose signal exits, bracket exits, or both within an experiment.
Signal exits can close or rotate a holding when the opposing condition fires.
Bracket exits require a stop-loss and take-profit measured from the entry fill.
An optional holding limit can also close the position. For rotation with bracket
exits, the opposing signal can select asset B while the strategy is flat.
Time Machine requires a holding limit of 1–48 bars to simulate the bracket.

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
