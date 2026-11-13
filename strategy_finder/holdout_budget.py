"""Estimate holdout work without reading sealed candles."""
import math


def reserve_seconds(unique_work, sealing_seconds=0):
    """Use a 2x cushion, plus a small fixed startup and reporting allowance."""
    return math.ceil(30 + 2*(sum(unique_work.values()) + sealing_seconds))


def candidate_seconds(metrics, development_bars, holdout_bars):
    # The evaluator also computes indicator history over the development prefix.
    return max(.12, metrics.get("evaluation_seconds", 0)*(1+holdout_bars/development_bars))


def add_work(work, selections, development_bars, holdout_bars):
    for picks in selections:
        for pick in picks:
            key = pick["id"]
            work[key] = candidate_seconds(pick["development"], development_bars, holdout_bars)
