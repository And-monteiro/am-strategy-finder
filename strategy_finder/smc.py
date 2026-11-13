"""Causal OHLC structure and zone events inspired by LuxAlgo SMC."""
from collections import deque

import numpy as np

INDICATORS = ("bos", "choch", "fvg", "order_block")


def structure(times, candles, step, length, wick):
    """Emit signed breaks and the corresponding confirmed pivot index."""
    high, low, close = candles[:, 1], candles[:, 2], candles[:, 3]
    events = np.zeros(len(times), dtype=np.int8)
    pivots = np.full(len(times), -1, dtype=int)
    upper = lower = -1
    bias = 0
    start = 0
    for i in range(len(times)):
        if i and times[i]-times[i-1] != step:
            start, upper, lower, bias = i, -1, -1, 0
        if i-start >= 2*length:
            p = i-length
            left = slice(p-length, p)
            right = slice(p+1, i+1)
            if high[p] > max(high[left].max(), high[right].max()):
                upper = p
            if low[p] < min(low[left].min(), low[right].min()):
                lower = p
        up = upper >= 0 and (high[i] if wick else close[i]) > high[upper]
        down = lower >= 0 and (low[i] if wick else close[i]) < low[lower]
        if up and down:
            upper = lower = -1
            bias = 0
        elif up or down:
            side = 1 if up else -1
            events[i] = side * (2 if bias == -side else 1)
            pivots[i] = upper if up else lower
            bias = side
            if up:
                upper = -1
            else:
                lower = -1
    return events, pivots


def zones(times, candles, step, rule, breaks=None):
    """Track formation, retests and invalidation, separately for both sides."""
    events = np.zeros((len(times), 2), dtype=bool)
    opening, high, low, close = candles[:, :4].T
    active = deque()
    start = 0
    for i in range(len(times)):
        if i and times[i]-times[i-1] != step:
            active.clear()
            start = i
        for _ in range(len(active)):
            born, side, bottom, top, touched = active.popleft()
            if i-born > rule.smc_max_age:
                continue
            invalid = ((low[i] if rule.smc_wick_invalidation else close[i]) <= bottom if side == 1
                       else (high[i] if rule.smc_wick_invalidation else close[i]) >= top)
            if invalid:
                continue
            level = (bottom+top)/2 if rule.smc_trigger == "midpoint" else (top if side == 1 else bottom)
            touch = low[i] <= level <= high[i]
            rejected = close[i] > top if side == 1 else close[i] < bottom
            qualifies = touch and (rule.smc_trigger != "rejection" or rejected)
            if qualifies:
                events[i, int(side < 0)] = True
                touched = True
            if not (rule.smc_first_retest and touched):
                active.append((born, side, bottom, top, touched))
        new = None
        if rule.indicator == "fvg" and i-start >= 2:
            if low[i] > high[i-2]:
                new = (1, high[i-2], low[i])
            elif high[i] < low[i-2]:
                new = (-1, high[i], low[i-2])
        elif rule.indicator == "order_block" and breaks[0][i]:
            side = int(np.sign(breaks[0][i]))
            earliest = max(start, int(breaks[1][i]), i-rule.smc_max_age)
            opposing = np.flatnonzero(side*(close[earliest:i]-opening[earliest:i]) < 0)
            if len(opposing):
                index = earliest+opposing[-1]
                bottom, top = (min(opening[index], close[index]), max(opening[index], close[index])) if rule.smc_body else (low[index], high[index])
                if (side == 1 and close[i] > top) or (side == -1 and close[i] < bottom):
                    new = (side, bottom, top)
        if new:
            side, bottom, top = new
            if top-bottom > 0 and (top-bottom)/close[i] >= rule.smc_min_size:
                if rule.smc_trigger == "formation":
                    events[i, int(side < 0)] = True
                else:
                    active.append((i, side, bottom, top, False))
    return events


def event_masks(times, candles, step, rule, cache):
    breaks = None
    if rule.indicator != "fvg":
        key = ("smc_structure", step, rule.lookback, rule.smc_wick_break)
        if key not in cache:
            store(cache, key, structure(times, candles, step, rule.lookback, rule.smc_wick_break))
        breaks = cache[key]
    if rule.indicator in ("bos", "choch"):
        kind = 1 if rule.indicator == "bos" else 2
        events = np.column_stack((breaks[0] == kind, breaks[0] == -kind))
    else:
        events = zones(times, candles, step, rule, breaks)
    if rule.smc_recent > 1:
        last = np.full(2, -rule.smc_recent)
        for i, row in enumerate(events):
            if i and times[i]-times[i-1] != step:
                last[:] = i-rule.smc_recent
            last[row] = i
            row[:] = i-last < rule.smc_recent
    return events.T


def store(cache, key, value):
    if len(cache) >= 256:
        cache.pop(next(iter(cache)))
    cache[key] = value
