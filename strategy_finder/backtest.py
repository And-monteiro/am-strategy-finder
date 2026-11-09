"""Long-only event simulation. Signals use the previous completed bar."""
from dataclasses import dataclass

import numpy as np

from .data import DAY, STEPS


class EvaluationDeadline(TimeoutError):
    """A candidate can safely be retried because no partial result was saved."""


@dataclass(frozen=True)
class Execution:
    capital: float = 1000
    allocation: float = 1.0
    risk: float = 0.005
    fee_bps: float = 10
    slippage_bps: float = 5

    def __post_init__(self):
        if not all(type(x) in (int, float) and np.isfinite(x) for x in vars(self).values()):
            raise ValueError("Execution values must be finite numbers.")
        if not self.capital > 0 or not 0 < self.allocation <= 1 or not 0 < self.risk <= 1:
            raise ValueError("Invalid capital, allocation or stop risk.")
        if not 0 <= self.fee_bps < 10000 or not 0 <= self.slippage_bps < 10000:
            raise ValueError("Invalid cost assumption.")


def time_machine(bars, times, i, leg, strategy, execution, seed, paths=192):
    """Joint block draws from completed history, anchored at signal close."""
    horizon = strategy.max_hold
    block = min(4, horizon)
    step = STEPS[strategy.timeframe]
    start = max(1, i - 730)
    candidates = np.arange(start, i - block + 2)
    if len(candidates):
        candidates = candidates[times[candidates + block - 1] - times[candidates - 1] == step * block]
    if len(candidates) < 40:
        return None
    rng = np.random.default_rng(np.random.SeedSequence([seed, int(times[i] // 3600)]))
    draws = rng.choice(candidates, (paths, (horizon + block - 1) // block))
    indexes = (draws[..., None] + np.arange(block)).reshape(paths, -1)[:, :horizon]
    ratios = bars[indexes, leg, :4] / bars[indexes - 1, leg, 3, None]
    growth = np.cumprod(ratios[:, :, 3], axis=1)
    preceding = np.column_stack((np.ones(paths), growth[:, :-1]))
    price_paths = ratios * preceding[:, :, None]
    slip = execution.slippage_bps / 10000
    fee = execution.fee_bps / 10000
    entry = price_paths[:, 0, 0] * (1 + slip)
    stop = entry * (1 - strategy.stop_pct)
    target = entry * (1 + strategy.target_pct)
    stop_hit = price_paths[:, :, 2] <= stop[:, None]
    target_hit = price_paths[:, :, 1] >= target[:, None]
    hit = stop_hit | target_hit
    first = hit.argmax(axis=1)
    rows = np.arange(paths)
    stopped = hit.any(axis=1) & stop_hit[rows, first]
    won = hit.any(axis=1) & ~stopped
    exits = np.where(stopped, np.minimum(stop, price_paths[rows, first, 0]),
                     np.where(won, target, growth[:, -1]))
    payoff = exits * (1-slip) * (1-fee) / (entry * (1+fee)) - 1
    return float(won.mean()), float(stopped.mean()), float((~(won | stopped)).mean()), float(payoff.mean())


def simulate(times, bars, signal, strategy, execution, start, stop, seed=19, tm_cache=None, deadline=None):
    import time

    cash = execution.capital
    leg = -1
    shares = basis = entry_price = 0.0
    entry_i = 0
    costs = turnover = 0.0
    step = STEPS[strategy.timeframe]
    slip = execution.slippage_bps / 10000
    fee_rate = execution.fee_bps / 10000
    trades, equity, forecasts = [], [], []
    tm_cache = {} if tm_cache is None else tm_cache
    occupied = 0

    def close(index, raw, reason):
        nonlocal cash, shares, leg, costs, turnover
        proceeds = shares * raw * (1-slip)
        fee = proceeds * fee_rate
        cash += proceeds - fee
        costs += fee + shares * raw * slip
        turnover += proceeds
        trades.append({"entry_timestamp": int(times[entry_i]), "exit_timestamp": int(times[index]),
                       "leg": leg, "quantity": shares, "entry": entry_price,
                       "exit": raw * (1-slip), "pnl": proceeds - fee - basis, "reason": reason})
        shares, leg = 0.0, -1

    def bracket(index):
        if leg < 0:
            return
        opening, high, low = bars[index, leg, :3]
        floor = entry_price * (1-strategy.stop_pct)
        ceiling = entry_price * (1+strategy.target_pct)
        if strategy.stop_pct and low <= floor:
            close(index, min(opening, floor), "stop")
        elif strategy.target_pct and high >= ceiling:
            close(index, ceiling, "target")

    for i in range(start, stop):
        if deadline is not None and i % 128 == 0 and time.time() >= deadline:
            raise EvaluationDeadline("Candidate exceeded its evaluation deadline.")
        continuous = i > 0 and times[i] - times[i-1] == step
        desired = int(signal[i-1]) if i > start and continuous else -2
        expired = leg >= 0 and strategy.max_hold and times[i] - times[entry_i] >= strategy.max_hold * step
        if expired or (leg >= 0 and desired != -2 and desired != leg):
            close(i, bars[i, leg, 0], "timeout" if expired else "signal")
        if leg < 0 and desired >= 0 and not expired:
            permitted = True
            forecast = None
            if strategy.time_machine:
                key = (i-1, desired, strategy.max_hold, strategy.stop_pct, strategy.target_pct)
                if key not in tm_cache:
                    if len(tm_cache) >= 60_000:
                        tm_cache.clear()
                    tm_cache[key] = time_machine(bars, times, i-1, desired, strategy, execution, seed)
                forecast = tm_cache[key]
                permitted = bool(forecast and forecast[0] >= strategy.time_machine and forecast[3] > 0)
                if forecast is not None:
                    forecasts.append({"timestamp": int(times[i]), "leg": desired,
                                      "p_target": forecast[0], "p_stop": forecast[1], "p_neither": forecast[2],
                                      "expected_net_return": forecast[3], "accepted": permitted})
            if permitted:
                allocation = min(execution.allocation, execution.risk / strategy.stop_pct) if strategy.stop_pct else execution.allocation
                basis = cash * allocation
                entry_price = bars[i, desired, 0] * (1+slip)
                shares = basis / (entry_price * (1+fee_rate))
                cash -= basis
                costs += shares * entry_price * fee_rate + shares * bars[i, desired, 0] * slip
                turnover += shares * entry_price
                leg, entry_i = desired, i
        occupied += leg >= 0
        bracket(i)
        mark = cash + (shares * bars[i, leg, 3] if leg >= 0 else 0)
        equity.append(float(mark))
    return {"equity": equity, "trades": trades, "forecasts": forecasts, "costs": float(costs),
            "turnover": float(turnover / execution.capital), "exposure": occupied / max(1, stop-start),
            "open_position": None if leg < 0 else {"leg": leg, "quantity": shares,
                "entry_timestamp": int(times[entry_i]), "entry": entry_price, "basis": basis}}


def daily_returns(times, equity, capital, step):
    days = (times + step - 1) // DAY
    ends = np.r_[np.flatnonzero(days[1:] != days[:-1]), len(days)-1]
    values = np.r_[capital, np.asarray(equity)[ends]]
    return values[1:] / values[:-1] - 1


def metrics(times, result, capital, step):
    equity = np.asarray(result["equity"])
    all_equity = np.r_[capital, equity]
    drawdown = 1 - all_equity / np.maximum.accumulate(all_equity)
    daily = daily_returns(times, equity, capital, step)
    sd = daily.std(ddof=1) if len(daily) > 1 else 0
    trades = result["trades"]
    pnl = np.array([t["pnl"] for t in trades])
    winners = pnl[pnl > 0]
    positive = float(winners.sum())
    return {
        "net_return": float(equity[-1]/capital - 1), "max_drawdown": float(drawdown.max()),
        "sharpe": float(daily.mean()/sd * np.sqrt(365.25)) if sd > 0 else None,
        "trades": len(trades), "win_rate": len(winners)/len(trades) if trades else None,
        "profit_concentration": float(winners.max()/positive) if positive else None,
        "exposure": result["exposure"], "turnover": result["turnover"], "costs": result["costs"],
        "elapsed_days": float((times[-1]-times[0]+step)/DAY),
    }
