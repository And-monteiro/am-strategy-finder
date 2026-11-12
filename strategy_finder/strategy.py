"""The supported rule language and its TradingView compiler."""
from dataclasses import dataclass, asdict

import numpy as np

from .data import STEPS


@dataclass(frozen=True)
class Rule:
    indicator: str
    lookback: int
    entry: float
    exit: float | None
    direction: str

    def __post_init__(self):
        if self.indicator not in {"zscore", "return", "sma_distance", "breakout", "volatility"}:
            raise ValueError("Unsupported indicator.")
        if type(self.lookback) is not int or not 2 <= self.lookback <= 365:
            raise ValueError("Lookback must be an integer from 2 to 365 bars.")
        if self.direction not in {"above", "below"}:
            raise ValueError("Invalid direction.")
        thresholds = (self.entry,) if self.exit is None else (self.entry, self.exit)
        if not all(type(x) in (int, float) and np.isfinite(x) for x in thresholds):
            raise ValueError("Thresholds must be finite numbers.")
        if self.exit is not None:
            ordered = self.entry > self.exit if self.direction == "above" else self.entry < self.exit
            if not ordered:
                raise ValueError("Entry and opposing thresholds must be ordered.")


@dataclass(frozen=True)
class Strategy:
    timeframe: str
    rule: Rule
    max_hold: int = 0
    stop_pct: float = 0.0
    target_pct: float = 0.0
    time_machine: float = 0.0
    filter_rule: Rule | None = None
    exit_policy: str = "signal"
    forecast_bars: int = 0

    def __post_init__(self):
        if self.timeframe not in STEPS:
            raise ValueError("Unsupported timeframe.")
        if self.exit_policy not in ("signal", "bracket"):
            raise ValueError("Exit policy must be signal or bracket.")
        if self.exit_policy == "signal" and self.rule.exit is None:
            raise ValueError("Signal exit policy requires an exit threshold.")
        if type(self.max_hold) is not int or not 0 <= self.max_hold <= 365:
            raise ValueError("Invalid maximum holding period.")
        if not all(type(v) in (int, float) and np.isfinite(v) and 0 <= v < 1
                   for v in (self.stop_pct, self.target_pct, self.time_machine)):
            raise ValueError("Invalid stop, target or Time Machine threshold.")
        if type(self.forecast_bars) is not int or not 0 <= self.forecast_bars <= 48:
            raise ValueError("Invalid Time Machine forecast horizon.")
        if self.time_machine and not (self.stop_pct and self.target_pct and 0 < self.forecast_horizon <= 48):
            raise ValueError("Time Machine requires both brackets and a 1–48 bar horizon.")
        if self.exit_policy == "bracket" and not (self.stop_pct and self.target_pct):
            raise ValueError("Bracket exit policy requires a stop and a take-profit.")

    @classmethod
    def from_dict(cls, raw):
        config = dict(raw)
        config["rule"] = Rule(**config["rule"])
        if config.get("filter_rule") is not None:
            config["filter_rule"] = Rule(**config["filter_rule"])
        return cls(**config)

    def to_dict(self):
        spec = asdict(self)
        if not self.forecast_bars:
            del spec["forecast_bars"]
        if self.exit_policy == "signal":
            # Keep existing signal strategy identities when resuming an investigation.
            del spec["exit_policy"]
        return spec

    @property
    def forecast_horizon(self):
        return self.forecast_bars or self.max_hold


def feature(prices, times, timeframe, rule):
    from numpy.lib.stride_tricks import sliding_window_view

    result = np.full(len(prices), np.nan)
    n = rule.lookback
    if len(prices) < n or (len(prices) == n and rule.indicator not in {"zscore", "sma_distance"}):
        return result
    step = STEPS[timeframe]
    if rule.indicator in {"zscore", "sma_distance"}:
        windows = sliding_window_view(prices, n)
        avg = windows.mean(axis=1)
        close = prices[n-1:]
        if rule.indicator == "zscore":
            sd = windows.std(axis=1)
            result[n-1:] = np.divide(close - avg, sd, out=np.full(len(avg), np.nan), where=sd > 0)
        else:
            result[n-1:] = close / avg - 1
        result[n-1:][times[n-1:] - times[:len(times)-n+1] != (n-1)*step] = np.nan
    else:
        if rule.indicator == "return":
            result[n:] = prices[n:] / prices[:-n] - 1
        elif rule.indicator == "breakout":
            previous_high = sliding_window_view(prices, n)[:-1].max(axis=1)
            result[n:] = prices[n:] / previous_high - 1
        else:
            changes = prices[1:] / prices[:-1] - 1
            result[n:] = sliding_window_view(changes, n).std(axis=1)
        result[n:][times[n:] - times[:-n] != n*step] = np.nan
    return result


def signals(times, bars, strategy, cache):
    series = bars[:, 0, 3] if bars.shape[1] == 1 else bars[:, 0, 3] / bars[:, 1, 3]
    def get(rule):
        key = (strategy.timeframe, rule.indicator, rule.lookback)
        if key not in cache:
            if len(cache) >= 256:
                cache.pop(next(iter(cache)))
            cache[key] = feature(series, times, strategy.timeframe, rule)
        return cache[key]

    rule = strategy.rule
    values = get(rule)
    other = np.zeros(len(values), dtype=bool)
    if rule.direction == "above":
        first = values > rule.entry
        if rule.exit is not None:
            other = values < rule.exit
    else:
        first = values < rule.entry
        if rule.exit is not None:
            other = values > rule.exit
    if strategy.filter_rule:
        gate = strategy.filter_rule
        score = get(gate)
        first &= (score > gate.entry) if gate.direction == "above" else (score < gate.entry)
    return np.where(first, 0, np.where(other, 1 if bars.shape[1] == 2 else -1, -2))


def pine(strategy, assets, capital, allocation, fee_bps, slippage_bps, start, end):
    """Return Pine source and an explicit parity classification."""
    def expr(rule):
        n = rule.lookback
        return {
            "zscore": f"(price - ta.sma(price, {n})) / ta.stdev(price, {n}, true)",
            "return": f"price / price[{n}] - 1",
            "sma_distance": f"price / ta.sma(price, {n}) - 1",
            "breakout": f"price / ta.highest(price[1], {n}) - 1",
            "volatility": f"ta.stdev(price / price[1] - 1, {n}, true)",
        }[rule.indicator]

    tf = {"1h": "60", "4h": "240", "1d": "D"}[strategy.timeframe]
    diagnostic = len(assets) == 2 or strategy.time_machine > 0
    price = "close"
    if len(assets) == 2:
        price = f'request.security("BINANCE:{assets[0]}", timeframe.period, close) / request.security("BINANCE:{assets[1]}", timeframe.period, close)'
    rule = strategy.rule
    entry, exit_ = (">", "<") if rule.direction == "above" else ("<", ">")
    gate = ""
    if strategy.filter_rule:
        extra = strategy.filter_rule
        gate = f" and ({expr(extra)}) {'>' if extra.direction == 'above' else '<'} {extra.entry}"
    allocation_adjusted = allocation / (1 + fee_bps / 10000) * 100
    header = 'indicator("Strategy Finder diagnostics", overlay=false)' if diagnostic else (
        f'strategy("Strategy Finder", initial_capital={capital}, currency=currency.EUR, '
        f'default_qty_type=strategy.percent_of_equity, default_qty_value={allocation_adjusted:.10g}, '
        f'commission_type=strategy.commission.percent, commission_value={fee_bps/100}, '
        'margin_long=100, pyramiding=0, process_orders_on_close=false)')
    lines = ["//@version=6", header, f'// Chart: BINANCE:{assets[0]}, timeframe {tf}.',
             f"// Engine slippage: {slippage_bps} bps per side. Pine slippage uses ticks.",
             f"price = {price}", f"score = {expr(rule)}",
             f"enter = score {entry} {rule.entry}{gate}",
             f"active = time >= {start*1000} and time_close < {end*1000} and barstate.isconfirmed"]
    if rule.exit is not None:
        lines.append(f"opposing = score {exit_} {rule.exit}")
    if diagnostic:
        lines.extend(["// Raw signal only. Python applies the portfolio and simulation rules.",
                      'plot(score, "Signal")',
                      'plotshape(active and enter, title="Asset A / long setup")'])
        if rule.exit is not None:
            title = "Asset B entry setup" if strategy.exit_policy == "bracket" else "Asset B / exit setup"
            lines.append(f'plotshape(active and opposing, title="{title}")')
        return "\n".join(lines) + "\n", "diagnostic"
    lines.extend(['if active and enter and strategy.position_size == 0', '    strategy.entry("Long", strategy.long)'])
    if strategy.exit_policy == "signal":
        lines.extend(['if active and opposing and strategy.position_size > 0', '    strategy.close("Long")'])
    if strategy.max_hold:
        lines.extend([f'if active and strategy.position_size > 0 and bar_index - strategy.opentrades.entry_bar_index(0) + 1 >= {strategy.max_hold}',
                      '    strategy.close("Long", comment="Time exit")'])
    if strategy.stop_pct or strategy.target_pct:
        lines.extend(['if strategy.position_size > 0',
                      f'    strategy.exit("Bracket", "Long", stop={("strategy.position_avg_price * " + str(1-strategy.stop_pct)) if strategy.stop_pct else "na"}, limit={("strategy.position_avg_price * " + str(1+strategy.target_pct)) if strategy.target_pct else "na"})'])
    # Broker emulator bracket/fill timing and percentage slippage may differ.
    status = "approximate_strategy" if slippage_bps or strategy.stop_pct or strategy.target_pct else "strategy_comparable"
    return "\n".join(lines) + "\n", status
