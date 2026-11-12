"""Precompute the common intraday Time Machine forecast grid once per snapshot."""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from hashlib import sha256

import numpy as np

from .backtest import Execution, forecast_paths
from .data import DAY, STEPS
from .research import encode, load_data

STOPS = tuple(x/1000 for x in range(5, 41, 2))
TARGETS = tuple(x/1000 for x in range(8, 81, 4))
WORKER = {}


def summarize_grid(price_paths, execution, stops=STOPS, targets=TARGETS):
    horizon = price_paths.shape[1]-1
    slip = execution.slippage_bps / 10000
    fee = execution.fee_bps / 10000
    entry = price_paths[:, 0, 0]*(1+slip)
    floors = entry[:, None]*(1-np.asarray(stops))
    ceilings = entry[:, None]*(1+np.asarray(targets))
    lows = price_paths[:, :horizon, 2, None] <= floors[:, None, :]
    highs = price_paths[:, :horizon, 1, None] >= ceilings[:, None, :]
    first_stop = np.where(lows.any(axis=1), lows.argmax(axis=1), horizon)
    first_target = np.where(highs.any(axis=1), highs.argmax(axis=1), horizon)
    stopped = (first_stop[:, :, None] < horizon) & (first_stop[:, :, None] <= first_target[:, None, :])
    won = (first_target[:, None, :] < horizon) & ~stopped
    openings = np.take_along_axis(price_paths[:, :, 0], first_stop, axis=1)
    exits = np.where(stopped, np.minimum(floors, openings)[:, :, None],
                     np.where(won, ceilings[:, None, :], price_paths[:, horizon, 0, None, None]))
    payoff = exits*(1-slip)*(1-fee)/(entry[:, None, None]*(1+fee))-1
    return np.stack((won.mean(axis=0), stopped.mean(axis=0), (~(won | stopped)).mean(axis=0),
                     np.ascontiguousarray(payoff.transpose(1, 2, 0)).mean(axis=-1)), axis=-1)


def grid_path(setup, config, seed):
    frame, = config["datasets"]
    inputs = (config["datasets"][frame]["development_hash"], config["execution"],
              config.get("evidence", {}).get("cost_multiplier", 1.5), DAY//STEPS[frame], STOPS, TARGETS)
    digest = sha256(encode(inputs).encode()).hexdigest()[:16]
    return setup/f"hourly-forecasts-{digest}-{seed}.npy"


def init_builder(setup, config, path, seed):
    frame, = config["datasets"]
    times, bars, _ = load_data(setup, config)[frame]
    execution = Execution(**config["execution"])
    multiplier = config.get("evidence", {}).get("cost_multiplier", 1.5)
    stressed = replace(execution, fee_bps=min(9999, execution.fee_bps*multiplier),
                       slippage_bps=min(9999, execution.slippage_bps*multiplier))
    WORKER.update(times=times, bars=bars, output=np.load(path, mmap_mode="r+"),
                  execution=(execution, stressed), seed=seed, step=STEPS[frame])


def fill_block(bounds):
    a, b = bounds
    times, bars = WORKER["times"], WORKER["bars"]
    for i in range(a, b):
        step = WORKER["step"]
        sampled = forecast_paths(bars, times, i, 0, DAY//step, step, True, WORKER["seed"])
        if sampled is None:
            WORKER["output"][i] = np.nan
            continue
        for j, execution in enumerate(WORKER["execution"]):
            WORKER["output"][i, j] = summarize_grid(sampled[0], execution)
    return b-a


def prepare_grid(setup, config, seed, workers):
    path = grid_path(setup, config, seed)
    if path.exists():
        return
    frame, = config["datasets"]
    count = config["datasets"][frame]["development_bars"]
    temporary = path.with_suffix(".tmp.npy")
    mapped = np.lib.format.open_memmap(temporary, mode="w+", dtype="float64",
                                       shape=(count, 2, len(STOPS), len(TARGETS), 4))
    del mapped
    blocks = [(a, min(a+128, count)) for a in range(0, count, 128)]
    with ProcessPoolExecutor(max_workers=workers, initializer=init_builder,
                             initargs=(setup, config, temporary, seed)) as pool:
        for _ in pool.map(fill_block, blocks):
            pass
    temporary.replace(path)


def load_lookup(setup, config, seed, stressed=False):
    path = grid_path(setup, config, seed)
    if not path.is_file():
        return None
    data = np.load(path, mmap_mode="r")
    stop_ids = {v: i for i, v in enumerate(STOPS)}
    target_ids = {v: i for i, v in enumerate(TARGETS)}

    def lookup(i, strategy):
        row = data[i, int(stressed), stop_ids[strategy.stop_pct], target_ids[strategy.target_pct]]
        return tuple(row.tolist()) if np.isfinite(row[0]) else None

    return lookup
