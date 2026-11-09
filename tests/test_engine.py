import tempfile
import unittest
from pathlib import Path

import numpy as np

from strategy_finder.backtest import Execution, metrics, simulate, time_machine
from strategy_finder.data import DAY, HOUR, Candle, align, load_snapshot, save_snapshot
from strategy_finder.research import assess, variants
from strategy_finder.planner import offline
from strategy_finder.strategy import Rule, Strategy, pine, signals


class EvaluationTests(unittest.TestCase):
    def test_next_open_costs_and_stop_precedence(self):
        times = np.arange(3) * HOUR
        bars = np.array([[[80, 105, 80, 100, 1]],
                         [[110, 140, 80, 111, 1]],
                         [[120, 130, 115, 125, 1]]], dtype=float)
        strategy = Strategy("1h", Rule("return", 2, .1, 0, "above"), stop_pct=.1, target_pct=.1)
        execution = Execution(allocation=.1, risk=.01, fee_bps=10, slippage_bps=0)
        result = simulate(times, bars, [0, -2, -2], strategy, execution, 0, 3)
        trade, = result["trades"]
        shares = 100 / (110 * 1.001)
        expected = 900 + shares * 99 * .999
        self.assertEqual(trade["entry"], 110)
        self.assertEqual(trade["reason"], "stop")
        self.assertAlmostEqual(result["equity"][-1], expected)
        self.assertAlmostEqual(metrics(times, result, 1000, HOUR)["max_drawdown"], 1 - expected/1000)

    def test_rotation_executes_two_real_fills_and_cash_is_not_reused_before_sale(self):
        times = np.arange(4) * HOUR
        bars = np.ones((4, 2, 5))
        for i, prices in enumerate(((100, 50), (100, 50), (120, 60), (150, 90))):
            bars[i, :, :4] = np.asarray(prices)[:, None]
        strategy = Strategy("1h", Rule("return", 2, .1, 0, "above"))
        result = simulate(times, bars, [0, 1, -2, -2], strategy,
                          Execution(fee_bps=0, slippage_bps=0), 0, 4)
        self.assertEqual(result["trades"][0]["leg"], 0)
        self.assertEqual(result["open_position"]["leg"], 1)
        self.assertEqual(result["equity"], [1000, 1000, 1200, 1800])

    def test_time_machine_and_signals_do_not_change_when_future_changes(self):
        times = np.arange(140) * HOUR
        closes = 100 + 10*np.sin(np.arange(140)/8)
        bars = np.repeat(closes[:, None, None], 5, axis=2)
        bars[:, 0, 1] *= 1.01
        bars[:, 0, 2] *= .99
        altered = bars.copy()
        altered[100:] *= 10
        strategy = Strategy("1h", Rule("zscore", 10, -1, 1, "below"), 8, .02, .03, .2)
        np.testing.assert_array_equal(signals(times, bars, strategy, {})[:100],
                                      signals(times, altered, strategy, {})[:100])
        self.assertEqual(time_machine(bars, times, 99, 0, strategy, Execution(), 19),
                         time_machine(altered, times, 99, 0, strategy, Execution(), 19))

    def test_complete_alignment_and_snapshot_round_trip(self):
        complete = [Candle(i*HOUR, 1, 2, .5, 1.2, 2) for i in range(8)]
        times, bars = align([complete, complete[:5]+complete[6:]], "4h")
        self.assertEqual(times.tolist(), [0])
        self.assertEqual(bars[0, 1].tolist(), [1, 2, .5, 1.2, 8])
        with tempfile.TemporaryDirectory() as location:
            path = Path(location) / "partition.parquet"
            save_snapshot(path, times, bars)
            t, b = load_snapshot(path)
            np.testing.assert_array_equal(t, times)
            np.testing.assert_array_equal(b, bars)

    def test_funnel_and_qualified_example(self):
        times = np.arange(365*4)*DAY
        i = np.arange(len(times))
        values = 100 + 12*np.sin(i*2*np.pi/24)
        bars = np.repeat(values[:, None, None], 5, axis=2)
        # Enter at troughs and exit at peaks, independent of the feature tests.
        rule = np.where(i % 24 == 18, 0, np.where(i % 24 == 6, -1, -2))
        spec = Strategy("1d", Rule("return", 2, .01, -.01, "above"))
        report, _ = assess(times, bars, rule, spec, Execution(), .3, 19)
        self.assertEqual(report["outcome"], "qualified")
        self.assertTrue(all(report["funnel"].values()))
        self.assertGreater(report["net_return_interval_80"][0], 0)

    def test_variant_control_and_pine_modes(self):
        choices = variants(offline(0, ["4h"]), 1000, 19, ["4h"])
        for item in choices:
            if item["time_machine"]:
                self.assertIn(dict(item, time_machine=0), choices)
        strategy = Strategy.from_dict(choices[0])
        code, status = pine(strategy, ["BTCEUR"], 1000, 1, 10, 5, 0, 10000)
        self.assertIn("strategy(", code)
        self.assertEqual(status, "approximate_strategy")
        code, status = pine(strategy, ["BTCEUR", "ETHEUR"], 1000, 1, 10, 5, 0, 10000)
        self.assertIn("request.security", code)
        self.assertEqual(status, "diagnostic")


if __name__ == "__main__":
    unittest.main()
