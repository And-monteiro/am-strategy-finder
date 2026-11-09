import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from strategy_finder.backtest import Execution, metrics, simulate, time_machine
from strategy_finder.data import DAY, HOUR, Candle, align, load_snapshot, save_snapshot
from strategy_finder.feedback import rich_context, validate_action
from strategy_finder.research import assess, variants
from strategy_finder.planner import offline
from strategy_finder.strategy import Rule, Strategy, pine, signals


class EvaluationTests(unittest.TestCase):
    def test_rich_feedback_keeps_older_winner_and_excludes_holdout(self):
        good = Strategy("1d", Rule("return", 3, .1, 0, "above")).to_dict()
        bad = Strategy("1d", Rule("zscore", 5, .5, 0, "above")).to_dict()
        def candidate(spec, n, ret, outcome):
            return {"strategy": spec, "experiment": n, "final": {"secret": True},
                    "development": {"net_return": ret, "outcome": outcome, "trades": 10,
                                    "max_drawdown": .1, "sharpe": ret, "periods": [],
                                    "funnel": {"profitable_after_costs": ret > 0}}}
        latest = [{"experiment": n, "funnel": {}, "top_parameters": [],
                   "proposal": {"hypothesis": "", "timeframe": "1d", "indicator": "return"}}
                  for n in range(2, 15)]
        context = rich_context(latest, [candidate(good, 1, .5, "qualified"),
                                        candidate(bad, 2, -.5, "rejected")])
        self.assertEqual(len(context["last_experiments"]), 10)
        self.assertEqual(context["strong_candidates"][0]["experiment"], 1)
        self.assertEqual(context["weak_candidates"][0]["experiment"], 2)
        self.assertNotIn("secret", str(context))
        validate_action({"action": "refinement", "parent": 1}, context, None)
        with self.assertRaises(ValueError):
            validate_action({"action": "refinement", "parent": 1}, context, "new_hypothesis")

    def test_bracket_ignores_opposing_signal_until_take_profit(self):
        times = np.arange(5)*HOUR
        bars = np.array([[[100, 101, 99, 100, 1]],
                         [[100, 104, 99, 103, 1]],
                         [[104, 105, 102, 103, 1]],
                         [[104, 121, 103, 118, 1]],
                         [[119, 120, 118, 119, 1]]], dtype=float)
        execution = Execution(fee_bps=0, slippage_bps=0)
        signal = [0, -1, -2, -2, -2]
        original = Strategy("1h", Rule("return", 2, .1, 0, "above"),
                            stop_pct=.1, target_pct=.2)
        for policy, timestamp, reason, price in (("signal", 2, "signal", 104),
                                                 ("bracket", 3, "target", 120)):
            spec = replace(original, exit_policy=policy)
            trade, = simulate(times, bars, signal, spec, execution, 0, 5)["trades"]
            self.assertEqual((trade["exit_timestamp"], trade["reason"]), (times[timestamp], reason))
            self.assertAlmostEqual(trade["exit"], price)

    def test_bracket_rotation_uses_second_signal_only_when_flat(self):
        times = np.arange(5)*HOUR
        bars = np.full((5, 2, 5), 100.)
        bars[3, 0, 2] = 89.
        strategy = Strategy("1h", Rule("return", 2, .1, 0, "above"),
                            stop_pct=.1, target_pct=.2, exit_policy="bracket")
        result = simulate(times, bars, [0, 1, 1, 1, -2], strategy, Execution(), 0, 5)
        trade, = result["trades"]
        self.assertEqual((trade["leg"], trade["reason"], trade["exit_timestamp"]), (0, "stop", times[3]))
        self.assertEqual(result["open_position"]["leg"], 1)
        self.assertEqual(result["open_position"]["entry_timestamp"], times[4])

    def test_proposed_bracket_has_no_signal_exit_and_keeps_time_machine_control(self):
        proposal = dict(offline(0, ["4h"]), exit_policies=["bracket"], exits=[],
                        lookbacks=[8], entries=[-1], stops=[.05], targets=[.1],
                        holding_periods=[12], time_machine_thresholds=[.3])
        choices = variants(proposal, 4, 19, ["4h"])
        self.assertEqual(len(choices), 2)
        for item in choices:
            self.assertEqual(item["exit_policy"], "bracket")
            self.assertIsNone(item["rule"]["exit"])
        ungated = Strategy.from_dict(next(item for item in choices if not item["time_machine"]))
        code, _ = pine(ungated, ["BTCEUR"], 1000, 1, 10, 5, 0, 10000)
        self.assertNotIn("opposing", code)
        self.assertIn('strategy.exit("Bracket"', code)
        self.assertIn('comment="Time exit"', code)

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
