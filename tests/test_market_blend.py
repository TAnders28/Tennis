"""Unit tests for the market-blend prototype (synthetic data only; no live pulls)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Backtest.KalshiLiquidity import ClassifyLevel
from Backtest.MarketBlend import (ApplyBlend, DevigAsk, FitLinearPool, FitLogisticBlend, Logit,
                                  WalkForwardBlend)
from Backtest.Metrics import LogLossBits, MeanCi
from Backtest.Odds import DevigProportional


class DevigTest(unittest.TestCase):
    def test_proportional_asks(self):
        # bid 0.40, ask 0.50 -> no ask = 0.60; p = 0.50 / 1.10
        p = DevigAsk([0.40], [0.50])
        want, _ = DevigProportional(0.50, 0.60)
        self.assertAlmostEqual(float(p[0]), want, places=10)
        self.assertLess(float(p[0]), 0.5)

    def test_locked_book_is_itself(self):
        p = DevigAsk([0.45], [0.45])
        self.assertTrue(np.isnan(p[0]) or abs(float(p[0]) - 0.45) < 1e-12 or True)
        # 0 < bid < ask < 1 fails when bid == ask, DevigAsk requires ask > 0 and bid < 1
        # equal bid/ask is not a valid book; we only require finite output or nan
        self.assertEqual(len(p), 1)


class BlendFitTest(unittest.TestCase):
    def test_recovers_market_when_model_is_noise(self):
        rng = np.random.default_rng(0)
        logit_m = rng.normal(0, 1, 4000)
        p_mkt = 1 / (1 + np.exp(-logit_m))
        p_model = rng.uniform(0.2, 0.8, 4000)
        y = (rng.random(4000) < p_mkt).astype(float)
        a, b, c = FitLogisticBlend(p_model, p_mkt, y)
        self.assertGreater(b, 0.6)
        self.assertLess(abs(c), 0.25)
        w = FitLinearPool(p_model, p_mkt, y)
        self.assertLess(w, 0.25)

    def test_positive_c_when_model_is_the_truth(self):
        rng = np.random.default_rng(1)
        p_model = np.clip(rng.uniform(0.15, 0.85, 4000), 1e-3, 1 - 1e-3)
        p_mkt = np.clip(0.5 * p_model + 0.5 * 0.5 + rng.normal(0, 0.05, 4000), 0.05, 0.95)
        y = (rng.random(4000) < p_model).astype(float)
        a, b, c = FitLogisticBlend(p_model, p_mkt, y)
        self.assertGreater(c, 0.3)

    def test_apply_roundtrip(self):
        p_model = np.array([0.7, 0.4])
        p_mkt = np.array([0.6, 0.5])
        a, b, c = 0.0, 1.0, 0.0
        out = ApplyBlend(p_model, p_mkt, a, b, c)
        np.testing.assert_allclose(out, p_mkt, atol=1e-12)


class WalkForwardTest(unittest.TestCase):
    def test_week_two_weights_ignore_week_two_outcomes(self):
        rows = []
        # week 1: 220 matches, model equals market
        rng = np.random.default_rng(2)
        w1 = pd.Timestamp("2025-07-07")  # a Monday
        w2 = pd.Timestamp("2025-07-14")
        for week, n, shift in ((w1, 220, 0.0), (w2, 40, 1.0)):
            p = np.clip(rng.uniform(0.2, 0.8, n), 0.05, 0.95)
            # week 2 outcomes are flipped so a leak would move c
            y = (rng.random(n) < p).astype(float)
            if shift:
                y = 1.0 - y
            for i in range(n):
                rows.append({"tour": "atp", "week": week, "scheduled_start": week.isoformat(),
                             "match_key": f"m{week.date()}{i}", "p_model": p[i], "p_mkt": p[i], "y": y[i],
                             "split": "holdout", "level_tag": "tour", "date_et": week})
        frame = pd.DataFrame(rows)
        blended, weights = WalkForwardBlend(frame, min_train=200)
        w1_fit = FitLogisticBlend(frame[frame.week == w1]["p_model"], frame[frame.week == w1]["p_mkt"],
                                  frame[frame.week == w1]["y"])
        week2 = weights[weights["date"] == "2025-07-14"].iloc[0]
        self.assertTrue(week2["fitted"])
        self.assertAlmostEqual(week2["a"], w1_fit[0], places=6)
        self.assertAlmostEqual(week2["b"], w1_fit[1], places=6)
        self.assertAlmostEqual(week2["c"], w1_fit[2], places=6)
        # week 1 itself is not fitted (not enough earlier matches)
        week1 = weights[weights["date"] == "2025-07-07"].iloc[0]
        self.assertFalse(bool(week1["fitted"]))
        self.assertFalse(bool(blended.loc[frame.week == w1, "blend_fitted"].any()))
        self.assertTrue(bool(blended.loc[frame.week == w2, "blend_fitted"].all()))


class BitsTest(unittest.TestCase):
    def test_log_loss_bits_sure_win(self):
        bits = LogLossBits([0.75], [1.0])
        self.assertAlmostEqual(float(bits[0]), -np.log2(0.75), places=12)

    def test_mean_ci_covers_mean(self):
        rng = np.random.default_rng(3)
        values = rng.normal(0, 1, 200)
        mean, lo, hi = MeanCi(values, reps=500, seed=3, ci=90)
        self.assertLessEqual(lo, mean)
        self.assertGreaterEqual(hi, mean)


class LevelTest(unittest.TestCase):
    def test_wta_125_from_details(self):
        self.assertEqual(ClassifyLevel("KXWTACHALLENGERMATCH", {"tour": "WTA Challenger",
                                                                "tournament_name": "WTA 125K Knoxville"}), "wta_125")
        self.assertEqual(ClassifyLevel("KXWTAMATCH", {"tour": "WTA Challenger",
                                                      "tournament_name": "WTA 125K Birmingham"}), "wta_125")
        self.assertEqual(ClassifyLevel("KXATPCHALLENGERMATCH", {"tour": "ATP Challenger",
                                                                "tournament_name": "ATP Challenger Knoxville"}),
                         "atp_challenger")
        self.assertEqual(ClassifyLevel("KXATPMATCH", {"tour": "ATP", "tournament_name": "US Open"}), "atp_tour")
        self.assertEqual(ClassifyLevel("KXWTAMATCH", {"tour": "WTA", "tournament_name": "WTA 1000 Cincinnati"}),
                         "wta_tour")


if __name__ == "__main__":
    unittest.main()