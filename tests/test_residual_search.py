"""Tests for residual-on-market helpers, SPA, and a tiny in-play Monte Carlo check."""
from __future__ import annotations

import unittest

import numpy as np

from Backtest.InPlayMonteCarlo import RunCell, Pack, Unpack
from Backtest.ResidualRatings import ResidualParams, Walk, BuildGrid, ResidualElo
from Backtest.ResidualSearch import DevigAsk, Residualize, FitW, WalkForwardW, ClvSeries, BitsIgn
from Backtest.SpaTest import HansenSpa, WhiteRealityCheck, PermuteProbGap


class TestGrid(unittest.TestCase):
    def test_size(self):
        grid = BuildGrid()
        self.assertEqual(len(grid), 3888)
        keys = {p.Key() for p in grid}
        self.assertEqual(len(keys), 3888)


class TestResidualMath(unittest.TestCase):
    def test_w_zero_equals_market(self):
        p_m = np.array([0.4, 0.6, 0.75])
        p_mod = np.array([0.2, 0.9, 0.1])
        out = Residualize(p_mod, p_m, 0.0)
        np.testing.assert_allclose(out, p_m, atol=1e-9)

    def test_w_one_equals_model(self):
        p_m = np.array([0.4, 0.6, 0.75])
        p_mod = np.array([0.2, 0.9, 0.1])
        out = Residualize(p_mod, p_m, 1.0)
        np.testing.assert_allclose(out, p_mod, atol=1e-9)

    def test_devig_ask_symmetric(self):
        # 0.48 bid / 0.52 ask, no side is 0.52 / 0.48; proportional mid is 0.5
        self.assertAlmostEqual(float(DevigAsk(0.48, 0.52)), 0.5, places=6)

    def test_fit_w_recovers_one(self):
        rng = np.random.default_rng(0)
        p_m = rng.uniform(0.2, 0.8, 2000)
        # outcomes from the model, which equals the market here, so w is weakly identified;
        # instead let p_model be better and y ~ p_model
        p_mod = np.clip(p_m + 0.15, 0.05, 0.95)
        y = (rng.random(2000) < p_mod).astype(float)
        w = FitW(p_mod, p_m, y)
        self.assertGreater(w, 0.5)

    def test_walk_forward_uses_prior_months_only(self):
        p_m = np.array([0.5, 0.5, 0.5, 0.5])
        p_mod = np.array([0.9, 0.9, 0.1, 0.1])
        y = np.array([1.0, 1.0, 0.0, 0.0])
        months = np.array(["2025-06", "2025-07", "2025-08", "2025-09"])
        p, w = WalkForwardW(p_mod, p_m, y, months, min_train=3)
        # first three months have < 3 prior rows, so default w=0.5
        np.testing.assert_array_equal(w[:3], [0.5, 0.5, 0.5])
        self.assertGreater(w[3], 0.5)

    def test_clv_sign(self):
        # model likes A more than early 0.40; close goes to 0.55 -> positive CLV
        clv = ClvSeries(np.array([0.6]), np.array([0.40]), np.array([0.55]))
        self.assertAlmostEqual(float(clv[0]), 0.15)
        clv_b = ClvSeries(np.array([0.3]), np.array([0.40]), np.array([0.55]))
        self.assertAlmostEqual(float(clv_b[0]), -0.15)


class TestLeak(unittest.TestCase):
    def test_same_group_does_not_see_other_result(self):
        # two matches, same group, four distinct players
        group = np.array([0, 0])
        winner = np.array([1, 3])
        loser = np.array([2, 4])
        surface = np.array(["Hard", "Hard"])
        best_of = np.array([3, 3])
        level = np.array(["tour", "tour"])
        days = np.array([20000, 20000])
        params = ResidualParams()
        p = Walk(group, winner, loser, surface, best_of, level, days, params)
        # unrated vs unrated, no fatigue history: both should be 0.5
        np.testing.assert_allclose(p, [0.5, 0.5], atol=1e-9)
        # a later group featuring player 1 (who won) vs a new player should exceed 0.5
        group2 = np.array([0, 0, 1])
        winner2 = np.array([1, 3, 1])
        loser2 = np.array([2, 4, 5])
        surface2 = np.array(["Hard", "Hard", "Hard"])
        best_of2 = np.array([3, 3, 3])
        level2 = np.array(["tour", "tour", "tour"])
        days2 = np.array([20000, 20000, 20001])
        p2 = Walk(group2, winner2, loser2, surface2, best_of2, level2, days2, params)
        self.assertGreater(p2[2], 0.5)


class TestSpa(unittest.TestCase):
    def test_noise_not_rejected(self):
        rng = np.random.default_rng(1)
        n, k = 400, 8
        days = np.repeat(np.arange(40), 10)
        d = rng.normal(0.0, 0.2, size=(n, k))
        spa = HansenSpa(d, days, reps=500, seed=1)
        rc = WhiteRealityCheck(d, days, reps=500, seed=1)
        self.assertGreater(spa["p_value"], 0.05)
        self.assertGreater(rc["p_value"], 0.05)

    def test_strong_signal_rejected(self):
        rng = np.random.default_rng(2)
        n, k = 400, 8
        days = np.repeat(np.arange(40), 10)
        d = rng.normal(0.0, 0.05, size=(n, k))
        d[:, 0] += 0.20
        spa = HansenSpa(d, days, reps=500, seed=2)
        self.assertLess(spa["p_value"], 0.05)
        self.assertEqual(spa["best_index"], 0)

    def test_permutation_null_on_noise(self):
        rng = np.random.default_rng(3)
        p = rng.uniform(0.3, 0.7, 300)
        y = (rng.random(300) < 0.5).astype(float)
        days = np.repeat(np.arange(30), 10)
        perm = PermuteProbGap(p, p, y, days, reps=400, seed=3)
        # identical forecasts: gap is exactly 0 under every permutation
        self.assertEqual(perm["t_obs"], 0.0)
        self.assertEqual(perm["p_value"], 1.0)


class TestInPlayMc(unittest.TestCase):
    def test_pack_roundtrip(self):
        k = Pack([1], [0], [6], [4], [1])
        sa, sb, ga, gb, sv = Unpack(k)
        self.assertEqual(tuple(map(int, [sa[0], sb[0], ga[0], gb[0], sv[0]])), (1, 0, 6, 4, 1))

    def test_mc_near_exact_even(self):
        pre, _ = RunCell(0.60, 0.60, 3, 20_000, 7, with_exact_inplay=False)
        self.assertEqual(pre["unfinished"], 0)
        self.assertAlmostEqual(pre["mc_p_win_a"], 0.5, delta=0.02)
        self.assertAlmostEqual(pre["exact_p_win_a"], 0.5, delta=1e-9)

    def test_mc_near_exact_favourite(self):
        pre, inp = RunCell(0.70, 0.58, 3, 20_000, 11, with_exact_inplay=True)
        self.assertEqual(pre["unfinished"], 0)
        self.assertLess(abs(pre["mc_minus_exact"]), 0.02)
        self.assertTrue(len(inp) > 0)


if __name__ == "__main__":
    unittest.main()
