"""Unit tests for the tennis match simulator and the power helpers (Sim/). No network, no data/ needed."""
import unittest

import numpy as np
import pandas as pd

from Backtest.Markov import GameFrom, MatchProb
from Sim.MatchSim import ExactMatchProb, ExactMatchProbForm, HoldProb, ParseScore, SimulateMatches
from Sim.Power import DesignEffect, KalshiFee, SimPower


class TestMatchSim(unittest.TestCase):
    def test_hold_prob_matches_markov(self):
        for p in (0.45, 0.6, 0.72):
            self.assertAlmostEqual(float(HoldProb(p)), GameFrom(p, 0, 0), places=10)

    def test_exact_matches_markov(self):
        cases = [(0.65, 0.62, 3, 7), (0.70, 0.60, 5, 10), (0.55, 0.66, 5, 7), (0.72, 0.55, 3, 7)]
        got = ExactMatchProb(np.array([c[0] for c in cases]), np.array([c[1] for c in cases]),
                             np.array([c[2] for c in cases]), np.array([c[3] for c in cases]))
        for g, (pa, pb, bo, ft) in zip(got, cases):
            self.assertAlmostEqual(g, MatchProb(pa, pb, bo, "tb10" if ft == 10 else "tb7"), places=9)

    def test_monte_carlo_agrees_with_exact(self):
        r = SimulateMatches(0.64, 0.61, 3, 7, np.random.default_rng(11), n=200_000)
        exact = ExactMatchProb(np.array([0.64]), np.array([0.61]), 3, 7)[0]
        self.assertAlmostEqual(r["a_won"].mean(), exact, delta=0.006)

    def test_scores_valid(self):
        r = SimulateMatches(0.6, 0.6, 5, 10, np.random.default_rng(1), n=5000)
        self.assertTrue(np.all(np.maximum(r["sets_a"], r["sets_b"]) == 3))
        self.assertTrue(np.all(r["total_games"] >= 18))
        self.assertTrue(np.all(r["straight"] == (np.minimum(r["sets_a"], r["sets_b"]) == 0)))

    def test_form_shock(self):
        pa, pb = np.array([0.68]), np.array([0.60])
        self.assertAlmostEqual(ExactMatchProbForm(pa, pb, 3, 7, 0.0)[0], ExactMatchProb(pa, pb, 3, 7)[0])
        flat = ExactMatchProbForm(pa, pb, 3, 7, 0.05)[0]
        self.assertLess(abs(flat - 0.5), abs(ExactMatchProb(pa, pb, 3, 7)[0] - 0.5))
        r = SimulateMatches(0.68, 0.60, 3, 7, np.random.default_rng(2), n=200_000, sigma=0.05)
        self.assertAlmostEqual(r["a_won"].mean(), flat, delta=0.006)

    def test_parse_score(self):
        self.assertEqual(ParseScore("6-7(5) 6-3 6-3"), [(6, 7, 1), (6, 3, 0), (6, 3, 0)])
        self.assertIsNone(ParseScore("6-4 2-1 RET"))
        self.assertIsNone(ParseScore("W/O"))


class TestPower(unittest.TestCase):
    def test_fee(self):
        self.assertAlmostEqual(float(KalshiFee(0.5)), 0.02)
        self.assertAlmostEqual(float(KalshiFee(0.95)), 0.01)

    def test_design_effect(self):
        pnl = np.r_[np.ones(50), -np.ones(50)]
        days = np.r_[np.zeros(50), np.ones(50)]
        self.assertGreater(DesignEffect(pnl, days), 10)
        self.assertAlmostEqual(DesignEffect(np.r_[1.0, -1.0] * 0 + np.array([1.0, -1.0]), np.array([0, 1])), 1.0)

    def test_sim_power_big_edge(self):
        ask = np.full(400, 0.5)
        days = np.repeat(np.arange(40), 10)
        self.assertGreater(SimPower(ask, days, 0.20, 400, np.random.default_rng(0), reps=100), 0.95)
        self.assertLess(SimPower(ask, days, 0.0, 400, np.random.default_rng(0), reps=200), 0.10)


if __name__ == "__main__":
    unittest.main()
