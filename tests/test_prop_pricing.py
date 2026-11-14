"""Synthetic tests for tennis prop projections, parsers, hold, and power. No live backtest."""
from __future__ import annotations

import math
import unittest

from Backtest.Markov import MatchProb, SetFrom
from Backtest.PropMarkets import Hold, ParsePropMarket, ValidBook
from Backtest.PropProjection import (
    AtLeastCount, GamePointsFrom, MatchProjection, PoissonTail, ProbExactSets,
    ProbGameHandicap, ProbTotalOver, SetProjection,
)
from Backtest.PropScore import ParseScore, OrientScore
from Backtest.RunPropPricing import MinDetectableEdge, PowerTable, QUAL


class ScoreTests(unittest.TestCase):
    def test_straight_sets_and_totals(self):
        s = ParseScore("6-4 6-3")
        self.assertEqual(s["sets"], ((6, 4), (6, 3)))
        self.assertEqual(s["sets_a"], 2)
        self.assertEqual(s["sets_b"], 0)
        self.assertEqual(s["total_games"], 19)
        self.assertEqual(s["game_diff"], 5)
        self.assertTrue(s["sweep"])
        self.assertFalse(s["incomplete"])

    def test_three_set_and_tiebreak(self):
        s = ParseScore("6-4 3-6 7-6(5)")
        self.assertEqual(s["sets_a"], 2)
        self.assertEqual(s["sets_b"], 1)
        self.assertEqual(s["total_games"], 32)
        self.assertEqual(s["n_tiebreaks"], 1)
        flipped = OrientScore(s, winner_is_a=False)
        self.assertEqual(flipped["sets_a"], 1)
        self.assertEqual(flipped["game_diff"], -s["game_diff"])

    def test_retirement_flag(self):
        s = ParseScore("6-4 2-1 RET")
        self.assertTrue(s["incomplete"])
        self.assertEqual(s["sets"][0], (6, 4))


class ProjectionIdentityTests(unittest.TestCase):
    def test_set_win_matches_setfrom(self):
        for pa, pb in ((0.65, 0.62), (0.70, 0.55), (0.80, 0.80)):
            for server in ("a", "b"):
                proj = SetProjection(pa, pb, server, "tb7")
                aa, ab, ba, bb = SetFrom(pa, pb, 0, 0, server, "tb7")
                self.assertAlmostEqual(proj["p_win"], aa + ab, places=9, msg=(pa, pb, server))
                self.assertAlmostEqual(sum(proj["scores"].values()), 1.0, places=9)

    def test_match_win_matches_matchprob(self):
        for pa, pb, bo in ((0.65, 0.62, 3), (0.72, 0.58, 3), (0.67, 0.64, 5)):
            proj = MatchProjection(pa, pb, best_of=bo)
            self.assertAlmostEqual(proj["p_win"], MatchProb(pa, pb, best_of=bo), places=8)
            self.assertAlmostEqual(sum(proj["set_scores"].values()), 1.0, places=8)
            self.assertAlmostEqual(sum(proj["total_games"].values()), 1.0, places=8)

    def test_equal_players_symmetric_and_over_line(self):
        proj = MatchProjection(0.65, 0.65, best_of=3)
        self.assertAlmostEqual(ProbExactSets(proj, 2, 0), ProbExactSets(proj, 0, 2), places=8)
        self.assertAlmostEqual(proj["e_game_diff"], 0.0, places=6)
        # A BO3 always has at least 12 games and at most 39 (7-6 7-6 7-6)
        self.assertAlmostEqual(ProbTotalOver(proj, 11.5), 1.0, places=8)
        self.assertAlmostEqual(ProbTotalOver(proj, 39.5), 0.0, places=8)
        self.assertGreater(ProbTotalOver(proj, 20.5), 0.0)
        self.assertLess(ProbTotalOver(proj, 20.5), 1.0)
        self.assertAlmostEqual(ProbGameHandicap(proj, 0.5, "a") + ProbGameHandicap(proj, 0.5, "b")
                               + sum(p for d, p in proj["game_diff"].items() if d == 0), 1.0, places=8)

    def test_game_points_positive(self):
        self.assertGreater(GamePointsFrom(0.65, 0, 0), 4.0)
        self.assertLess(GamePointsFrom(0.90, 0, 0), GamePointsFrom(0.55, 0, 0))


class MarketAndPowerTests(unittest.TestCase):
    def test_hold_is_the_spread(self):
        h = Hold(0.48, 0.52)
        self.assertAlmostEqual(h["spread"], 0.04)
        self.assertAlmostEqual(h["ask_overround"], 0.04)
        self.assertFalse(ValidBook(0.0, 0.5))

    def test_parse_total_and_spread_and_exact_and_aces(self):
        tot = ParsePropMarket({"ticker": "KXATPGTOTAL-26NOV14FRITIR-30", "series": "KXATPGTOTAL",
                               "title": "Over 29.5 games", "yes_sub_title": "Over 29.5 games"})
        self.assertEqual(tot["kind"], "total_games")
        self.assertEqual(tot["line"], 29.5)
        spr = ParsePropMarket({"ticker": "KXATPGSPREAD-26NOV14FRITIR-FRI7", "series": "KXATPGSPREAD",
                               "title": "Will Taylor Fritz win at least 6.5 more games than Thiago Agustin Tirante?",
                               "yes_sub_title": "Taylor Fritz -6.5 games"})
        self.assertEqual(spr["kind"], "game_handicap")
        self.assertEqual(spr["line"], -6.5)
        self.assertIn("Fritz", spr["player"])
        ex = ParsePropMarket({"ticker": "KXATPEXACTMATCH-26NOV16MEDCOB-MED21", "series": "KXATPEXACTMATCH",
                              "title": "Will Daniil Medvedev win the match by a set score of 2-1?",
                              "yes_sub_title": "Daniil Medvedev wins 2-1"})
        self.assertEqual(ex["kind"], "exact_score")
        self.assertEqual((ex["sets_a"], ex["sets_b"]), (2, 1))
        ace = ParsePropMarket({"ticker": "KXATPACES-26JUL01FRTKYP-FRI19", "series": "KXATPACES",
                               "title": "Will Taylor Fritz record at least 19 total match aces in the Taylor Fritz vs Patrick Kypson match?",
                               "yes_sub_title": "Taylor Fritz: 19+"})
        self.assertEqual(ace["kind"], "aces")
        self.assertEqual(ace["line"], 19.0)

    def test_poisson_and_power(self):
        self.assertAlmostEqual(PoissonTail(0, 2.0), 1.0)
        lam, p = AtLeastCount(100.0, 0.08, 0)
        self.assertAlmostEqual(lam, 8.0)
        self.assertAlmostEqual(p, 1.0)
        self.assertLess(MinDetectableEdge(1600, 0.5), MinDetectableEdge(62, 0.5))
        table = PowerTable([62, 1600])
        self.assertFalse(bool(table.loc[table["n"] == 62, "powered_for_3c_sigma0.50"].iloc[0]))
        self.assertEqual(QUAL, 0.03)


class RefuseRunTests(unittest.TestCase):
    def test_main_without_run_does_not_backtest(self):
        from Backtest.RunPropPricing import Main
        code = Main([])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()