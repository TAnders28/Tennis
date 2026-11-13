import unittest

from Backtest.Markov import GameFrom, LiveWinProb, MatchFrom, MatchProb, MatchProbTable, SlamFinalSetRule, TiebreakFrom


class MarkovTest(unittest.TestCase):
    def test_game_known_value(self):
        self.assertAlmostEqual(GameFrom(0.6, 0, 0), 0.735729, places=5)
        self.assertAlmostEqual(GameFrom(0.5, 0, 0), 0.5, places=12)

    def test_symmetry(self):
        for best_of in (3, 5):
            for rule in ("tb7", "adv", "tb10", "tb7_12"):
                self.assertAlmostEqual(MatchProb(0.62, 0.62, best_of, rule), 0.5, places=9)
                self.assertAlmostEqual(MatchProb(0.66, 0.6, best_of, rule) + MatchProb(0.6, 0.66, best_of, rule), 1.0, places=9)

    def test_serve_first_invariance(self):
        # Newton & Keller: with iid points, who serves first does not change match win probability
        self.assertAlmostEqual(MatchFrom(0.7, 0.55, 0, 0, "a", 5, "tb10"), MatchFrom(0.7, 0.55, 0, 0, "b", 5, "tb10"), places=9)

    def test_bo5_amplifies_favourite(self):
        self.assertGreater(MatchProb(0.66, 0.6, 5), MatchProb(0.66, 0.6, 3))

    def test_live_start_equals_prematch(self):
        self.assertAlmostEqual(LiveWinProb(0.66, 0.6, 0, 0, 0, 0, 0, 0, "a", 3), MatchProb(0.66, 0.6, 3), places=9)

    def test_live_terminal_like_states(self):
        self.assertGreater(LiveWinProb(0.62, 0.62, 1, 0, 5, 0, 3, 0, "a", 3), 0.999)
        self.assertLess(LiveWinProb(0.62, 0.62, 0, 1, 0, 5, 0, 3, "b", 3), 0.001)

    def test_tiebreak_symmetry(self):
        self.assertAlmostEqual(TiebreakFrom(0.6, 0.6, 0, 0, "a", 7), 0.5, places=9)
        self.assertAlmostEqual(TiebreakFrom(0.6, 0.6, 0, 0, "a", 10), 0.5, places=9)

    def test_table_matches_exact(self):
        table = MatchProbTable(lo=0.55, hi=0.70, step=0.005)  # same step as production
        self.assertAlmostEqual(table.Prob(0.65, 0.60, 3), MatchProb(0.65, 0.60, 3), places=6)
        self.assertAlmostEqual(table.Prob(0.6525, 0.6025, 3), MatchProb(0.6525, 0.6025, 3), places=3)

    def test_final_set_rules(self):
        self.assertEqual(SlamFinalSetRule("wimbledon", 2018), "adv")
        self.assertEqual(SlamFinalSetRule("wimbledon", 2020), "tb7_12")
        self.assertEqual(SlamFinalSetRule("usopen", 2015), "tb7")
        self.assertEqual(SlamFinalSetRule("frenchopen", 2023), "tb10")


if __name__ == "__main__":
    unittest.main()