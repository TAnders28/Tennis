import unittest

import pandas as pd

from Backtest.Elo import EloParams
from Backtest.Engine import RunModels
from Backtest.KalshiResults import Combine, RoundToStage, SyntheticId


class KalshiResultsTest(unittest.TestCase):
    def test_round_codes(self):
        self.assertEqual(RoundToStage("Round Of 32"), "R32")
        self.assertEqual(RoundToStage("Quarterfinals"), "QF")
        self.assertEqual(RoundToStage("Semifinal"), "SF")
        self.assertEqual(RoundToStage("Final"), "F")
        self.assertEqual(RoundToStage("Qualification Round 2"), "Q2")
        self.assertEqual(RoundToStage("Qualification Final"), "Q3")
        self.assertEqual(RoundToStage("1/16 Finals"), "R32")

    def test_synthetic_id_stable_negative(self):
        self.assertEqual(SyntheticId("atp", "New Player"), SyntheticId("atp", "new  player"))
        self.assertLess(SyntheticId("atp", "New Player"), 0)

    def test_seed_and_same_day_leak_free(self):
        base = {"tour": "atp", "season": 2026, "tourney_id": "t", "tourney_name": "x", "tourney_level": "K", "level_tag": "challenger",
                "surface_norm": "Hard", "best_of": 3, "match_num": 0, "file_kind": "kalshi", **{c: float("nan") for c in ("w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon")}}
        rows = [dict(base, match_key="a", tourney_date=20260601, date=pd.Timestamp("2026-06-01"), stage=12, winner_id=1, loser_id=-5),
                dict(base, match_key="b", tourney_date=20260601, date=pd.Timestamp("2026-06-01"), stage=12, winner_id=-5, loser_id=2)]
        frame = Combine(pd.DataFrame(rows).iloc[:0], pd.DataFrame(rows))
        out = RunModels(frame, EloParams(), None, "atp", seed_ratings={-5: 1300.0})
        # same group: second match predicted before the first result is applied, seeded player starts at 1300
        self.assertAlmostEqual(out["elo"].rating[-5] > 1300, True)
        self.assertLess(out["p_elo"][1], 0.5)


if __name__ == "__main__":
    unittest.main()