import datetime as dt
import unittest

from Backtest.DeepDivePanel import NormKey, ParseRound, ParseTournament, SurfaceFor


class DeepDivePanelTests(unittest.TestCase):
    def test_parse_challenger(self):
        r = "If Evan Zhu wins the Zhu vs Liu professional tennis match in the 2026 ATP Challenger Jingshan Round Of 32 after a ball has been played, then the market resolves to Yes."
        self.assertEqual(ParseTournament(r), ("Jingshan", "R32"))

    def test_parse_wta125_qualifying(self):
        r = "If X wins the X vs Y professional tennis match in the 2026 WTA 125K Madrid Qualification Round 1 after a ball has been played"
        self.assertEqual(ParseTournament(r), ("Madrid", "Q"))

    def test_parse_missing(self):
        self.assertEqual(ParseTournament("match originally scheduled for Aug 1, 2025 after a ball"), ("", ""))

    def test_rounds(self):
        self.assertEqual(ParseRound("2025 WTA Chennai semifinal"), "SF")
        self.assertEqual(ParseRound("2026 ATP Tokyo Final"), "F")
        self.assertEqual(ParseRound("2025 Next Gen ATP Finals Men Singles Red Group"), "RR")

    def test_normkey(self):
        self.assertEqual(NormKey("Buenos Aires 1 CH"), "buenos aires")
        self.assertEqual(NormKey("Antalya 125 2"), "antalya")
        self.assertEqual(NormKey("Shymkent II"), "shymkent")

    def test_surface_nearest_time_of_year(self):
        table = {"madrid": [(120, "Clay"), (300, "Hard")]}
        self.assertEqual(SurfaceFor("Madrid", dt.date(2026, 5, 2), table), ("Clay", "city_map"))
        self.assertEqual(SurfaceFor("Madrid", dt.date(2026, 10, 25), table), ("Hard", "city_map"))
        self.assertEqual(SurfaceFor("Madrid", dt.date(2026, 2, 1), table), ("", ""))
        self.assertEqual(SurfaceFor("US Open", dt.date(2026, 9, 1), {}), ("Hard", "major"))


if __name__ == "__main__":
    unittest.main()