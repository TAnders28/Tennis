"""Tests for Archetypes.ServeReturn (serve/return style matchup study)."""
import os, tempfile, unittest
import numpy as np
import pandas as pd

from Archetypes import ServeReturn as SR


class TestServeReturn(unittest.TestCase):
    def test_double_centre(self):
        w = np.array([[10., 20., 5.], [3., 8., 12.]])
        g = SR.DoubleCentre(np.array([[1., 2., 3.], [4., 5., 7.]]), w)
        self.assertTrue(np.allclose((g * w).sum(0), 0, atol=1e-6))
        self.assertTrue(np.allclose((g * w).sum(1), 0, atol=1e-6))

    def test_bh(self):
        self.assertTrue(np.allclose(SR.BH([0.01, 0.04, 0.03, 0.5]), [0.04, 0.16 / 3, 0.16 / 3, 0.5]))

    def test_profiles_and_rates(self):
        s = pd.DataFrame({"season": 2024, "match": [f"m{i}" for i in range(10)], "server": 1, "returner": 2, "svpt": 80.0, "ace": 8.0, "df": 4.0,
                          "in1": 48.0, "won1": 36.0, "won2": 16.0, "svgms": 12.0, "bpf": 3.0})
        s2 = s.assign(server=2, returner=1, ace=2.0, won1=30.0, won2=12.0)
        p = SR.Profiles(pd.concat([s, s2]), min_pts=300).set_index("player")
        self.assertAlmostEqual(p.loc[1, "s_ace"], 0.1)
        self.assertAlmostEqual(p.loc[1, "s_1won"], 0.75)
        self.assertAlmostEqual(p.loc[1, "s_2won"], 0.5)
        self.assertAlmostEqual(p.loc[1, "r_1won"], 1 - 30 / 48)   # player 1 returning player 2's serve
        self.assertAlmostEqual(p.loc[1, "r_ace_against"], 2 / 80)

    def test_prior_season_labels_only(self):
        # labels for season T must come from season T-1 profiles (shift by +1 before merging)
        L = pd.DataFrame({"season": [2023, 2024], "player": [1, 1], "serve_arch": [0, 1]})
        L2 = L.copy(); L2["season"] += 1
        m = pd.DataFrame({"season": [2024, 2025], "server": [1, 1]}).merge(L2.rename(columns={"player": "server"}), on=["season", "server"])
        self.assertEqual(m.set_index("season").loc[2024, "serve_arch"], 0)
        self.assertEqual(m.set_index("season").loc[2025, "serve_arch"], 1)


if __name__ == "__main__":
    unittest.main()
