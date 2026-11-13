import unittest

from Backtest.NameMatch import NameIndex, ParseTennisDataName
from Backtest.Odds import ContractPnl, DevigPower, DevigProportional, Edge, KalshiFee


class OddsTest(unittest.TestCase):
    def test_devig_sums_to_one(self):
        a, b = DevigProportional(1 / 1.5, 1 / 2.7)
        self.assertAlmostEqual(a + b, 1.0)
        a, b = DevigPower(1 / 1.5, 1 / 2.7)
        self.assertAlmostEqual(a + b, 1.0, places=8)
        self.assertGreater(a, DevigProportional(1 / 1.5, 1 / 2.7)[0])  # power shades the longshot

    def test_edge_and_pnl(self):
        self.assertAlmostEqual(Edge(0.60, 0.55), 0.03)
        self.assertAlmostEqual(KalshiFee(0.5), 0.0175)
        pnl, cost = ContractPnl(0.55, True)
        self.assertAlmostEqual(pnl, 0.43)
        self.assertAlmostEqual(cost, 0.57)


class NameMatchTest(unittest.TestCase):
    def setUp(self):
        self.index = NameIndex([(1, "Roger Federer"), (2, "Juan Martin del Potro"), (3, "Albert Ramos"),
                                (4, "Roberto Bautista Agut"), (5, "Alexander Zverev"), (6, "Mischa Zverev"),
                                (7, "Su Wei Hsieh"), (8, "Andrey Kuznetsov"), (9, "Alex Kuznetsov")])

    def test_tennis_data_names(self):
        self.assertEqual(ParseTennisDataName("Del Potro J.M."), ("del potro", "j"))
        self.assertEqual(self.index.LookupTennisData("Federer R."), {1})
        self.assertEqual(self.index.LookupTennisData("Del Potro J.M."), {2})
        self.assertEqual(self.index.LookupTennisData("Ramos-Vinolas A."), {3})
        self.assertEqual(self.index.LookupTennisData("Bautista Agut R."), {4})
        self.assertEqual(self.index.LookupTennisData("Zverev A."), {5})
        self.assertEqual(self.index.LookupTennisData("Hsieh S.W."), {7})
        self.assertEqual(self.index.LookupTennisData("Kuznetsov A."), {8, 9})  # ambiguous -> join disambiguates

    def test_full_names(self):
        self.assertEqual(self.index.LookupFull("Alexander Zverev"), {5})
        self.assertEqual(self.index.LookupFull("Juan Martín del Potro"), {2})


if __name__ == "__main__":
    unittest.main()