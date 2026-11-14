"""Synthetic tests for the Challenger maker fill rule and market_fav side (no live data)."""
from __future__ import annotations

import unittest

from Backtest.ChallengerMaker import MakerFilled
from Backtest.MarketBlend import DevigAsk


class MakerFillTest(unittest.TestCase):
    def test_unfilled_if_never_trades_through(self):
        candles = [
            {"end_ts": 100, "last": 0.40, "yes_bid": 0.39, "yes_ask": 0.41},  # the T-60 quote itself
            {"end_ts": 200, "last": 0.42, "yes_bid": 0.40, "yes_ask": 0.43},
            {"end_ts": 400, "last": 0.45, "yes_bid": 0.44, "yes_ask": 0.46},  # at or after start, ignored
        ]
        filled, reason = MakerFilled(candles, quote_end_ts=100, start_ts=400, side="a", t60_bid=0.39, t60_ask=0.41)
        self.assertFalse(filled)
        self.assertEqual(reason, "unfilled")

    def test_filled_when_last_trades_through_bid(self):
        candles = [
            {"end_ts": 100, "last": 0.40, "yes_bid": 0.39, "yes_ask": 0.41},
            {"end_ts": 200, "last": 0.38, "yes_bid": 0.37, "yes_ask": 0.40},
        ]
        filled, reason = MakerFilled(candles, 100, 400, "a", 0.39, 0.41)
        self.assertTrue(filled)
        self.assertEqual(reason, "last_through_bid")

    def test_buy_no_fills_when_yes_last_reaches_ask(self):
        candles = [
            {"end_ts": 100, "last": 0.60, "yes_bid": 0.59, "yes_ask": 0.62},
            {"end_ts": 200, "last": 0.62, "yes_bid": 0.61, "yes_ask": 0.63},
        ]
        filled, reason = MakerFilled(candles, 100, 400, "b", 0.59, 0.62)
        self.assertTrue(filled)
        self.assertEqual(reason, "last_through_bid")

    def test_ignores_candles_after_start_or_at_quote(self):
        candles = [
            {"end_ts": 100, "last": 0.20, "yes_bid": 0.39, "yes_ask": 0.41},
            {"end_ts": 400, "last": 0.20, "yes_bid": 0.19, "yes_ask": 0.21},
        ]
        filled, reason = MakerFilled(candles, 100, 400, "a", 0.39, 0.41)
        self.assertFalse(filled)


class MarketFavTest(unittest.TestCase):
    def test_favourite_is_the_devigged_side_at_or_above_half(self):
        p = float(DevigAsk([0.40], [0.50])[0])  # yes ~ 0.4545, favourite is no
        self.assertLess(p, 0.5)
        fav = "a" if p >= 0.5 else "b"
        self.assertEqual(fav, "b")
        p2 = float(DevigAsk([0.55], [0.60])[0])
        self.assertGreaterEqual(p2, 0.5)


if __name__ == "__main__":
    unittest.main()