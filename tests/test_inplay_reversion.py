"""Synthetic tests for the pre-registered in-play mid-jump design. No real candle results."""
from __future__ import annotations

import datetime as dt
import unittest
from zoneinfo import ZoneInfo

from Backtest.InPlayReversion import (
    HOLDOUT_START, BuyYes, CellSeries, ExtractJumps, FmtEdge, JumpsFrame,
    MakerPnl, Mid, ScoreJump, SortMinutes, TakerPnl, TuneCandidates, ValidBook,
    Verdict, ExitAt, MC_LARGE_LOSS, QUAL,
)

ET = ZoneInfo("America/New_York")


def Candle(end, bid, ask, last=None, volume=0.0):
    return {"end_ts": end, "yes_bid": bid, "yes_ask": ask, "last": last, "volume": volume}


def Payload(start_iso, minutes, ticker="KXATPCHALLENGERMATCH-TEST"):
    return {
        "event_ticker": ticker,
        "ticker": ticker + "-AAA",
        "scheduled_start": start_iso,
        "series": "KXATPCHALLENGERMATCH",
        "minute": minutes,
    }


class ValidBookTests(unittest.TestCase):
    def test_rejects_crossed_or_settled(self):
        self.assertFalse(ValidBook(0.0, 0.5))
        self.assertFalse(ValidBook(0.4, 0.4))
        self.assertFalse(ValidBook(0.6, 0.4))
        self.assertFalse(ValidBook(0.4, 1.0))
        self.assertFalse(ValidBook(None, 0.5))
        self.assertTrue(ValidBook(0.40, 0.42))
        self.assertAlmostEqual(Mid(0.40, 0.42), 0.41)


class JumpExtractionTests(unittest.TestCase):
    def test_official_requires_exact_60s_gap(self):
        t0 = 1_700_000_000
        minutes = [
            Candle(t0, 0.40, 0.42),
            Candle(t0 + 60, 0.50, 0.52),  # +0.10 official
            Candle(t0 + 180, 0.62, 0.64),  # +0.12 but gap 120s -> stale, not official
        ]
        start = dt.datetime(2026, 7, 1, 12, 0, tzinfo=ET).isoformat()
        official, diag = ExtractJumps(Payload(start, minutes))
        self.assertEqual(len(official), 1)
        self.assertEqual(official[0]["gap_s"], 60)
        self.assertAlmostEqual(official[0]["abs_delta"], 0.10)
        self.assertEqual(diag["n_stale_jumps"], 1)
        self.assertEqual(diag["n_gaps"], 1)
        self.assertEqual(official[0]["split"], "tune")

    def test_holdout_split_uses_scheduled_start(self):
        t0 = 1_800_000_000
        minutes = [Candle(t0, 0.20, 0.22), Candle(t0 + 60, 0.40, 0.42)]
        start = dt.datetime(2026, 8, 13, 0, 0, tzinfo=ET).isoformat()
        official, _ = ExtractJumps(Payload(start, minutes))
        self.assertEqual(official[0]["split"], "holdout")
        start_before = dt.datetime(2026, 8, 12, 23, 59, tzinfo=ET).isoformat()
        official_b, _ = ExtractJumps(Payload(start_before, minutes))
        self.assertEqual(official_b[0]["split"], "tune")
        self.assertEqual(HOLDOUT_START, dt.datetime(2026, 8, 13, 0, 0, 0, tzinfo=ET))

    def test_invalid_book_breaks_the_1min_chain(self):
        t0 = 1_700_000_000
        minutes = [
            Candle(t0, 0.40, 0.42),
            Candle(t0 + 60, 0.0, 1.0),  # invalid / empty
            Candle(t0 + 120, 0.60, 0.62),
        ]
        official, diag = ExtractJumps(Payload(dt.datetime(2026, 7, 1, tzinfo=ET).isoformat(), minutes))
        self.assertEqual(official, [])
        self.assertGreaterEqual(diag["n_empty_or_invalid"], 1)


class ExitAndPnlTests(unittest.TestCase):
    def test_exit_window_is_first_valid_at_or_within_60s_of_horizon(self):
        t0 = 1000
        minutes = [
            Candle(t0, 0.40, 0.42),
            Candle(t0 + 60, 0.50, 0.52),
            Candle(t0 + 60 + 180, 0.48, 0.50),  # +3 min exact
            Candle(t0 + 60 + 241, 0.30, 0.32),  # too late for +3 (window ends +240)
        ]
        jump_end = t0 + 60
        ext = ExitAt(minutes, jump_end, 3)
        self.assertIsNotNone(ext)
        self.assertEqual(ext["end_ts"], t0 + 240)
        self.assertIsNone(ExitAt(minutes, jump_end, 1))  # no candle in [+60, +120]
        self.assertIsNone(ExitAt(minutes, jump_end, 5))

    def test_taker_and_maker_pnl_signs(self):
        # Up jump: mid 0.41 -> 0.51. Continuation buys yes; reversion buys no.
        self.assertTrue(BuyYes(0.10, "continuation"))
        self.assertFalse(BuyYes(0.10, "reversion"))
        self.assertFalse(BuyYes(-0.10, "continuation"))
        self.assertTrue(BuyYes(-0.10, "reversion"))
        # Taker continuation up: exit_bid 0.48 - ask 0.52 - 0.02 = -0.06
        self.assertAlmostEqual(TakerPnl(True, 0.50, 0.52, 0.48, 0.50, 0.02), -0.06)
        # Taker reversion up (buy no): bid 0.50 - exit_ask 0.50 - 0.02 = -0.02
        self.assertAlmostEqual(TakerPnl(False, 0.50, 0.52, 0.48, 0.50, 0.02), -0.02)
        # Maker fee 0 buy yes: exit_bid 0.48 - bid 0.50 = -0.02
        self.assertAlmostEqual(MakerPnl(True, 0.50, 0.52, 0.48, 0.50, 0.0), -0.02)
        # Maker buy no: jump_ask 0.52 - exit_ask 0.50 = +0.02
        self.assertAlmostEqual(MakerPnl(False, 0.50, 0.52, 0.48, 0.50, 0.0), 0.02)

    def test_maker_fill_requires_later_trade_through_before_exit(self):
        t0 = 2_000_000_000
        # Jump up to bid/ask 0.50/0.52. +1 exit at 0.48/0.50. A between-candle last 0.50 fills yes-bid.
        minutes = [
            Candle(t0, 0.40, 0.42),
            Candle(t0 + 60, 0.50, 0.52, last=0.51),
            Candle(t0 + 90, 0.49, 0.51, last=0.50),  # trades through 0.50 bid
            Candle(t0 + 120, 0.48, 0.50, last=0.49),
        ]
        start = dt.datetime(2026, 7, 1, 12, 0, tzinfo=ET).isoformat()
        official, _ = ExtractJumps(Payload(start, minutes))
        self.assertEqual(len(official), 1)
        row = official[0]
        self.assertTrue(row["h1_exit_ok"])
        self.assertTrue(row["h1_continuation_maker_filled"])
        self.assertIsNotNone(row["h1_continuation_maker_pnl"])
        # Reversion rests at no bid = 1-0.52=0.48, fill if last through that (last >= 0.52) or bid>=0.52
        self.assertFalse(row["h1_reversion_maker_filled"])
        self.assertIsNone(row["h1_reversion_maker_pnl"])

    def test_maker_unfilled_when_no_through_print(self):
        t0 = 2_000_000_100
        minutes = [
            Candle(t0, 0.40, 0.42),
            Candle(t0 + 60, 0.50, 0.52, last=0.51),
            Candle(t0 + 90, 0.51, 0.53, last=0.52),  # never trades through 0.50
            Candle(t0 + 120, 0.48, 0.50, last=0.49),
        ]
        start = dt.datetime(2026, 7, 1, 12, 0, tzinfo=ET).isoformat()
        official, _ = ExtractJumps(Payload(start, minutes))
        row = official[0]
        self.assertFalse(row["h1_continuation_maker_filled"])
        self.assertIsNone(row["h1_continuation_maker_pnl"])
        self.assertAlmostEqual(row["h1_continuation_taker_pnl"], 0.48 - 0.52 - 0.02)


class CellAndVerdictTests(unittest.TestCase):
    def test_cell_series_nests_thresholds_and_drops_unscored(self):
        start = dt.datetime(2026, 9, 1, 12, 0, tzinfo=ET).isoformat()
        t0 = 2_100_000_000
        minutes = [
            Candle(t0, 0.30, 0.32),
            Candle(t0 + 60, 0.40, 0.42),  # +0.10
            Candle(t0 + 120, 0.41, 0.43),
            Candle(t0 + 60 + 180, 0.39, 0.41),
            Candle(t0 + 60 + 300, 0.38, 0.40),
            Candle(t0 + 60 + 600, 0.37, 0.39),
        ]
        official, _ = ExtractJumps(Payload(start, minutes))
        frame = JumpsFrame(official)
        s08 = CellSeries(frame, 0.08, 1, "continuation", "taker")
        s12 = CellSeries(frame, 0.12, 1, "continuation", "taker")
        s20 = CellSeries(frame, 0.20, 1, "continuation", "taker")
        self.assertEqual(int(s08.notna().sum()), 1)
        self.assertEqual(int(s12.notna().sum()), 0)
        self.assertEqual(int(s20.notna().sum()), 0)

    def test_tune_candidates_ignore_holdout_means(self):
        import pandas as pd
        summary = pd.DataFrame([
            {"cell": "x0.08_h1_reversion_taker", "split": "tune", "timing": "all",
             "mean_edge": 0.04, "qualify_3c": True, "n_scored": 10},
            {"cell": "x0.20_h10_continuation_maker", "split": "holdout", "timing": "all",
             "mean_edge": 0.50, "qualify_3c": True, "n_scored": 10},
            {"cell": "x0.08_h3_continuation_taker", "split": "tune", "timing": "all",
             "mean_edge": 0.01, "qualify_3c": False, "n_scored": 10},
        ])
        cand = TuneCandidates(summary)
        self.assertEqual(list(cand["cell"]), ["x0.08_h1_reversion_taker"])

    def test_verdict_defaults_to_pass(self):
        import pandas as pd
        spa = {"p_value": 0.40, "best_index": 0, "t_obs": 0.01, "mean_best": 0.01}
        rc = {"p_value": 0.50, "best_index": 0, "t_obs": 0.01}
        verdict, why = Verdict(spa, rc, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        self.assertEqual(verdict, "PASS")
        self.assertIn("No tune cell", why)

    def test_verdict_bet_requires_all_three_bars(self):
        import pandas as pd
        spa = {"p_value": 0.01, "best_index": 0, "t_obs": 2.0, "mean_best": 0.05}
        rc = {"p_value": 0.20, "best_index": 0, "t_obs": 0.05}
        cand = pd.DataFrame([{"cell": "x0.08_h1_reversion_taker"}])
        hold = pd.DataFrame([
            {"cell": "x0.08_h1_reversion_taker", "split": "holdout", "timing": "all",
             "ci_lo": 0.01, "ci_hi": 0.08, "mean_edge": 0.04},
        ])
        bank = pd.DataFrame([
            {"cell": "x0.08_h1_reversion_taker", "style": "flat100", "p5": 9_000.0},
        ])
        verdict, _ = Verdict(spa, rc, cand, hold, bank)
        self.assertEqual(verdict, "BET")
        bank_bad = pd.DataFrame([
            {"cell": "x0.08_h1_reversion_taker", "style": "flat100", "p5": MC_LARGE_LOSS - 1},
        ])
        verdict_bad, why = Verdict(spa, rc, cand, hold, bank_bad)
        self.assertEqual(verdict_bad, "PASS")
        self.assertIn("LARGE LOSS", why)
        self.assertEqual(QUAL, 0.03)


if __name__ == "__main__":
    unittest.main()