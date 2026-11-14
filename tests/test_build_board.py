"""Offline unit tests for Dashboard.BuildBoard (synthetic data only; no live pulls)."""
from __future__ import annotations

import datetime as dt
import json
import unittest

import pandas as pd

from Dashboard.BuildBoard import (
    AttachProps, BuildBoard, BuildContext, BuildGame, EdgeCents, EventSuffix,
    GameStatus, HeadToHead, InDateWindow, JsonSafe, KalshiEdgeCents, OpenPrice,
    PropRecord, RecentForm, RestDays, WriteBoard,
)


class EdgeTest(unittest.TestCase):
    def test_tim_fee_cents(self):
        # (0.60 - 0.50 - 0.02) * 100 = 8.0
        self.assertEqual(EdgeCents(0.60, 0.50), 8.0)

    def test_missing_or_invalid_is_null(self):
        self.assertIsNone(EdgeCents(None, 0.50))
        self.assertIsNone(EdgeCents(0.60, None))
        self.assertIsNone(EdgeCents(0.60, 0.0))
        self.assertIsNone(EdgeCents(0.60, 1.0))

    def test_kalshi_fee_differs_from_tim_estimate(self):
        # Kalshi fee at p=0.50 is 0.07*0.5*0.5 = 0.0175; edge = (0.60-0.50-0.0175)*100 = 8.25
        self.assertEqual(KalshiEdgeCents(0.60, 0.50), 8.25)
        self.assertNotEqual(KalshiEdgeCents(0.60, 0.50), EdgeCents(0.60, 0.50))


class BuildGameTest(unittest.TestCase):
    def test_schema_keys_and_edges(self):
        game = BuildGame(
            game_id="KXATPMATCH-26NOV14AAABBB",
            start_time="2026-11-14T17:00:00-05:00",
            home="Ada",
            away="Bea",
            status="scheduled",
            score=None,
            market={"home_yes_ask": 0.50, "away_yes_ask": 0.52, "source": "kalshi_public", "as_of": "2026-11-14T12:00:00+00:00"},
            model_prob=0.60,
            context={"injuries": None, "recent_form": None, "rest_days": None, "head_to_head": None, "notes": ["fee estimate"]},
            props=[],
        )
        for key in ("game_id", "start_time", "home", "away", "status", "score", "market",
                    "model", "edge_home_c", "edge_away_c", "context", "props"):
            self.assertIn(key, game)
        for key in ("home_yes_ask", "away_yes_ask", "open_home_yes", "source", "as_of"):
            self.assertIn(key, game["market"])
        for key in ("home_win_prob", "model_name", "tested"):
            self.assertIn(key, game["model"])
        self.assertEqual(game["home"], "Ada")
        self.assertEqual(game["away"], "Bea")
        self.assertEqual(game["model"]["model_name"], "surface_elo")
        self.assertTrue(game["model"]["tested"])
        self.assertEqual(game["model"]["fee_basis"], "tim_est_0.02")
        self.assertEqual(game["edge_home_c"], 8.0)
        # away prob 0.40 - ask 0.52 - 0.02 = -0.14 -> -14.0 cents
        self.assertEqual(game["edge_away_c"], -14.0)
        self.assertIsNone(game["market"]["open_home_yes"])
        self.assertIsNone(game["score"])
        self.assertIsNone(game["context"]["injuries"])
        self.assertIn("lost to Kalshi", game["model"]["holdout_note"])

    def test_unrated_leaves_prob_and_edges_null(self):
        game = BuildGame(
            game_id="x", start_time=None, home="Ada", away="Bea", status="scheduled",
            score=None, market={"home_yes_ask": 0.50, "away_yes_ask": 0.50}, model_prob=None,
        )
        self.assertIsNone(game["model"]["home_win_prob"])
        self.assertIsNone(game["edge_home_c"])
        self.assertIsNone(game["edge_away_c"])
        self.assertIsNone(game["edge_home_c_kalshi"])

    def test_open_price_only_from_named_field(self):
        self.assertIsNone(OpenPrice({"last": 0.55, "last_price_dollars": 0.55, "yes_ask": 0.60}))
        self.assertEqual(OpenPrice({"yes_open_dollars": "0.48"}), 0.48)
        game = BuildGame(
            game_id="x", start_time=None, home="A", away="B", status="open", score=None,
            market={"home_yes_ask": 0.4, "away_yes_ask": 0.6, "last": 0.55}, model_prob=0.5,
        )
        self.assertIsNone(game["market"]["open_home_yes"])


class ContextTest(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {"date": pd.Timestamp("2026-05-20"), "winner_id": 1, "loser_id": 2, "score": "6-3 6-2"},
            {"date": pd.Timestamp("2026-05-28"), "winner_id": 2, "loser_id": 1, "score": "7-5 6-4"},
            {"date": pd.Timestamp("2026-06-01"), "winner_id": 1, "loser_id": 3, "score": "6-1 6-1"},
        ])
        self.last = dt.date(2026, 6, 1)

    def test_stale_form_and_rest_are_null(self):
        ctx = BuildContext(self.frame, 1, 2, dt.date(2026, 11, 14), self.last)
        self.assertIsNone(ctx["injuries"])
        self.assertIsNone(ctx["recent_form"])
        self.assertIsNone(ctx["rest_days"])
        self.assertIsNotNone(ctx["head_to_head"])
        self.assertEqual(ctx["head_to_head"]["home_wins"], 1)
        self.assertEqual(ctx["head_to_head"]["away_wins"], 1)
        self.assertEqual(ctx["head_to_head"]["through"], "2026-05-28")  # last meeting, not the file cutoff
        self.assertFalse(ctx["head_to_head"]["complete"])
        blob = " ".join(ctx["notes"])
        self.assertIn("stale", blob.lower())
        self.assertIn("2026-06-01", blob)
        self.assertIn("lost to Kalshi", blob)
        self.assertIn("fee estimate", blob.lower())

    def test_fresh_form_is_filled(self):
        ctx = BuildContext(self.frame, 1, 2, dt.date(2026, 6, 5), self.last)
        self.assertIsNotNone(ctx["recent_form"])
        self.assertEqual(ctx["recent_form"]["home"]["results"][-1], "W")
        self.assertEqual(ctx["rest_days"]["home"], 4)
        self.assertEqual(ctx["rest_days"]["away"], 8)

    def test_no_h2h_is_null(self):
        ctx = BuildContext(self.frame, 1, 99, dt.date(2026, 6, 5), self.last)
        self.assertIsNone(ctx["head_to_head"])
        self.assertTrue(any("no head-to-head" in n for n in ctx["notes"]))

    def test_helpers_on_synthetic_rows(self):
        self.assertEqual(RecentForm(self.frame, 1)["results"], "WLW")  # all local matches, oldest to newest
        self.assertEqual(RestDays(self.frame, 2, dt.date(2026, 6, 1)), 4)
        h2h = HeadToHead(self.frame, 1, 2)
        self.assertEqual(h2h["meetings"], 2)


class BoardAndPropsTest(unittest.TestCase):
    def test_top_level_and_window(self):
        start = dt.datetime(2026, 11, 14, 17, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5)))
        self.assertTrue(InDateWindow(start, dt.date(2026, 11, 14), dt.date(2026, 11, 15)))
        later = dt.datetime(2026, 11, 16, 17, 0, tzinfo=dt.timezone(dt.timedelta(hours=-5)))
        self.assertFalse(InDateWindow(later, dt.date(2026, 11, 14), dt.date(2026, 11, 15)))
        board = BuildBoard([], "2026-11-14T18:00:00-05:00")
        self.assertEqual(board["sport"], "TENNIS")
        self.assertEqual(board["games"], [])
        self.assertEqual(board["generated_at"], "2026-11-14T18:00:00-05:00")

    def test_props_join_on_suffix_and_are_prices_only(self):
        games = [BuildGame(
            game_id="KXATPMATCH-26NOV14AAABBB", start_time=None, home="Ada", away="Bea",
            status="scheduled", score=None, market={"home_yes_ask": 0.4, "away_yes_ask": 0.6},
            model_prob=0.55,
        )]
        props = [{
            "ticker": "KXATPGTOTAL-26NOV14AAABBB-22",
            "event_ticker": "KXATPGTOTAL-26NOV14AAABBB",
            "series": "KXATPGTOTAL",
            "title": "Ada vs Bea: Total Games",
            "yes_sub_title": "Over 22.5 games",
            "yes_ask": 0.48,
            "yes_bid": 0.44,
            "fetched_at": "2026-11-14T12:00:00+00:00",
        }]
        AttachProps(games, props)
        self.assertEqual(len(games[0]["props"]), 1)
        rec = games[0]["props"][0]
        self.assertEqual(rec["kind"], "total_games")
        self.assertEqual(rec["line"], 22.5)
        self.assertEqual(rec["yes_ask"], 0.48)
        self.assertNotIn("home_win_prob", rec)
        self.assertNotIn("model", rec)
        self.assertEqual(EventSuffix("KXATPMATCH-26NOV14AAABBB"), "26NOV14AAABBB")

    def test_write_latest_and_dated_copy(self):
        import tempfile
        from pathlib import Path
        payload = BuildBoard([], "2026-11-14T18:00:00-05:00")
        with tempfile.TemporaryDirectory() as tmp:
            latest = Path(tmp) / "board_latest.json"
            out, dated = WriteBoard(payload, latest, "2026-11-14")
            self.assertTrue(out.exists())
            self.assertTrue(dated.exists())
            self.assertEqual(dated.name, "board_2026-11-14.json")
            loaded = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(loaded["sport"], "TENNIS")
            self.assertIsNone(JsonSafe(float("nan")))

    def test_status_scheduled_vs_final(self):
        future = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=3)
        self.assertEqual(GameStatus(future, "open"), "scheduled")
        self.assertEqual(GameStatus(None, "finalized"), "final")


if __name__ == "__main__":
    unittest.main()
