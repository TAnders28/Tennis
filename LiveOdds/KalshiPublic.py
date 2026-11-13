r"""Read-only Kalshi public API client for tennis match markets (no auth, no orders - ever).

Kalshi's market list gives top-of-book yes bid/ask and sizes but NO quote timestamp, so
every quote carries fetched_at (our clock, UTC) and price_time=None (source time unknown).
Robinhood lists Kalshi contracts; Robinhood-vs-Kalshi price differences are unmeasured.
"""
from __future__ import annotations

import datetime as dt

from DataPull.Http import PoliteSession

API = "https://api.elections.kalshi.com/trade-api/v2"
MATCH_SERIES = {"KXATPMATCH": ("atp", "tour"), "KXWTAMATCH": ("wta", "tour"),
                "KXATPCHALLENGERMATCH": ("atp", "challenger"), "KXWTACHALLENGERMATCH": ("wta", "challenger")}


def _F(value):
    try:
        return float(value) if value not in (None, "") else None
    except (TypeError, ValueError):
        return None


class KalshiPublic:
    def __init__(self, min_interval_s=0.2):
        self.session = PoliteSession(min_interval_s=min_interval_s, max_retries=3, timeout_s=30)

    def OpenMarkets(self, series):
        out, cursor = [], ""
        fetched_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        while True:
            params = {"series_ticker": series, "status": "open", "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            response = self.session.Get(API + "/markets", params=params)
            if response.status_code != 200:
                break
            payload = response.json()
            for m in payload.get("markets", []):
                out.append({"ticker": m.get("ticker"), "event_ticker": m.get("event_ticker"), "series": series,
                            "player": m.get("yes_sub_title"), "title": m.get("title"), "rules": m.get("rules_primary"),
                            "yes_bid": _F(m.get("yes_bid_dollars")), "yes_ask": _F(m.get("yes_ask_dollars")),
                            "yes_bid_size": _F(m.get("yes_bid_size_fp")), "yes_ask_size": _F(m.get("yes_ask_size_fp")),
                            "last": _F(m.get("last_price_dollars")), "status": m.get("status"),
                            "occurrence_datetime": m.get("occurrence_datetime"), "fetched_at": fetched_at,
                            "price_time": None})
            cursor = payload.get("cursor") or ""
            if not cursor or not payload.get("markets"):
                break
        return out

    def Milestones(self, since_iso):
        rows, cursor = [], ""
        while True:
            params = {"type": "tennis_tournament_singles", "limit": 500, "minimum_start_date": since_iso}
            if cursor:
                params["cursor"] = cursor
            response = self.session.Get(API + "/milestones", params=params)
            if response.status_code != 200:
                break
            payload = response.json()
            rows.extend(payload.get("milestones", []))
            cursor = payload.get("cursor") or ""
            if not cursor or not payload.get("milestones"):
                break
        starts = {}
        for item in rows:
            for ticker in set((item.get("primary_event_tickers") or []) + (item.get("related_event_tickers") or [])):
                starts[ticker] = {"start_date": item.get("start_date"), "details": item.get("details") or {}, "title": item.get("title")}
        return starts

    def Slate(self, series_list=None, since_iso=None):
        """Open match events: {event_ticker: {series, tour, level, markets: [...], start, details}}."""
        series_list = series_list or list(MATCH_SERIES)
        since_iso = since_iso or (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=1)).strftime("%Y-%m-%dT%H:%M:%SZ")
        starts = self.Milestones(since_iso)
        events = {}
        for series in series_list:
            tour, level = MATCH_SERIES[series]
            for m in self.OpenMarkets(series):
                event = events.setdefault(m["event_ticker"], {"series": series, "tour": tour, "level": level, "markets": []})
                event["markets"].append(m)
        for ticker, event in events.items():
            meta = starts.get(ticker, {})
            event["start"] = meta.get("start_date") or (event["markets"][0].get("occurrence_datetime") if event["markets"] else None)
            event["start_source"] = "milestone" if meta else "occurrence_datetime"
            event["details"] = meta.get("details", {})
        return events