r"""Pull Kalshi tennis match markets and hourly price candles (public, read-only, no auth).

Kalshi is Tim's venue (Robinhood prediction markets list Kalshi contracts). This
script only reads public market data; it never authenticates or places orders.

  markets  -> data/raw/kalshi/markets/<series>.jsonl   (historical + recently settled)
  candles  -> data/raw/kalshi/candles/<series>/<event_ticker>.json  (1-min T-180m..T+5m, hourly earlier; yes_bid/yes_ask closes)

Candles carry top-of-book bid/ask at each hour close but no size, so a backtest
price from them is "executable-ish", not a guaranteed fill.
Resumable: candle files that exist are skipped.

Usage:
  .venv\Scripts\python.exe -m DataPull.PullKalshi --markets
  .venv\Scripts\python.exe -m DataPull.PullKalshi --candles --from 2025-06-01 --to 2026-07-15
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

from DataPull.Http import RAW_DIR, PoliteSession, WriteJsonAtomic

API = "https://api.elections.kalshi.com/trade-api/v2"
MATCH_SERIES = ["KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH", "KXCHALLENGERMATCH", "KXWTACHALLENGERMATCH"]
SERIES_TOUR = {"KXATPMATCH": "atp", "KXWTAMATCH": "wta", "KXATPCHALLENGERMATCH": "atp",
               "KXCHALLENGERMATCH": "atp", "KXWTACHALLENGERMATCH": "wta"}
SERIES_LEVEL = {"KXATPMATCH": "tour", "KXWTAMATCH": "tour", "KXATPCHALLENGERMATCH": "challenger",
                "KXCHALLENGERMATCH": "challenger", "KXWTACHALLENGERMATCH": "challenger"}
KEEP_FIELDS = ["ticker", "event_ticker", "title", "yes_sub_title", "no_sub_title", "rules_primary", "status", "result",
               "open_time", "close_time", "occurrence_datetime", "expected_expiration_time", "settlement_ts",
               "last_price_dollars", "volume_fp", "custom_strike"]


def ParseTs(text):
    if not text:
        return None
    return dt.datetime.fromisoformat(text.replace("Z", "+00:00"))


def ListMarkets(session, series):
    markets = {}
    for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"}), ("/markets", {"status": "open"})):
        cursor = ""
        while True:
            params = {"series_ticker": series, "limit": 1000, **extra}
            if cursor:
                params["cursor"] = cursor
            response = session.Get(API + path, params=params)
            if response.status_code != 200:
                break
            payload = response.json()
            for market in payload.get("markets", []):
                row = {key: market.get(key) for key in KEEP_FIELDS}
                row["series"] = series
                row["source_endpoint"] = path
                markets[row["ticker"]] = row
            cursor = payload.get("cursor") or ""
            if not cursor or not payload.get("markets"):
                break
    return list(markets.values())


def PullMarkets(session, series_list):
    for series in series_list:
        rows = ListMarkets(session, series)
        out_path = RAW_DIR / "kalshi" / "markets" / f"{series}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as handle:
            for row in sorted(rows, key=lambda r: r["ticker"]):
                handle.write(json.dumps(row, sort_keys=True) + "\n")
        print(f"{series}: {len(rows):,} markets -> {out_path}", flush=True)


def PullMilestones(session, since):
    """Kalshi milestones carry the scheduled start (start_date), round and tournament for each tennis match event."""
    rows, cursor = [], ""
    while True:
        params = {"type": "tennis_tournament_singles", "limit": 500, "minimum_start_date": since}
        if cursor:
            params["cursor"] = cursor
        response = session.Get(API + "/milestones", params=params)
        if response.status_code != 200:
            break
        payload = response.json()
        for item in payload.get("milestones", []):
            rows.append({"id": item.get("id"), "start_date": item.get("start_date"), "end_date": item.get("end_date"),
                         "title": item.get("title"), "details": item.get("details"),
                         "event_tickers": sorted(set((item.get("primary_event_tickers") or []) + (item.get("related_event_tickers") or [])))})
        cursor = payload.get("cursor") or ""
        if not cursor or not payload.get("milestones"):
            break
    out_path = RAW_DIR / "kalshi" / "milestones_tennis.jsonl"
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"milestones: {len(rows):,} -> {out_path}", flush=True)


def LoadStartTimes():
    path = RAW_DIR / "kalshi" / "milestones_tennis.jsonl"
    starts = {}
    if path.exists():
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                for ticker in row["event_tickers"]:
                    starts[ticker] = row["start_date"]
    return starts


def LoadMarkets(series):
    path = RAW_DIR / "kalshi" / "markets" / f"{series}.jsonl"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def PullCandles(session, series_list, date_from, date_to, hours_before):
    """One market per event (the first ticker): its yes side prices player A, and its no side
    (no_ask = 1 - yes_bid) prices player B, so a single book covers both sides.
    One request per event: 1-minute candles T-hours_before..T+5m (older files may also carry an hourly list),
    where T = the Kalshi milestone start_date (scheduled start; tennis starts can slip later)."""
    done = 0
    starts = LoadStartTimes()
    for series in series_list:
        events = {}
        for market in LoadMarkets(series):
            events.setdefault(market["event_ticker"], []).append(market)
        for event_ticker, markets in sorted(events.items()):
            market = sorted(markets, key=lambda m: m["ticker"])[0]
            occurrence = ParseTs(starts.get(event_ticker) or market.get("occurrence_datetime"))
            if occurrence is None or not (date_from <= occurrence.date() <= date_to):
                continue
            if market.get("result") in (None, ""):
                continue
            out_path = RAW_DIR / "kalshi" / "candles" / series / f"{event_ticker}.json"
            if out_path.exists():
                continue
            opened = ParseTs(market["open_time"])
            windows = {"minute": (occurrence - dt.timedelta(hours=hours_before), occurrence + dt.timedelta(minutes=5), 1)}
            payload = {"event_ticker": event_ticker, "ticker": market["ticker"], "series": series,
                       "scheduled_start": occurrence.isoformat(), "start_source": "milestone" if event_ticker in starts else "occurrence_datetime",
                       "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}
            for name, (start, end, period) in windows.items():
                start = max(start, opened)
                if end <= start:
                    payload[name] = []
                    continue
                params = {"start_ts": int(start.timestamp()), "end_ts": int(end.timestamp()), "period_interval": period}
                candles = []
                for endpoint in (f"/historical/markets/{market['ticker']}/candlesticks",
                                 f"/series/{series}/markets/{market['ticker']}/candlesticks"):
                    response = session.Get(API + endpoint, params=params)
                    if response.status_code == 200:
                        candles = response.json().get("candlesticks") or []
                        if candles:
                            break
                payload[name] = [CompactCandle(c) for c in candles]
            WriteJsonAtomic(out_path, payload)
            done += 1
            if done % 250 == 0:
                print(f"  events {done:,} ({series}) last {event_ticker} throttled={session.throttled}", flush=True)
    print(f"events fetched: {done:,}")


def CompactCandle(candle):
    def Close(key):
        block = candle.get(key) or {}
        value = block.get("close") if isinstance(block, dict) else None
        return float(value) if value not in (None, "") else None
    price = candle.get("price") or {}
    return {"end_ts": candle.get("end_period_ts"), "yes_bid": Close("yes_bid"), "yes_ask": Close("yes_ask"),
            "last": float(price["close"]) if price.get("close") not in (None, "") else None,
            "volume": float(candle.get("volume") or 0)}

def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--series", nargs="+", default=MATCH_SERIES)
    parser.add_argument("--markets", action="store_true")
    parser.add_argument("--candles", action="store_true")
    parser.add_argument("--milestones", action="store_true", help="pull scheduled start times (run before --candles)")
    parser.add_argument("--from", dest="date_from", default="2025-06-01")
    parser.add_argument("--to", dest="date_to", default=dt.date.today().isoformat())
    parser.add_argument("--hours-before", type=float, default=24.0)
    parser.add_argument("--interval", type=float, default=0.15, help="seconds between requests (Kalshi public tier allows more)")
    args = parser.parse_args(argv)
    session = PoliteSession(min_interval_s=args.interval)
    if args.markets:
        PullMarkets(session, args.series)
    if args.milestones:
        PullMilestones(session, args.date_from + "T00:00:00Z")
    if args.candles:
        PullCandles(session, args.series, dt.date.fromisoformat(args.date_from), dt.date.fromisoformat(args.date_to), args.hours_before)


if __name__ == "__main__":
    Main()