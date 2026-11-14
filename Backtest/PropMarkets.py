"""Kalshi tennis-prop contract parsers, vig/hold, and open-to-close mid moves.

No network I/O. Series coverage constants are from the 2026-11-14 public catalog audit.
"""
from __future__ import annotations

import re

from Backtest.KalshiJoin import QuoteAt, STALE_S

# Current (live or recently active) series, plus retired aliases.
SERIES = {
    "total_games": ("KXATPGTOTAL", "KXWTAGTOTAL", "KXATPGAMETOTAL"),
    "game_handicap": ("KXATPGSPREAD", "KXATPGAMESPREAD"),
    "set_betting": ("KXATPSETWINNER", "KXWTASETWINNER"),
    "exact_score": ("KXATPEXACTMATCH", "KXWTAEXACTMATCH"),
    "set_handicap": ("KXATPSSPREAD",),
    "total_sets": ("KXATPTOTALSETS",),
    "aces": ("KXATPACES", "KXWTAACES"),
    "double_faults": (),  # no series in the 2026-11-14 catalog
}

# Prefer the current ticker when a retired alias also exists.
PRIMARY = {
    "KXATPGAMETOTAL": "KXATPGTOTAL",
    "KXATPGAMESPREAD": "KXATPGSPREAD",
}

OVER_GAMES = re.compile(r"over\s+(\d+(?:\.\d+)?)\s+games", re.I)
OVER_SETS = re.compile(r"over\s+(\d+(?:\.\d+)?)\s+(?:total\s+)?sets", re.I)
HANDICAP = re.compile(r"(.+?)\s+(-?\d+(?:\.\d+)?)\s+games", re.I)
SET_HANDICAP = re.compile(r"wins by over\s+(\d+(?:\.\d+)?)\s+sets", re.I)
EXACT = re.compile(r"wins\s+(\d+)\s*-\s*(\d+)", re.I)
SET_WINNER = re.compile(r"win set\s+(\d+)", re.I)
ACES_AT_LEAST = re.compile(r"at least\s+(\d+)\s+total match aces|:\s*(\d+)\+", re.I)
TICKER_TOTAL = re.compile(r"-(\d+)$")
TICKER_EXACT = re.compile(r"-([A-Z]{2,})(\d)(\d)$")
TICKER_SPREAD = re.compile(r"-([A-Z]{2,})(\d+)$")
TICKER_SET = re.compile(r"-(\d)-([A-Z0-9]+)$")


def ValidBook(bid, ask):
    return bid is not None and ask is not None and 0.0 < bid < ask < 1.0


def Hold(yes_bid, yes_ask):
    """Kalshi two-sided hold at the asks equals the spread: yes_ask + no_ask - 1 = ask - bid."""
    if not ValidBook(yes_bid, yes_ask):
        return None
    spread = yes_ask - yes_bid
    take_both = yes_ask + (1.0 - yes_bid)  # 1 + spread
    return {"spread": spread, "ask_overround": take_both - 1.0, "yes_bid": yes_bid, "yes_ask": yes_ask}


def Mid(yes_bid, yes_ask):
    if not ValidBook(yes_bid, yes_ask):
        return None
    return 0.5 * (yes_bid + yes_ask)


def ParsePropMarket(row):
    """Identify a tennis prop contract from a Kalshi market or candle sidecar row."""
    ticker = row.get("ticker") or ""
    title = row.get("title") or ""
    yes = row.get("yes_sub_title") or ""
    series = row.get("series") or row.get("series_ticker") or ticker.split("-")[0]
    blob = f"{title} {yes}"
    event = row.get("event_ticker")
    parsed = {"ticker": ticker, "event_ticker": event, "series": series, "title": title,
              "yes_sub_title": yes, "kind": None, "side_label": None, "line": None,
              "player": None, "sets_a": None, "sets_b": None, "set_no": None}

    m = OVER_GAMES.search(blob)
    if m and "total" in (series or "").lower() or (m and "games" in blob.lower() and "spread" not in blob.lower()):
        if m:
            parsed.update(kind="total_games", line=float(m.group(1)), side_label="over")
            return parsed
    m = OVER_SETS.search(blob)
    if m:
        parsed.update(kind="total_sets", line=float(m.group(1)), side_label="over")
        return parsed
    m = HANDICAP.search(blob)
    if m and "set" not in blob.lower():
        parsed.update(kind="game_handicap", player=m.group(1).strip(), line=float(m.group(2)), side_label="cover")
        return parsed
    m = SET_HANDICAP.search(blob)
    if m:
        player = (yes.split("wins")[0].strip() if "wins" in yes.lower() else yes) or title
        parsed.update(kind="set_handicap", player=player, line=float(m.group(1)), side_label="cover")
        return parsed
    m = EXACT.search(blob)
    if m:
        parsed.update(kind="exact_score", sets_a=int(m.group(1)), sets_b=int(m.group(2)),
                      player=(yes.split("wins")[0].strip() if "wins" in yes.lower() else None),
                      side_label="exact")
        return parsed
    m = SET_WINNER.search(blob)
    if m:
        parsed.update(kind="set_betting", set_no=int(m.group(1)), player=yes.strip() or None, side_label="win_set")
        return parsed
    m = ACES_AT_LEAST.search(blob)
    if m:
        k = m.group(1) or m.group(2)
        player = (title.split("record")[0].replace("Will", "").strip() if "record" in title.lower() else yes)
        player = re.sub(r":\s*\d+\+\s*$", "", player).strip()
        parsed.update(kind="aces", line=float(k), player=player, side_label="at_least")
        return parsed
    # ticker fallbacks
    if "GTOTAL" in series or "GAMETOTAL" in series:
        tm = TICKER_TOTAL.search(ticker)
        if tm:
            parsed.update(kind="total_games", line=int(tm.group(1)) - 0.5, side_label="over")
            return parsed
    return parsed


def OpenToClose(candles, open_ts, start_ts, stale_s=STALE_S):
    """Return (open_mid, close_mid, move) from 1-min candles. close is last valid mid at or before start.

    Candles may not exist for prop series on disk; callers should treat missing as unscored.
    """
    if not candles or open_ts is None:
        return None
    first = QuoteAt(candles, open_ts, stale_s=stale_s) if open_ts else None
    # QuoteAt looks at or before target; for the open mid, walk forward to the first valid book at/after open
    open_mid = None
    if first and ValidBook(first.get("yes_bid"), first.get("yes_ask")) and first.get("end_ts", 0) >= open_ts - 60:
        open_mid = Mid(first["yes_bid"], first["yes_ask"])
    if open_mid is None:
        for candle in candles:
            end = candle.get("end_ts")
            if end is None or end < open_ts:
                continue
            if ValidBook(candle.get("yes_bid"), candle.get("yes_ask")):
                open_mid = Mid(candle["yes_bid"], candle["yes_ask"])
                break
    close = QuoteAt(candles, start_ts, stale_s=stale_s) if start_ts else None
    close_mid = Mid(close["yes_bid"], close["yes_ask"]) if close and ValidBook(close.get("yes_bid"), close.get("yes_ask")) else None
    if open_mid is None or close_mid is None:
        return {"open_mid": open_mid, "close_mid": close_mid, "move": None}
    return {"open_mid": open_mid, "close_mid": close_mid, "move": close_mid - open_mid}