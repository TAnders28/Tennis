r"""Join Kalshi tennis match events to Sackmann matches and extract pre-match quotes.

Event -> match: both players resolved by name (NameIndex.LookupFull), and the Kalshi
scheduled start date must fall in [tourney_date - 3 days, tourney_date + 16 days]
(Sackmann only stores the event week). Unmatched events are logged with a reason.
One book per event: the fetched market's yes side prices player A; buying B costs
1 - yes_bid(A) (the no side of the same book).

Quotes (1-min candles; yes_bid/yes_ask at minute close, no size):
  T-60  entry quote: last candle closing <= T-60m, age <= 3 min, 0 < bid < ask < 1
  T-5   closing reference: same rule at T-5m
  early earliest candle in [T-24h, T-3h] with both sides (an "opening-ish" price; hourly list if present, else 1-min)
T = Kalshi milestone start_date = the SCHEDULED start. Tennis starts slip ("followed by"),
so T-60 can be several hours before the real first ball; and if a match started EARLY the
quote could be in-play. Events whose mid moved > 0.25 between T-60 and T-5 are flagged
possible_inplay and excluded from bets.

Usage: .venv\Scripts\python.exe -m Backtest.KalshiJoin --tour atp
Writes data/backtest/<tour>/kalshi_quotes.parquet and kalshi_unmatched.csv
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Backtest.NameMatch import NameIndex

REPO_ROOT = Path(__file__).resolve().parents[1]
KALSHI_DIR = REPO_ROOT / "data" / "raw" / "kalshi"
OUT_ROOT = REPO_ROOT / "data" / "backtest"
SERIES_BY_TOUR = {"atp": [("KXATPMATCH", "tour"), ("KXATPCHALLENGERMATCH", "challenger")],
                  "wta": [("KXWTAMATCH", "tour"), ("KXWTACHALLENGERMATCH", "challenger")]}
STALE_S = 180
INPLAY_JUMP = 0.25


def LoadJsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def QuoteAt(candles, target_ts, stale_s=STALE_S):
    best = None
    for candle in candles:
        if candle["end_ts"] is None or candle["end_ts"] > target_ts:
            break
        bid, ask = candle["yes_bid"], candle["yes_ask"]
        if bid is not None and ask is not None and 0 < bid < ask < 1:
            best = candle
    if best is None or target_ts - best["end_ts"] > stale_s:
        return None
    return best


def EarlyQuote(hourly, start_ts):
    for candle in hourly:
        if candle["end_ts"] is None or candle["end_ts"] < start_ts - 24 * 3600:
            continue
        if candle["end_ts"] > start_ts - 3 * 3600:
            break
        bid, ask = candle["yes_bid"], candle["yes_ask"]
        if bid is not None and ask is not None and 0 < bid < ask < 1:
            return candle
    return None


def BuildQuotes(tour, preds):
    index = NameIndex(pd.concat([preds[["winner_id", "winner_name"]].set_axis(["id", "name"], axis=1),
                                 preds[["loser_id", "loser_name"]].set_axis(["id", "name"], axis=1)])
                      .drop_duplicates("id").itertuples(index=False, name=None))
    by_pair = {}
    dates = preds["date"].to_numpy()
    for i, (w, l) in enumerate(zip(preds["winner_id"].to_numpy(), preds["loser_id"].to_numpy())):
        by_pair.setdefault(frozenset((int(w), int(l))), []).append(i)
    milestones = {}
    for row in LoadJsonl(KALSHI_DIR / "milestones_tennis.jsonl"):
        for ticker in row["event_tickers"]:
            milestones[ticker] = row
    rows, unmatched = [], []
    for series, kalshi_level in SERIES_BY_TOUR[tour]:
        markets = LoadJsonl(KALSHI_DIR / "markets" / f"{series}.jsonl")
        events = {}
        for market in markets:
            events.setdefault(market["event_ticker"], []).append(market)
        candle_dir = KALSHI_DIR / "candles" / series
        for event_ticker, ms in events.items():
            path = candle_dir / f"{event_ticker}.json"
            if not path.exists():
                continue
            payload = json.loads(path.read_text())
            ms = sorted(ms, key=lambda m: m["ticker"])
            market_a = next((m for m in ms if m["ticker"] == payload["ticker"]), ms[0])
            others = [m for m in ms if m["ticker"] != market_a["ticker"]]
            name_a = market_a.get("yes_sub_title")
            name_b = others[0].get("yes_sub_title") if others else market_a.get("no_sub_title")
            start = dt.datetime.fromisoformat(payload["scheduled_start"])
            base = {"event_ticker": event_ticker, "series": series, "kalshi_level": kalshi_level,
                    "name_a": name_a, "name_b": name_b, "scheduled_start": start.isoformat()}
            ids_a, ids_b = index.LookupFull(name_a), index.LookupFull(name_b)
            if not ids_a or not ids_b:
                unmatched.append({**base, "reason": "name_a" if not ids_a else "name_b"})
                continue
            start_day = np.datetime64(start.date())
            found = []
            for a in ids_a:
                for b in ids_b:
                    for i in by_pair.get(frozenset((int(a), int(b))), []):
                        lag = (start_day - dates[i].astype("datetime64[D]")).astype(int)
                        if -3 <= lag <= 16:
                            found.append((abs(lag - 3), i, int(a)))
            if not found:
                unmatched.append({**base, "reason": "no_sackmann_match_in_window"})
                continue
            found.sort()
            _, i, id_a = found[0]
            start_ts = start.timestamp()
            q60 = QuoteAt(payload.get("minute", []), start_ts - 3600)
            q5 = QuoteAt(payload.get("minute", []), start_ts - 300)
            early = EarlyQuote(payload.get("hourly") or payload.get("minute", []), start_ts)
            row = {**base, "row": int(i), "match_key": preds["match_key"].iat[i], "id_a": id_a,
                   "a_won": int(preds["winner_id"].iat[i]) == id_a, "kalshi_result_a": market_a.get("result"),
                   "candidates": len(found)}
            for label, quote in (("t60", q60), ("t5", q5), ("early", early)):
                row[f"{label}_bid"] = quote["yes_bid"] if quote else np.nan
                row[f"{label}_ask"] = quote["yes_ask"] if quote else np.nan
                row[f"{label}_age_s"] = (start_ts - (3600 if label == "t60" else 300) - quote["end_ts"]) if (quote and label != "early") else np.nan
            mid60 = (row["t60_bid"] + row["t60_ask"]) / 2
            mid5 = (row["t5_bid"] + row["t5_ask"]) / 2
            row["possible_inplay"] = bool(np.isfinite(mid60) and np.isfinite(mid5) and abs(mid5 - mid60) > INPLAY_JUMP)
            rows.append(row)
    quotes = pd.DataFrame(rows)
    if len(quotes):
        quotes["result_agrees"] = np.where(quotes["kalshi_result_a"].isin(["yes", "no"]),
                                           (quotes["kalshi_result_a"] == "yes") == quotes["a_won"], np.nan)
        quotes = quotes.sort_values("candidates").drop_duplicates("match_key", keep="first")
    return quotes, pd.DataFrame(unmatched)


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tour", choices=["atp", "wta"], required=True)
    args = parser.parse_args(argv)
    out_dir = OUT_ROOT / args.tour
    preds = pd.read_parquet(out_dir / "predictions.parquet")
    quotes, unmatched = BuildQuotes(args.tour, preds)
    quotes.to_parquet(out_dir / "kalshi_quotes.parquet", index=False)
    unmatched.to_csv(out_dir / "kalshi_unmatched.csv", index=False)
    total = len(quotes) + len(unmatched)
    print(f"{args.tour}: events with candles {total:,}; matched {len(quotes):,} ({len(quotes) / max(total, 1):.1%}); "
          f"unmatched {len(unmatched):,}")
    if len(unmatched):
        print(unmatched.groupby(["kalshi_level", "reason"]).size().to_string())
    if len(quotes):
        print("result agreement:", quotes["result_agrees"].mean(), " t60 quote present:", quotes["t60_ask"].notna().mean(),
              " possible in-play flagged:", quotes["possible_inplay"].mean())


if __name__ == "__main__":
    Main()