r"""Re-score the Kalshi-reality backtest on Jun-Nov 2026, using ratings kept current from Kalshi settled results.

Leak-free: one chronological walk (Sackmann through early June 2026, then Kalshi results by ET day);
each match is predicted from the state before its day's group. Frozen params, no re-tuning.
Serve/return is frozen at the last Sackmann stats (no point stats after June), so p_sr / blend for
June+ matches carry stale serve/return info; Elo is the cleanest read. New players start from the
level default (KalshiResults.NewPlayerSeeds, computed from pre-June ratings only).
Quotes and bet rules are exactly KalshiJoin / KalshiReality (T-60 1-min candle, 30-min staleness, strict
3-min subset, in-play jump filter, ask and mid fills, Tim $0.02 est and Kalshi 0.07p(1-p) fees).

Usage: .venv\Scripts\python.exe -m Backtest.KalshiRecent --from 2026-06-01 --to 2026-11-12
Writes data/backtest/kalshi_recent/{market_compare,summary}.csv and bets.parquet
"""
from __future__ import annotations

import argparse
import datetime as dt
import json

import numpy as np
import pandas as pd

from Backtest.Elo import EloParams
from Backtest.Engine import Blend, RunModels
from Backtest.FrozenParams import FROZEN
from Backtest.KalshiJoin import INPLAY_JUMP, KALSHI_DIR, EarlyQuote, QuoteAt
from Backtest.KalshiReality import BT_DIR, ET, MODELS, THRESHOLDS, MarketCompare, SimBets, StrictSubset, Summary
from Backtest.KalshiResults import SERIES, BuildKalshiRows, Combine, NewPlayerSeeds
from Backtest.LoadMatches import LoadMatches
from Backtest.Odds import FEES
from Backtest.ServeReturn import ServeReturnParams

OUT_DIR = BT_DIR / "kalshi_recent"


def TourFrame(tour, date_from, date_to):
    sackmann = LoadMatches(tour)
    params = FROZEN[tour]
    elo_p, sr_p = EloParams(**params["elo"]), ServeReturnParams(**params["sr"])
    kalshi, cover = BuildKalshiRows(tour, sackmann, until=date_to)
    pre = RunModels(sackmann, elo_p, None, tour)
    seeds, _ = NewPlayerSeeds(pre["elo"], sackmann, kalshi)
    frame = Combine(sackmann, kalshi)
    result = RunModels(frame, elo_p, sr_p, tour, seed_ratings=seeds)
    frame["p_elo"], frame["p_sr"] = result["p_elo"], result["p_sr"]
    frame["p_blend"] = [Blend(e, s, params["blend_weight_sr"]) for e, s in zip(frame["p_elo"], frame["p_sr"])]
    recent = frame[(frame["file_kind"] == "kalshi") & (frame["date"] >= pd.Timestamp(date_from))
                   & (frame["date"] <= pd.Timestamp(date_to))]
    recent = recent.merge(kalshi[["match_key", "event_ticker", "winner_new", "loser_new"]], on="match_key", how="left")
    series_of = {}
    for series, _level in SERIES[tour]:
        for path in (KALSHI_DIR / "candles" / series).glob("*.json"):
            series_of[path.stem] = path
    rows, missing = [], 0
    for r in recent.itertuples():
        path = series_of.get(r.event_ticker)
        if path is None:
            missing += 1
            continue
        payload = json.loads(path.read_text())
        start = dt.datetime.fromisoformat(payload["scheduled_start"])
        start_ts = start.timestamp()
        q60 = QuoteAt(payload.get("minute", []), start_ts - 3600)
        q5 = QuoteAt(payload.get("minute", []), start_ts - 300)
        early = EarlyQuote(payload.get("hourly") or payload.get("minute", []), start_ts)
        rows.append({"match_key": r.match_key, "event_ticker": r.event_ticker, "ticker_a": payload.get("ticker"),
                     "tour": tour, "level_tag": r.level_tag, "split": "kalshi_2026H2", "date": r.date,
                     "scheduled_start": start.isoformat(), "p_elo": r.p_elo, "p_sr": r.p_sr, "p_blend": r.p_blend,
                     "new_player": bool(r.winner_new or r.loser_new),
                     **{f"{lab}_{k}": (q[f"yes_{k}"] if q else np.nan) for lab, q in (("t60", q60), ("t5", q5), ("early", early)) for k in ("bid", "ask")},
                     "t60_age_s": (start_ts - 3600 - q60["end_ts"]) if q60 else np.nan})
    quotes = pd.DataFrame(rows)
    # which side is market A? payload ticker's market; its yes_sub_title vs the winner
    winners = {}
    for series, _level in SERIES[tour]:
        with open(KALSHI_DIR / "markets" / f"{series}.jsonl", encoding="utf-8") as handle:
            for line in handle:
                m = json.loads(line)
                winners[m["ticker"]] = m.get("result")
    quotes["y_a"] = (quotes["ticker_a"].map(winners) == "yes").astype(float)
    for model in MODELS:  # p_* is the winner's probability
        quotes[f"pa_{model}"] = np.where(quotes["y_a"] == 1, quotes[f"p_{model}"], 1 - quotes[f"p_{model}"])
    for label in ("t60", "t5", "early"):
        quotes[f"{label}_mid"] = (quotes[f"{label}_bid"] + quotes[f"{label}_ask"]) / 2
    quotes["possible_inplay"] = (quotes["t60_mid"] - quotes["t5_mid"]).abs().gt(INPLAY_JUMP).fillna(False)
    quotes["date_et"] = pd.to_datetime(quotes["scheduled_start"], utc=True).dt.tz_convert(ET).dt.date
    stats = {"tour": tour, "kalshi_result_rows": int(len(recent)), "with_candles": int(len(quotes)), "no_candles": missing,
             "t60_quote": int(quotes["t60_ask"].notna().sum()), "results_cover": cover}
    return quotes, stats


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="date_from", default="2026-06-01")
    parser.add_argument("--to", dest="date_to", default="2026-11-12")
    args = parser.parse_args(argv)
    parts, stats = [], []
    for tour in ("atp", "wta"):
        q, s = TourFrame(tour, args.date_from, args.date_to)
        parts.append(q)
        stats.append(s)
        print(s, flush=True)
    frame = pd.concat(parts, ignore_index=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    market = MarketCompare(frame)
    market.to_csv(OUT_DIR / "market_compare.csv", index=False)
    bets = []
    for model in MODELS:
        for entry in ("t60", "early"):
            for fill in ("ask", "mid"):
                for fee_name in FEES:
                    for threshold in THRESHOLDS:
                        bets.append(SimBets(frame, model, entry, fill, fee_name, threshold))
    bets = pd.concat(bets, ignore_index=True)
    bets = pd.concat([bets, StrictSubset(bets, frame)], ignore_index=True)
    bets.to_parquet(OUT_DIR / "bets.parquet", index=False)
    summary = Summary(bets)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    (OUT_DIR / "stats.json").write_text(json.dumps(stats, indent=1, default=str))
    pd.set_option("display.width", 250)
    print(market.to_string(index=False))
    view = summary[(summary.entry.isin(["t60", "t60_strict3min"])) & (summary.threshold == 0.0) & summary.model.isin(["blend", "elo"])]
    print(view[["tour", "level", "model", "entry", "fill", "fee_basis", "n_bets", "roi", "avg_edge", "clv_mean"]].to_string(index=False))


if __name__ == "__main__":
    Main()