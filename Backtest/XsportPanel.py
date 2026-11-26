"""Cross-sport look-back panel for tennis (Sportsbot, 2026-11-26). Read-only on data; writes one CSV.

One row per Kalshi tennis match event with a settled yes/no result and a pre-match quote. Columns follow the
cross-sport panel contract read by basketball `Analysis.CrossSportHistory` ("home" = player A = the YES side of the
candle file, "away" = player B):
  date (ET day of the scheduled start), sport, game_id (event ticker), season, split (tune < 2026-01-01 <= holdout),
  y (1 = A won, from the Kalshi settlement), k_bid/k_ask (A YES at scheduled start - 60 min, <= 30 min old),
  k2_bid/k2_ask (A at T-5), p_kalshi (A T-60 mid), p_kalshi5 (A T-5 mid),
  p_elo / p_blend (frozen walk-forward surface Elo and Elo+serve/return blend, from data/backtest/<tour>/predictions.parquet;
  only for events matched to Sackmann results, which end 2026-06-02), ctx_* (tour, level, surface, round, best_of,
  ATP/WTA ranks from the Sackmann CSVs, elo_diff), has_model.
Excluded: events whose T-60 and T-5 mids differ by > 0.25 (possible in-play). Everything here is tested:false.
"""
import argparse
import csv
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from Backtest.KalshiJoin import INPLAY_JUMP, KALSHI_DIR, LoadJsonl, OUT_ROOT, QuoteAt, SERIES_BY_TOUR

ET = ZoneInfo("America/New_York")
SACK = Path(__file__).resolve().parents[1] / "data" / "raw" / "sackmann"
SPLIT = dt.date(2026, 1, 1)


def RankTable(tour):
    out = {}
    for p in sorted((SACK / tour).glob(f"{tour}_matches*_202[456].csv")):
        try:
            d = pd.read_csv(p, low_memory=False, usecols=["tourney_id", "round", "winner_id", "loser_id", "winner_rank", "loser_rank"])
        except ValueError:
            continue
        for r in d.itertuples(index=False):
            out[f"{tour}:{r.tourney_id}:{r.round}:{r.winner_id}-{r.loser_id}"] = (r.winner_rank, r.loser_rank)
    return out


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    stats = {}
    for tour, series_list in SERIES_BY_TOUR.items():
        preds = pd.read_parquet(OUT_ROOT / tour / "predictions.parquet",
                                columns=["match_key", "winner_id", "loser_id", "p_elo", "p_blend", "surface_norm", "level_tag",
                                         "round", "best_of", "tourney_level", "elo_diff", "date"]).drop_duplicates("match_key").set_index("match_key")
        quotes = pd.read_parquet(OUT_ROOT / tour / "kalshi_quotes.parquet").set_index("event_ticker")
        ranks = RankTable(tour)
        n_model = n_mkt = n_inplay = n_noresult = 0
        for series, level in series_list:
            mpath = KALSHI_DIR / "markets" / f"{series}.jsonl"
            if not mpath.exists() or mpath.stat().st_size == 0:
                continue
            events = {}
            for m in LoadJsonl(mpath):
                events.setdefault(m["event_ticker"], []).append(m)
            cdir = KALSHI_DIR / "candles" / series
            for ev, ms in events.items():
                path = cdir / f"{ev}.json"
                if not path.exists():
                    continue
                payload = json.loads(path.read_text())
                ms = sorted(ms, key=lambda m: m["ticker"])
                ma = next((m for m in ms if m["ticker"] == payload["ticker"]), ms[0])
                if ma.get("result") not in ("yes", "no"):
                    n_noresult += 1
                    continue
                start = dt.datetime.fromisoformat(payload["scheduled_start"])
                ts = start.timestamp()
                q60 = QuoteAt(payload.get("minute", []), ts - 3600)
                q5 = QuoteAt(payload.get("minute", []), ts - 300)
                if not q60:
                    continue
                mid60 = (q60["yes_bid"] + q60["yes_ask"]) / 2
                mid5 = (q5["yes_bid"] + q5["yes_ask"]) / 2 if q5 else None
                if mid5 is not None and abs(mid5 - mid60) > INPLAY_JUMP:
                    n_inplay += 1
                    continue
                d = start.astimezone(ET).date()
                others = [m for m in ms if m["ticker"] != ma["ticker"]]
                row = {"date": d.isoformat(), "sport": "TENNIS", "game_id": ev, "season": d.year,
                       "split": "holdout" if d >= SPLIT else "tune", "home": ma.get("yes_sub_title"),
                       "away": others[0].get("yes_sub_title") if others else ma.get("no_sub_title"),
                       "y": int(ma["result"] == "yes"), "k_bid": q60["yes_bid"], "k_ask": q60["yes_ask"],
                       "k_time": dt.datetime.fromtimestamp(q60["end_ts"], tz=dt.timezone.utc).isoformat(), "p_kalshi": mid60,
                       "ctx_tour": tour, "ctx_level": level, "has_model": 0}
                if q5:
                    row["k2_bid"], row["k2_ask"], row["p_kalshi5"] = q5["yes_bid"], q5["yes_ask"], mid5
                if ev in quotes.index:
                    q = quotes.loc[ev]
                    if isinstance(q, pd.DataFrame):
                        q = q.iloc[0]
                    mk = q["match_key"]
                    if mk in preds.index and q.get("result_agrees", 1.0) == 1.0:
                        p = preds.loc[mk]
                        a_is_w = int(q["id_a"]) == int(p["winner_id"])
                        row["p_elo"] = float(p["p_elo"] if a_is_w else 1 - p["p_elo"])
                        row["p_blend"] = float(p["p_blend"] if a_is_w else 1 - p["p_blend"])
                        row["ctx_surface"] = p["surface_norm"]
                        row["ctx_round"] = p["round"]
                        row["ctx_best_of"] = int(p["best_of"])
                        row["ctx_sack_level"] = p["level_tag"]
                        row["ctx_elo_diff_a"] = float(p["elo_diff"] if a_is_w else -p["elo_diff"])
                        rk = ranks.get(mk)
                        if rk:
                            wr, lr = rk
                            row["ctx_rank_a"], row["ctx_rank_b"] = (wr, lr) if a_is_w else (lr, wr)
                        row["has_model"] = 1
                        n_model += 1
                n_mkt += 1
                rows.append(row)
        stats[tour] = {"events": n_mkt, "with_model": n_model, "possible_inplay_excluded": n_inplay, "no_result": n_noresult}
    rows.sort(key=lambda r: (r["date"], r["game_id"]))
    fields = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"rows": len(rows), "first": rows[0]["date"] if rows else None, "last": rows[-1]["date"] if rows else None, **stats}))


if __name__ == "__main__":
    Main()