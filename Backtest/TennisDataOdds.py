r"""tennis-data.co.uk odds: join to Sackmann matches and score vs. bookmaker prices.

Needs Tim's manual downloads (see DataPull.PullTennisData) ingested to
data/processed/tennis_data_<tour>.parquet. If that file is missing this module says so and exits 0.

Columns and what "closing" means (per tennis-data.co.uk's notes; re-check notes.txt when downloaded):
  PSW/PSL    Pinnacle odds; treated as Pinnacle CLOSING (the sharp reference)
  MaxW/MaxL  best price across books (Oddsportal); OPTIMISTIC - nobody reliably gets every max
  AvgW/AvgL  average across books (Oddsportal); fallback market when PS is missing
  B365W/L    Bet365 pre-match
There is no opening price, so true open->close CLV is NOT measurable from tennis-data alone;
Kalshi early-vs-T-5 quotes (Backtest.KalshiReality) are the only open->close measure here.

Join: names (NameIndex.LookupTennisData, all candidate ids) + tennis-data Date within
[tourney_date - 3, tourney_date + 16] days. Matches without odds still feed the ratings; they are just not scored for P&L.
Betting: edge = p_model - 1/decimal_odds - 0.02 (Tim's estimate) > threshold; cost per $1 contract = 1/odds + 0.02.

Usage: .venv\Scripts\python.exe -m Backtest.TennisDataOdds --tour atp
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from Backtest.NameMatch import NameIndex
from Backtest.Odds import DevigPower, DevigProportional, TIM_FEE

REPO_ROOT = Path(__file__).resolve().parents[1]
BT_DIR = REPO_ROOT / "data" / "backtest"
PROCESSED = REPO_ROOT / "data" / "processed"
THRESHOLDS = [0.0, 0.03, 0.05]


def Join(tour, preds, td):
    index = NameIndex(pd.concat([preds[["winner_id", "winner_name"]].set_axis(["i", "n"], axis=1),
                                 preds[["loser_id", "loser_name"]].set_axis(["i", "n"], axis=1)])
                      .drop_duplicates("i").itertuples(index=False, name=None))
    by_pair = {}
    for i, (w, l) in enumerate(zip(preds["winner_id"].to_numpy(), preds["loser_id"].to_numpy())):
        by_pair.setdefault((int(w), int(l)), []).append(i)
    dates = preds["date"].to_numpy().astype("datetime64[D]")
    cache = {}
    rows = []
    for k, r in enumerate(td.itertuples(index=False)):
        for name in (r.Winner, r.Loser):
            if name not in cache:
                cache[name] = index.LookupTennisData(name)
        hits = []
        if pd.notna(r.Date):
            day = np.datetime64(pd.Timestamp(r.Date).date())
            for w in cache[r.Winner]:
                for l in cache[r.Loser]:
                    for i in by_pair.get((int(w), int(l)), []):
                        lag = int((day - dates[i]).astype(int))
                        if -3 <= lag <= 16:
                            hits.append((abs(lag - 3), i))
        rows.append({"td_row": k, "row": min(hits)[1] if hits else -1, "n_hits": len(hits)})
    joined = pd.DataFrame(rows)
    return joined


def Score(tour):
    td_path = PROCESSED / f"tennis_data_{tour}.parquet"
    if not td_path.exists():
        print(f"{tour}: no tennis-data file ({td_path}); download manually, run DataPull.PullTennisData --ingest. Skipping.")
        return None
    preds = pd.read_parquet(BT_DIR / tour / "predictions.parquet")
    preds = preds[preds["level_tag"] == "tour"].reset_index(drop=True)
    td = pd.read_parquet(td_path)
    td = td[~td["Comment"].fillna("").str.contains("Walkover|W/O", case=False)].reset_index(drop=True)
    joined = Join(tour, preds, td)
    td["row"] = joined["row"].to_numpy()
    rate = td.assign(matched=td["row"] >= 0).groupby("td_year")["matched"].agg(["size", "mean"])
    rate.to_csv(BT_DIR / tour / "tennis_data_match_rate.csv")
    print(f"{tour} tennis-data match rate by year:\n{rate.to_string()}")
    m = td[td["row"] >= 0].drop_duplicates("row").copy()
    p = preds.iloc[m["row"].to_numpy()]
    for col in ("season", "split", "p_blend", "p_elo", "p_sr"):
        m[col] = p[col].to_numpy()
    out = []
    for (season, split), part in m.groupby(["season", "split"]):
        row = {"tour": tour, "season": int(season), "split": split, "n_matched": len(part)}
        ps_ok = part["PSW"].gt(1) & part["PSL"].gt(1)
        avg_ok = part["AvgW"].gt(1) & part["AvgL"].gt(1)
        mk = np.full(len(part), np.nan)
        src = np.where(ps_ok, "ps", np.where(avg_ok, "avg", ""))
        for j, (pw, pl, aw, al, s) in enumerate(zip(part["PSW"], part["PSL"], part["AvgW"], part["AvgL"], src)):
            if s == "ps":
                mk[j] = DevigPower(1 / pw, 1 / pl)[0]
            elif s == "avg":
                mk[j] = DevigPower(1 / aw, 1 / al)[0]
        ok = np.isfinite(mk)
        pm = part["p_blend"].to_numpy()
        row.update({"n_market": int(ok.sum()), "market_source_ps_share": float((src[ok] == "ps").mean()) if ok.any() else np.nan,
                    "market_log_loss": float(-np.mean(np.log(mk[ok]))) if ok.any() else np.nan,
                    "market_fav_acc": float(np.mean(mk[ok] > 0.5)) if ok.any() else np.nan,
                    "model_log_loss_same": float(-np.mean(np.log(np.clip(pm[ok], 1e-6, 1)))) if ok.any() else np.nan,
                    "model_acc_same": float(np.mean(pm[ok] > 0.5)) if ok.any() else np.nan})
        for book, cw, cl in (("pinnacle_close", "PSW", "PSL"), ("max_optimistic", "MaxW", "MaxL")):
            for th in THRESHOLDS:
                n = pnl = cost = 0.0
                for pw_model, ow, ol in zip(pm, part[cw], part[cl]):
                    if not (ow > 1 and ol > 1):
                        continue
                    edge_w = pw_model - 1 / ow - TIM_FEE
                    edge_l = (1 - pw_model) - 1 / ol - TIM_FEE
                    if max(edge_w, edge_l) <= th:
                        continue
                    price = 1 / ow if edge_w >= edge_l else 1 / ol
                    c = price + TIM_FEE
                    n += 1
                    cost += c
                    pnl += (1 - c) if edge_w >= edge_l else -c
                row[f"{book}_th{th:g}_n"] = int(n)
                row[f"{book}_th{th:g}_roi"] = pnl / cost if cost else np.nan
        out.append(row)
    result = pd.DataFrame(out)
    result.to_csv(BT_DIR / tour / "tennis_data_scores.csv", index=False)
    print(result.to_string(index=False))
    return result


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tour", choices=["atp", "wta"], required=True)
    args = parser.parse_args(argv)
    Score(args.tour)


if __name__ == "__main__":
    Main()