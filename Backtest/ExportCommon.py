r"""Export the tennis backtest in the village common formats (Degen Bot sports-ops plan, 2026-11-13, sections 2-5).

  .venv\Scripts\python.exe -m Backtest.ExportCommon --date 2026-11-13

Inputs (already leak-free, frozen params): data/backtest/<tour>/predictions.parquet + kalshi_quotes.parquet
(holdout Jun-Dec 2025, forward Jan-Jun 2026), and the Jun-Nov 2026 Kalshi-results extension
(Backtest.KalshiRecent; Elo current, serve/return frozen) written to SEPARATE "_ext-2026H2" files.
Model: frozen blend (tennis_blend_v1). price_basis kalshi_T60: last 1-min candle at or before T-60 (<= 30 min old).
Fixed side player_a = alphabetically first surname (last name token). market_prob_devig = mid of player_a's book
(one book, both sides sum to 1). Fill = the bet side's ask (B's ask = 1 - A's bid) + Tim's $0.02.
Bets: tour level only. tour_qual and challenger appear in calibration only, labelled in the model name.
Writes E:\AgentDocs\Tennibot\<date>_calibration\ and <date>_bankroll-10k\.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from Backtest.FrozenParams import MODEL_VERSION
from Backtest.KalshiReality import ET, LoadTour
from Backtest.NameMatch import Normalise
from Backtest.Odds import KalshiFee

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTDOCS = Path(r"E:\AgentDocs\Tennibot")
FEE = 0.02
QUAL = 0.03
START = 10_000.0
FLAT = 100.0
MODEL = "tennis_blend_v1"
MARKET = "market_kalshi_T60"
BINS = np.linspace(0, 1, 11)
RNG = np.random.default_rng(20261113)


def Surname(name):
    toks = Normalise(name).split()
    return toks[-1] if toks else ""


def Prepare(frame):
    f = frame[frame["t60_ask"].notna() & frame["t60_bid"].notna()].copy()
    dropped_inplay = int(f["possible_inplay"].sum())
    f = f[~f["possible_inplay"]]
    # fixed side: alphabetically first surname -> player_a
    swap = np.array([(Surname(b), b) < (Surname(a), a) for a, b in zip(f["name_a"], f["name_b"])])
    f["player_a"] = np.where(swap, f["name_b"], f["name_a"])
    f["player_b"] = np.where(swap, f["name_a"], f["name_b"])
    mid = (f["t60_bid"] + f["t60_ask"]) / 2
    f["p_model"] = np.where(swap, 1 - f["pa_blend"], f["pa_blend"])
    f["p_market"] = np.where(swap, 1 - mid, mid)
    f["ask_a"] = np.where(swap, 1 - f["t60_bid"], f["t60_ask"])
    f["ask_b"] = np.where(swap, f["t60_ask"], 1 - f["t60_bid"])
    f["y"] = np.where(swap, 1 - f["y_a"], f["y_a"])
    start = pd.to_datetime(f["scheduled_start"], utc=True)
    f["start_time"] = start.dt.tz_convert(ET).map(lambda t: t.isoformat())
    f["price_time"] = (start - pd.to_timedelta(3600 + f["t60_age_s"], unit="s")).dt.tz_convert(ET).map(lambda t: t.isoformat())
    f["date"] = start.dt.tz_convert(ET).dt.date.astype(str)
    f["season_label"] = start.dt.tz_convert(ET).dt.year.astype(str)
    return f.sort_values("start_time").reset_index(drop=True), dropped_inplay


def Calibration(f, sport, model_name, p_col, season, split, last_date):
    p, y = f[p_col].to_numpy(), f["y"].to_numpy()
    rows, ece = [], 0.0
    idx = np.clip(np.digitize(p, BINS[1:-1]), 0, 9)
    for b in range(10):
        m = idx == b
        n = int(m.sum())
        row = {"date": last_date, "sport": sport, "series": f"{sport}_{model_name}", "model": model_name, "season": season, "split": split,
               "bin_lo": round(BINS[b], 1), "bin_hi": round(BINS[b + 1], 1), "n": n,
               "mean_pred": round(float(p[m].mean()), 6) if n else None, "obs_rate": round(float(y[m].mean()), 6) if n else None}
        if n:
            ece += n / len(p) * abs(p[m].mean() - y[m].mean())
        rows.append(row)
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    summary = {"date": last_date, "sport": sport, "series": f"{sport}_{model_name}", "model": model_name, "season": season, "split": split,
               "n": len(p), "accuracy": round(float(np.mean((p > 0.5) == (y == 1))), 6), "brier": round(float(np.mean((p - y) ** 2)), 6),
               "log_loss": round(float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))), 6), "ece": round(float(ece), 6)}
    return rows, summary


def Edges(f, sport, split_col="split"):
    ea = f["p_model"] - f["p_market"] - FEE
    eb = (1 - f["p_model"]) - (1 - f["p_market"]) - FEE
    take_a = ea >= eb
    out = pd.DataFrame({
        "date": f["date"], "start_time": f["start_time"], "sport": sport, "season": f["season_label"], "split": f[split_col],
        "event_id": f["event_ticker"], "event": f["player_a"] + " vs " + f["player_b"],
        "side": np.where(take_a, f["player_a"], f["player_b"]), "model": MODEL,
        "model_prob": np.where(take_a, f["p_model"], 1 - f["p_model"]).round(6),
        "market_prob_devig": np.where(take_a, f["p_market"], 1 - f["p_market"]).round(6),
        "fill_price": np.where(take_a, f["ask_a"], f["ask_b"]).round(4),
        "price_basis": "kalshi_T60", "price_time": f["price_time"], "fee": FEE, "fee_basis": "tim_est_0.02"})
    out["edge"] = (out["model_prob"] - out["market_prob_devig"] - FEE).round(6)
    out["edge_vs_fill"] = (out["model_prob"] - out["fill_price"] - FEE).round(6)
    out["qualifies"] = out["edge"] >= QUAL - 1e-12
    out["result"] = np.where(take_a, f["y"], 1 - f["y"]).astype(int)
    out["pnl_at_flat_stake"] = (FLAT / (out["fill_price"] + FEE) * out["result"] - FLAT).round(4)
    # extra columns (plan allows a verified fee as a second column, never instead)
    out["fee_kalshi_verified"] = out["fill_price"].map(KalshiFee).round(6)
    out["edge_vs_fill_kalshi_fee"] = (out["model_prob"] - out["fill_price"] - out["fee_kalshi_verified"]).round(6)
    out["pnl_at_flat_stake_kalshi_fee"] = (FLAT / (out["fill_price"] + out["fee_kalshi_verified"]) * out["result"] - FLAT).round(4)
    out["fav_side_fill"] = np.where(f["p_market"] >= 0.5, f["ask_a"], f["ask_b"]).round(4)
    out["fav_result"] = np.where(f["p_market"] >= 0.5, f["y"], 1 - f["y"]).astype(int)
    return out


def Bankroll(edges, sport, strategy, price_basis="kalshi_T60"):
    eq, peak, x, bets_cum, rows, stakes, pnls, ruined = START, START, 0, 0, [], [], [], False
    for season in sorted(edges["season"].unique()):
        part = edges[edges["season"] == season]
        for day, d in part.groupby("date", sort=True):
            x += 1
            sod, day_pnl, nb = eq, 0.0, 0
            for r in d.itertuples():
                if ruined or eq < FLAT:
                    ruined = True
                    break
                if strategy == "flat100":
                    if not r.qualifies:
                        continue
                    stake, cost, won = FLAT, r.fill_price + FEE, r.result
                elif strategy == "qkelly":
                    if not r.qualifies:
                        continue
                    cost = r.fill_price + FEE
                    frac = 0.25 * (r.model_prob - cost) / (1 - cost)
                    if frac <= 0:
                        continue
                    stake, won = min(frac, 0.02) * sod, r.result
                else:  # market_fav
                    stake, cost, won = FLAT, r.fav_side_fill + FEE, r.fav_result
                pnl = stake / cost * won - stake
                day_pnl += pnl
                nb += 1
                stakes.append(stake)
                pnls.append(pnl)
            eq += day_pnl
            bets_cum += nb
            peak = max(peak, eq)
            rows.append({"date": day, "sport": sport, "series": f"{sport}_{strategy}_{price_basis}", "strategy": strategy,
                         "price_basis": price_basis, "season": season, "split": d["split"].iloc[0], "x": x, "bets_day": nb,
                         "bets_cum": bets_cum, "equity": round(eq, 2), "drawdown_pct": round(eq / peak - 1, 6)})
    stakes, pnls = np.array(stakes), np.array(pnls)
    roi = float(pnls.sum() / stakes.sum()) if len(stakes) else float("nan")
    ci = [float("nan"), float("nan")]
    if len(stakes) > 1:
        idx = RNG.integers(0, len(stakes), size=(2000, len(stakes)))
        boots = pnls[idx].sum(1) / stakes[idx].sum(1)
        ci = [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))]
    dd = min((r["drawdown_pct"] for r in rows), default=0.0)
    return pd.DataFrame(rows), {"sport": sport, "strategy": strategy, "n_bets": int(len(stakes)), "final_equity": round(eq, 2),
                                "staked": round(float(stakes.sum()), 2), "roi": roi, "roi_ci90": ci, "max_drawdown_pct": dd, "ruined": ruined}


def ExtensionFrame(tour):
    from Backtest.KalshiRecent import TourFrame
    from Backtest.KalshiResults import SERIES, KALSHI_DIR
    q, _ = TourFrame(tour, "2026-06-01", "2026-11-12")
    names = {}
    for series, _lvl in SERIES[tour]:
        with open(KALSHI_DIR / "markets" / f"{series}.jsonl", encoding="utf-8") as h:
            for line in h:
                m = json.loads(line)
                names.setdefault(m["event_ticker"], {})[m["ticker"]] = m.get("yes_sub_title")
    q["name_a"] = [names.get(e, {}).get(t) for e, t in zip(q["event_ticker"], q["ticker_a"])]
    q["name_b"] = [next((n for k, n in names.get(e, {}).items() if k != t), None) for e, t in zip(q["event_ticker"], q["ticker_a"])]
    q["split"] = "forward"
    return q[q["name_a"].notna() & q["name_b"].notna()]


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default="2026-11-13")
    args = parser.parse_args(argv)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    cal_dir, bank_dir = AGENTDOCS / f"{args.date}_calibration", AGENTDOCS / f"{args.date}_bankroll-10k"
    cal_dir.mkdir(parents=True, exist_ok=True)
    bank_dir.mkdir(parents=True, exist_ok=True)
    meta = {"head": head, "model_version": MODEL_VERSION, "tours": {}}
    sets = {"main": {}, "ext": {}}
    for tour in ("atp", "wta"):
        raw = LoadTour(tour)
        raw = raw[raw["split"].isin(["holdout", "forward"]) & raw["level_tag"].isin(["tour", "tour_qual", "challenger"])]
        f, inplay = Prepare(raw)
        sets["main"][tour] = f
        ext, inplay_ext = Prepare(ExtensionFrame(tour))
        sets["ext"][tour] = ext
        meta["tours"][tour] = {"main_no_t60": int(raw["t60_ask"].isna().sum()), "main_inplay_dropped": inplay,
                               "ext_inplay_dropped": inplay_ext, "ext_rows": int(len(ext))}
        print(tour, meta["tours"][tour], flush=True)
    for key, suffix in (("main", ""), ("ext", "_ext-2026H2")):
        cal_rows, cal_sum, edge_parts, bank_parts, bank_stats = [], [], [], [], []
        for tour, f in sets[key].items():
            sport = f"tennis_{tour}"
            for level in ("tour", "tour_qual", "challenger"):
                lv = f[f["level_tag"] == level]
                if lv.empty:
                    continue
                tag = "" if level == "tour" else f"@{level}"
                groups = [(s, sp, g) for (s, sp), g in lv.groupby(["season_label", "split"])] + [("pooled_oos", "holdout+forward" if key == "main" else "forward", lv)]
                for season, split, g in groups:
                    if len(g) < 1:
                        continue
                    for name, col in ((MODEL + tag, "p_model"), (MARKET + tag, "p_market")):
                        rows, summ = Calibration(g, sport, name, col, season, split, g["date"].max())
                        cal_rows += rows
                        cal_sum.append(summ)
            tour_rows = f[f["level_tag"] == "tour"]
            edges = Edges(tour_rows, sport)
            edge_parts.append(edges)
            for strategy in ("flat100", "qkelly", "market_fav"):
                curve, stats = Bankroll(edges, sport, strategy)
                bank_parts.append(curve)
                stats["scope"] = key
                bank_stats.append(stats)
        pd.DataFrame(cal_rows).to_csv(cal_dir / f"{args.date}_calibration{suffix}.csv", index=False)
        pd.DataFrame(cal_sum).to_csv(cal_dir / f"{args.date}_calibration_summary{suffix}.csv", index=False)
        pd.concat(bank_parts).to_csv(bank_dir / f"{args.date}_bankroll-10k{suffix}.csv", index=False)
        pd.concat(edge_parts).to_csv(bank_dir / f"{args.date}_bankroll-10k_edges{suffix}.csv", index=False)
        meta[f"bank_{key}"] = bank_stats
        meta[f"cal_{key}"] = [s for s in cal_sum if s["season"] == "pooled_oos"]
        # kalshi-fee flat100 ROI (extra column) per tour
        for e in edge_parts:
            q = e[e["qualifies"]]
            meta.setdefault(f"flat100_kalshi_fee_{key}", {})[e["sport"].iloc[0]] = {
                "n": int(len(q)), "roi": float(q["pnl_at_flat_stake_kalshi_fee"].sum() / (FLAT * len(q))) if len(q) else None}
    (bank_dir / f"{args.date}_bankroll-10k_stats.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    Main()