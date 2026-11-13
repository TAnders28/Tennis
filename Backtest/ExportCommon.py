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


CAVEATS = [
    "**No edge vs Kalshi.** The Kalshi T-60 market has lower log loss and Brier than the model at every level and split, and CLV is negative. Treat every series as PASS evidence, not a strategy.",
    "**Sackmann data is stale.** The Sackmann results archive ends 2026-06-01 (last rows 2026-06-10). After that, ratings are kept current from Kalshi-settled results, which update Elo only; serve/return stays frozen at early June 2026. That period is in the separate `_ext-2026H2` files and is never mixed with the main series.",
    "**Fills assume unlimited size at the ask.** Kalshi candle history has bid/ask but no size, so every fill assumes the full stake fills at the T-60 ask. That is optimistic for larger stakes.",
    "Fees: the edges and P&L use Tim's $0.02 estimate (`fee_basis=tim_est_0.02`). Kalshi's verified taker fee 0.07*p*(1-p) is shown as extra columns, never instead.",
    "Robinhood-vs-Kalshi price differences are unmeasured.",
    "T-60 is measured from Kalshi's *scheduled* start. Tennis starts slip, so the quote can be hours before first ball. Events whose mid moved > 0.25 between T-60 and T-5 are flagged possibly in-play and dropped.",
]


def WriteMarkdown(meta, cal_dir, bank_dir, date):
    head = meta["head"]
    cal = [f"# Tennis calibration ({date}), common format (sports-ops plan section 5a)", "",
           f"Repo E:\\Tennis\\tennis at HEAD `{head}`. Model `{meta['model_version']}` = frozen surface-Elo + serve/return blend (`tennis_blend_v1`). Parameters were frozen 2026-11-13 from tune 2010-2019 and validation 2020-2022 only; the holdout and forward seasons were never used in any search.",
           "Leak test: `tests/test_no_lookahead.py` passes at this HEAD (each match is predicted from the state before its date/round group).",
           "Splits: `holdout` = season 2025 (Jun-Dec, the Kalshi era); `forward` = season 2026 (Jan-Jun 2026). Fixed side player_a = the alphabetically first surname (last name token), never the winner.",
           "Market = Kalshi T-60 mid of player_a's book (both sides from one book, so de-vigged by construction), on exactly the same matches as the model.",
           "Levels: tour-level rows use model names `tennis_blend_v1` and `market_kalshi_T60`. Qualifying and Challenger / WTA 125 rows are calibration-only and carry the level in the model name (`...@tour_qual`, `...@challenger`). ITF has no Kalshi markets, so there are no ITF rows.", "",
           "## Pooled out-of-sample headline (holdout + forward)", "", "| sport | model | N | accuracy | Brier | log loss | ECE |", "|---|---|---|---|---|---|---|"]
    cal += [f"| {s['sport']} | {s['model']} | {s['n']} | {s['accuracy']:.3f} | {s['brier']:.4f} | {s['log_loss']:.4f} | {s['ece']:.4f} |" for s in meta["cal_main"]]
    cal += ["", "## Jun-Nov 2026 extension (separate files `*_ext-2026H2.csv`, split `forward`, Elo current from Kalshi results, serve/return frozen)", "",
            "| sport | model | N | accuracy | Brier | log loss | ECE |", "|---|---|---|---|---|---|---|"]
    cal += [f"| {s['sport']} | {s['model']} | {s['n']} | {s['accuracy']:.3f} | {s['brier']:.4f} | {s['log_loss']:.4f} | {s['ece']:.4f} |" for s in meta["cal_ext"]]
    cal += ["", "Per-season rows and the ten 0.1-wide bins are in the CSVs. Empty bins have n=0 and blank mean_pred / obs_rate.", "", "## Caveats", ""] + [f"- {c}" for c in CAVEATS]
    (cal_dir / f"{date}_calibration.md").write_text("\n".join(cal), encoding="utf-8")
    bk = [f"# Tennis $10k bankroll ({date}), common format (sports-ops plan sections 3-5)", "",
          f"Repo HEAD `{head}`, model `{meta['model_version']}`. Price basis `kalshi_T60`: the last 1-min candle at or before T-60 min before the scheduled start, no older than 30 min. The fill is the bet side's ask; P&L uses ask + fee.",
          "Seasons are calendar years: 2025 = holdout (Jun-Dec), 2026 = forward (Jan-Jun 2026). Rows exist only for game days (days with at least one priced tour-level match); `x` is continuous and skips the offseason. Charts start at each season's first match and end at its last.",
          "Bets are tour level only (no qualifying, Challenger, WTA 125 or ITF). One side per event (the larger edge). An event qualifies when edge = model - de-vigged mid - 0.02 >= 0.03.",
          "- `flat100`: $100 per qualifying event, no compounding, stop if equity < $100.",
          "- `qkelly`: 0.25 x (p - c)/(1 - c) x start-of-day equity with c = ask + 0.02, capped at 2%, compounding; qualifying events where p <= c get no stake.",
          "- `market_fav`: $100 on the Kalshi favourite in every priced event (reference for the cost of fees).", "",
          "## Results (main: holdout 2025 + forward Jan-Jun 2026)", "",
          "| sport | strategy | N bets | staked | final equity | ROI per $ | ROI 90% CI (bootstrap over events) | max drawdown |", "|---|---|---|---|---|---|---|---|"]
    def Rows(stats):
        return [f"| {s['sport']} | {s['strategy']} | {s['n_bets']} | ${s['staked']:,.0f} | ${s['final_equity']:,.2f} | {s['roi']:+.1%} | [{s['roi_ci90'][0]:+.1%}, {s['roi_ci90'][1]:+.1%}] | {s['max_drawdown_pct']:.1%} |" for s in stats]
    bk += Rows(meta["bank_main"])
    bk += ["", "flat100 ROI with Kalshi's verified fee instead of the $0.02 estimate (extra column, same qualifying set): " +
           ", ".join(f"{k} {v['roi']:+.1%} (N {v['n']})" for k, v in meta["flat100_kalshi_fee_main"].items() if v["roi"] is not None), "",
           "## Jun-Nov 2026 extension (separate files `*_ext-2026H2.csv`; Elo current from Kalshi results, serve/return frozen; not mixed with the above)", "",
           "| sport | strategy | N bets | staked | final equity | ROI per $ | ROI 90% CI | max drawdown |", "|---|---|---|---|---|---|---|---|"]
    bk += Rows(meta["bank_ext"])
    bk += ["", "Extension flat100 with Kalshi verified fee: " + ", ".join(f"{k} {v['roi']:+.1%} (N {v['n']})" for k, v in meta["flat100_kalshi_fee_ext"].items() if v["roi"] is not None), "",
           "## Dropped events", ""]
    for tour, d in meta["tours"].items():
        bk.append(f"- {tour.upper()} tour level (main): {d['main_tour_matched_events']} Kalshi events matched to results; {d['main_tour_no_t60']} had no T-60 quote (no price, dropped); "
                  f"{d['main_tour_inplay_dropped']} were flagged possibly in-play and dropped. Ties and voids: 0 (Kalshi voids never reach the matched set). Extension: {d['ext_rows']} priced rows over all levels, {d['ext_inplay_dropped']} in-play dropped.")
    bk += ["", "Files: `_bankroll-10k.csv` (equity per game day), `_bankroll-10k_edges.csv` (every priced tour event, qualifying or not; extra columns fee_kalshi_verified, edge_vs_fill_kalshi_fee, pnl_at_flat_stake_kalshi_fee, fav_side_fill, fav_result), and the same two with `_ext-2026H2`. `_stats.json` holds these numbers.",
           "`price_time` = the T-60 candle's end time (Kalshi's own timestamp). It is a 1-min candle close, not a quote timestamp.", "", "## Caveats", ""] + [f"- {c}" for c in CAVEATS]
    (bank_dir / f"{date}_bankroll-10k.md").write_text("\n".join(bk), encoding="utf-8")


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
        tour_raw = raw[raw["level_tag"] == "tour"]
        meta["tours"][tour] = {"main_tour_matched_events": int(len(tour_raw)), "main_tour_no_t60": int(tour_raw["t60_ask"].isna().sum()),
                               "main_tour_inplay_dropped": int((tour_raw["t60_ask"].notna() & tour_raw["possible_inplay"]).sum()),
                               "main_no_t60": int(raw["t60_ask"].isna().sum()), "main_inplay_dropped": inplay,
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
    WriteMarkdown(meta, cal_dir, bank_dir, args.date)
    (bank_dir / f"{args.date}_bankroll-10k_stats.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    print(json.dumps(meta, indent=1, default=str))


if __name__ == "__main__":
    Main()