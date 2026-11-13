r"""Kalshi-reality backtest: the FROZEN models priced at Kalshi's own pre-match quotes.

Window: Kalshi tennis match history starts June 2025; Sackmann results end early June 2026,
so this covers the 2025 holdout (Jun-Dec) and the 2026 forward window (Jan-Jun). Simulation only; no orders.

Entry quotes (see Backtest.KalshiJoin): t60 = 1-min candle at scheduled start - 60m (primary),
early = first hourly candle in [T-24h, T-3h]; closing reference = t5 mid (T-5m).
Side A uses the book's yes side; side B costs 1 - yes_bid (the no side of the same book).
Fill conventions: "ask" (executable top of book, no size known) and "mid" (Tim's convention).
Fee bases: tim_est_0.02 (Tim's estimate, labelled) and kalshi_verified (0.07 * p * (1 - p)).
Rule: bet the side with the larger edge = p_model - price - fee if edge > threshold; one side per
match; 1 contract per bet (pays $1). Thresholds 0 (primary), 0.03, 0.05 are fixed, not tuned.
CLV = closing mid of the bet side - entry price (positive = beat the close).
Events flagged possible_inplay (mid moved > 0.25 between T-60 and T-5) are excluded.

Usage: .venv\Scripts\python.exe -m Backtest.KalshiReality
Writes data/backtest/kalshi_reality/{bets.parquet, summary.csv, market_compare.csv, equity.csv}
"""
from __future__ import annotations

import argparse
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from Backtest.Odds import FEES

REPO_ROOT = Path(__file__).resolve().parents[1]
BT_DIR = REPO_ROOT / "data" / "backtest"
OUT_DIR = BT_DIR / "kalshi_reality"
MODELS = ["blend", "elo", "sr"]
THRESHOLDS = [0.0, 0.03, 0.05]
ET = ZoneInfo("America/New_York")


def LoadTour(tour):
    preds = pd.read_parquet(BT_DIR / tour / "predictions.parquet")
    quotes = pd.read_parquet(BT_DIR / tour / "kalshi_quotes.parquet")
    merged = quotes.merge(preds[["match_key", "season", "split", "level_tag", "winner_id", "p_elo", "p_sr", "p_blend"]],
                          on="match_key", how="left")
    a_is_w = merged["id_a"] == merged["winner_id"]
    for model in MODELS:
        merged[f"pa_{model}"] = np.where(a_is_w, merged[f"p_{model}"], 1 - merged[f"p_{model}"])
    merged["y_a"] = merged["a_won"].astype(float)
    merged["tour"] = tour
    merged["date_et"] = pd.to_datetime(merged["scheduled_start"], utc=True).dt.tz_convert(ET).dt.date
    for label in ("t60", "t5", "early"):
        merged[f"{label}_mid"] = (merged[f"{label}_bid"] + merged[f"{label}_ask"]) / 2
    return merged


def LogLoss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def MarketCompare(frame):
    rows = []
    usable = frame[frame["t60_mid"].notna() & ~frame["possible_inplay"]]
    for (tour, level, split), part in usable.groupby(["tour", "level_tag", "split"]):
        y = part["y_a"].to_numpy()
        row = {"tour": tour, "level": level, "split": split, "n": len(part)}
        for label in ("t60", "t5"):
            mid = part[f"{label}_mid"].to_numpy()
            ok = np.isfinite(mid)
            row[f"market_{label}_n"] = int(ok.sum())
            row[f"market_{label}_log_loss"] = LogLoss(mid[ok], y[ok]) if ok.any() else np.nan
            row[f"market_{label}_fav_acc"] = float(np.mean((mid[ok] > 0.5) == (y[ok] == 1))) if ok.any() else np.nan
        for model in MODELS:
            p = part[f"pa_{model}"].to_numpy()
            row[f"{model}_log_loss"] = LogLoss(p, y)
            row[f"{model}_acc"] = float(np.mean((p > 0.5) == (y == 1)))
            row[f"{model}_brier"] = float(np.mean((p - y) ** 2))
        rows.append(row)
    return pd.DataFrame(rows)


def SimBets(frame, model, entry, fill, fee_name, threshold):
    fee_fn = FEES[fee_name]
    bid, ask = frame[f"{entry}_bid"].to_numpy(), frame[f"{entry}_ask"].to_numpy()
    mid = (bid + ask) / 2
    pa = frame[f"pa_{model}"].to_numpy()
    price_a = ask if fill == "ask" else mid
    price_b = (1 - bid) if fill == "ask" else (1 - mid)
    edge_a = pa - price_a - np.array([fee_fn(p) for p in price_a])
    edge_b = (1 - pa) - price_b - np.array([fee_fn(p) for p in price_b])
    take_a = edge_a >= edge_b
    edge = np.where(take_a, edge_a, edge_b)
    price = np.where(take_a, price_a, price_b)
    ok = np.isfinite(edge) & ~frame["possible_inplay"].to_numpy() & (edge > threshold)
    won = np.where(take_a, frame["y_a"].to_numpy() == 1, frame["y_a"].to_numpy() == 0)
    fee = np.array([fee_fn(p) if np.isfinite(p) else np.nan for p in price])
    cost = price + fee
    pnl = np.where(won, 1 - cost, -cost)
    close_mid = frame["t5_mid"].to_numpy()
    close_side = np.where(take_a, close_mid, 1 - close_mid)
    bets = pd.DataFrame({"match_key": frame["match_key"].to_numpy(), "tour": frame["tour"].to_numpy(),
                         "level": frame["level_tag"].to_numpy(), "split": frame["split"].to_numpy(),
                         "date_et": frame["date_et"].to_numpy(), "side": np.where(take_a, "a", "b"),
                         "p_model": np.where(take_a, pa, 1 - pa), "price": price, "fee": fee, "edge": edge,
                         "won": won, "pnl": pnl, "cost": cost, "clv": close_side - price})[ok]
    bets["model"], bets["entry"], bets["fill"], bets["fee_basis"], bets["threshold"] = model, entry, fill, fee_name, threshold
    return bets


def StrictSubset(bets, frame, max_age_s=180):
    ages = frame.groupby("match_key")["t60_age_s"].first()
    keep = bets["match_key"].map(ages).le(max_age_s)
    strict = bets[keep & (bets["entry"] == "t60")].copy()
    strict["entry"] = "t60_strict3min"
    return strict


def Summary(bets):
    rows = []
    keys = ["tour", "level", "split", "model", "entry", "fill", "fee_basis", "threshold"]
    for key, part in bets.groupby(keys):
        clv = part["clv"].dropna()
        rows.append({**dict(zip(keys, key)), "n_bets": len(part), "wins": int(part["won"].sum()),
                     "staked": float(part["cost"].sum()), "pnl": float(part["pnl"].sum()),
                     "roi": float(part["pnl"].sum() / part["cost"].sum()), "avg_edge": float(part["edge"].mean()),
                     "avg_price": float(part["price"].mean()), "clv_mean": float(clv.mean()) if len(clv) else np.nan,
                     "clv_n": int(len(clv)), "clv_pos_share": float((clv > 0).mean()) if len(clv) else np.nan})
    return pd.DataFrame(rows)


def Equity(bets, start_bank=100.0):
    rows = []
    for (tour, fill), part in bets.groupby(["tour", "fill"]):
        daily = part.groupby("date_et")["pnl"].sum().sort_index()
        equity = start_bank + daily.cumsum()
        peak = np.maximum.accumulate(np.concatenate([[start_bank], equity.to_numpy()]))[1:]
        for date, eq, pk in zip(daily.index, equity.to_numpy(), peak):
            rows.append({"date": str(date), "series": f"{tour}_kalshi_{fill}_blend_tim_est_0.02", "equity": round(float(eq), 4),
                         "drawdown_pct": round(float(eq / pk - 1), 6)})
    return pd.DataFrame(rows)


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.parse_args(argv)
    frame = pd.concat([LoadTour(t) for t in ("atp", "wta")], ignore_index=True)
    frame = frame[frame["split"].isin(["holdout", "forward"])]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    market = MarketCompare(frame)
    market.to_csv(OUT_DIR / "market_compare.csv", index=False)
    all_bets = []
    for model in MODELS:
        for entry in ("t60", "early"):
            for fill in ("ask", "mid"):
                for fee_name in FEES:
                    for threshold in THRESHOLDS:
                        all_bets.append(SimBets(frame, model, entry, fill, fee_name, threshold))
    bets = pd.concat(all_bets, ignore_index=True)
    bets = pd.concat([bets, StrictSubset(bets, frame)], ignore_index=True)
    bets.to_parquet(OUT_DIR / "bets.parquet", index=False)
    summary = Summary(bets)
    summary.to_csv(OUT_DIR / "summary.csv", index=False)
    primary = bets[(bets.model == "blend") & (bets.entry == "t60") & (bets.fee_basis == "tim_est_0.02") & (bets.threshold == 0.0)
                   & (bets.level == "tour")]
    equity = Equity(primary)
    equity.to_csv(OUT_DIR / "equity.csv", index=False)
    print(market.to_string(index=False))
    view = summary[(summary.model == "blend") & (summary.entry == "t60") & (summary.threshold == 0.0)]
    print(view[["tour", "level", "split", "fill", "fee_basis", "n_bets", "roi", "avg_edge", "clv_mean"]].to_string(index=False))


if __name__ == "__main__":
    Main()