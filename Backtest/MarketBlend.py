r"""Market-relative Ignorance, a walk-forward logistic blend into Kalshi, and $10k bankrolls.

Tasks 1-3 of the LSE-style market-blend brief:

  1. Log-loss gap vs de-vigged Kalshi T-60, in bits per match, ATP/WTA x holdout/forward,
     with a 90% match-level bootstrap CI and N.
  2. Weekly walk-forward logistic blend: logit p_blend = a + b logit p_mkt + c logit p_model,
     fit on past weeks only. Report weights and out-of-sample log loss vs Kalshi alone.
  3. Edges off p_blend at the ask + fee, 3-cent minimum. From $10k: flat $100 and
     quarter-Kelly capped at 2%, per tour, for Tim's $0.02 estimate and Kalshi's verified fee.

De-vig: proportional on the two executable asks (yes ask, no ask = 1 - yes bid).
Fills at the ask. possible_inplay rows are dropped. Simulation only; no orders.

Usage:
  .venv\Scripts\python.exe -m Backtest.MarketBlend --out-dir E:\AgentDocs\Tennibot\2026-11-13_market-blend
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from Backtest.KalshiReality import LoadTour
from Backtest.Metrics import LogLossBits, MeanCi
from Backtest.Odds import FEES, DevigProportional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(r"E:\AgentDocs\Tennibot\2026-11-13_market-blend")
MIN_TRAIN = 200
BOOT_REPS = 2000
BOOT_SEED = 20261113
FEE_EST = 0.02
QUAL = 0.03
START_BANK = 10_000.0
FLAT = 100.0
KELLY_FRAC = 0.25
KELLY_CAP = 0.02


def Logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def Sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def DevigAsk(bid, ask):
    """P(yes) after proportional de-vig of the two asks (yes ask, no ask = 1 - bid)."""
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    out = np.full(bid.shape, np.nan)
    ok = np.isfinite(bid) & np.isfinite(ask) & (ask > 0) & (bid < 1)
    if ok.any():
        yes, _no = DevigProportional(ask[ok], 1.0 - bid[ok])
        out[ok] = yes
    return out if out.ndim else float(out)


def UsableFrame(tours=("atp", "wta")):
    frame = pd.concat([LoadTour(t) for t in tours], ignore_index=True)
    frame = frame[frame["split"].isin(["holdout", "forward"])].copy()
    ok = frame["t60_bid"].notna() & frame["t60_ask"].notna() & ~frame["possible_inplay"]
    frame = frame.loc[ok].copy()
    frame["p_mkt"] = DevigAsk(frame["t60_bid"].to_numpy(), frame["t60_ask"].to_numpy())
    frame["p_model"] = frame["pa_blend"].to_numpy(dtype=float)
    frame["ask_a"] = frame["t60_ask"].to_numpy(dtype=float)
    frame["ask_b"] = (1.0 - frame["t60_bid"]).to_numpy(dtype=float)
    frame["y"] = frame["y_a"].astype(float)
    frame = frame[np.isfinite(frame["p_mkt"]) & np.isfinite(frame["p_model"])].copy()
    frame["date_et"] = pd.to_datetime(frame["date_et"])
    iso = frame["date_et"].dt.isocalendar()
    frame["week"] = pd.to_datetime(iso["year"].astype(str) + iso["week"].astype(str).str.zfill(2) + "1",
                                   format="%G%V%u")
    return frame.sort_values(["date_et", "scheduled_start", "match_key"]).reset_index(drop=True)


def GapRows(frame, last_date):
    """ign_rel_bits = mean(IGN_model - IGN_market). Negative means the model beats Kalshi."""
    rows = []

    def One(part, tour, split, level):
        if len(part) == 0:
            return
        ign_m = LogLossBits(part["p_model"], part["y"])
        ign_k = LogLossBits(part["p_mkt"], part["y"])
        gap = ign_m - ign_k
        mean, lo, hi = MeanCi(gap, reps=BOOT_REPS, seed=BOOT_SEED, ci=90)
        rows.append({
            "date": last_date,
            "series": f"{tour}_{split}_{level}",
            "tour": tour,
            "split": split,
            "level": level,
            "n": int(len(part)),
            "ign_model_bits": float(ign_m.mean()),
            "ign_market_bits": float(ign_k.mean()),
            "ign_rel_bits": mean,
            "ci90_lo": lo,
            "ci90_hi": hi,
            "growth_per_event": float(2.0 ** (-mean)) if np.isfinite(mean) else np.nan,
        })

    for (tour, split, level), part in frame.groupby(["tour", "split", "level_tag"]):
        One(part, tour, split, level)
    for (tour, split), part in frame.groupby(["tour", "split"]):
        One(part, tour, split, "all")
    for tour, part in frame.groupby("tour"):
        One(part[part["split"].isin(["holdout", "forward"])], tour, "holdout+forward", "all")
        tour_only = part[part["level_tag"] == "tour"]
        One(tour_only, tour, "holdout+forward", "tour")
    return pd.DataFrame(rows)


def FitLogisticBlend(p_model, p_mkt, y, l2_c=0.0):
    xm, xd, y = Logit(p_mkt), Logit(p_model), np.asarray(y, dtype=float)
    if len(y) < 2 or np.unique(y).size < 2:
        return 0.0, 1.0, 0.0

    def Nll(theta):
        a, b, c = theta
        p = np.clip(Sigmoid(a + b * xm + c * xd), 1e-6, 1.0 - 1e-6)
        loss = -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))
        if l2_c:
            loss += l2_c * c * c
        return loss

    res = minimize(Nll, x0=np.array([0.0, 1.0, 0.0]), method="L-BFGS-B")
    if not res.success:
        return 0.0, 1.0, 0.0
    return tuple(float(x) for x in res.x)


def ApplyBlend(p_model, p_mkt, a, b, c):
    return Sigmoid(a + b * Logit(p_mkt) + c * Logit(p_model))


def FitLinearPool(p_model, p_mkt, y):
    p_model = np.asarray(p_model, dtype=float)
    p_mkt = np.asarray(p_mkt, dtype=float)
    y = np.asarray(y, dtype=float)

    def Nll(weight):
        w = weight[0]
        p = np.clip(w * p_model + (1.0 - w) * p_mkt, 1e-6, 1.0 - 1e-6)
        return -np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))

    res = minimize(Nll, x0=np.array([0.0]), bounds=[(0.0, 1.0)], method="L-BFGS-B")
    return float(res.x[0]) if res.success else 0.0


def WalkForwardBlend(frame, min_train=MIN_TRAIN, l2_c=0.0):
    """Fit on all strictly earlier weeks; apply to the current week. Per tour."""
    pieces, weight_rows = [], []
    for tour, tf in frame.groupby("tour", sort=False):
        tf = tf.sort_values(["week", "scheduled_start", "match_key"]).copy()
        weeks = list(tf["week"].unique())
        preds, flags = [], []
        for week in weeks:
            train = tf[tf["week"] < week]
            apply = tf[tf["week"] == week]
            if len(train) < min_train:
                a, b, c, w_pool, fitted = np.nan, np.nan, np.nan, np.nan, False
                p = apply["p_mkt"].to_numpy()
            else:
                a, b, c = FitLogisticBlend(train["p_model"], train["p_mkt"], train["y"], l2_c=l2_c)
                w_pool = FitLinearPool(train["p_model"], train["p_mkt"], train["y"])
                fitted = True
                p = ApplyBlend(apply["p_model"], apply["p_mkt"], a, b, c)
            preds.append(pd.Series(p, index=apply.index))
            flags.append(pd.Series(fitted, index=apply.index, dtype=bool))
            ign_b = LogLossBits(p, apply["y"]) if len(apply) else np.array([np.nan])
            ign_k = LogLossBits(apply["p_mkt"], apply["y"]) if len(apply) else np.array([np.nan])
            weight_rows.append({
                "date": pd.Timestamp(week).date().isoformat(),
                "series": f"{tour}_blend_weights",
                "tour": tour,
                "a": a, "b": b, "c": c,
                "linear_pool_w": w_pool,
                "n_train": int(len(train)),
                "n_week": int(len(apply)),
                "fitted": fitted,
                "ign_blend_bits": float(np.mean(ign_b)),
                "ign_market_bits": float(np.mean(ign_k)),
                "ign_rel_bits": float(np.mean(ign_b - ign_k)),
            })
        out = tf.copy()
        out["p_blend_mkt"] = pd.concat(preds).sort_index()
        out["blend_fitted"] = pd.concat(flags).sort_index()
        pieces.append(out)
    blended = pd.concat(pieces, ignore_index=False).sort_index()
    return blended, pd.DataFrame(weight_rows)


def BlendOosGap(blended, last_date):
    """Out-of-sample gap of the walk-forward blend vs Kalshi, on weeks with a fitted blend."""
    rows = []
    fitted = blended[blended["blend_fitted"]]
    for (tour, split, level), part in list(fitted.groupby(["tour", "split", "level_tag"])) + \
            [((t, s, "all"), g) for (t, s), g in fitted.groupby(["tour", "split"])] + \
            [((t, "holdout+forward", "all"), g) for t, g in fitted.groupby("tour")] + \
            [((t, "holdout+forward", "tour"), g[g["level_tag"] == "tour"]) for t, g in fitted.groupby("tour")]:
        if len(part) == 0:
            continue
        gap = LogLossBits(part["p_blend_mkt"], part["y"]) - LogLossBits(part["p_mkt"], part["y"])
        mean, lo, hi = MeanCi(gap, reps=BOOT_REPS, seed=BOOT_SEED, ci=90)
        rows.append({
            "date": last_date,
            "series": f"{tour}_{split}_{level}_blend_oos",
            "tour": tour, "split": split, "level": level,
            "n": int(len(part)),
            "ign_blend_bits": float(LogLossBits(part["p_blend_mkt"], part["y"]).mean()),
            "ign_market_bits": float(LogLossBits(part["p_mkt"], part["y"]).mean()),
            "ign_rel_bits": mean, "ci90_lo": lo, "ci90_hi": hi,
            "growth_per_event": float(2.0 ** (-mean)) if np.isfinite(mean) else np.nan,
        })
    return pd.DataFrame(rows)


def Bankroll(frame, tour, level, strategy, fee_name, last_date, start=START_BANK):
    fee_fn = FEES[fee_name]
    part = frame[(frame["tour"] == tour) & (frame["blend_fitted"])].copy()
    if level != "all":
        part = part[part["level_tag"] == level]
    part = part.sort_values(["date_et", "scheduled_start", "match_key"])
    eq, peak, ruined = start, start, False
    x, bets_cum, rows, stakes, pnls = 0, 0, [], [], []
    for day, d in part.groupby("date_et", sort=True):
        x += 1
        sod, day_pnl, nb = eq, 0.0, 0
        if ruined or eq < FLAT:
            ruined = True
        else:
            for r in d.itertuples():
                price_a, price_b = r.ask_a, r.ask_b
                p = r.p_blend_mkt
                fee_a, fee_b = fee_fn(price_a), fee_fn(price_b)
                edge_a = p - price_a - fee_a
                edge_b = (1.0 - p) - price_b - fee_b
                take_a = edge_a >= edge_b
                edge, price, fee = (edge_a, price_a, fee_a) if take_a else (edge_b, price_b, fee_b)
                if edge < QUAL:
                    continue
                cost = price + fee
                if strategy == "flat100":
                    stake = FLAT
                else:
                    frac = KELLY_FRAC * ( (p if take_a else 1.0 - p) - cost ) / (1.0 - cost)
                    if frac <= 0:
                        continue
                    stake = min(frac, KELLY_CAP) * sod
                won = (r.y == 1) if take_a else (r.y == 0)
                pnl = stake / cost * (1.0 if won else 0.0) - stake
                day_pnl += pnl
                nb += 1
                stakes.append(stake)
                pnls.append(pnl)
        eq += day_pnl
        bets_cum += nb
        peak = max(peak, eq)
        rows.append({
            "date": pd.Timestamp(day).date().isoformat(),
            "series": f"{tour}_{level}_{strategy}_{fee_name}",
            "tour": tour, "level": level, "strategy": strategy, "fee_basis": fee_name,
            "split": d["split"].iloc[0],
            "x": x, "bets_day": nb, "bets_cum": bets_cum,
            "equity": round(eq, 2), "drawdown_pct": round(eq / peak - 1, 6),
        })
    stakes, pnls = np.array(stakes, dtype=float), np.array(pnls, dtype=float)
    roi = float(pnls.sum() / stakes.sum()) if len(stakes) else float("nan")
    if len(stakes) > 1:
        _, lo, hi = MeanCi(pnls / stakes, reps=BOOT_REPS, seed=BOOT_SEED, ci=90)
        # stake-weighted ROI CI (resample bets)
        rng = np.random.default_rng(BOOT_SEED)
        idx = rng.integers(0, len(stakes), size=(BOOT_REPS, len(stakes)))
        boots = pnls[idx].sum(1) / stakes[idx].sum(1)
        lo, hi = float(np.percentile(boots, 5)), float(np.percentile(boots, 95))
    else:
        lo, hi = float("nan"), float("nan")
    stats = {
        "date": last_date,
        "series": f"{tour}_{level}_{strategy}_{fee_name}",
        "tour": tour, "level": level, "strategy": strategy, "fee_basis": fee_name,
        "n_bets": int(len(stakes)),
        "final_equity": round(eq, 2),
        "staked": round(float(stakes.sum()), 2) if len(stakes) else 0.0,
        "pnl": round(float(pnls.sum()), 2) if len(stakes) else 0.0,
        "roi": roi, "roi_ci90_lo": lo, "roi_ci90_hi": hi,
        "max_drawdown_pct": min((r["drawdown_pct"] for r in rows), default=0.0),
        "ruined": ruined, "start_bank": start,
    }
    return pd.DataFrame(rows), stats


def RunBankrolls(blended, last_date, levels=("tour",)):
    curves, stats = [], []
    for tour in ("atp", "wta"):
        for level in levels:
            for strategy in ("flat100", "qkelly"):
                for fee_name in ("tim_est_0.02", "kalshi_verified"):
                    curve, row = Bankroll(blended, tour, level, strategy, fee_name, last_date)
                    if len(curve):
                        curves.append(curve)
                    stats.append(row)
    curve = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return curve, pd.DataFrame(stats)


def LastDate(frame):
    return pd.to_datetime(frame["date_et"]).max().date().isoformat()


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base", default="2026-11-13_market-blend")
    parser.add_argument("--min-train", type=int, default=MIN_TRAIN)
    parser.add_argument("--l2-c", type=float, default=0.0)
    parser.add_argument("--levels", nargs="+", default=["tour"], help="Bankroll universes (tour, challenger, all, ...)")
    args = parser.parse_args(argv)
    frame = UsableFrame()
    last_date = LastDate(frame)
    gap = GapRows(frame, last_date)
    blended, weights = WalkForwardBlend(frame, min_train=args.min_train, l2_c=args.l2_c)
    oos = BlendOosGap(blended, last_date)
    curve, bank_stats = RunBankrolls(blended, last_date, levels=tuple(args.levels))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gap_path = args.out_dir / f"{args.base}_gap.csv"
    w_path = args.out_dir / f"{args.base}_weights.csv"
    b_path = args.out_dir / f"{args.base}_bankroll.csv"
    s_path = args.out_dir / f"{args.base}_bankroll_stats.csv"
    oos_path = args.out_dir / f"{args.base}_blend_oos.csv"
    pred_cols = [c for c in [
        "match_key", "tour", "level_tag", "split", "season", "date_et", "week",
        "scheduled_start", "event_ticker", "series", "kalshi_level", "name_a", "name_b",
        "p_model", "p_mkt", "p_blend_mkt", "blend_fitted", "y", "ask_a", "ask_b",
        "t60_bid", "t60_ask", "t60_age_s", "t5_bid", "t5_ask", "possible_inplay",
        "a_won", "id_a", "winner_id",
    ] if c in blended.columns]
    preds_path = args.out_dir / f"{args.base}_blend_preds.parquet"
    blended[pred_cols].to_parquet(preds_path, index=False)
    gap.to_csv(gap_path, index=False)
    weights.to_csv(w_path, index=False)
    curve.to_csv(b_path, index=False)
    bank_stats.to_csv(s_path, index=False)
    oos.to_csv(oos_path, index=False)
    print("=== ign_rel_bits (model vs de-vigged T-60; negative = model better) ===")
    print(gap.to_string(index=False))
    print("=== blend OOS ===")
    print(oos.to_string(index=False))
    print("=== bankroll stats ===")
    print(bank_stats.to_string(index=False))
    print("wrote", gap_path, w_path, b_path, s_path, oos_path, preds_path, sep="\n  ")
    return {"gap": gap, "weights": weights, "oos": oos, "bankroll": curve, "bank_stats": bank_stats,
            "blended": blended, "last_date": last_date}


if __name__ == "__main__":
    Main()