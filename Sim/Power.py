r"""Power, error bands and selection-luck for the watch patterns (Sportsbot, 2026-11-26; research, tested:false).

Reads the cross-sport panels (real Kalshi T-60 quotes and real outcomes). Synthetic outcomes are drawn only to
measure how the TEST behaves (power, false positives); they are never treated as real results or as evidence of edge.

Per pattern (one side bought at the T-60 ask, Kalshi taker fee ceil(0.07*C*P*(1-P)) at C=100, `venue_schedule:kalshi`;
Tim's flat 0.02 as a sensitivity):
  * real N, monthly volume (last 180 days of data), mean and SD of P&L per contract, day-cluster design effect;
  * bets (and months) for 80% power at two-sided alpha 0.05 to detect a true +1/+2/+3/+5c edge after fees,
    analytic (z-test with the design effect) and checked by simulation (planted edge: y ~ Bernoulli(ask + fee + edge),
    test = cluster-robust z on days);
  * error band: day-block bootstrap 95% CI of the real mean vs the parametric band from 'fair market' worlds
    (y ~ Bernoulli(mid)), and where the real mean falls in those worlds;
  * selection luck: in 2,000 fair-market worlds, the tune-period P&L of the best of the 18 favourite/underdog price
    buckets (what our tune-pick step would choose by luck alone), next to what was actually picked.

Usage: E:\AgentWorkspaces\Tennis\tennis\.venv\Scripts\python.exe -m Sim.Power --out <json>
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

XS = Path(r"E:\AgentDocs\Sportsbot\2026-11-26_cross-sport-history")
NBA = Path(r"E:\AgentDocs\Basketbot\2026-11-25_nba-full-analysis\2026-11-25_nba-full-analysis_games.csv")
Z = 1.959964 + 0.841621
EDGES = (0.01, 0.02, 0.03, 0.05)
BUCKETS = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 0.75), (0.75, 0.80), (0.80, 0.85), (0.85, 0.90), (0.90, 1.01)]


def KalshiFee(p):
    return np.ceil(0.07 * 100 * np.asarray(p) * (1 - np.asarray(p)) - 1e-9) / 100.0


def Sides(df, date, y, bid, ask, abid=None, aask=None, split=None, extra=None):
    """-> one row per game: date, split, fav_mid, fav_ask, fav_y, dog_ask, dog_y (+ extra columns)."""
    b, a = df[bid].astype(float), df[ask].astype(float)
    mid = (b + a) / 2
    a_ask_other = df[aask].astype(float) if aask and aask in df else 1 - b
    a_ask_other = a_ask_other.fillna(1 - b)
    yy = df[y].astype(float)
    fav_home = mid >= 0.5
    out = pd.DataFrame({"date": pd.to_datetime(df[date]).dt.date.astype(str), "split": df[split] if split else "all",
                        "fav_mid": np.where(fav_home, mid, 1 - mid), "fav_ask": np.where(fav_home, a, a_ask_other),
                        "fav_y": np.where(fav_home, yy, 1 - yy), "dog_ask": np.where(fav_home, a_ask_other, a),
                        "dog_y": np.where(fav_home, 1 - yy, yy), "home_ask": a, "away_ask": a_ask_other, "home_y": yy, "away_y": 1 - yy, "home_mid": mid})
    for c in extra or []:
        out[c] = df[c].to_numpy()
    return out[np.isfinite(out["fav_ask"]) & np.isfinite(out["fav_y"]) & (out["fav_ask"] > 0) & (out["fav_ask"] < 1)].reset_index(drop=True)


def DesignEffect(pnl, days):
    pnl = np.asarray(pnl, float)
    s = pd.Series(pnl - pnl.mean()).groupby(np.asarray(days)).sum()
    v_cl = float((s ** 2).sum()) / len(pnl) ** 2
    v_iid = float(pnl.var()) / len(pnl)
    return max(v_cl / v_iid, 1.0) if v_iid > 0 else 1.0


def BlockCi(pnl, days, n=2000, seed=1):
    rng = np.random.default_rng(seed)
    g = pd.Series(np.asarray(pnl, float)).groupby(np.asarray(days))
    s, c = g.sum().to_numpy(), g.count().to_numpy()
    m = np.array([s[i].sum() / c[i].sum() for i in (rng.integers(0, len(s), len(s)) for _ in range(n))])
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def SimPower(ask, days, edge, n_bets, rng, reps=400, fee_fn=KalshiFee):
    """Planted-edge power: resample whole days until n_bets, y ~ Bernoulli(ask + fee + edge), cluster z-test."""
    ask = np.asarray(ask, float)
    days = np.asarray(days)
    ud, inv = np.unique(days, return_inverse=True)
    members = [np.flatnonzero(inv == k) for k in range(len(ud))]
    per_day = len(ask) / len(ud)
    hits = 0
    for _ in range(reps):
        k = int(math.ceil(n_bets / per_day))
        pick = rng.integers(0, len(ud), k)
        idx = np.concatenate([members[j] for j in pick])
        dd = np.repeat(np.arange(k), [len(members[j]) for j in pick])
        a = ask[idx]
        fee = fee_fn(a)
        q = np.clip(a + fee + edge, 0.0005, 0.9995)
        y = (rng.random(len(a)) < q).astype(float)
        r = y - a - fee
        s = pd.Series(r - r.mean()).groupby(dd).sum()
        se = math.sqrt(float((s ** 2).sum())) / len(r)
        hits += (r.mean() / se > 1.959964) if se > 0 else 0
    return hits / reps


def PerMonth(rows, sport_rows, window_days=365):
    """Pattern rows per ACTIVE month (calendar months in which the sport's priced panel has games), last 12 months."""
    last = pd.to_datetime(sport_rows["date"]).max()
    cut = last - pd.Timedelta(days=window_days)
    sr = pd.to_datetime(sport_rows["date"])
    months = sr[sr >= cut].dt.to_period("M").nunique()
    n = int((pd.to_datetime(rows["date"]) >= cut).sum())
    return n / months if months else 0.0


def Pattern(name, rows, side, rng, sport_rows=None):
    ask = rows[f"{side}_ask"].to_numpy()
    y = rows[f"{side}_y"].to_numpy()
    mid = np.where(side == "fav", rows["fav_mid"].to_numpy(), 1 - rows["fav_mid"].to_numpy()) if side in ("fav", "dog") else rows["home_mid"].to_numpy() if side == "home" else 1 - rows["home_mid"].to_numpy()
    fee = KalshiFee(ask)
    pnl = y - ask - fee
    pnl_flat = y - ask - 0.02
    days = rows["date"].to_numpy()
    per_month = PerMonth(rows, sport_rows if sport_rows is not None else rows)
    deff = DesignEffect(pnl, days)
    sd = float(pnl.std())
    out = {"pattern": name, "n": int(len(rows)), "first": str(min(days)), "last": str(max(days)), "per_month": per_month,
           "mean_pnl": float(pnl.mean()), "mean_pnl_flat": float(pnl_flat.mean()), "sd_pnl": sd, "design_effect": deff,
           "win_rate": float(y.mean()), "mean_mid": float(np.nanmean(mid)), "mean_ask": float(ask.mean()),
           "boot_ci": BlockCi(pnl, days)}
    # fair-market worlds: y ~ Bernoulli(mid)
    reps = 2000
    sims = np.empty(reps)
    for i in range(reps):
        ys = (rng.random(len(mid)) < np.clip(mid, 0, 1)).astype(float)
        sims[i] = (ys - ask - fee).mean()
    out["fair_world_mean"] = float(sims.mean())
    out["fair_world_band"] = [float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))]
    out["fair_world_p_ge_real"] = float((sims >= pnl.mean()).mean())
    out["boot_width"] = out["boot_ci"][1] - out["boot_ci"][0]
    out["fair_band_width"] = out["fair_world_band"][1] - out["fair_world_band"][0]
    # analytic N uses the binomial SD at break-even (q0 = ask + fee), not the realised SD (which is ~0 for a
    # bucket with no losses, e.g. NBA 90c+ at 69-0)
    q0 = np.clip(ask + fee, 0.0005, 0.9995)
    sd0 = float(np.sqrt(np.mean(q0 * (1 - q0))))
    out["sd_breakeven"] = sd0
    for e in EDGES:
        n_need = (Z * sd0 / e) ** 2 * deff
        out[f"n_for_{int(e*100)}c"] = float(n_need)
        out[f"months_for_{int(e*100)}c"] = float(n_need / per_month) if per_month > 0 else None
    # simulation check: smallest N (bisection on a log grid) with >= 80% power for a planted +3c / +1c edge
    for e in (0.03, 0.01):
        lo, hi = 10.0, 60000.0
        if SimPower(ask, days, e, hi, rng, reps=150) < 0.8:
            out[f"sim_n80_{int(e*100)}c"] = None
            continue
        for _ in range(9):
            mid_n = math.sqrt(lo * hi)
            if SimPower(ask, days, e, mid_n, rng, reps=200) >= 0.8:
                hi = mid_n
            else:
                lo = mid_n
        out[f"sim_n80_{int(e*100)}c"] = float(hi)
        out[f"sim_months80_{int(e*100)}c"] = float(hi / per_month) if per_month > 0 else None
    return out


def SelectionLuck(name, rows, rng, reps=2000):
    """Fair worlds: best tune-period bucket P&L among 18 fav/dog buckets vs real tune-pick."""
    tune = rows[rows["split"] == "tune"].reset_index(drop=True)
    if len(tune) < 50:
        return None
    fm = tune["fav_mid"].to_numpy()
    fa, da = tune["fav_ask"].to_numpy(), tune["dog_ask"].to_numpy()
    ff, df_ = KalshiFee(fa), KalshiFee(da)
    masks = [((fm >= lo) & (fm < hi)) for lo, hi in BUCKETS]

    def Best(yf):
        best = -9.0
        for m in masks:
            if m.sum() >= 15:
                best = max(best, (yf[m] - fa[m] - ff[m]).mean(), ((1 - yf[m]) - da[m] - df_[m]).mean())
        return best

    real = Best(tune["fav_y"].to_numpy())
    sims = np.array([Best((rng.random(len(fm)) < fm).astype(float)) for _ in range(reps)])
    return {"pattern": name, "tune_n": int(len(tune)), "real_best_tune_bucket_pnl": float(real),
            "fair_world_best_median": float(np.median(sims)), "fair_world_best_p95": float(np.percentile(sims, 95)),
            "p_fair_best_ge_real": float((sims >= real).mean())}


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=20261126)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    ten = pd.read_csv(XS / "2026-11-26_cross-sport-history_tennis_games.csv", low_memory=False)
    t = Sides(ten, "date", "y", "k_bid", "k_ask", split="split", extra=["ctx_level"])
    nfl = pd.read_csv(XS / "2026-11-26_cross-sport-history_nfl_games.csv", low_memory=False)
    nfl = nfl[nfl["k_ask"].notna()]
    n = Sides(nfl, "date", "y", "k_bid", "k_ask", "ka_bid", "ka_ask", split="split", extra=["ctx_roof"])
    nba = pd.read_csv(NBA, low_memory=False)
    nba = nba[nba["k60_ask"].notna()]
    b = Sides(nba, "date", "home_won", "k60_bid", "k60_ask", split="split")
    pats = []
    t90 = t[t["fav_mid"] >= 0.90]
    pats.append(Pattern("tennis_fav90_all", t90, "fav", rng, t))
    pats.append(Pattern("tennis_fav90_challenger", t90[t90["ctx_level"] == "challenger"], "fav", rng, t))
    pats.append(Pattern("tennis_fav90_tour", t90[t90["ctx_level"] == "tour"], "fav", rng, t))
    pats.append(Pattern("nba_fav90", b[b["fav_mid"] >= 0.90], "fav", rng, b))
    pats.append(Pattern("nfl_all_favourites", n, "fav", rng, n))
    pats.append(Pattern("nfl_dog_when_fav_50_55", n[(n["fav_mid"] >= 0.5) & (n["fav_mid"] < 0.55)], "dog", rng, n))
    dome = n[n["ctx_roof"].isin(["dome", "closed"])]
    pats.append(Pattern("nfl_away_at_dome_home", dome, "away", rng, n))
    pats.append(Pattern("nba_dog_when_fav_50_55", b[(b["fav_mid"] >= 0.5) & (b["fav_mid"] < 0.55)], "dog", rng, b))
    luck = [x for x in (SelectionLuck("NFL", n, rng), SelectionLuck("TENNIS", t, rng), SelectionLuck("NBA", b, rng)) if x]
    res = {"patterns": pats, "selection_luck": luck, "z": Z, "fee_basis": "venue_schedule:kalshi (C=100); flat 0.02 sensitivity in mean_pnl_flat"}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for p in pats:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items() if k in ("pattern", "n", "per_month", "mean_pnl", "sd_pnl", "design_effect", "boot_ci", "fair_world_band", "n_for_1c", "n_for_3c", "months_for_1c", "months_for_3c", "sim_n80_3c", "sim_n80_1c", "sim_months80_3c", "sim_months80_1c", "fair_world_p_ge_real")})
    for l in luck:
        print(l)


if __name__ == "__main__":
    Main()
