r"""Residual-on-market grid search against Kalshi T-60 (Pinnacle fallback).

PRE-REGISTERED (2026-11-13), from date coverage only, before any model scores:
  Market: Kalshi T-60 de-vigged asks. tennis-data.co.uk / Pinnacle closing files
  were not present under data/raw/tennis_data or data/processed/tennis_data_*.parquet.
  Fit / selection: market matches with date_et < 2026-01-01 (2025 H2).
  Holdout (frozen): date_et >= 2026-01-01.
  Residual: logit(p) = logit(p_mkt) + w * (logit(p_model) - logit(p_mkt)),
  with w fit walk-forward by month on earlier 2025 matches (default w=0.5 until
  200 training matches), then frozen from all of 2025 for the 2026 holdout.
  Selection: maximum mean bits gap vs T-60 on the 2025 walk-forward predictions.
  SPA / Reality Check: 10,000 day-block bootstraps on that 2025 sample.
  Permutation null: 10,000 shuffles of holdout outcomes for the frozen best config.

Default verdict is PASS unless the holdout 95% day-block CI for the bits gap
excludes 0 in the model's favour and the SPA p-value is small.

Usage:
  .venv\Scripts\python.exe -m Backtest.ResidualSearch --out-dir E:\AgentDocs\Tennibot\2026-11-13_residual-search --workers 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

from Backtest.FrozenParams import FROZEN
from Backtest.KalshiReality import LoadTour
from Backtest.LoadMatches import LoadMatches
from Backtest.Odds import FEES, DevigProportional, TIM_FEE
from Backtest.ResidualRatings import ResidualParams, BuildGrid, Walk, Logit as LogitS, Sigmoid as SigmoidS
from Backtest.SpaTest import DayBlockMeans, HansenSpa, PermuteProbGap, WhiteRealityCheck

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(r"E:\AgentDocs\Tennibot\2026-11-13_residual-search")
DEFAULT_BASE = "2026-11-13_residual-search"
ET = ZoneInfo("America/New_York")
HOLDOUT_START = dt.date(2026, 1, 1)
MIN_TRAIN = 200
DEFAULT_W = 0.5
SPA_REPS = 10_000
MC_PATHS = 10_000
BOOT_SEED = 20261113
VALUE_LINE = 0.03
START_BANK = 10_000.0
FLAT = 100.0
KELLY_FRAC = 0.25
KELLY_CAP = 0.02
W_BOUNDS = (-0.5, 2.0)

_STATE = {}


def DevigAsk(bid, ask):
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)
    out = np.full(np.shape(bid), np.nan, dtype=float)
    ok = np.isfinite(bid) & np.isfinite(ask) & (ask > 0) & (bid < 1)
    if np.any(ok):
        yes, _no = DevigProportional(ask[ok], 1.0 - bid[ok])
        out[ok] = yes
    return out


def Logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def Sigmoid(z):
    z = np.clip(np.asarray(z, dtype=float), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-z))


def BitsIgn(p, y):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype=float)
    return -(y * np.log2(p) + (1.0 - y) * np.log2(1.0 - p))


def Blend(p_elo, p_sr, weight):
    p_elo = np.asarray(p_elo, dtype=float)
    p_sr = np.asarray(p_sr, dtype=float)
    out = p_elo.copy()
    ok = np.isfinite(p_elo) & np.isfinite(p_sr)
    if weight and np.any(ok):
        out[ok] = Sigmoid((1.0 - weight) * Logit(p_elo[ok]) + weight * Logit(p_sr[ok]))
    return out


def Residualize(p_model, p_mkt, w):
    return Sigmoid(Logit(p_mkt) + w * (Logit(p_model) - Logit(p_mkt)))


def FitW(p_model, p_mkt, y):
    p_model = np.asarray(p_model, dtype=float)
    p_mkt = np.asarray(p_mkt, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(p_model) & np.isfinite(p_mkt) & np.isfinite(y)
    if ok.sum() < 2 or np.unique(y[ok]).size < 2:
        return DEFAULT_W
    delta = Logit(p_model[ok]) - Logit(p_mkt[ok])
    yk, mk = y[ok], p_mkt[ok]

    def nll(w):
        p = np.clip(Sigmoid(Logit(mk) + w * delta), 1e-6, 1.0 - 1e-6)
        return -np.sum(yk * np.log(p) + (1.0 - yk) * np.log(1.0 - p))

    res = minimize_scalar(nll, bounds=W_BOUNDS, method="bounded")
    return float(res.x) if res.success else DEFAULT_W


def PinnaclePresent():
    raw = REPO_ROOT / "data" / "raw" / "tennis_data"
    proc = [REPO_ROOT / "data" / "processed" / f"tennis_data_{t}.parquet" for t in ("atp", "wta")]
    has_raw = raw.exists() and any(raw.rglob("*"))
    has_proc = any(p.exists() for p in proc)
    return bool(has_raw or has_proc)


def LoadArrays(tour):
    frame = LoadMatches(tour)
    days = frame["date"].to_numpy().astype("datetime64[D]").astype("int64")
    packed = {
        "group": frame["group_id"].to_numpy(),
        "winner": frame["winner_id"].to_numpy(dtype="int64"),
        "loser": frame["loser_id"].to_numpy(dtype="int64"),
        "surface": frame["surface_norm"].astype(str).to_numpy(),
        "best_of": frame["best_of"].to_numpy(dtype="int64"),
        "level": frame["level_tag"].astype(str).to_numpy(),
        "days": days,
        "match_key": frame["match_key"].to_numpy(),
    }
    return frame, packed


def LoadMarket():
    parts = []
    for tour in ("atp", "wta"):
        q = LoadTour(tour)
        q = q[q["t60_bid"].notna() & q["t60_ask"].notna() & ~q["possible_inplay"]].copy()
        q["p_mkt"] = DevigAsk(q["t60_bid"].to_numpy(), q["t60_ask"].to_numpy())
        q["p_early"] = DevigAsk(q["early_bid"].to_numpy(), q["early_ask"].to_numpy())
        q["p_close"] = DevigAsk(q["t5_bid"].to_numpy(), q["t5_ask"].to_numpy())
        q["ask_a"] = q["t60_ask"].to_numpy(dtype=float)
        q["ask_b"] = (1.0 - q["t60_bid"]).to_numpy(dtype=float)
        q["y"] = q["y_a"].astype(float)
        q["date_et"] = pd.to_datetime(q["date_et"])
        q = q[np.isfinite(q["p_mkt"])].copy()
        parts.append(q)
    market = pd.concat(parts, ignore_index=True)
    market["is_holdout"] = market["date_et"].dt.date >= HOLDOUT_START
    market["month"] = market["date_et"].dt.to_period("M").astype(str)
    market["day"] = market["date_et"].dt.strftime("%Y-%m-%d")
    return market.sort_values(["date_et", "scheduled_start", "match_key"]).reset_index(drop=True)


def WalkForwardW(p_model, p_mkt, y, months, min_train=MIN_TRAIN):
    """Fit w on earlier months; apply to the current month. Returns p_resid and per-row w."""
    p_res = np.full(len(y), np.nan)
    w_row = np.full(len(y), np.nan)
    order = sorted(set(months))
    for month in order:
        mask = months == month
        train = np.array([m < month for m in months])
        if train.sum() < min_train:
            w = DEFAULT_W
        else:
            w = FitW(p_model[train], p_mkt[train], y[train])
        p_res[mask] = Residualize(p_model[mask], p_mkt[mask], w)
        w_row[mask] = w
    return p_res, w_row


def DayBlockCi(values, days, reps=SPA_REPS, seed=BOOT_SEED, q=(2.5, 97.5)):
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    values, days = values[ok], np.asarray(days)[ok]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    means = DayBlockMeans(values, days, reps, seed)
    return float(values.mean()), float(np.percentile(means, q[0])), float(np.percentile(means, q[1]))


def ClvSeries(p_side, p_early, p_close):
    """CLV of the side favoured by p_side vs the early market, marked to the T-5 close."""
    p_side, p_early, p_close = (np.asarray(x, dtype=float) for x in (p_side, p_early, p_close))
    take_a = p_side >= p_early
    return np.where(take_a, p_close - p_early, p_early - p_close)


def _Init(packed, idx):
    _STATE.clear()
    _STATE.update(packed)
    _STATE["idx"] = np.asarray(idx, dtype=np.int64)
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _Eval(params: ResidualParams):
    packed = _STATE
    p = Walk(packed["group"], packed["winner"], packed["loser"], packed["surface"],
             packed["best_of"], packed["level"], packed["days"], params,
             predict_index=packed["idx"])
    return asdict(params), p[packed["idx"]]


def UniqueWalkGrid(grid):
    seen = {}
    for p in grid:
        key = tuple((k, getattr(p, k)) for k in asdict(p) if k != "sr_weight")
        if key not in seen:
            seen[key] = replace(p, sr_weight=0.0)
    return list(seen.values())


def AlignWinnerToA(p_winner, winner_id, id_a):
    return np.where(np.asarray(winner_id) == np.asarray(id_a), p_winner, 1.0 - p_winner)


def ScoreBlock(p, p_mkt, y, days, p_early=None, p_close=None):
    ign_m = BitsIgn(p, y)
    ign_k = BitsIgn(p_mkt, y)
    gap = ign_k - ign_m
    mean, lo, hi = DayBlockCi(gap, days)
    out = {
        "n": int(np.isfinite(p).sum()),
        "bits_gap": mean, "bits_gap_ci95_lo": lo, "bits_gap_ci95_hi": hi,
        "ign_model_bits": float(np.mean(ign_m)), "ign_market_bits": float(np.mean(ign_k)),
    }
    if p_early is not None and p_close is not None:
        ok = np.isfinite(p_early) & np.isfinite(p_close)
        if ok.any():
            clv = ClvSeries(p[ok], p_early[ok], p_close[ok])
            cmean, clo, chi = DayBlockCi(clv, np.asarray(days)[ok])
            out.update({"clv_n": int(ok.sum()), "clv": cmean, "clv_ci95_lo": clo, "clv_ci95_hi": chi})
        else:
            out.update({"clv_n": 0, "clv": np.nan, "clv_ci95_lo": np.nan, "clv_ci95_hi": np.nan})
    return out


def BankrollActual(part, strategy, fee_fn, start=START_BANK):
    part = part.sort_values(["date_et", "scheduled_start", "match_key"])
    eq, peak, ruined = start, start, False
    rows, stakes, pnls = [], [], []
    bets_cum, x = 0, 0
    for day, d in part.groupby("day", sort=True):
        x += 1
        sod, day_pnl, nb = eq, 0.0, 0
        if ruined or eq < FLAT:
            ruined = True
        else:
            for r in d.itertuples():
                fee_a, fee_b = fee_fn(r.ask_a), fee_fn(r.ask_b)
                edge_a = r.p_resid - r.ask_a - fee_a
                edge_b = (1.0 - r.p_resid) - r.ask_b - fee_b
                take_a = edge_a >= edge_b
                edge, price, fee = (edge_a, r.ask_a, fee_a) if take_a else (edge_b, r.ask_b, fee_b)
                if edge < VALUE_LINE:
                    continue
                cost = price + fee
                if strategy == "flat100":
                    stake = FLAT
                else:
                    frac = KELLY_FRAC * ((r.p_resid if take_a else 1.0 - r.p_resid) - cost) / (1.0 - cost)
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
        rows.append({"date": day, "series": f"{part['tour'].iloc[0]}_{strategy}",
                     "tour": part["tour"].iloc[0], "strategy": strategy,
                     "x": x, "bets_day": nb, "bets_cum": bets_cum,
                     "equity": round(eq, 2), "drawdown_pct": round(eq / peak - 1, 6)})
    return pd.DataFrame(rows), np.array(stakes, dtype=float), np.array(pnls, dtype=float), ruined, eq


def DayBets(part, fee_fn):
    """Precompute qualifying bets per day for Monte Carlo replay (qkelly uses start-of-day equity)."""
    days = []
    for day, d in part.sort_values(["date_et", "scheduled_start", "match_key"]).groupby("day", sort=True):
        bets = []
        for r in d.itertuples():
            fee_a, fee_b = fee_fn(r.ask_a), fee_fn(r.ask_b)
            edge_a = r.p_resid - r.ask_a - fee_a
            edge_b = (1.0 - r.p_resid) - r.ask_b - fee_b
            take_a = edge_a >= edge_b
            edge, price, fee = (edge_a, r.ask_a, fee_a) if take_a else (edge_b, r.ask_b, fee_b)
            if edge < VALUE_LINE:
                continue
            cost = price + fee
            p = r.p_resid if take_a else (1.0 - r.p_resid)
            won = (r.y == 1) if take_a else (r.y == 0)
            bets.append((cost, p, won))
        days.append(bets)
    return days


def Replay(days, order, strategy, start=START_BANK):
    eq, ruined = start, False
    staked = pnl = 0.0
    n = 0
    for j in order:
        if ruined or eq < FLAT:
            ruined = True
            break
        sod = eq
        for cost, p, won in days[j]:
            if strategy == "flat100":
                stake = FLAT
            else:
                frac = KELLY_FRAC * (p - cost) / (1.0 - cost)
                if frac <= 0:
                    continue
                stake = min(frac, KELLY_CAP) * sod
            this = stake / cost * (1.0 if won else 0.0) - stake
            eq += this
            staked += stake
            pnl += this
            n += 1
    return eq, (pnl / staked if staked else np.nan), n, ruined


def BankrollMC(part, strategy, fee_fn, reps=MC_PATHS, seed=BOOT_SEED):
    days = DayBets(part, fee_fn)
    n_days = len(days)
    actual_eq, actual_roi, actual_n, actual_ruined = Replay(days, np.arange(n_days), strategy)
    rng = np.random.default_rng(seed)
    finals = np.empty(reps)
    rois = np.empty(reps)
    for i in range(reps):
        order = rng.integers(0, n_days, size=n_days) if n_days else np.array([], dtype=int)
        eq, roi, _n, _r = Replay(days, order, strategy)
        finals[i] = eq
        rois[i] = roi
    return {
        "final_equity": float(actual_eq), "roi": float(actual_roi), "n_bets": int(actual_n),
        "ruined": bool(actual_ruined),
        "mc_equity_p05": float(np.nanpercentile(finals, 5)),
        "mc_equity_p50": float(np.nanpercentile(finals, 50)),
        "mc_equity_p95": float(np.nanpercentile(finals, 95)),
        "mc_roi_p05": float(np.nanpercentile(rois, 5)),
        "mc_roi_p50": float(np.nanpercentile(rois, 50)),
        "mc_roi_p95": float(np.nanpercentile(rois, 95)),
        "mc_paths": int(reps),
    }


def RunTourWalks(tour, market, workers, grid_walk, smoke=False):
    frame, packed = LoadArrays(tour)
    keys = set(market.loc[market["tour"] == tour, "match_key"])
    idx = np.flatnonzero(np.isin(packed["match_key"], list(keys)))
    print(f"{tour}: {len(frame):,} matches, {len(idx):,} market rows, {len(grid_walk)} walks, workers={workers}",
          flush=True)
    if smoke:
        out = []
        _Init(packed, idx)
        for params in grid_walk:
            out.append(_Eval(params))
        return packed["match_key"][idx], out
    with Pool(workers, initializer=_Init, initargs=(packed, idx)) as pool:
        results = pool.map(_Eval, grid_walk, chunksize=1)
    return packed["match_key"][idx], results


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--spa-reps", type=int, default=SPA_REPS)
    parser.add_argument("--mc-paths", type=int, default=MC_PATHS)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)

    cap = max(1, (os.cpu_count() or 28) // 4)
    workers = max(1, min(int(args.workers), cap))
    pinnacle = PinnaclePresent()
    print(f"pinnacle/tennis-data present: {pinnacle}; market=Kalshi T-60; "
          f"holdout starts {HOLDOUT_START.isoformat()}; workers={workers}", flush=True)

    market = LoadMarket()
    print("market rows", len(market),
          "fit", int((~market["is_holdout"]).sum()),
          "holdout", int(market["is_holdout"].sum()), flush=True)

    full_grid = BuildGrid()
    if args.smoke:
        full_grid = full_grid[:8]
        args.spa_reps = min(args.spa_reps, 200)
        args.mc_paths = min(args.mc_paths, 200)
    walk_grid = UniqueWalkGrid(full_grid)
    print(f"configs={len(full_grid)} unique_walks={len(walk_grid)}", flush=True)

    # p_model_winner keyed by match_key per unique walk, then blend sr_weight in-process
    walk_preds = {}  # tour -> (keys, list of (params_dict, p_winner))
    t0 = dt.datetime.now()
    for tour in ("atp", "wta"):
        keys, results = RunTourWalks(tour, market, workers, walk_grid, smoke=args.smoke)
        walk_preds[tour] = (keys, results)
        print(f"{tour} walks done in {(dt.datetime.now()-t0).total_seconds():.0f}s", flush=True)

    # Assemble p_model (P of side A) for every config
    fit_mask = (~market["is_holdout"]).to_numpy()
    hold_mask = market["is_holdout"].to_numpy()
    y = market["y"].to_numpy()
    p_mkt = market["p_mkt"].to_numpy()
    p_early = market["p_early"].to_numpy()
    p_close = market["p_close"].to_numpy()
    days = market["day"].to_numpy()
    months = market["month"].to_numpy()
    pa_sr = market["pa_sr"].to_numpy(dtype=float) if "pa_sr" in market.columns else np.full(len(market), np.nan)

    # map tour, match_key -> p_elo_a for each walk
    elo_by_key = {}
    for tour, (keys, results) in walk_preds.items():
        sub = market["tour"].to_numpy() == tour
        id_a = market.loc[sub, "id_a"].to_numpy()
        winner_id = market.loc[sub, "winner_id"].to_numpy()
        mk = market.loc[sub, "match_key"].to_numpy()
        pos = {k: i for i, k in enumerate(keys)}
        order = np.array([pos[k] for k in mk])
        for params_d, p_winner in results:
            pa = AlignWinnerToA(p_winner[order], winner_id, id_a)
            key = tuple(sorted((k, params_d[k]) for k in params_d if k != "sr_weight"))
            elo_by_key.setdefault(key, np.full(len(market), np.nan))
            elo_by_key[key][np.flatnonzero(sub)] = pa

    config_rows = []
    d_fit = []
    p_fit_models = []
    meta = []
    for i, params in enumerate(full_grid):
        d = asdict(params)
        key = tuple(sorted((k, d[k]) for k in d if k != "sr_weight"))
        pa_elo = elo_by_key[key]
        p_model = Blend(pa_elo, pa_sr, params.sr_weight)
        p_wf, w_wf = WalkForwardW(p_model[fit_mask], p_mkt[fit_mask], y[fit_mask], months[fit_mask])
        w_frozen = FitW(p_model[fit_mask], p_mkt[fit_mask], y[fit_mask])
        p_hold = Residualize(p_model[hold_mask], p_mkt[hold_mask], w_frozen)
        fit_score = ScoreBlock(p_wf, p_mkt[fit_mask], y[fit_mask], days[fit_mask],
                               p_early[fit_mask], p_close[fit_mask])
        hold_score = ScoreBlock(p_hold, p_mkt[hold_mask], y[hold_mask], days[hold_mask],
                                p_early[hold_mask], p_close[hold_mask])
        row = {**{k: d[k] for k in d}, "w_frozen": w_frozen, "w_wf_mean": float(np.nanmean(w_wf))}
        row.update({f"fit_{k}": v for k, v in fit_score.items()})
        row.update({f"holdout_{k}": v for k, v in hold_score.items()})
        config_rows.append(row)
        d_fit.append(BitsIgn(p_mkt[fit_mask], y[fit_mask]) - BitsIgn(p_wf, y[fit_mask]))
        p_fit_models.append(p_wf)
        meta.append((params, w_frozen, p_model, p_wf, p_hold))
        if (i + 1) % 500 == 0 or i + 1 == len(full_grid):
            print(f"scored {i+1}/{len(full_grid)} configs", flush=True)

    grid_table = pd.DataFrame(config_rows).sort_values("fit_bits_gap", ascending=False).reset_index(drop=True)
    grid_table["fit_rank"] = np.arange(1, len(grid_table) + 1)
    best_i = int(np.argmax([r["fit_bits_gap"] for r in config_rows]))
    # iloc after sort changes index; use original argmax on config_rows
    best_params, best_w, best_pmodel, best_pwf, best_phold = meta[best_i]
    print("best fit bits_gap", config_rows[best_i]["fit_bits_gap"], "w", best_w, best_params, flush=True)

    d_mat = np.column_stack(d_fit)
    spa = HansenSpa(d_mat, days[fit_mask], reps=args.spa_reps, seed=BOOT_SEED)
    rc = WhiteRealityCheck(d_mat, days[fit_mask], reps=args.spa_reps, seed=BOOT_SEED)
    perm = PermuteProbGap(best_phold, p_mkt[hold_mask], y[hold_mask], days[hold_mask],
                          reps=args.spa_reps, seed=BOOT_SEED)
    print("SPA", spa, "RC", rc, "PERM", perm, flush=True)

    # attach frozen residual to market for bankroll (holdout only)
    m = market.copy()
    m["p_model"] = best_pmodel
    m["p_resid"] = np.nan
    m.loc[fit_mask, "p_resid"] = best_pwf
    m.loc[hold_mask, "p_resid"] = best_phold
    hold = m[hold_mask].copy()

    bank_curves, bank_stats = [], []
    for tour in ("atp", "wta"):
        for level, part0 in (("all", hold[hold["tour"] == tour]),
                             ("tour", hold[(hold["tour"] == tour) & (hold["level_tag"] == "tour")])):
            if len(part0) == 0:
                continue
            for strategy in ("flat100", "qkelly"):
                for fee_name, fee_fn in FEES.items():
                    curve, stakes, pnls, ruined, eq = BankrollActual(part0, strategy, fee_fn)
                    mc = BankrollMC(part0, strategy, fee_fn, reps=args.mc_paths, seed=BOOT_SEED)
                    if len(curve):
                        curve = curve.copy()
                        curve["series"] = f"{tour}_{level}_{strategy}_{fee_name}"
                        curve["level"] = level
                        curve["fee_basis"] = fee_name
                        bank_curves.append(curve)
                    bank_stats.append({
                        "tour": tour, "level": level, "strategy": strategy, "fee_basis": fee_name,
                        "n_priced": int(len(part0)), "ruined_actual": ruined,
                        "staked": float(stakes.sum()) if len(stakes) else 0.0,
                        "pnl": float(pnls.sum()) if len(pnls) else 0.0,
                        **mc,
                    })
    bank_curve = pd.concat(bank_curves, ignore_index=True) if bank_curves else pd.DataFrame()
    bank_stats_f = pd.DataFrame(bank_stats)

    hold_score = ScoreBlock(best_phold, p_mkt[hold_mask], y[hold_mask], days[hold_mask],
                            p_early[hold_mask], p_close[hold_mask])
    fit_score = ScoreBlock(best_pwf, p_mkt[fit_mask], y[fit_mask], days[fit_mask],
                           p_early[fit_mask], p_close[fit_mask])
    # PASS unless holdout CI excludes 0 in the model's favour AND spa p is small
    gap, lo, hi = hold_score["bits_gap"], hold_score["bits_gap_ci95_lo"], hold_score["bits_gap_ci95_hi"]
    beats = bool(np.isfinite(gap) and lo > 0 and spa["p_value"] < 0.05)
    verdict = "PASS"
    if beats:
        verdict = "INVESTIGATE"  # still not an auto BET; default stance stays cautious
    headline = {
        "pre_registered_holdout_start": HOLDOUT_START.isoformat(),
        "market": "kalshi_t60_devig_ask" + ("" if not pinnacle else "+pinnacle_present_but_unused"),
        "pinnacle_tennis_data_present": pinnacle,
        "n_configs": len(full_grid),
        "n_unique_walks": len(walk_grid),
        "n_fit": int(fit_mask.sum()),
        "n_holdout": int(hold_mask.sum()),
        "best_params": asdict(best_params),
        "w_frozen": best_w,
        "fit": fit_score,
        "holdout": hold_score,
        "spa": spa,
        "white_rc": rc,
        "permutation_holdout": perm,
        "verdict": verdict,
        "verdict_reason": ("holdout 95% CI excludes 0 and SPA p<0.05; still a recommendation only"
                           if beats else
                           "default PASS: holdout gap CI includes 0 or SPA does not reject the market"),
        "workers": workers,
        "elapsed_s": (dt.datetime.now() - t0).total_seconds(),
    }

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    base = args.base
    grid_table.to_csv(out_dir / f"{base}_grid.csv", index=False)
    spa_rows = pd.DataFrame([
        {"date": dt.date.today().isoformat(), "series": "hansen_spa", **{k: spa[k] for k in spa}},
        {"date": dt.date.today().isoformat(), "series": "white_rc", **{k: rc[k] for k in rc}},
        {"date": dt.date.today().isoformat(), "series": "permutation_holdout", **{k: perm[k] for k in perm}},
    ])
    spa_rows.to_csv(out_dir / f"{base}_spa.csv", index=False)
    if len(bank_curve):
        bank_curve.to_csv(out_dir / f"{base}_bankroll.csv", index=False)
    bank_stats_f.to_csv(out_dir / f"{base}_bankroll_stats.csv", index=False)
    # tidy daily holdout series for the best config
    daily = hold.groupby(["day", "tour"], sort=True).apply(
        lambda g: pd.Series({
            "n": len(g),
            "bits_gap": float(np.mean(BitsIgn(g["p_mkt"], g["y"]) - BitsIgn(g["p_resid"], g["y"]))),
        }), include_groups=False).reset_index().rename(columns={"day": "date"})
    daily["series"] = daily["tour"].astype(str) + "_holdout_bits_gap"
    daily.to_csv(out_dir / f"{base}.csv", index=False)
    (out_dir / f"{base}_best.json").write_text(json.dumps(headline, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: headline[k] for k in ("verdict", "n_fit", "n_holdout", "w_frozen", "holdout", "spa", "permutation_holdout")},
                     indent=2, default=str), flush=True)
    return headline


if __name__ == "__main__":
    Main()
