r"""Validate the tennis match simulator (Sportsbot, 2026-11-26; research, tested:false).

Inputs (all local): data/backtest/<tour>/predictions.parquet (frozen walk-forward serve-point probabilities
p_serve_w / p_serve_l, known before each match), raw Sackmann CSVs (scores, in-match serve stats),
data/backtest/<tour>/kalshi_quotes.parquet (T-60 Kalshi quotes joined to Sackmann match keys).
Levels: tour, tour_qual, challenger (ITF has no point stats); Davis Cup and retirements/walkovers excluded.
Splits: tune = 2020-2022 (model 'validation' seasons), holdout = 2023-2025, forward = 2026 to 2026-06-02.
Player A is the winner or the loser by a hash of the match key (so y is not always 1).

Per match: exact pre-match P(A wins) (Sim.MatchSim.ExactMatchProb), and from Monte Carlo cells (serve probabilities
rounded to 0.01, `--sims` matches per cell): P(straight sets), P(any tiebreak), the total-games distribution and the
PIT of the real total; 'near-miss' P(A wins) replaying the match's own observed serve-point rates, shrunk toward the
pre-match rates by k pseudo-points (k in NM_K).

Usage: .venv\Scripts\python.exe -m Sim.ValidateTennis --out-dir <dir> --base <name> [--sims 2000]
Writes <base>_tennis_matches.csv.
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

from Sim.MatchSim import ExactMatchProb, ExactMatchProbForm, ParseScore, SimulateMatches
SIGMA_GRID = (0.0, 0.03, 0.04, 0.05, 0.06)
LAM_GRID = (1.0, 1.25, 1.5, 1.75, 2.0)


def Stretch(d, lam):
    """Scale the pre-match serve-rate gap by lam around its midpoint (lam=1: unchanged)."""
    out = d.copy()
    gap = (d["p_a_serve"] - d["p_b_serve"]) * (lam - 1) / 2
    out["p_a_serve"] = (d["p_a_serve"] + gap).clip(0.05, 0.98)
    out["p_b_serve"] = (d["p_b_serve"] - gap).clip(0.05, 0.98)
    return out


def Ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(bins) if (b == k).any()))

REPO = Path(__file__).resolve().parents[1]
SACK = REPO / "data" / "raw" / "sackmann"
LEVELS = ("tour", "tour_qual", "challenger")
NM_K = (0, 30, 100)
MAX_GAMES = 80


def RawStats(tour, years):
    rows = []
    for y in years:
        for kind in (f"{tour}_matches_{y}.csv", f"{tour}_matches_qual_chall_{y}.csv", f"{tour}_matches_qual_itf_{y}.csv"):
            p = SACK / tour / kind
            if not p.exists():
                continue
            d = pd.read_csv(p, low_memory=False, usecols=lambda c: c in {"tourney_id", "round", "winner_id", "loser_id", "w_svpt", "w_1stWon",
                                                                           "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"})
            d["match_key"] = tour + ":" + d["tourney_id"].astype(str) + ":" + d["round"].astype(str) + ":" + d["winner_id"].astype(str) + "-" + d["loser_id"].astype(str)
            rows.append(d[["match_key", "w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon"]])
    return pd.concat(rows).drop_duplicates("match_key").set_index("match_key")


def Load():
    frames = []
    for tour in ("atp", "wta"):
        p = pd.read_parquet(REPO / "data" / "backtest" / tour / "predictions.parquet",
                            columns=["match_key", "tour", "date", "level_tag", "tourney_level", "tourney_name", "round", "surface_norm", "best_of",
                                     "winner_id", "loser_id", "score", "is_ret", "p_serve_w", "p_serve_l", "p_blend", "p_elo"])
        p = p[(p["date"] >= "2020-01-01") & p["level_tag"].isin(LEVELS) & ~p["is_ret"] & (p["tourney_level"] != "D")]
        p = p.dropna(subset=["p_serve_w", "p_serve_l"]).drop_duplicates("match_key")
        st = RawStats(tour, range(2020, 2027))
        p = p.join(st, on="match_key")
        q = pd.read_parquet(REPO / "data" / "backtest" / tour / "kalshi_quotes.parquet")
        q = q[(q["result_agrees"] == 1.0) & ~q["possible_inplay"].astype(bool)].drop_duplicates("match_key")
        q = q.set_index("match_key")[["event_ticker", "kalshi_level", "id_a", "t60_bid", "t60_ask"]]
        p = p.join(q, on="match_key")
        frames.append(p)
    d = pd.concat(frames, ignore_index=True)
    d["year"] = pd.to_datetime(d["date"]).dt.year
    d["split"] = np.where(d["year"] <= 2022, "tune", np.where(d["year"] <= 2025, "holdout", "forward"))
    sets = d["score"].map(ParseScore)
    d = d[sets.notna()].copy()
    sets = sets[sets.notna()]
    d["real_total_games"] = sets.map(lambda s: sum(w + l for w, l, _ in s))
    d["real_sets"] = sets.map(len)
    d["real_tb"] = sets.map(lambda s: int(any(t for _, _, t in s)))
    need = (d["best_of"] + 1) // 2
    d = d[(d["real_sets"] >= need) & (d["real_sets"] <= d["best_of"])].copy()
    d["real_straight"] = (d["real_sets"] == (d["best_of"] + 1) // 2).astype(int)
    d["final_tb"] = np.where((d["tourney_level"] == "G") & (d["year"] >= 2022), 10, 7)
    a_is_w = d["match_key"].map(lambda k: zlib.crc32(k.encode()) % 2 == 0).to_numpy()
    d["a_is_winner"] = a_is_w.astype(int)
    d["y"] = a_is_w.astype(float)
    d["p_a_serve"] = np.where(a_is_w, d["p_serve_w"], d["p_serve_l"])
    d["p_b_serve"] = np.where(a_is_w, d["p_serve_l"], d["p_serve_w"])
    d["p_blend_a"] = np.where(a_is_w, d["p_blend"], 1 - d["p_blend"])
    # observed in-match serve rates
    w_won = d["w_1stWon"] + d["w_2ndWon"]
    l_won = d["l_1stWon"] + d["l_2ndWon"]
    d["obs_a_won"] = np.where(a_is_w, w_won, l_won)
    d["obs_a_pts"] = np.where(a_is_w, d["w_svpt"], d["l_svpt"])
    d["obs_b_won"] = np.where(a_is_w, l_won, w_won)
    d["obs_b_pts"] = np.where(a_is_w, d["l_svpt"], d["w_svpt"])
    # Kalshi T-60 mid for A (id_a is Kalshi's YES player)
    has_k = d["id_a"].notna()
    yes_is_w = d["id_a"].astype("float") == d["winner_id"].astype("float")
    mid = (d["t60_bid"] + d["t60_ask"]) / 2
    yes_is_a = np.where(a_is_w, yes_is_w, ~yes_is_w)
    d["p_kalshi_a"] = np.where(has_k, np.where(yes_is_a, mid, 1 - mid), np.nan)
    d["k_ask_a"] = np.where(has_k, np.where(yes_is_a, d["t60_ask"], 1 - d["t60_bid"]), np.nan)
    d["k_ask_b"] = np.where(has_k, np.where(yes_is_a, 1 - d["t60_bid"], d["t60_ask"]), np.nan)
    return d.reset_index(drop=True)


def CellSims(d, sims, seed, sigma=0.0):
    rng = np.random.default_rng(seed)
    key = pd.DataFrame({"pa": (d["p_a_serve"] * 100).round().astype(int), "pb": (d["p_b_serve"] * 100).round().astype(int),
                        "bo": d["best_of"].astype(int), "ft": d["final_tb"].astype(int)})
    cells = key.drop_duplicates().reset_index(drop=True)
    cells["cell"] = np.arange(len(cells))
    key = key.merge(cells, on=["pa", "pb", "bo", "ft"], how="left")
    hist = np.zeros((len(cells), MAX_GAMES + 1))
    p_win = np.zeros(len(cells))
    p_straight = np.zeros(len(cells))
    p_tb = np.zeros(len(cells))
    batch = max(1, 1_000_000 // sims)
    for s in range(0, len(cells), batch):
        c = cells.iloc[s:s + batch]
        rep = np.repeat(np.arange(len(c)), sims)
        r = SimulateMatches(np.repeat(c["pa"].to_numpy() / 100, sims), np.repeat(c["pb"].to_numpy() / 100, sims),
                            np.repeat(c["bo"].to_numpy(), sims), np.repeat(c["ft"].to_numpy(), sims), rng, sigma=sigma)
        tg = np.minimum(r["total_games"], MAX_GAMES)
        h = np.zeros((len(c), MAX_GAMES + 1))
        np.add.at(h, (rep, tg), 1.0)
        hist[s:s + len(c)] = h / sims
        p_win[s:s + len(c)] = np.bincount(rep, weights=r["a_won"], minlength=len(c)) / sims
        p_straight[s:s + len(c)] = np.bincount(rep, weights=r["straight"], minlength=len(c)) / sims
        p_tb[s:s + len(c)] = np.bincount(rep, weights=(r["tiebreaks"] > 0), minlength=len(c)) / sims
    ci = key["cell"].to_numpy()
    t = d["real_total_games"].to_numpy().astype(int)
    cdf = np.cumsum(hist, axis=1)
    below = np.where(t > 0, cdf[ci, np.clip(t - 1, 0, MAX_GAMES)], 0.0)
    at = hist[ci, np.clip(t, 0, MAX_GAMES)]
    out = pd.DataFrame({"sim_p_a_cell": p_win[ci], "sim_p_straight": p_straight[ci], "sim_p_tb": p_tb[ci],
                        "sim_total_mean": (hist * np.arange(MAX_GAMES + 1)).sum(axis=1)[ci], "pit_total": below + 0.5 * at})
    return out, len(cells)


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20261126)
    a = ap.parse_args()
    t0 = time.time()
    d = Load()
    print("loaded", len(d), round(time.time() - t0), flush=True)
    d["p_sim"] = ExactMatchProb(d["p_a_serve"].to_numpy(), d["p_b_serve"].to_numpy(), d["best_of"].to_numpy(), d["final_tb"].to_numpy())
    cs, n_cells = CellSims(d, a.sims, a.seed)
    d = pd.concat([d, cs], axis=1)
    # form-shock tuning on the tune split only (2020-2022). Two knobs: sigma (match-level form shock) and lam
    # (stretch of the pre-match serve-rate gap, since a shrunk estimate plus symmetric noise is too flat).
    # Criterion (pre-declared): |straight-sets err| + |tiebreak err| (best-of-3) + |upset err| + ECE, on tune.
    tune = d[(d["split"] == "tune")].reset_index(drop=True)
    b3 = (tune["best_of"] == 3).to_numpy()
    grid = []
    for lam in LAM_GRID:
        tt = Stretch(tune, lam)
        for sg in SIGMA_GRID:
            c, _ = CellSims(tt, max(500, a.sims // 2), a.seed + 1, sigma=sg)
            pf = ExactMatchProbForm(tt["p_a_serve"].to_numpy(), tt["p_b_serve"].to_numpy(), tt["best_of"].to_numpy(), tt["final_tb"].to_numpy(), sg, nodes=7)
            y = tt["y"].to_numpy()
            upset_err = abs(np.where(pf >= 0.5, y < 0.5, y > 0.5).mean() - np.minimum(pf, 1 - pf).mean())
            ece = Ece(pf, y)
            err = (abs(c["sim_p_straight"][b3].mean() - tt["real_straight"][b3].mean()) + abs(c["sim_p_tb"][b3].mean() - tt["real_tb"][b3].mean())
                   + upset_err + ece)
            grid.append({"lam": lam, "sigma": sg, "err": float(err), "sim_straight_bo3": float(c["sim_p_straight"][b3].mean()),
                         "real_straight_bo3": float(tt["real_straight"][b3].mean()), "sim_tb_bo3": float(c["sim_p_tb"][b3].mean()),
                         "real_tb_bo3": float(tt["real_tb"][b3].mean()), "upset_err": float(upset_err), "ece": float(ece),
                         "brier": float(np.mean((pf - y) ** 2)), "cover_total_80": float((np.abs(c["pit_total"] - 0.5) <= 0.4).mean())})
            print("tune", grid[-1], flush=True)
    best = min(grid, key=lambda r: r["err"])
    lam, sigma = best["lam"], best["sigma"]
    dd = Stretch(d, lam)
    cf, _ = CellSims(dd, a.sims, a.seed + 2, sigma=sigma)
    cf.columns = [c + "_form" for c in cf.columns]
    d = pd.concat([d, cf], axis=1)
    d["p_sim_form"] = ExactMatchProbForm(dd["p_a_serve"].to_numpy(), dd["p_b_serve"].to_numpy(), d["best_of"].to_numpy(), d["final_tb"].to_numpy(), sigma)
    d["sigma_form"] = sigma
    d["lam_form"] = lam
    (Path(a.out_dir) / f"{a.base}_tennis_form_grid.json").write_text(json.dumps({"grid": grid, "chosen": best}, indent=1))
    print("cells", n_cells, round(time.time() - t0), flush=True)
    ok = d["obs_a_pts"].gt(0) & d["obs_b_pts"].gt(0) & (d["obs_a_won"] <= d["obs_a_pts"]) & (d["obs_b_won"] <= d["obs_b_pts"])
    for k in NM_K:
        ra = ((d["obs_a_won"] + k * d["p_a_serve"]) / (d["obs_a_pts"] + k)).clip(0.05, 0.98)
        rb = ((d["obs_b_won"] + k * d["p_b_serve"]) / (d["obs_b_pts"] + k)).clip(0.05, 0.98)
        v = np.full(len(d), np.nan)
        m = ok.to_numpy()
        v[m] = ExactMatchProb(ra[m].to_numpy(), rb[m].to_numpy(), d.loc[m, "best_of"].to_numpy(), d.loc[m, "final_tb"].to_numpy())
        d[f"p_nm_k{k}"] = v
    print("near-miss done", round(time.time() - t0), flush=True)
    cols = ["match_key", "tour", "date", "split", "level_tag", "tourney_level", "tourney_name", "round", "surface_norm", "best_of", "final_tb",
            "score", "a_is_winner", "y", "p_a_serve", "p_b_serve", "p_sim", "sim_p_a_cell", "p_blend_a", "sim_p_straight", "real_straight",
            "sim_p_tb", "real_tb", "sim_total_mean", "real_total_games", "pit_total", "obs_a_won", "obs_a_pts", "obs_b_won", "obs_b_pts",
            "event_ticker", "kalshi_level", "p_kalshi_a", "k_ask_a", "k_ask_b"] + [f"p_nm_k{k}" for k in NM_K] + [
            "sigma_form", "lam_form", "p_sim_form", "sim_p_a_cell_form", "sim_p_straight_form", "sim_p_tb_form", "sim_total_mean_form", "pit_total_form"]
    od = Path(a.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    d[cols].to_csv(od / f"{a.base}_tennis_matches.csv", index=False)
    print(json.dumps({"matches": len(d), "cells": n_cells, "with_kalshi": int(d["p_kalshi_a"].notna().sum()), "seconds": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    Main()
