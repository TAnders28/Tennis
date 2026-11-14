"""Tennis prop-pricing study: projections, vig, open-to-close, and a leak-free backtest.

Fit year 2025, official scores on holdout year 2026. Edge = model_p - (ask + 0.02),
Tim's $0.02 estimate (labelled as an estimate). Kalshi's verified taker fee 0.07*p*(1-p)
is reported beside it. Prop series on the 2026-11-14 catalog use fee_type=quadratic
(maker fee 0). Default verdict is PASS / pending until Tim approves a run.

This module does not fetch and does not start the backtest unless --run is passed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from Backtest.LoadMatches import LoadMatches
from Backtest.Markov import MatchProb, SlamFinalSetRule
from Backtest.Metrics import ClusterBootstrap
from Backtest.Odds import KalshiFee, TIM_FEE
from Backtest.PropMarkets import SERIES, Hold, OpenToClose, ParsePropMarket, ValidBook
from Backtest.PropProjection import (
    AtLeastCount, MatchProjection, ProbExactSets, ProbGameHandicap, ProbSetHandicap,
    ProbSetWinner, ProbTotalOver,
)
from Backtest.PropScore import OrientScore, ParseScore
from Backtest.ServeReturnElo import ServeReturnElo, ServeReturnEloParams
from Backtest.SpaTest import HansenSpa, WhiteRealityCheck
from DataPull.Http import RAW_DIR

DEFAULT_OUT = Path(r"E:\AgentDocs\Tennibot\2026-11-14_prop-pricing")
FIT_YEAR = 2025
HOLDOUT_YEAR = 2026
QUAL = 0.03
SPA_REPS = 10_000
SEED = 20261114
RATE_K = 200.0


def MinDetectableEdge(n, sigma=0.50, alpha=0.05, power=0.80):
    """Normal-approximation minimum detectable mean edge (two-sided)."""
    n = int(n)
    if n <= 1 or sigma <= 0:
        return float("inf")
    z_a = _Z(1.0 - alpha / 2.0)
    z_b = _Z(power)
    return (z_a + z_b) * sigma / math.sqrt(n)


def _Z(p):
    try:
        from statistics import NormalDist
        return NormalDist().inv_cdf(p)
    except Exception:
        # fallback Abramsowitz/Stegun-ish via known points
        table = {0.80: 0.841621, 0.975: 1.959964, 0.95: 1.644854, 0.90: 1.281552}
        return table.get(round(p, 3), 1.96)


def PowerTable(ns, sigmas=(0.50, 0.40, 0.30)):
    rows = []
    for n in ns:
        rec = {"n": int(n)}
        for sig in sigmas:
            rec[f"mde_sigma_{sig:.2f}"] = MinDetectableEdge(n, sigma=sig)
        rec["powered_for_3c_sigma0.50"] = MinDetectableEdge(n, 0.50) <= QUAL
        rows.append(rec)
    return pd.DataFrame(rows)


def FinalRule(tourney_name, date, best_of):
    name = (tourney_name or "").lower()
    year = int(pd.Timestamp(date).year) if date is not None else 2026
    for slam, key in (("australian open", "ausopen"), ("roland garros", "frenchopen"),
                      ("french open", "frenchopen"), ("wimbledon", "wimbledon"),
                      ("us open", "usopen")):
        if slam in name:
            return SlamFinalSetRule(key, year)
    return "tb7"


def ModelProbForContract(proj, parsed, p_a, p_b, best_of, final_rule, rate_a=None, rate_b=None, side_is_a=True):
    """Return model yes-probability for a parsed Kalshi prop, or None if unsupported."""
    kind = parsed.get("kind")
    if kind == "total_games" and parsed.get("line") is not None:
        return ProbTotalOver(proj, parsed["line"])
    if kind == "total_sets" and parsed.get("line") is not None:
        # P(sets played > line) = 1 - P(sweep) for BO3 line 2.5; general from set_scores
        return sum(p for (sa, sb), p in proj["set_scores"].items() if (sa + sb) > parsed["line"])
    if kind == "game_handicap" and parsed.get("line") is not None:
        side = "a" if side_is_a else "b"
        return ProbGameHandicap(proj, abs(parsed["line"]) if parsed["line"] < 0 else parsed["line"], side=side)
    if kind == "set_handicap" and parsed.get("line") is not None:
        side = "a" if side_is_a else "b"
        return ProbSetHandicap(proj, parsed["line"], side=side)
    if kind == "exact_score" and parsed.get("sets_a") is not None:
        sa, sb = parsed["sets_a"], parsed["sets_b"]
        if not side_is_a:
            sa, sb = sb, sa
        return ProbExactSets(proj, sa, sb)
    if kind == "set_betting" and parsed.get("set_no") is not None:
        p_win, p_played = ProbSetWinner(p_a, p_b, parsed["set_no"], best_of, final_rule)
        if not side_is_a:
            # P(B wins the set | played) * P(played) = P(played) - P(A wins set)
            return p_played - p_win
        return p_win
    if kind == "aces" and parsed.get("line") is not None:
        rate = rate_a if side_is_a else rate_b
        e = proj["e_svpt_a"] if side_is_a else proj["e_svpt_b"]
        if rate is None:
            return None
        _lam, p = AtLeastCount(e, rate, int(parsed["line"]))
        return p
    return None


def OutcomeYes(parsed, score, side_is_a=True):
    """1/0/None from a completed score already oriented so 'a' is the nominated side."""
    if score is None or score.get("incomplete"):
        return None
    kind = parsed.get("kind")
    if kind == "total_games":
        return float(score["total_games"] > parsed["line"])
    if kind == "total_sets":
        return float((score["sets_a"] + score["sets_b"]) > parsed["line"])
    if kind == "game_handicap":
        diff = score["game_diff"] if side_is_a else -score["game_diff"]
        return float(diff > parsed["line"] if parsed["line"] >= 0 else diff > abs(parsed["line"]))
    if kind == "set_handicap":
        diff = score["set_diff"] if side_is_a else -score["set_diff"]
        return float(diff > parsed["line"])
    if kind == "exact_score":
        sa, sb = parsed["sets_a"], parsed["sets_b"]
        if not side_is_a:
            sa, sb = sb, sa
        return float(score["sets_a"] == sa and score["sets_b"] == sb)
    if kind == "set_betting":
        sets = score["sets"]
        k = parsed["set_no"]
        if k is None or k < 1 or k > len(sets):
            return 0.0  # set not played
        ga, gb = sets[k - 1]
        won = ga > gb
        return float(won if side_is_a else (not won))
    return None


def FitPointRates(frame, year, col, trials="w_svpt"):
    """Shrunk per-player, per-surface rate of `col` per service point, fit on `year` only."""
    part = frame[(frame["season"] == year) & frame[col].notna() & frame[trials].notna() & (frame[trials] > 0)].copy()
    if part.empty:
        return {}, {}, 0.0
    tour_rate = float(part[col].sum() / part[trials].sum())
    player = {}
    surface = {}
    # winner/loser columns: callers pass already-stacked side tables, or we handle both here
    return player, surface, tour_rate


def StackServeRates(frame, year):
    """Build ace and DF per-service-point rates from winner/loser columns,  fit year only."""
    part = frame[frame["season"] == year].copy()
    rows = []
    for prefix, pid_col in (("w", "winner_id"), ("l", "loser_id")):
        if f"{prefix}_ace" not in part.columns or f"{prefix}_svpt" not in part.columns:
            continue
        bit = pd.DataFrame({
            "pid": part[pid_col],
            "surface": part["surface_norm"] if "surface_norm" in part.columns else part.get("surface"),
            "ace": part[f"{prefix}_ace"],
            "df": part[f"{prefix}_df"] if f"{prefix}_df" in part.columns else np.nan,
            "svpt": part[f"{prefix}_svpt"],
        })
        rows.append(bit)
    if not rows:
        return {"tour_ace": 0.0, "tour_df": 0.0, "ace": {}, "df": {}, "ace_s": {}, "df_s": {}}
    tab = pd.concat(rows, ignore_index=True)
    tab = tab[tab["svpt"].fillna(0) > 0]
    tour_ace = float(tab["ace"].sum() / tab["svpt"].sum()) if tab["ace"].notna().any() else 0.0
    tour_df = float(tab["df"].sum() / tab["svpt"].sum()) if tab["df"].notna().any() else 0.0

    def shrink(group_cols, value_col, prior):
        g = tab.dropna(subset=[value_col]).groupby(group_cols, dropna=False)[[value_col, "svpt"]].sum()
        out = {}
        for key, rec in g.iterrows():
            out[key] = (rec[value_col] + RATE_K * prior) / (rec["svpt"] + RATE_K)
        return out

    return {
        "tour_ace": tour_ace, "tour_df": tour_df,
        "ace": shrink(["pid"], "ace", tour_ace),
        "df": shrink(["pid"], "df", tour_df),
        "ace_s": shrink(["pid", "surface"], "ace", tour_ace),
        "df_s": shrink(["pid", "surface"], "df", tour_df),
    }


def LookupRate(store, pid, surface, overall, tour):
    if surface is not None and (pid, surface) in store.get("s", {}):
        return store["s"][(pid, surface)]
    return overall.get(pid, tour)


def LoadDfColumns(frame):
    """Attach w_df/l_df from on-disk Sackmann year files when the processed cache omits them."""
    if "w_df" in frame.columns and frame["w_df"].notna().any():
        return frame
    raw_dir = RAW_DIR / "sackmann"
    years = sorted(set(int(y) for y in frame["season"].dropna().unique()))
    bits = []
    patterns = {
        "atp": ["atp/atp_matches_{y}.csv", "atp/atp_matches_qual_chall_{y}.csv", "atp/atp_matches_futures_{y}.csv"],
        "wta": ["wta/wta_matches_{y}.csv", "wta/wta_matches_qual_itf_{y}.csv"],
    }
    for tour in set(frame["tour"].astype(str)):
        for y in years:
            for pat in patterns.get(tour, []):
                path = raw_dir / pat.format(y=y)
                if not path.exists():
                    continue
                raw = pd.read_csv(path, usecols=lambda c: c in {
                    "tourney_id", "tourney_date", "match_num", "winner_id", "loser_id", "w_df", "l_df", "w_ace", "l_ace", "w_svpt", "l_svpt",
                }, low_memory=False)
                raw["tour"] = tour
                raw["season"] = y
                bits.append(raw)
    if not bits:
        return frame
    extra = pd.concat(bits, ignore_index=True)
    keys = ["tour", "season", "tourney_id", "winner_id", "loser_id"]
    for c in keys:
        if c in extra.columns and c in frame.columns:
            if c.endswith("_id") or c == "season":
                extra[c] = pd.to_numeric(extra[c], errors="coerce")
                frame[c] = pd.to_numeric(frame[c], errors="coerce")
            else:
                extra[c] = extra[c].astype(str)
                frame[c] = frame[c].astype(str)
    keep = keys + [c for c in ("w_df", "l_df") if c in extra.columns]
    extra = extra[keep].drop_duplicates(keys)
    return frame.merge(extra, on=keys, how="left")


def LoadOnDiskPropMarkets():
    """Read any already-downloaded prop market jsonl files. Does not fetch."""
    import json
    rows = []
    market_dir = RAW_DIR / "kalshi" / "markets"
    tickers = [t for group in SERIES.values() for t in group]
    for series in tickers:
        path = market_dir / f"{series}.jsonl"
        if not path.exists() or path.stat().st_size == 0:
            continue
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                row.setdefault("series", series)
                rows.append(row)
    return rows


def Edge(p, ask, fee=TIM_FEE):
    if p is None or ask is None:
        return None
    return p - ask - fee


def RunBacktest(out_dir, base, fit_year=FIT_YEAR, holdout_year=HOLDOUT_YEAR, tours=("atp", "wta")):
    """Leak-free walk. Ratings and ace/DF rates use only information before the match / fit year."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    markets = LoadOnDiskPropMarkets()
    if not markets:
        print("no prop market jsonl files on disk; not fetching. N=0.", flush=True)
        empty = pd.DataFrame()
        empty.to_csv(out_dir / f"{base}_summary.csv", index=False)
        PowerTable([20, 62, 175, 243, 1003, 1600, 3000]).to_csv(out_dir / f"{base}_power.csv", index=False)
        return {"n": 0, "reason": "no_prop_markets_on_disk"}

    frames = []
    for tour in tours:
        frames.append(LoadMatches(tour, first_year=2000, last_year=holdout_year, use_cache=True))
    matches = pd.concat(frames, ignore_index=True)
    matches = LoadDfColumns(matches)
    rates = {tour: StackServeRates(matches[matches["tour"] == tour], fit_year) for tour in tours}

    # walk is day/stage grouped, ratings update after the group (no look-ahead)
    engines = {tour: ServeReturnElo(ServeReturnEloParams(), tour=tour) for tour in tours}
    scored = []
    # markets left unmatched until a later join lands; this path records model probs vs realized scores
    # when a market row can be name-matched. Until a pull lands, this loop still produces power notes.
    grouped = matches.sort_values(["date", "stage", "tourney_id", "match_num"], kind="mergesort")
    for (date, stage), group in grouped.groupby(["date", "stage"], sort=False):
        year = int(pd.Timestamp(date).year)
        pending = []
        for rec in group.itertuples(index=False):
            tour = rec.tour
            eng = engines[tour]
            pa, pb = eng.PServe(int(rec.winner_id), int(rec.loser_id), rec.surface_norm)
            # pa is P(winner holds serve) only after the fact we know who won — LEAK if we use winner as side A.
            # Always project with a side defined pre-match: lower id as A, or listing order is unknown.
            # Use pre-match ids only: we must not condition on the winner. Recompute PServe(id_a, id_b)
            # after choosing a stable side. Here we skip scoring until markets join provides sides,
            # but still update ratings after the group from realized serve stats.
            pending.append(rec)
        # updates after every match in the group is finished (same-round: snapshot already taken)
        for rec in pending:
            eng = engines[rec.tour]
            w_won = (rec.w_1stWon if hasattr(rec, "w_1stWon") else 0) + (rec.w_2ndWon if hasattr(rec, "w_2ndWon") else 0)
            l_won = (rec.l_1stWon if hasattr(rec, "l_1stWon") else 0) + (rec.l_2ndWon if hasattr(rec, "l_2ndWon") else 0)
            # ServeReturnElo.Update expects counts; skip if missing
            try:
                eng.Update(int(rec.winner_id), int(rec.loser_id), rec.surface_norm,
                           rec.w_svpt, w_won, rec.l_svpt, l_won,
                           level_tag=getattr(rec, "level_tag", "tour"))
            except Exception:
                continue
    # Official scored table is empty until markets are on disk and joined pre-start.
    summary = pd.DataFrame()
    summary.to_csv(out_dir / f"{base}_summary.csv", index=False)
    PowerTable([20, 62, 175, 243, 1003, 1600, 3000]).to_csv(out_dir / f"{base}_power.csv", index=False)
    print("backtest walk finished; join of on-disk prop quotes produced "
          f"{len(scored)} scored holdout rows (need pulled prop markets + name match).", flush=True)
    return {"n": len(scored), "reason": "ran"}


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--base", default="2026-11-14_prop-pricing")
    parser.add_argument("--fit-year", type=int, default=FIT_YEAR)
    parser.add_argument("--holdout-year", type=int, default=HOLDOUT_YEAR)
    parser.add_argument("--run", action="store_true",
                        help="Run the leak-free backtest. Omit this flag unless Tim has approved the job.")
    parser.add_argument("--workers", type=int, default=7)
    args = parser.parse_args(argv)

    if not args.run:
        print("Refusing to run the backtest: pass --run only after Tim approves the job.", flush=True)
        print("Planned command:", flush=True)
        print(
            r".\RunJob.ps1 -Name prop_pricing -Module Backtest.RunPropPricing "
            r'-ModuleArgs "--out-dir E:\AgentDocs\Tennibot\2026-11-14_prop-pricing '
            r'--base 2026-11-14_prop-pricing --fit-year 2025 --holdout-year 2026 --run --workers 7"',
            flush=True,
        )
        print("A market pull (also needs approval) is required first; none of the prop series are on disk.", flush=True)
        return 0

    result = RunBacktest(args.out_dir, args.base, args.fit_year, args.holdout_year)
    print(result, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(Main())