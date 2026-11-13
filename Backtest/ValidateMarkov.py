r"""Validate the Markov in-play win probability on Grand Slam point-by-point data.

Inputs: Sackmann tennis_slam_pointbypoint (mirror; 2011-2024, AO/RG missing for 2022-2024,
Wimbledon 2020 cancelled), plus each player's PRE-MATCH serve-point probability from the
frozen serve/return model (data/backtest/<tour>/predictions.parquet, leak-free).
For every point we rebuild the score state from the point winners/servers with the
event's rules (final-set rule by slam and year) and compute P(player1 wins).
In-match learning (optional, tuned): p_serve <- (m * p_prior + serve points won) / (m + serve points).

Splits by slam year: tune 2011-2019 (pick m), validation 2020-2021, holdout 2022-2024.
Validated := on the holdout, (a) ECE <= 0.03, (b) every reliability bin with >= 50 matches has
|observed - predicted| <= 0.05 or the match-level bootstrap CI covers the prediction, and
(c) live log loss beats the constant pre-match probability.
Matches that retired, could not be joined, or whose replay disagrees with the recorded
set numbers / final winner are dropped and counted (a data-quality filter, not a model one).

Usage: .venv\Scripts\python.exe -m Backtest.ValidateMarkov --workers 8
Writes data/backtest/markov/{points_scored.parquet, summary.json, reliability.csv}
"""
from __future__ import annotations

import argparse
import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from Backtest import Markov
from Backtest.Metrics import ClusterBootstrap, Ece, NaiveCi, Reliability
from Backtest.NameMatch import NameIndex

REPO_ROOT = Path(__file__).resolve().parents[1]
PBP_DIR = REPO_ROOT / "data" / "raw" / "sackmann" / "slam_pbp"
BT_DIR = REPO_ROOT / "data" / "backtest"
OUT_DIR = BT_DIR / "markov"
SLAM_NAMES = {"ausopen": "Australian Open", "frenchopen": "Roland Garros", "wimbledon": "Wimbledon", "usopen": "Us Open"}
M_GRID = [None, 400.0, 150.0, 60.0]
SPLIT = lambda year: "tune" if year <= 2019 else ("validation" if year <= 2021 else "holdout")
QUANT = 0.0025


def Q(p):
    return round(round(p / QUANT) * QUANT, 6)


class Scorer:
    """Replays a match from point winners. State before each point is exposed via State()."""

    def __init__(self, best_of, final_rule):
        self.best_of, self.final_rule = best_of, final_rule
        self.need = best_of // 2 + 1
        self.sets = [0, 0]
        self.games = [0, 0]
        self.points = [0, 0]
        self.set_no = 1
        self.done = False

    def Rule(self):
        return self.final_rule if self.sets[0] == self.need - 1 and self.sets[1] == self.need - 1 else "tb7"

    def InTiebreak(self):
        tb_at = Markov._TbAt(self.Rule())[0]
        return tb_at is not None and self.games[0] == tb_at and self.games[1] == tb_at

    def Point(self, winner):
        w, l = winner, 1 - winner
        self.points[w] += 1
        if self.InTiebreak():
            target = Markov._TbAt(self.Rule())[1]
            if self.points[w] >= target and self.points[w] - self.points[l] >= 2:
                self._SetWon(w)
            return
        if self.points[w] >= 4 and self.points[w] - self.points[l] >= 2:
            self.points = [0, 0]
            self.games[w] += 1
            if self.games[w] >= 6 and self.games[w] - self.games[l] >= 2:
                self._SetWon(w)

    def _SetWon(self, w):
        self.sets[w] += 1
        self.games = [0, 0]
        self.points = [0, 0]
        self.set_no += 1
        if self.sets[w] >= self.need:
            self.done = True
            self.winner = w


def LoadSlam(year, slam):
    matches_path = PBP_DIR / f"{year}-{slam}-matches.csv"
    points_path = PBP_DIR / f"{year}-{slam}-points.csv"
    if not (matches_path.exists() and points_path.exists()):
        return None, None
    matches = pd.read_csv(matches_path, encoding="utf-8", encoding_errors="replace")
    points = pd.read_csv(points_path, low_memory=False, encoding="utf-8", encoding_errors="replace")
    points = points[["match_id", "SetNo", "PointNumber", "PointWinner", "PointServer"]].copy()
    points["PointWinner"] = pd.to_numeric(points["PointWinner"], errors="coerce")
    points["PointServer"] = pd.to_numeric(points["PointServer"], errors="coerce")
    points = points[points["PointWinner"].isin([1, 2]) & points["PointServer"].isin([1, 2])]
    return matches, points


def JoinPrematch(year, slam, matches, preds_by_tour, index_by_tour):
    joined = []
    for row in matches.itertuples(index=False):
        num = int(row.match_num)
        if num >= 3000:
            continue  # doubles / mixed / juniors
        tour = "atp" if num < 2000 else "wta"
        preds = preds_by_tour[tour]
        slam_rows = preds[(preds["season"] == year) & (preds["tourney_level"] == "G")
                          & preds["tourney_name"].str.lower().str.contains(SLAM_NAMES[slam].lower()[:9])]
        ids1 = index_by_tour[tour].LookupFull(row.player1)
        ids2 = index_by_tour[tour].LookupFull(row.player2)
        hit = slam_rows[(slam_rows["winner_id"].isin(ids1) & slam_rows["loser_id"].isin(ids2))
                        | (slam_rows["winner_id"].isin(ids2) & slam_rows["loser_id"].isin(ids1))]
        if len(hit) != 1:
            joined.append({"match_id": row.match_id, "status": "no_join" if len(hit) == 0 else "ambiguous"})
            continue
        h = hit.iloc[0]
        p1_is_winner = h["winner_id"] in ids1
        joined.append({"match_id": row.match_id, "status": "retired" if h["is_ret"] else "ok", "tour": tour,
                       "best_of": int(h["best_of"]), "p1_won": bool(p1_is_winner),
                       "p_serve_1": float(h["p_serve_w"] if p1_is_winner else h["p_serve_l"]),
                       "p_serve_2": float(h["p_serve_l"] if p1_is_winner else h["p_serve_w"]),
                       "p_pre_1": float(h["p_blend"] if p1_is_winner else 1 - h["p_blend"]),
                       "p_markov_pre_1": float(h["p_sr"] if p1_is_winner else 1 - h["p_sr"])})
    return pd.DataFrame(joined)


def ScoreMatch(points, info, final_rule, m_values):
    scorer = Scorer(info["best_of"], final_rule)
    served = [0, 0]
    won = [0, 0]
    rows = []
    for set_no, pw, ps in zip(points["SetNo"].to_numpy(), points["PointWinner"].to_numpy(), points["PointServer"].to_numpy()):
        if scorer.done:
            return None, "points_after_end"
        try:
            if int(set_no) != scorer.set_no:
                return None, "set_mismatch"
        except (TypeError, ValueError):
            return None, "set_mismatch"
        server = "a" if ps == 1 else "b"
        record = {"set_no": scorer.set_no, "sets_1": scorer.sets[0], "sets_2": scorer.sets[1],
                  "games_1": scorer.games[0], "games_2": scorer.games[1], "in_tb": scorer.InTiebreak()}
        for m in m_values:
            p1, p2 = info["p_serve_1"], info["p_serve_2"]
            if m is not None:
                p1 = (m * p1 + won[0]) / (m + served[0])
                p2 = (m * p2 + won[1]) / (m + served[1])
            p1, p2 = Q(min(max(p1, 0.35), 0.92)), Q(min(max(p2, 0.35), 0.92))
            try:
                prob = Markov.LiveWinProb(p1, p2, scorer.sets[0], scorer.sets[1], scorer.games[0], scorer.games[1],
                                          scorer.points[0], scorer.points[1], server, info["best_of"], final_rule)
            except (ValueError, RecursionError):
                return None, "invalid_state"
            record[f"p_m{'inf' if m is None else int(m)}"] = prob
        rows.append(record)
        idx = 0 if ps == 1 else 1
        served[idx] += 1
        if (pw == 1) == (ps == 1):
            won[idx] += 1
        scorer.Point(0 if pw == 1 else 1)
    if not scorer.done:
        return None, "unfinished"
    if (scorer.winner == 0) != info["p1_won"]:
        return None, "winner_mismatch"
    return rows, "ok"


def _Work(args):
    year, slam = args
    matches, points = LoadSlam(year, slam)
    if matches is None:
        return [], []
    preds = {t: pd.read_parquet(BT_DIR / t / "predictions.parquet",
                                columns=["season", "tourney_level", "tourney_name", "winner_id", "winner_name", "loser_id",
                                         "loser_name", "is_ret", "best_of", "p_serve_w", "p_serve_l", "p_blend", "p_sr"])
             for t in ("atp", "wta")}
    for t in preds:
        preds[t] = preds[t][preds[t]["tourney_level"] == "G"]
    index = {t: NameIndex(pd.concat([preds[t][["winner_id", "winner_name"]].set_axis(["i", "n"], axis=1),
                                     preds[t][["loser_id", "loser_name"]].set_axis(["i", "n"], axis=1)])
                          .drop_duplicates("i").itertuples(index=False, name=None)) for t in preds}
    joined = JoinPrematch(year, slam, matches, preds, index)
    final_rule = Markov.SlamFinalSetRule(slam, year)
    scored, status = [], []
    by_match = dict(tuple(points.groupby("match_id", sort=False)))
    for info in joined.to_dict("records"):
        if info["status"] != "ok":
            status.append({"match_id": info["match_id"], "year": year, "slam": slam, "status": info["status"]})
            continue
        pts = by_match.get(info["match_id"])
        if pts is None or len(pts) < 20:
            status.append({"match_id": info["match_id"], "year": year, "slam": slam, "status": "no_points"})
            continue
        rows, result = ScoreMatch(pts, info, final_rule, M_GRID)
        for cache in (Markov.GameFrom, Markov.TiebreakFrom, Markov.SetFrom, Markov.MatchFrom):
            cache.cache_clear()
        status.append({"match_id": info["match_id"], "year": year, "slam": slam, "status": result, "tour": info["tour"]})
        if rows is None:
            continue
        for k, r in enumerate(rows):
            r.update({"match_id": info["match_id"], "year": year, "slam": slam, "tour": info["tour"], "point_index": k,
                      "y": 1.0 if info["p1_won"] else 0.0, "p_pre": info["p_pre_1"], "p_markov_pre": info["p_markov_pre_1"]})
        scored.extend(rows)
    return scored, status


def LogLoss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def Brier(p, y):
    return float(np.mean((p - y) ** 2))


def Summarise(points, column):
    out = {}
    for split, part in points.groupby("split"):
        p, y = part[column].to_numpy(), part["y"].to_numpy()
        ll_values = -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))
        out[split] = {"n_points": int(len(part)), "n_matches": int(part["match_id"].nunique()),
                      "log_loss": LogLoss(p, y), "brier": Brier(p, y), "ece": Ece(p, y),
                      "log_loss_ci_naive": NaiveCi(ll_values), "log_loss_ci_match_bootstrap": ClusterBootstrap(ll_values, part["match_id"]),
                      "pregame_const_log_loss": LogLoss(part["p_pre"].to_numpy(), y),
                      "pregame_const_brier": Brier(part["p_pre"].to_numpy(), y)}
    return out


def ReliabilityWithCi(part, column, bins=10):
    rows = []
    p, y, mid = part[column].to_numpy(), part["y"].to_numpy(), part["match_id"].to_numpy()
    for r in Reliability(p, y, bins):
        mask = (p >= r["bin_lo"]) & ((p < r["bin_hi"]) if r["bin_hi"] < 1 else (p <= r["bin_hi"]))
        r["n_matches"] = int(len(np.unique(mid[mask])))
        r["ci_naive"] = NaiveCi(y[mask]) if mask.sum() > 1 else (np.nan, np.nan)
        r["ci_match_bootstrap"] = ClusterBootstrap(y[mask], mid[mask], reps=400) if mask.sum() > 1 else (np.nan, np.nan)
        rows.append(r)
    return rows


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--years", nargs=2, type=int, default=[2011, 2024])
    args = parser.parse_args(argv)
    jobs = [(y, s) for y in range(args.years[0], args.years[1] + 1) for s in SLAM_NAMES]
    with Pool(min(args.workers, (os.cpu_count() or 2) // 2)) as pool:
        results = pool.map(_Work, jobs, chunksize=1)
    points = pd.DataFrame([r for scored, _ in results for r in scored])
    status = pd.DataFrame([s for _, st in results for s in st])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    points["split"] = points["year"].map(SPLIT)
    points["phase"] = np.where(points["point_index"] == 0, "pre_first_point",
                               np.where(points["in_tb"], "tiebreak", "set_" + points["set_no"].astype(str)))
    points.to_parquet(OUT_DIR / "points_scored.parquet", index=False)
    columns = [c for c in points.columns if c.startswith("p_m")]
    tune = points[points["split"] == "tune"]
    tune_ll = {c: LogLoss(tune[c].to_numpy(), tune["y"].to_numpy()) for c in columns}
    chosen = min(tune_ll, key=tune_ll.get)
    holdout = points[points["split"] == "holdout"]
    reliability = ReliabilityWithCi(holdout, chosen)
    phases = []
    for (split, phase), part in points.groupby(["split", "phase"]):
        phases.append({"split": split, "phase": phase, "n_points": int(len(part)), "n_matches": int(part["match_id"].nunique()),
                       "log_loss": LogLoss(part[chosen].to_numpy(), part["y"].to_numpy()),
                       "brier": Brier(part[chosen].to_numpy(), part["y"].to_numpy()),
                       "pregame_const_log_loss": LogLoss(part["p_pre"].to_numpy(), part["y"].to_numpy())})
    summary = Summarise(points, chosen)
    h = summary.get("holdout", {})
    bins_ok = all(abs(r["observed"] - r["mean_pred"]) <= 0.05 or (r["ci_match_bootstrap"][0] <= r["mean_pred"] <= r["ci_match_bootstrap"][1])
                  for r in reliability if r["n_matches"] >= 50)
    validated = bool(h and h["ece"] <= 0.03 and bins_ok and h["log_loss"] < h["pregame_const_log_loss"])
    result = {"chosen_column": chosen, "tune_log_loss_by_m": tune_ll, "summary": summary, "phases": phases,
              "holdout_reliability": reliability, "validated": validated,
              "criteria": "holdout ECE <= 0.03; bins with >=50 matches within 0.05 or match-bootstrap CI; live log loss < constant pre-match",
              "match_status_counts": status.groupby(["status"]).size().to_dict(),
              "match_status_by_split": status.assign(split=status["year"].map(SPLIT)).groupby(["split", "status"]).size().unstack(fill_value=0).to_dict(orient="index"),
              "note": "pre-match serve probabilities come from the frozen SR model; points within a match are correlated, so N(matches) is the effective sample size"}
    (OUT_DIR / "summary.json").write_text(json.dumps(result, indent=1, default=float))
    pd.DataFrame(reliability).to_csv(OUT_DIR / "reliability.csv", index=False)
    pd.DataFrame(phases).to_csv(OUT_DIR / "phases.csv", index=False)
    print(json.dumps({"chosen": chosen, "validated": validated, "tune_ll": tune_ll, "holdout": h,
                      "status": result["match_status_counts"]}, indent=1, default=float))


if __name__ == "__main__":
    Main()