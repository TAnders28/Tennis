r"""Tune -> validate -> freeze -> score once on the untouched holdout.

Splits (same for ATP and WTA): tune 2010-2019, validation 2020-2022, holdout 2023-2025,
forward 2026 (partial season, Sackmann snapshot ends early June 2026).
Ranking metric: log loss on tour-level main-draw matches; configs ranked by their WORST tune
season; the top 5 then go to validation and the one with the best worst-validation-season wins.
The search never computes holdout or forward metrics; those come only from the frozen run.

Writes data/backtest/<tour>/: search_elo.csv, search_sr.csv, search_blend.csv, frozen.json,
predictions.parquet, metrics.csv (per season x level x model).

Usage:
  .venv\Scripts\python.exe -m Backtest.RunBacktest --tour atp --workers 10
  .venv\Scripts\python.exe -m Backtest.RunBacktest --tour wta --workers 10
  .venv\Scripts\python.exe -m Backtest.RunBacktest --tour atp --frozen-only   (re-score with data/backtest/atp/frozen.json)
"""
from __future__ import annotations

import argparse
import datetime as dt
import itertools
import json
import os
from dataclasses import asdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from Backtest.Elo import EloParams
from Backtest.Engine import Blend, RunModels
from Backtest.LoadMatches import LoadMatches
from Backtest.Metrics import Score
from Backtest.ServeReturn import ServeReturnParams

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = REPO_ROOT / "data" / "backtest"
TUNE = list(range(2010, 2020))
VALIDATION = [2020, 2021, 2022]
HOLDOUT = [2023, 2024, 2025]
FORWARD = [2026]
SPLIT_OF = {**{s: "tune" for s in TUNE}, **{s: "validation" for s in VALIDATION},
            **{s: "holdout" for s in HOLDOUT}, **{s: "forward" for s in FORWARD}}
RANK_LEVEL = "tour"
TOP_N = 5
BLEND_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]


def EloGrid():
    return [EloParams(k_a=a, k_b=5.0, k_c=c, surface_weight=w, bo5_mult=bo5, lower_k_mult=low)
            for a, c, w, bo5, low in itertools.product([150.0, 250.0, 350.0], [0.3, 0.45], [0.0, 0.25, 0.5, 0.75, 1.0],
                                                       [1.0, 1.15], [1.0, 0.75])]


def SrGrid():
    return [ServeReturnParams(shrink_k=k, half_life=h, overall_weight=w)
            for k, h, w in itertools.product([200.0, 800.0, 3000.0], [50.0, 150.0, 1e9], [0.2, 0.5, 1.0])]


def RefineEloGrid(center):
    """Round 2: a local grid around the round-1 pick (incl. the pick itself)."""
    c = center
    grid = set()
    for a in (c["k_a"] * 0.67, c["k_a"], c["k_a"] * 1.33):
        for kc in (c["k_c"] - 0.1, c["k_c"], c["k_c"] + 0.1):
            for w in (c["surface_weight"] - 0.1, c["surface_weight"], c["surface_weight"] + 0.1):
                for bo5 in (c["bo5_mult"], c["bo5_mult"] + 0.15):
                    grid.add(EloParams(k_a=round(a, 1), k_b=c["k_b"], k_c=round(max(kc, 0.05), 3),
                                       surface_weight=round(min(max(w, 0.0), 1.0), 3), bo5_mult=round(bo5, 3),
                                       lower_k_mult=c["lower_k_mult"]))
    return sorted(grid, key=lambda p: asdict(p).values().__repr__())


def RefineSrGrid(center):
    c = center
    grid = set()
    halves = (c["half_life"],) if c["half_life"] >= 1e8 else (c["half_life"] * 0.6, c["half_life"], c["half_life"] * 1.6)
    for k in (c["shrink_k"] * 0.5, c["shrink_k"], c["shrink_k"] * 2):
        for h in halves:
            for w in (c["overall_weight"] * 0.5, c["overall_weight"], min(c["overall_weight"] * 2, 1.0)):
                grid.add(ServeReturnParams(shrink_k=round(k, 1), half_life=round(h, 1), overall_weight=round(w, 3)))
    return sorted(grid, key=lambda p: asdict(p).values().__repr__())


_FRAME = None
_TOUR = None


def _Init(tour):
    global _FRAME, _TOUR
    _TOUR = tour
    _FRAME = LoadMatches(tour)
    try:
        import psutil  # optional
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def SeasonScores(frame, p, seasons, level=RANK_LEVEL):
    out = {}
    mask_level = frame["level_tag"].to_numpy() == level
    season_arr = frame["season"].to_numpy()
    for season in seasons:
        mask = mask_level & (season_arr == season)
        out[season] = Score(p[mask])
    return out


def _EvalElo(params):
    result = RunModels(_FRAME, params, None, _TOUR)
    scores = SeasonScores(_FRAME, result["p_elo"], TUNE + VALIDATION)
    return asdict(params), scores


def _EvalSr(params):
    result = RunModels(_FRAME, None, params, _TOUR)
    scores = SeasonScores(_FRAME, result["p_sr"], TUNE + VALIDATION)
    return asdict(params), scores


def SearchTable(results):
    rows = []
    for params, scores in results:
        row = dict(params)
        tune_ll = [scores[s]["log_loss"] for s in TUNE]
        val_ll = [scores[s]["log_loss"] for s in VALIDATION]
        row.update({"tune_worst_ll": max(tune_ll), "tune_mean_ll": float(np.mean(tune_ll)),
                    "val_worst_ll": max(val_ll), "val_mean_ll": float(np.mean(val_ll)),
                    "tune_mean_acc": float(np.mean([scores[s]["accuracy"] for s in TUNE]))})
        rows.append(row)
    table = pd.DataFrame(rows).sort_values("tune_worst_ll").reset_index(drop=True)
    table["tune_rank"] = np.arange(1, len(table) + 1)
    return table


def PickFromTable(table):
    top = table.head(TOP_N).sort_values("val_worst_ll")
    return top.iloc[0].to_dict()


def MetricsTable(frame, preds):
    rows = []
    for (season, level), index in frame.groupby(["season", "level_tag"]).groups.items():
        index = np.asarray(index)
        for model in ("elo", "sr", "blend"):
            score = Score(preds[f"p_{model}"][index])
            rows.append({"season": int(season), "split": SPLIT_OF.get(int(season), "warmup"), "level": level,
                         "model": model, **score})
    return pd.DataFrame(rows).sort_values(["level", "model", "season"])


def FrozenRun(tour, frame, frozen):
    elo_params = EloParams(**frozen["elo"])
    sr_params = ServeReturnParams(**frozen["sr"])
    result = RunModels(frame, elo_params, sr_params, tour, record_state=True)
    weight = frozen["blend_weight_sr"]
    p_blend = np.array([Blend(a, b, weight) for a, b in zip(result["p_elo"], result["p_sr"])])
    preds = frame[["match_key", "tour", "season", "group_id", "tourney_id", "tourney_name", "tourney_level", "level_tag",
                   "date", "round", "surface_norm", "best_of", "winner_id", "winner_name", "loser_id", "loser_name",
                   "score", "is_ret"]].copy()
    preds["p_elo"] = result["p_elo"]
    preds["p_sr"] = result["p_sr"]
    preds["p_blend"] = p_blend
    preds["p_serve_w"] = result["p_serve_w"]
    preds["p_serve_l"] = result["p_serve_l"]
    preds["elo_diff"] = result["elo_diff"]
    preds["split"] = preds["season"].map(SPLIT_OF).fillna("warmup")
    return preds, result


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tour", choices=["atp", "wta"], required=True)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--frozen-only", action="store_true")
    parser.add_argument("--refine", action="store_true", help="round 2: local grid around data/backtest/<tour>/frozen.json")
    args = parser.parse_args(argv)
    out_dir = OUT_ROOT / args.tour
    out_dir.mkdir(parents=True, exist_ok=True)
    frame = LoadMatches(args.tour)
    if args.frozen_only:
        frozen = json.loads((out_dir / "frozen.json").read_text())
    else:
        workers = min(args.workers, max(1, (os.cpu_count() or 2) // 2))
        print(f"{args.tour}: {len(frame):,} matches; workers={workers}", flush=True)
        suffix, previous = "", None
        elo_grid, sr_grid = EloGrid(), SrGrid()
        if args.refine:
            previous = json.loads((out_dir / "frozen.json").read_text())
            (out_dir / "frozen_r1.json").write_text(json.dumps(previous, indent=1))
            elo_grid, sr_grid, suffix = RefineEloGrid(previous["elo"]), RefineSrGrid(previous["sr"]), "_r2"
        with Pool(workers, initializer=_Init, initargs=(args.tour,)) as pool:
            elo_results = pool.map(_EvalElo, elo_grid, chunksize=1)
            print("elo grid done", flush=True)
            sr_results = pool.map(_EvalSr, sr_grid, chunksize=1)
            print("sr grid done", flush=True)
        elo_table = SearchTable(elo_results)
        sr_table = SearchTable(sr_results)
        elo_table.to_csv(out_dir / f"search_elo{suffix}.csv", index=False)
        sr_table.to_csv(out_dir / f"search_sr{suffix}.csv", index=False)
        best_elo = PickFromTable(elo_table)
        best_sr = PickFromTable(sr_table)
        elo_params = {k: best_elo[k] for k in asdict(EloParams())}
        sr_params = {k: best_sr[k] for k in asdict(ServeReturnParams())}
        # blend weight on validation only
        both = RunModels(frame, EloParams(**elo_params), ServeReturnParams(**sr_params), args.tour)
        blend_rows = []
        for weight in BLEND_GRID:
            p = np.array([Blend(a, b, weight) for a, b in zip(both["p_elo"], both["p_sr"])])
            tune_scores = SeasonScores(frame, p, TUNE)
            val_scores = SeasonScores(frame, p, VALIDATION)
            blend_rows.append({"blend_weight_sr": weight,
                               "tune_worst_ll": max(s["log_loss"] for s in tune_scores.values()),
                               "val_worst_ll": max(s["log_loss"] for s in val_scores.values()),
                               "val_mean_ll": float(np.mean([s["log_loss"] for s in val_scores.values()]))})
        blend_table = pd.DataFrame(blend_rows)
        blend_table.to_csv(out_dir / f"search_blend{suffix}.csv", index=False)
        weight = float(blend_table.sort_values("val_worst_ll").iloc[0]["blend_weight_sr"])
        frozen = {"tour": args.tour, "elo": elo_params, "sr": sr_params, "blend_weight_sr": weight,
                  "configs_tried": {"elo": len(elo_table), "sr": len(sr_table), "blend": len(blend_table)},
                  "round": 2 if args.refine else 1,
                  "configs_tried_all_rounds": (previous.get("configs_tried_all_rounds", previous["configs_tried"]) if previous else None),
                  "blend_val_worst_ll": float(blend_table["val_worst_ll"].min()),
                  "selection": f"rank by worst tune-season log loss ({TUNE[0]}-{TUNE[-1]}, level={RANK_LEVEL}), "
                               f"top {TOP_N} -> best worst validation-season log loss ({VALIDATION[0]}-{VALIDATION[-1]}); "
                               "blend weight by worst validation-season log loss",
                  "splits": {"tune": TUNE, "validation": VALIDATION, "holdout": HOLDOUT, "forward": FORWARD},
                  "frozen_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                  "data_last_date": str(frame["date"].max().date())}
        (out_dir / "frozen.json").write_text(json.dumps(frozen, indent=1))
        print("frozen:", json.dumps(frozen), flush=True)
    preds, _ = FrozenRun(args.tour, frame, frozen)
    preds.to_parquet(out_dir / "predictions.parquet", index=False)
    metrics = MetricsTable(frame, {"p_elo": preds["p_elo"].to_numpy(), "p_sr": preds["p_sr"].to_numpy(),
                                   "p_blend": preds["p_blend"].to_numpy()})
    metrics.to_csv(out_dir / "metrics.csv", index=False)
    view = metrics[(metrics.level == "tour") & metrics.split.isin(["holdout", "forward", "validation"])]
    print(view.to_string(index=False))


if __name__ == "__main__":
    Main()