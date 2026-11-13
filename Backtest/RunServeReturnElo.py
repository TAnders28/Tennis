r"""Walk the Elo-style serve/return prototype and compare leak-free holdout log loss
to the current frozen model (p_elo, p_sr, p_blend in predictions.parquet).

Sackmann point stats run through 2026-06-01; later matches are dropped. Defaults are
not tuned on the holdout. Scores are reported in nats (repo convention) and bits.

Usage:
  .venv\Scripts\python.exe -m Backtest.RunServeReturnElo --out-dir E:\AgentDocs\Tennibot\2026-11-13_market-blend
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from Backtest.LoadMatches import LoadMatches
from Backtest.Metrics import LogLossBits, Score
from Backtest.RunBacktest import SPLIT_OF
from Backtest.ServeReturnElo import RunServeReturnElo, ServeReturnEloParams

REPO_ROOT = Path(__file__).resolve().parents[1]
BT_DIR = REPO_ROOT / "data" / "backtest"
DEFAULT_OUT = Path(r"E:\AgentDocs\Tennibot\2026-11-13_market-blend")
SACKMANN_END = dt.date(2026, 6, 1)
MODELS = ("srelo", "sr", "elo", "blend")


def WalkTour(tour, params, end=SACKMANN_END):
    frame = LoadMatches(tour)
    frame = frame[frame["date"] <= np.datetime64(end)].copy().reset_index(drop=True)
    out = RunServeReturnElo(frame, params, tour)
    frame["p_srelo"] = out["p_srelo"]
    preds = pd.read_parquet(BT_DIR / tour / "predictions.parquet")
    keep = ["match_key", "split", "p_elo", "p_sr", "p_blend"]
    merged = frame.merge(preds[keep], on="match_key", how="left")
    if "split" not in merged.columns or merged["split"].isna().any():
        merged["split"] = merged["season"].map(SPLIT_OF)
    merged["tour"] = tour
    return merged


def SliceScores(part, last_date):
    rows = []
    y = np.ones(len(part))  # p_* is the winner's pre-match probability
    for name in MODELS:
        col = f"p_{name}"
        if col not in part.columns:
            continue
        p = part[col].to_numpy(dtype=float)
        ok = np.isfinite(p)
        scored = Score(p[ok])
        bits = LogLossBits(p[ok], y[ok]) if ok.any() else np.array([])
        rows.append({
            "date": last_date,
            "tour": part["tour"].iloc[0] if len(part) else "",
            "split": part["split"].iloc[0] if len(part) else "",
            "level": part["level_tag"].iloc[0] if "level_tag" in part.columns and part["level_tag"].nunique() == 1 else "all",
            "model": name,
            "n": scored["n"],
            "accuracy": scored["accuracy"],
            "log_loss": scored["log_loss"],
            "log_loss_bits": float(bits.mean()) if len(bits) else np.nan,
            "brier": scored["brier"],
        })
    return rows


def Compare(frame, last_date):
    rows = []
    for (tour, split, level), part in frame.groupby(["tour", "split", "level_tag"], dropna=False):
        part = part.copy()
        part["tour"], part["split"], part["level_tag"] = tour, split, level
        rows.extend(SliceScores(part, last_date))
    for (tour, split), part in frame.groupby(["tour", "split"], dropna=False):
        part = part.copy()
        part["tour"], part["split"], part["level_tag"] = tour, split, "all"
        rows.extend(SliceScores(part, last_date))
    return pd.DataFrame(rows)


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base", default="2026-11-13_market-blend")
    parser.add_argument("--end", default="2026-06-01")
    args = parser.parse_args(argv)
    end = dt.date.fromisoformat(args.end)
    params = ServeReturnEloParams()
    frames = [WalkTour(tour, params, end) for tour in ("atp", "wta")]
    frame = pd.concat(frames, ignore_index=True)
    last_date = pd.to_datetime(frame["date"]).max().date().isoformat()
    scores = Compare(frame, last_date)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.base}_serve-return-elo.csv"
    scores.to_csv(out, index=False)
    view = scores[(scores.split == "holdout") & (scores.level == "all")]
    print(view.to_string(index=False))
    print(f"wrote {out}")
    return scores


if __name__ == "__main__":
    Main()