r"""Load Sackmann match CSVs into one chronologically ordered table per tour.

Ordering (the leak-free contract used by Backtest.Engine):
  1. tourney_date (Sackmann: the Monday/start of the event week)
  2. round stage, earliest first (Q1 .. Q4, R128 .. R16, QF, SF, F)
Matches with the same (tourney_date, stage) form one *group*. Every prediction in
a group is made from the rating snapshot taken before the group starts, and the
group's results are applied only after all of its predictions are made. So no
match is ever scored with its own result, a same-round result, or any later
round of the same tournament.

Level tags: tour (main draw of tour events incl. slams, Masters/1000, 500/250, Finals,
Davis/BJK Cup, Olympics), tour_qual (tour-event qualifying), challenger (ATP
Challenger, WTA 125, incl. their qualifying), itf (ITF / Futures).
Walkovers (W/O, no ball struck) are dropped entirely.

Usage: .venv\Scripts\python.exe -m Backtest.LoadMatches --tour atp   (builds data/processed/matches_atp.parquet)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw" / "sackmann"
PROCESSED_DIR = REPO_ROOT / "data" / "processed"

STAGE = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "R128": 10, "R64": 11, "ER": 11, "R32": 12, "R16": 13,
         "RR": 14, "QF": 15, "SF": 16, "BR": 17, "F": 17}
UNKNOWN_STAGE = 12
STAT_COLUMNS = ["w_svpt", "w_1stWon", "w_2ndWon", "l_svpt", "l_1stWon", "l_2ndWon", "w_ace", "l_ace"]
TOUR_LEVELS = {"G", "M", "A", "F", "D", "O", "P", "PM", "I", "W", "T1", "T2", "T3", "T4", "T5", "E", "J"}
FILE_KINDS = {"atp": ["atp_matches_{y}.csv", "atp_matches_qual_chall_{y}.csv", "atp_matches_futures_{y}.csv"],
              "wta": ["wta_matches_{y}.csv", "wta_matches_qual_itf_{y}.csv"]}


def NormaliseSurface(surface, carpet_as="Hard"):
    if not isinstance(surface, str) or not surface:
        return "Unknown"
    surface = surface.strip().title()
    if surface == "Carpet":
        return carpet_as
    return surface if surface in ("Hard", "Clay", "Grass") else "Unknown"


def LevelTag(tourney_level, round_name, file_kind):
    level = str(tourney_level)
    if file_kind == "futures" or level[:1].isdigit() or level in ("S", "15", "25"):
        return "itf"
    if level == "C":
        return "challenger"
    if round_name.startswith("Q"):
        return "tour_qual"
    return "tour"


def ReadTourFrame(tour, years, kinds=("main", "qual", "itf")):
    frames = []
    for year in years:
        for index, pattern in enumerate(FILE_KINDS[tour]):
            kind = ["main", "qual", "itf"][index] if tour == "atp" else ["main", "qual"][index]
            if kind not in kinds and not (tour == "wta" and kind == "qual" and "itf" in kinds):
                continue
            path = RAW_DIR / tour / pattern.format(y=year)
            if not path.exists():
                continue
            frame = pd.read_csv(path, low_memory=False, encoding="utf-8", encoding_errors="replace")
            frame["season"] = year
            frame["file_kind"] = "futures" if "futures" in pattern else kind
            frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"no Sackmann files for {tour} {years[0]}-{years[-1]} under {RAW_DIR / tour}")
    return pd.concat(frames, ignore_index=True)


def PrepareMatches(raw: pd.DataFrame, tour: str, carpet_as="Hard") -> pd.DataFrame:
    frame = raw.copy()
    frame["score"] = frame["score"].astype("string").fillna("")
    walkover = frame["score"].str.contains("W/O|Walkover|walkover", regex=True) | (frame["score"].str.strip() == "")
    frame = frame.loc[~walkover].copy()
    frame["round"] = frame["round"].astype("string").fillna("").str.strip()
    frame["tourney_level"] = frame["tourney_level"].astype("string").fillna("")
    frame["level_tag"] = [LevelTag(level, rnd, kind) for level, rnd, kind in
                          zip(frame["tourney_level"], frame["round"], frame["file_kind"])]
    frame["stage"] = frame["round"].map(STAGE).fillna(UNKNOWN_STAGE).astype(int)
    frame["surface_norm"] = [NormaliseSurface(s, carpet_as) for s in frame["surface"]]
    frame["tourney_date"] = pd.to_numeric(frame["tourney_date"], errors="coerce").astype("Int64")
    frame = frame.loc[frame["tourney_date"].notna()].copy()
    frame["date"] = pd.to_datetime(frame["tourney_date"].astype(int).astype(str), format="%Y%m%d")
    frame["best_of"] = pd.to_numeric(frame["best_of"], errors="coerce").fillna(3).astype(int)
    frame["is_ret"] = frame["score"].str.contains("RET|DEF|Def|ABD|unfinished", regex=True)
    frame["winner_id"] = pd.to_numeric(frame["winner_id"], errors="coerce").astype("Int64")
    frame["loser_id"] = pd.to_numeric(frame["loser_id"], errors="coerce").astype("Int64")
    frame = frame.loc[frame["winner_id"].notna() & frame["loser_id"].notna()].copy()
    for column in STAT_COLUMNS + ["match_num", "winner_rank", "loser_rank"]:
        frame[column] = pd.to_numeric(frame.get(column), errors="coerce")
    frame["tour"] = tour
    frame["match_num"] = frame["match_num"].fillna(0)
    frame = frame.sort_values(["tourney_date", "stage", "tourney_id", "match_num"], kind="mergesort").reset_index(drop=True)
    frame["group_id"] = frame.groupby(["tourney_date", "stage"], sort=False).ngroup()
    frame["match_key"] = (frame["tour"] + ":" + frame["tourney_id"].astype(str) + ":" + frame["round"].astype(str) + ":"
                          + frame["winner_id"].astype(str) + "-" + frame["loser_id"].astype(str))
    keep = ["match_key", "tour", "season", "group_id", "tourney_id", "tourney_name", "tourney_level", "level_tag",
            "tourney_date", "date", "stage", "round", "surface", "surface_norm", "best_of", "draw_size",
            "winner_id", "winner_name", "loser_id", "loser_name", "winner_rank", "loser_rank", "score", "is_ret",
            "file_kind", "match_num"] + STAT_COLUMNS
    frame["draw_size"] = pd.to_numeric(frame["draw_size"], errors="coerce")
    for column in ("winner_name", "loser_name", "tourney_name", "tourney_id", "surface"):
        frame[column] = frame[column].astype("string")
    return frame[keep]


def LoadMatches(tour, first_year=2000, last_year=2026, kinds=("main", "qual", "itf"), use_cache=True, carpet_as="Hard"):
    tag = "-".join(kinds)
    cache_path = PROCESSED_DIR / f"matches_{tour}_{first_year}_{last_year}_{tag}_{carpet_as}.parquet"
    if use_cache and cache_path.exists():
        return pd.read_parquet(cache_path)
    raw = ReadTourFrame(tour, list(range(first_year, last_year + 1)), kinds)
    frame = PrepareMatches(raw, tour, carpet_as)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return frame


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tour", choices=["atp", "wta"], required=True)
    parser.add_argument("--years", nargs=2, type=int, default=[2000, 2026])
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args(argv)
    frame = LoadMatches(args.tour, args.years[0], args.years[1], use_cache=not args.no_cache)
    print(frame.groupby(["season", "level_tag"]).size().unstack(fill_value=0).to_string())
    print("last date:", frame["date"].max(), "rows:", len(frame))


if __name__ == "__main__":
    Main()