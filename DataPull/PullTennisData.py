r"""tennis-data.co.uk historical results + bookmaker odds: MANUAL download, local ingest.

Why manual: tennis-data.co.uk says its data is for private individuals and not for
automated bots/scrapers/AI, and on 2026-11-13 it answered HTTP 403 to scripted
requests from Tomputer. So this script never downloads from that site. Tim
downloads the yearly files in his own browser and drops them here:

  data/raw/tennis_data/atp/<year>.xlsx   (or .xls)  from http://www.tennis-data.co.uk/<year>/<year>.xlsx
  data/raw/tennis_data/wta/<year>.xlsx   (or .xls)  from http://www.tennis-data.co.uk/<year>w/<year>.xlsx

(ATP files exist from 2000/2001, WTA from 2007; years up to 2012 are .xls.)
This script lists what is present / missing and converts each file to
data/processed/tennis_data_<tour>.parquet for Backtest.JoinOdds.

Usage:
  .venv\Scripts\python.exe -m DataPull.PullTennisData --list
  .venv\Scripts\python.exe -m DataPull.PullTennisData --ingest
"""
from __future__ import annotations

import argparse
import datetime as dt

import pandas as pd

from DataPull.Http import DATA_DIR, RAW_DIR

FIRST_YEAR = {"atp": 2001, "wta": 2007}
ODDS_COLUMNS = ["PSW", "PSL", "B365W", "B365L", "MaxW", "MaxL", "AvgW", "AvgL"]
KEEP_COLUMNS = ["Date", "Tournament", "Location", "Series", "Tier", "Court", "Surface", "Round",
                "Best of", "Winner", "Loser", "WRank", "LRank", "Comment"] + ODDS_COLUMNS


def SourceUrl(tour: str, year: int) -> str:
    extension = "xls" if year <= 2012 else "xlsx"
    folder = f"{year}" if tour == "atp" else f"{year}w"
    return f"http://www.tennis-data.co.uk/{folder}/{year}.{extension}"


def FindLocalFile(tour: str, year: int):
    folder = RAW_DIR / "tennis_data" / tour
    for extension in ("xlsx", "xls", "csv"):
        path = folder / f"{year}.{extension}"
        if path.exists():
            return path
    return None


def ReadOne(path) -> pd.DataFrame:
    if path.suffix == ".csv":
        frame = pd.read_csv(path, encoding="latin-1")
    else:
        frame = pd.read_excel(path)
    for column in KEEP_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    frame = frame[KEEP_COLUMNS].copy()
    frame["Date"] = pd.to_datetime(frame["Date"], errors="coerce")
    for column in ODDS_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    for column in ["Winner", "Loser", "Tournament", "Round", "Surface", "Comment", "Series", "Tier", "Location", "Court"]:
        frame[column] = frame[column].astype("string").str.strip()
    frame["Best of"] = pd.to_numeric(frame["Best of"], errors="coerce")
    return frame


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--ingest", action="store_true")
    parser.add_argument("--last-year", type=int, default=dt.date.today().year)
    args = parser.parse_args(argv)
    for tour in ("atp", "wta"):
        present, missing, frames = [], [], []
        for year in range(FIRST_YEAR[tour], args.last_year + 1):
            path = FindLocalFile(tour, year)
            if path is None:
                missing.append(year)
                continue
            present.append(year)
            if args.ingest:
                frame = ReadOne(path)
                frame["td_year"] = year
                frame["tour"] = tour
                frames.append(frame)
        print(f"{tour}: present {present or 'none'}")
        if missing:
            print(f"{tour}: missing {len(missing)} years; download in a browser, e.g. {SourceUrl(tour, missing[-1])}")
        if args.ingest and frames:
            out_path = DATA_DIR / "processed" / f"tennis_data_{tour}.parquet"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            combined = pd.concat(frames, ignore_index=True)
            combined.to_parquet(out_path, index=False)
            print(f"{tour}: wrote {out_path} rows={len(combined):,}")


if __name__ == "__main__":
    Main()