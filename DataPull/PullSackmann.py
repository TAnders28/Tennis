r"""Pull Jeff Sackmann's ATP/WTA match CSVs and Grand Slam point-by-point files.

The original JeffSackmann/tennis_atp and tennis_wta GitHub repos return 404 since
about 2026-10-14. The default source is the data-only archival mirror
Aneeshers/tennis-sackmann-archive (snapshot of upstream, June 2026), which keeps
Sackmann's CC BY-NC-SA 4.0 licence. Non-commercial use only; credit Jeff Sackmann.

Files land in data/raw/sackmann/<tour>/<original filename> (year-aware names).
Resumable: existing files are skipped unless --refresh (or the year is >= --refresh-from).

Usage:
  .venv\Scripts\python.exe -m DataPull.PullSackmann --tours atp wta --kinds main qual_chall qual_itf futures --years 2000 2026
  .venv\Scripts\python.exe -m DataPull.PullSackmann --pbp --pbp-years 2011 2024
"""
from __future__ import annotations

import argparse
import datetime as dt

from DataPull.Http import RAW_DIR, PoliteSession, WriteBytesAtomic, WriteJsonAtomic

DEFAULT_BASE = "https://raw.githubusercontent.com/Aneeshers/tennis-sackmann-archive/main"
SLAMS = ["ausopen", "frenchopen", "wimbledon", "usopen"]

# kind -> filename pattern per tour (None = not covered by that tour's repo)
KIND_PATTERNS = {
    "main": {"atp": "atp_matches_{year}.csv", "wta": "wta_matches_{year}.csv"},
    "qual_chall": {"atp": "atp_matches_qual_chall_{year}.csv", "wta": None},
    "futures": {"atp": "atp_matches_futures_{year}.csv", "wta": None},
    "qual_itf": {"atp": None, "wta": "wta_matches_qual_itf_{year}.csv"},
}


def PullMatches(session, base, tours, kinds, years, refresh_from):
    results = []
    for tour in tours:
        for kind in kinds:
            pattern = KIND_PATTERNS[kind][tour]
            if pattern is None:
                continue
            for year in years:
                file_name = pattern.format(year=year)
                out_path = RAW_DIR / "sackmann" / tour / file_name
                if out_path.exists() and year < refresh_from:
                    results.append({"file": file_name, "status": "exists", "bytes": out_path.stat().st_size})
                    continue
                response = session.Get(f"{base}/{tour}/{file_name}")
                if response.status_code == 404:
                    results.append({"file": file_name, "status": "missing_404"})
                    continue
                WriteBytesAtomic(out_path, response.content)
                results.append({"file": file_name, "status": "downloaded", "bytes": len(response.content)})
                print(f"  {tour} {file_name} {len(response.content):,} bytes", flush=True)
    for tour in tours:
        file_name = f"{tour}_players.csv"
        out_path = RAW_DIR / "sackmann" / tour / file_name
        if not out_path.exists() or refresh_from <= dt.date.today().year:
            response = session.Get(f"{base}/{tour}/{file_name}")
            if response.status_code == 200:
                WriteBytesAtomic(out_path, response.content)
                results.append({"file": file_name, "status": "downloaded", "bytes": len(response.content)})
    return results


def PullSlamPbp(session, base, years):
    results = []
    for year in years:
        for slam in SLAMS:
            for part in ("matches", "points"):
                file_name = f"{year}-{slam}-{part}.csv"
                out_path = RAW_DIR / "sackmann" / "slam_pbp" / file_name
                if out_path.exists():
                    results.append({"file": file_name, "status": "exists"})
                    continue
                response = session.Get(f"{base}/slam_pointbypoint/{file_name}")
                if response.status_code == 404:
                    results.append({"file": file_name, "status": "missing_404"})
                    continue
                WriteBytesAtomic(out_path, response.content)
                results.append({"file": file_name, "status": "downloaded", "bytes": len(response.content)})
                print(f"  pbp {file_name} {len(response.content):,} bytes", flush=True)
    return results


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--tours", nargs="+", default=["atp", "wta"], choices=["atp", "wta"])
    parser.add_argument("--kinds", nargs="+", default=["main", "qual_chall", "qual_itf"], choices=list(KIND_PATTERNS))
    parser.add_argument("--years", nargs=2, type=int, default=[2000, dt.date.today().year])
    parser.add_argument("--refresh-from", type=int, default=9999, help="re-download years >= this (e.g. the current year)")
    parser.add_argument("--pbp", action="store_true", help="also pull Grand Slam point-by-point")
    parser.add_argument("--pbp-years", nargs=2, type=int, default=[2011, 2024])
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between requests")
    args = parser.parse_args(argv)
    session = PoliteSession(min_interval_s=args.interval)
    years = list(range(args.years[0], args.years[1] + 1))
    results = PullMatches(session, args.base, args.tours, args.kinds, years, args.refresh_from)
    if args.pbp:
        results += PullSlamPbp(session, args.base, list(range(args.pbp_years[0], args.pbp_years[1] + 1)))
    manifest = {
        "source_base": args.base,
        "licence": "CC BY-NC-SA 4.0 (Jeff Sackmann); non-commercial, attribution, share-alike",
        "pulled_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "files": results,
    }
    WriteJsonAtomic(RAW_DIR / "sackmann" / "manifest.json", manifest)
    counts = {}
    for row in results:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    print("done", counts)


if __name__ == "__main__":
    Main()