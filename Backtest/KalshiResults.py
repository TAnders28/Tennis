r"""Recent results from Kalshi settled tennis match markets (fills the gap after the Sackmann snapshot).

Source: Kalshi public trade API (read-only, no account), series KXATPMATCH / KXWTAMATCH /
KXATPCHALLENGERMATCH / KXWTACHALLENGERMATCH. Each settled event has two markets, one per player;
the market that settled "yes" is the winner. Kalshi settles on the official result once a ball
is played (retirements count for the player who advances); events that never started resolve at a
"fair price" and are dropped here (no yes/no pair).
Dates: Kalshi milestone start_date (scheduled start) converted to an ET calendar day.
Round and tournament: milestone details. Surface: Backtest.Surfaces lookup from the tournament name.
No serve/return point stats, so these rows update Elo only; the serve/return model stays frozen at the
last Sackmann result (early June 2026).

Players: matched to Sackmann ids by full name. Players unknown to Sackmann get a stable synthetic
negative id and a level-appropriate starting rating (the 25th percentile of the overall Elo of players
active at that level in 2026 H1). They are flagged as new.
Events already present in Sackmann (same player pair, start within [tourney_date - 3, + 16] days) are skipped.

Usage: .venv\Scripts\python.exe -m Backtest.KalshiResults --tour atp   (prints coverage)
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import zlib
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from Backtest.LoadMatches import STAGE, UNKNOWN_STAGE, LoadMatches
from Backtest.NameMatch import NameIndex, Normalise
from Backtest.Surfaces import SurfaceLookup

REPO_ROOT = Path(__file__).resolve().parents[1]
KALSHI_DIR = REPO_ROOT / "data" / "raw" / "kalshi"
ET = ZoneInfo("America/New_York")
SERIES = {"atp": [("KXATPMATCH", "tour"), ("KXATPCHALLENGERMATCH", "challenger")],
          "wta": [("KXWTAMATCH", "tour"), ("KXWTACHALLENGERMATCH", "challenger")]}
SLAMS = ("australian open", "french open", "roland garros", "wimbledon", "us open")
NEW_PLAYER_PERCENTILE = 25


def LoadJsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def Milestones():
    starts = {}
    for path in sorted(KALSHI_DIR.glob("milestones_tennis*.jsonl")):
        for row in LoadJsonl(path):
            for ticker in row["event_tickers"]:
                starts[ticker] = row
    return starts


def RoundToStage(text):
    t = str(text or "").lower()
    for key, code in (("1/32", "R64"), ("1/16", "R32"), ("1/8", "R16")):
        if key in t:
            return code
    if "qualif" in t and "final" in t:
        return "Q3"
    if "qualif" in t:
        digits = [c for c in t if c.isdigit()]
        return f"Q{digits[-1] if digits else 1}"
    for key, code in (("round of 128", "R128"), ("round of 64", "R64"), ("round of 32", "R32"), ("round of 16", "R16"),
                      ("quarter", "QF"), ("semi", "SF"), ("robin", "RR"), ("group", "RR"), ("final", "F")):
        if key in t:
            return code
    if "1st round" in t or "first round" in t or "round 1" in t:
        return "R32"
    if "2nd round" in t or "second round" in t or "round 2" in t:
        return "R16"
    return ""


def SyntheticId(tour, name):
    return -int(zlib.crc32(f"{tour}:{Normalise(name)}".encode()) % 2_000_000_000) - 1


def BuildKalshiRows(tour, sackmann, since="2026-05-20", until=None):
    """Return (rows DataFrame in LoadMatches schema, coverage dict)."""
    starts = Milestones()
    names = pd.concat([sackmann[["winner_id", "winner_name", "date"]].set_axis(["i", "n", "d"], axis=1),
                       sackmann[["loser_id", "loser_name", "date"]].set_axis(["i", "n", "d"], axis=1)])
    last_seen = names.groupby("i")["d"].max()
    index = NameIndex(names.drop_duplicates("i")[["i", "n"]].itertuples(index=False, name=None))
    surfaces = SurfaceLookup(sackmann)
    by_pair = {}
    dates = sackmann["date"].to_numpy().astype("datetime64[D]")
    for i, (w, l) in enumerate(zip(sackmann["winner_id"].to_numpy(), sackmann["loser_id"].to_numpy())):
        by_pair.setdefault(frozenset((int(w), int(l))), []).append(i)
    since_d = dt.date.fromisoformat(since)
    until_d = dt.date.fromisoformat(until) if until else dt.date(2100, 1, 1)
    rows, cover = [], {"events": 0, "settled_pairs": 0, "in_sackmann": 0, "used": 0, "new_players": set(), "no_start": 0, "not_settled": 0}

    def Resolve(name):
        ids = index.LookupFull(name)
        if len(ids) == 1:
            return next(iter(ids)), False
        if len(ids) > 1:
            return max(ids, key=lambda i: last_seen.get(i, pd.Timestamp(0))), False
        return SyntheticId(tour, name), True

    for series, level in SERIES[tour]:
        events = {}
        for m in LoadJsonl(KALSHI_DIR / "markets" / f"{series}.jsonl"):
            events.setdefault(m["event_ticker"], []).append(m)
        for ticker, ms in events.items():
            meta = starts.get(ticker)
            start_text = (meta or {}).get("start_date") or ms[0].get("occurrence_datetime")
            if not start_text:
                cover["no_start"] += 1
                continue
            start = dt.datetime.fromisoformat(start_text.replace("Z", "+00:00"))
            day = start.astimezone(ET).date()
            if not (since_d <= day <= until_d):
                continue
            cover["events"] += 1
            results = sorted((m.get("result") or "") for m in ms)
            if len(ms) != 2 or results != ["no", "yes"]:
                cover["not_settled"] += 1
                continue
            cover["settled_pairs"] += 1
            win_m = next(m for m in ms if m["result"] == "yes")
            lose_m = next(m for m in ms if m["result"] == "no")
            (w_id, w_new), (l_id, l_new) = Resolve(win_m["yes_sub_title"]), Resolve(lose_m["yes_sub_title"])
            hit = False
            for i in by_pair.get(frozenset((int(w_id), int(l_id))), []):
                lag = int((np.datetime64(day) - dates[i]).astype(int))
                if -3 <= lag <= 16:
                    hit = True
                    break
            if hit:
                cover["in_sackmann"] += 1
                continue
            details = (meta or {}).get("details") or {}
            competition = details.get("tournament_name") or ""
            round_code = RoundToStage(details.get("round"))
            surface, surface_source = surfaces.Get(competition)
            lvl = "tour_qual" if (level == "tour" and round_code.startswith("Q")) else level
            best_of = 5 if (tour == "atp" and level == "tour" and not round_code.startswith("Q")
                            and any(s in competition.lower() for s in SLAMS)) else 3
            for flag, name in ((w_new, win_m["yes_sub_title"]), (l_new, lose_m["yes_sub_title"])):
                if flag:
                    cover["new_players"].add(name)
            rows.append({"match_key": f"{tour}:kalshi:{ticker}", "tour": tour, "season": day.year, "tourney_id": f"kalshi:{competition}",
                         "tourney_name": competition, "tourney_level": "K", "level_tag": lvl,
                         "tourney_date": int(day.strftime("%Y%m%d")), "date": pd.Timestamp(day), "stage": STAGE.get(round_code, UNKNOWN_STAGE),
                         "round": round_code, "surface": surface, "surface_source": surface_source,
                         "surface_unknown": surface_source == "guess_default_hard", "surface_norm": surface, "best_of": best_of, "draw_size": np.nan,
                         "winner_id": int(w_id), "winner_name": win_m["yes_sub_title"], "loser_id": int(l_id),
                         "loser_name": lose_m["yes_sub_title"], "winner_rank": np.nan, "loser_rank": np.nan, "score": "",
                         "is_ret": False, "file_kind": "kalshi", "match_num": 0, "event_ticker": ticker,
                         "scheduled_start": start.isoformat(), "winner_new": w_new, "loser_new": l_new})
            cover["used"] += 1
    frame = pd.DataFrame(rows)
    cover["new_players"] = len(cover["new_players"])
    return frame, cover


def NewPlayerSeeds(sackmann_state_elo, sackmann, kalshi_rows, percentile=NEW_PLAYER_PERCENTILE):
    """Level-appropriate starting ratings for synthetic ids, computed only from pre-June (Sackmann) ratings."""
    recent = sackmann[sackmann["season"] == sackmann["season"].max()]
    defaults = {}
    for level in ("tour", "tour_qual", "challenger", "itf"):
        ids = pd.unique(recent.loc[recent["level_tag"] == level, ["winner_id", "loser_id"]].to_numpy().ravel())
        ratings = [sackmann_state_elo.rating[i] for i in ids if i in sackmann_state_elo.rating]
        defaults[level] = float(np.percentile(ratings, percentile)) if ratings else sackmann_state_elo.params.init
    seeds = {}
    if len(kalshi_rows):
        for row in kalshi_rows.sort_values("tourney_date").itertuples():
            for pid, new in ((row.winner_id, row.winner_new), (row.loser_id, row.loser_new)):
                if new and pid not in seeds:
                    seeds[pid] = defaults.get(row.level_tag, defaults["challenger"])
    return seeds, defaults


def Combine(sackmann, kalshi_rows):
    if not len(kalshi_rows):
        return sackmann.copy()
    keep = list(sackmann.columns)
    extra = kalshi_rows[[c for c in keep if c in kalshi_rows.columns]].copy()
    for c in keep:
        if c not in extra.columns:
            extra[c] = np.nan
    combined = pd.concat([sackmann, extra[keep]], ignore_index=True)
    combined = combined.sort_values(["tourney_date", "stage", "tourney_id", "match_num"], kind="mergesort").reset_index(drop=True)
    combined["group_id"] = combined.groupby(["tourney_date", "stage"], sort=False).ngroup()
    return combined


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tour", choices=["atp", "wta"], required=True)
    parser.add_argument("--since", default="2026-05-20")
    args = parser.parse_args(argv)
    sackmann = LoadMatches(args.tour)
    rows, cover = BuildKalshiRows(args.tour, sackmann, args.since)
    print(args.tour, cover)
    if len(rows):
        rows["month"] = rows["date"].dt.strftime("%Y-%m")
        print(rows.groupby(["month", "level_tag"]).size().unstack(fill_value=0).to_string())
        print(rows.groupby(["level_tag", "surface_source"]).size().unstack(fill_value=0).to_string())
        guessed = rows.loc[rows.surface_unknown, "tourney_name"].value_counts()
        print("tournaments with surface guessed (flagged surface_unknown):", guessed.to_dict())
        print("rows with a new player:", int((rows.winner_new | rows.loser_new).sum()), "last date:", rows["date"].max().date())


if __name__ == "__main__":
    Main()