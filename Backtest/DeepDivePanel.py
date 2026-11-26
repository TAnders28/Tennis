"""Tennis deep-dive price panel (Sportsbot, 2026-11-26). Read-only on data; writes one CSV.

One row per settled Kalshi tennis match event (same universe and in-play exclusion as Backtest.XsportPanel), with:
  player A (YES side of the candle file) quotes at scheduled start minus 240/180/120/60/30/15/5 minutes
  (q{m}_bid / q{m}_ask; last two-sided candle at or before that time, <= 30 min old; no look-ahead),
  tournament and round parsed from the market rules text, surface (Sackmann-matched match first, else the
  tournament city looked up in the Sackmann 2023-2026 tournament list at the nearest time of year, else blank),
  scheduled start hour (ET), start_source, event volume (sum of both markets' lifetime volume_fp, INCLUDES in-play
  trading, so it is outcome-dependent and must not be used as a pre-match filter), vol_pre60 (A-market candle volume
  traded up to T-60, a legitimate pre-match feature), y (1 = A won).
Used by basketball `Analysis.CrossSportDeepDive`. Everything here is tested:false.
"""
import argparse
import csv
import datetime as dt
import json
import re
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from Backtest.KalshiJoin import INPLAY_JUMP, KALSHI_DIR, OUT_ROOT, QuoteAt, SERIES_BY_TOUR, LoadJsonl

ET = ZoneInfo("America/New_York")
SACK = Path(__file__).resolve().parents[1] / "data" / "raw" / "sackmann"
OFFSETS = (240, 180, 120, 60, 30, 15, 5)
SPLIT = dt.date(2026, 1, 1)
RULES_RE = re.compile(r"match in the (.*?) after a ball")
ROUND_PATTERNS = (
    (re.compile(r"qualif", re.I), "Q"),
    (re.compile(r"round of 128", re.I), "R128"),
    (re.compile(r"round of 64", re.I), "R64"),
    (re.compile(r"round of 32", re.I), "R32"),
    (re.compile(r"round of 16", re.I), "R16"),
    (re.compile(r"quarter", re.I), "QF"),
    (re.compile(r"semi", re.I), "SF"),
    (re.compile(r"group", re.I), "RR"),
    (re.compile(r"\bfinals?\b", re.I), "F"),
)
MAJORS = {"us open": "Hard", "australian open": "Hard", "wimbledon": "Grass", "french open": "Clay", "roland garros": "Clay"}


def ParseRound(text):
    for rx, tag in ROUND_PATTERNS:
        if rx.search(text or ""):
            if tag == "F" and re.search(r"(next gen atp finals|atp finals|wta finals)", text, re.I) and not re.search(r"\bfinal\b\s*$", text, re.I):
                continue
            return tag
    return ""


def ParseTournament(rules):
    """'... match in the 2026 ATP Challenger Jingshan Round Of 32 after a ball ...' -> ('Jingshan', 'R32')."""
    m = RULES_RE.search(rules or "")
    if not m:
        return "", ""
    full = m.group(1)
    rnd = ParseRound(full)
    name = re.split(r"\s(?:Round [Oo]f|Qualification|Quarterfinal|quarterfinal|Semifinal|semifinal|Final|final|Men|Women|Group)\b", full)[0]
    name = re.sub(r"\b(19|20)\d{2}\b", " ", name)
    name = re.sub(r"\b(ATP|WTA)\b|\bChallenger\b|\b125K?\b", " ", name)
    return " ".join(name.split()), rnd


def NormKey(name):
    """Tournament city key shared by Kalshi and Sackmann names ('Buenos Aires 1 CH' -> 'buenos aires')."""
    s = (name or "").lower()
    s = re.sub(r"^[mw]\d{2,3}\s+", " ", s)
    s = re.sub(r"\b(ch|125|wta|atp|challenger|open|ii|iii|iv)\b", " ", s)
    s = re.sub(r"[^a-z ]", " ", s)
    return " ".join(s.split())


def SurfaceTable():
    rows = []
    for p in sorted(SACK.glob("*/*_matches*_202[3456].csv")):
        try:
            d = pd.read_csv(p, usecols=["tourney_name", "surface", "tourney_date"], low_memory=False)
        except ValueError:
            continue
        d = d.dropna().drop_duplicates(["tourney_name", "tourney_date"])
        for r in d.itertuples(index=False):
            k = NormKey(r.tourney_name)
            if k:
                doy = dt.datetime.strptime(str(int(r.tourney_date)), "%Y%m%d").timetuple().tm_yday
                rows.append((k, doy, r.surface))
    table = {}
    for k, doy, s in rows:
        table.setdefault(k, []).append((doy, s))
    return table


def SurfaceFor(tournament, day, table):
    low = (tournament or "").lower()
    for name, surf in MAJORS.items():
        if name in low:
            return surf, "major"
    k = NormKey(tournament)
    cands = table.get(k)
    if not cands:
        return "", ""
    doy = day.timetuple().tm_yday
    best = min(cands, key=lambda c: min(abs(c[0] - doy), 366 - abs(c[0] - doy)))
    gap = min(abs(best[0] - doy), 366 - abs(best[0] - doy))
    return (best[1], "city_map") if gap <= 21 else ("", "")


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--xsport-panel", default="", help="optional XsportPanel CSV to take Sackmann-matched surface/round from")
    a = ap.parse_args()
    matched = {}
    if a.xsport_panel:
        x = pd.read_csv(a.xsport_panel, usecols=["game_id", "ctx_surface", "ctx_round"])
        matched = {r.game_id: (r.ctx_surface, r.ctx_round) for r in x.itertuples() if isinstance(r.ctx_surface, str)}
    table = SurfaceTable()
    rows, stats = [], {}
    for tour, series_list in SERIES_BY_TOUR.items():
        for series, level in series_list:
            mpath = KALSHI_DIR / "markets" / f"{series}.jsonl"
            if not mpath.exists() or mpath.stat().st_size == 0:
                continue
            events = {}
            for m in LoadJsonl(mpath):
                events.setdefault(m["event_ticker"], []).append(m)
            cdir = KALSHI_DIR / "candles" / series
            n_in = n_ok = 0
            for ev, ms in events.items():
                path = cdir / f"{ev}.json"
                if not path.exists():
                    continue
                payload = json.loads(path.read_text())
                ma = next((m for m in ms if m["ticker"] == payload["ticker"]), None)
                if ma is None or ma.get("result") not in ("yes", "no"):
                    continue
                start = dt.datetime.fromisoformat(payload["scheduled_start"])
                ts = start.timestamp()
                cand = payload.get("minute", [])
                q = {mm: QuoteAt(cand, ts - 60 * mm) for mm in OFFSETS}
                if not q[60]:
                    continue
                if q[5] and abs((q[5]["yes_bid"] + q[5]["yes_ask"]) / 2 - (q[60]["yes_bid"] + q[60]["yes_ask"]) / 2) > INPLAY_JUMP:
                    n_in += 1
                    continue
                d = start.astimezone(ET).date()
                tourn, rnd = ParseTournament(ma.get("rules_primary", ""))
                surf, src = "", ""
                if ev in matched:
                    surf, src = matched[ev][0], "sackmann_match"
                    rnd = rnd or str(matched[ev][1])
                if not surf:
                    surf, src = SurfaceFor(tourn, d, table)
                vpre = sum(float(c.get("volume") or 0) for c in cand if c.get("end_ts") is not None and c["end_ts"] <= ts - 3600)
                vol = 0.0
                for m in ms:
                    try:
                        vol += float(m.get("volume_fp") or 0)
                    except ValueError:
                        pass
                row = {"date": d.isoformat(), "game_id": ev, "split": "holdout" if d >= SPLIT else "tune", "tour": tour,
                       "level": level, "tournament": tourn, "round": rnd, "surface": surf, "surface_source": src,
                       "start_hour_et": start.astimezone(ET).hour, "start_source": payload.get("start_source", ""),
                       "volume": round(vol, 2), "vol_pre60": round(vpre, 2), "y": int(ma["result"] == "yes")}
                for mm in OFFSETS:
                    row[f"q{mm}_bid"] = q[mm]["yes_bid"] if q[mm] else ""
                    row[f"q{mm}_ask"] = q[mm]["yes_ask"] if q[mm] else ""
                rows.append(row)
                n_ok += 1
            stats[series] = {"events": n_ok, "possible_inplay_excluded": n_in}
    rows.sort(key=lambda r: (r["date"], r["game_id"]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    surf_cov = sum(1 for r in rows if r["surface"]) / max(len(rows), 1)
    round_cov = sum(1 for r in rows if r["round"]) / max(len(rows), 1)
    print(json.dumps({"rows": len(rows), "surface_coverage": round(surf_cov, 3), "round_coverage": round(round_cov, 3), **stats}))


if __name__ == "__main__":
    Main()