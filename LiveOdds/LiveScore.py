r"""Live tennis score from Kalshi's public live_data endpoint (read-only, no auth).

GET /trade-api/v2/live_data/tennis_tournament_singles/milestone/<milestone_id> returns set scores
(round_scores with outcome winner/loser/ongoing), the current game's points (0/15/30/40 plus an
"advantage" competitor id, or raw points in a tiebreak), the server's competitor id, match_status
and live serve stats. It covers every Kalshi-listed match (tour, Challenger, WTA 125).
It carries NO source timestamp: state_time is null and fetched_at (our clock, UTC) is logged instead.
Competitor 1 is the milestone's first_competitor_id, i.e. the first name in the milestone title.
"""
from __future__ import annotations

import datetime as dt

API = "https://api.elections.kalshi.com/trade-api/v2"
POINTS = {0: 0, 15: 1, 30: 2, 40: 3}


def FetchLive(session, milestone_id):
    response = session.Get(f"{API}/live_data/tennis_tournament_singles/milestone/{milestone_id}")
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if response.status_code != 200:
        return None, fetched_at
    return (response.json().get("live_data") or {}).get("details"), fetched_at


def ParseState(d):
    """Return a state dict with competitor 1 as 'a', or None when the match is not in play."""
    if not d or d.get("status") != "live":
        return None
    c1, c2 = d.get("competitor1_id"), d.get("competitor2_id")
    r1, r2 = d.get("competitor1_round_scores") or [], d.get("competitor2_round_scores") or []
    sets_a = sum(1 for r in r1 if r.get("outcome") == "winner")
    sets_b = sum(1 for r in r2 if r.get("outcome") == "winner")
    done = [(a.get("score"), b.get("score")) for a, b in zip(r1, r2) if a.get("outcome") != "ongoing"]
    ongoing = [(a.get("score") or 0, b.get("score") or 0) for a, b in zip(r1, r2) if a.get("outcome") == "ongoing"]
    games_a, games_b = ongoing[-1] if ongoing else (0, 0)
    raw_a, raw_b = d.get("competitor1_current_round_score") or 0, d.get("competitor2_current_round_score") or 0
    tiebreak = games_a == 6 and games_b == 6
    if tiebreak:
        pts_a, pts_b = int(raw_a), int(raw_b)
    else:
        if raw_a not in POINTS or raw_b not in POINTS:
            return None
        pts_a, pts_b = POINTS[raw_a], POINTS[raw_b]
        adv = d.get("advantage") or ""
        if pts_a == 3 and pts_b == 3 and adv:
            pts_a, pts_b = (4, 3) if adv == c1 else (3, 4)
    server = d.get("server")
    if server not in (c1, c2):
        return None
    sets_text = " ".join(f"{a}-{b}" for a, b in done + [(games_a, games_b)])
    point_text = f"{pts_a}-{pts_b} (tb)" if tiebreak else f"{raw_a}-{raw_b}" + (" ad" if (d.get("advantage") or "") else "")
    return {"sets_a": sets_a, "sets_b": sets_b, "games_a": int(games_a), "games_b": int(games_b), "points_a": pts_a,
            "points_b": pts_b, "server": "a" if server == c1 else "b", "tiebreak": tiebreak,
            "period": d.get("match_status"), "score": f"{sets_text}, {point_text}"}