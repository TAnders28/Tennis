"""Parse Sackmann-style tennis scores into set lines, totals, and handicaps."""
from __future__ import annotations

import re

SET_RE = re.compile(
    r"^\s*(\d+)\s*-\s*(\d+)(?:\s*\(\s*\d+\s*\))?\s*$",
    re.IGNORECASE,
)
RET_RE = re.compile(r"\b(RET|DEF|ABD|W/O|Walkover|unfinished)\b", re.IGNORECASE)


def IsIncomplete(score):
    text = "" if score is None else str(score)
    return bool(RET_RE.search(text))


def ParseScore(score):
    """Return a dict or None if the score cannot be parsed into completed sets.

    Keys: sets ((a,b), ...), sets_a, sets_b, games_a, games_b, total_games,
    game_diff (a minus b), n_tiebreaks, incomplete.
    Player a is the listed winner in a Sackmann score string.
    """
    if score is None:
        return None
    text = str(score).strip()
    if not text:
        return None
    incomplete = IsIncomplete(text)
    cleaned = RET_RE.sub(" ", text).replace(",", " ")
    parts = [p for p in cleaned.split() if p and p not in ("RET", "DEF", "Def")]
    sets = []
    for part in parts:
        m = SET_RE.match(part)
        if not m:
            # keep tokens like 7-6(5)
            m = re.match(r"^\s*(\d+)\s*-\s*(\d+)", part)
            if not m:
                continue
        ga, gb = int(m.group(1)), int(m.group(2))
        sets.append((ga, gb))
    if not sets:
        return None
    sets_a = sum(1 for ga, gb in sets if ga > gb)
    sets_b = sum(1 for ga, gb in sets if gb > ga)
    games_a = sum(ga for ga, gb in sets)
    games_b = sum(gb for ga, gb in sets)
    n_tb = sum(1 for ga, gb in sets if (ga >= 6 and gb >= 6 and abs(ga - gb) == 1) or (ga + gb >= 13 and abs(ga - gb) == 1))
    return {
        "sets": tuple(sets),
        "sets_a": sets_a,
        "sets_b": sets_b,
        "games_a": games_a,
        "games_b": games_b,
        "total_games": games_a + games_b,
        "game_diff": games_a - games_b,
        "set_diff": sets_a - sets_b,
        "n_tiebreaks": n_tb,
        "sweep": sets_b == 0 and sets_a > 0,
        "incomplete": incomplete,
    }


def OrientScore(parsed, winner_is_a=True):
    """Re-orient a winner-based Sackmann parse so 'a' is a nominated side."""
    if parsed is None:
        return None
    if winner_is_a:
        return dict(parsed)
    out = dict(parsed)
    out["sets"] = tuple((gb, ga) for ga, gb in parsed["sets"])
    out["sets_a"], out["sets_b"] = parsed["sets_b"], parsed["sets_a"]
    out["games_a"], out["games_b"] = parsed["games_b"], parsed["games_a"]
    out["game_diff"] = -parsed["game_diff"]
    out["set_diff"] = -parsed["set_diff"]
    out["sweep"] = out["sets_b"] == 0 and out["sets_a"] > 0
    return out