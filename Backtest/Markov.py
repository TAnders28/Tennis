r"""Exact Markov chain for a tennis match: point -> game -> tiebreak -> set -> match.

Inputs are the two serve-point win probabilities p_a (A serving) and p_b (B serving),
assumed i.i.d. within the match. All functions return P(A wins).

Set rules (string codes):
  "tb7"     tiebreak to 7 at 6-6 (standard)
  "adv"     no tiebreak, win by two games (old AO/RG/Wimbledon final sets)
  "tb7_12"  tiebreak to 7 at 12-12 (Wimbledon final set 2019-2021)
  "tb10"    tiebreak to 10 at 6-6 (all Grand Slam final sets from 2022)

State convention for live use: sets_a, sets_b, games_a, games_b (current set),
points_a, points_b (current game or tiebreak, as point counts), server ("a"/"b").
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np


def _Q(p_a, p_b, server):
    """P(A wins the next point) given who serves."""
    return p_a if server == "a" else 1.0 - p_b


def _Other(server):
    return "b" if server == "a" else "a"


@lru_cache(maxsize=None)
def GameFrom(p, x, y):
    """P(server wins game) from server points x, receiver points y."""
    if x >= 4 and x - y >= 2:
        return 1.0
    if y >= 4 and y - x >= 2:
        return 0.0
    if x >= 3 and y >= 3:
        deuce = p * p / (p * p + (1 - p) * (1 - p))
        if x == y:
            return deuce
        if x > y:
            return p + (1 - p) * deuce
        return p * deuce
    return p * GameFrom(p, x + 1, y) + (1 - p) * GameFrom(p, x, y + 1)


def TiebreakServer(first_server, points_played):
    return first_server if ((points_played + 1) // 2) % 2 == 0 else _Other(first_server)


@lru_cache(maxsize=None)
def TiebreakFrom(p_a, p_b, a, b, first_server, target):
    """P(A wins tiebreak to `target` (win by 2) from a-b, first_server served point 1."""
    if a >= target and a - b >= 2:
        return 1.0
    if b >= target and b - a >= 2:
        return 0.0
    if a >= target - 1 and b >= target - 1 and a == b:
        q = p_a * (1 - p_b)
        r = (1 - p_a) * p_b
        return q / (q + r)
    server = TiebreakServer(first_server, a + b)
    q = _Q(p_a, p_b, server)
    return q * TiebreakFrom(p_a, p_b, a + 1, b, first_server, target) + (1 - q) * TiebreakFrom(p_a, p_b, a, b + 1, first_server, target)


def _TbAt(rule):
    return {"tb7": (6, 7), "tb10": (6, 10), "tb7_12": (12, 7), "adv": (None, None)}[rule]


@lru_cache(maxsize=None)
def SetFrom(p_a, p_b, ga, gb, server, rule):
    """Return (A wins & next set first server a, A wins & next b, B wins & next a, B wins & next b)
    from games ga-gb at the start of a game served by `server`."""
    tb_at, tb_target = _TbAt(rule)
    if tb_at is not None and (ga > tb_at + 1 or gb > tb_at + 1 or (ga == tb_at + 1 and gb == tb_at + 1)):
        raise ValueError(f"invalid game score {ga}-{gb} for rule {rule}")
    if ga >= 6 and ga - gb >= 2:
        nxt = server  # the game just finished was served by Other(server); next set starts with `server`
        return (1.0, 0.0, 0.0, 0.0) if nxt == "a" else (0.0, 1.0, 0.0, 0.0)
    if gb >= 6 and gb - ga >= 2:
        return (0.0, 0.0, 1.0, 0.0) if server == "a" else (0.0, 0.0, 0.0, 1.0)
    if tb_at is not None and ga == tb_at and gb == tb_at:
        t = TiebreakFrom(p_a, p_b, 0, 0, server, tb_target)
        nxt = _Other(server)
        return (t, 0.0, 1 - t, 0.0) if nxt == "a" else (0.0, t, 0.0, 1 - t)
    if rule == "adv" and ga == gb and ga >= 5:
        hold_a = GameFrom(p_a, 0, 0)
        break_a = 1.0 - GameFrom(p_b, 0, 0)
        q = hold_a * break_a
        r = (1 - hold_a) * (1 - break_a)
        win = q / (q + r)
        # an even number of further games is played, so the next set starts with the current server
        return (win, 0.0, 1 - win, 0.0) if server == "a" else (0.0, win, 0.0, 1 - win)
    if server == "a":
        g = GameFrom(p_a, 0, 0)
    else:
        g = 1.0 - GameFrom(p_b, 0, 0)
    after_win = SetFrom(p_a, p_b, ga + 1, gb, _Other(server), rule)
    after_loss = SetFrom(p_a, p_b, ga, gb + 1, _Other(server), rule)
    return tuple(g * w + (1 - g) * l for w, l in zip(after_win, after_loss))


@lru_cache(maxsize=None)
def MatchFrom(p_a, p_b, sa, sb, server, best_of, final_rule, normal_rule="tb7"):
    """P(A wins match) at the start of a set with sets sa-sb and `server` to serve first."""
    need = best_of // 2 + 1
    if sa >= need:
        return 1.0
    if sb >= need:
        return 0.0
    rule = final_rule if (sa == need - 1 and sb == need - 1) else normal_rule
    aa, ab, ba, bb = SetFrom(p_a, p_b, 0, 0, server, rule)
    return (aa * MatchFrom(p_a, p_b, sa + 1, sb, "a", best_of, final_rule, normal_rule)
            + ab * MatchFrom(p_a, p_b, sa + 1, sb, "b", best_of, final_rule, normal_rule)
            + ba * MatchFrom(p_a, p_b, sa, sb + 1, "a", best_of, final_rule, normal_rule)
            + bb * MatchFrom(p_a, p_b, sa, sb + 1, "b", best_of, final_rule, normal_rule))


def MatchProb(p_a, p_b, best_of=3, final_rule="tb7"):
    """Pre-match P(A wins), averaging over who serves first (coin toss)."""
    p_a = round(float(p_a), 6)
    p_b = round(float(p_b), 6)
    return 0.5 * (MatchFrom(p_a, p_b, 0, 0, "a", best_of, final_rule) + MatchFrom(p_a, p_b, 0, 0, "b", best_of, final_rule))


def LiveWinProb(p_a, p_b, sets_a, sets_b, games_a, games_b, points_a, points_b, server, best_of=3,
                final_rule="tb7", normal_rule="tb7"):
    """P(A wins) from an in-play state; `server` serves the current point."""
    p_a = round(float(p_a), 6)
    p_b = round(float(p_b), 6)
    need = best_of // 2 + 1
    rule = final_rule if (sets_a == need - 1 and sets_b == need - 1) else normal_rule
    tb_at, tb_target = _TbAt(rule)
    if tb_at is not None and games_a == tb_at and games_b == tb_at:
        n = points_a + points_b
        first_server = server if ((n + 1) // 2) % 2 == 0 else _Other(server)
        t = TiebreakFrom(p_a, p_b, points_a, points_b, first_server, tb_target)
        nxt = _Other(first_server)
        return (t * MatchFrom(p_a, p_b, sets_a + 1, sets_b, nxt, best_of, final_rule, normal_rule)
                + (1 - t) * MatchFrom(p_a, p_b, sets_a, sets_b + 1, nxt, best_of, final_rule, normal_rule))
    if server == "a":
        g = GameFrom(p_a, points_a, points_b)
    else:
        g = 1.0 - GameFrom(p_b, points_b, points_a)
    nxt_server = _Other(server)
    total = 0.0
    for prob, ga, gb in ((g, games_a + 1, games_b), (1 - g, games_a, games_b + 1)):
        if prob == 0.0:
            continue
        aa, ab, ba, bb = SetFrom(p_a, p_b, ga, gb, nxt_server, rule)
        total += prob * (aa * MatchFrom(p_a, p_b, sets_a + 1, sets_b, "a", best_of, final_rule, normal_rule)
                         + ab * MatchFrom(p_a, p_b, sets_a + 1, sets_b, "b", best_of, final_rule, normal_rule)
                         + ba * MatchFrom(p_a, p_b, sets_a, sets_b + 1, "a", best_of, final_rule, normal_rule)
                         + bb * MatchFrom(p_a, p_b, sets_a, sets_b + 1, "b", best_of, final_rule, normal_rule))
    return total


def SlamFinalSetRule(slam, year):
    """Final-set rule by Grand Slam and year (men bo5 and women bo3 share it)."""
    if year >= 2022:
        return "tb10"
    if slam == "usopen":
        return "tb7"
    if slam == "wimbledon":
        return "tb7_12" if year >= 2019 else "adv"
    if slam == "ausopen":
        return "tb10" if year >= 2019 else "adv"
    return "adv"  # frenchopen before 2022


class MatchProbTable:
    """Pre-computed pre-match P(A wins) on a p_a x p_b grid with bilinear interpolation (fast for backtests)."""

    def __init__(self, lo=0.30, hi=0.95, step=0.005):
        self.lo, self.hi, self.step = lo, hi, step
        self.grid = np.round(np.arange(lo, hi + step / 2, step), 6)
        self.tables = {}

    def Table(self, best_of):
        if best_of not in self.tables:
            n = len(self.grid)
            table = np.empty((n, n))
            for i, p_a in enumerate(self.grid):
                for j, p_b in enumerate(self.grid):
                    table[i, j] = MatchProb(p_a, p_b, best_of)
                GameFrom.cache_clear(); TiebreakFrom.cache_clear(); SetFrom.cache_clear(); MatchFrom.cache_clear()
            self.tables[best_of] = table
        return self.tables[best_of]

    def Prob(self, p_a, p_b, best_of=3):
        table = self.Table(5 if best_of == 5 else 3)
        n = len(self.grid)
        x = (min(max(p_a, self.lo), self.hi) - self.lo) / self.step
        y = (min(max(p_b, self.lo), self.hi) - self.lo) / self.step
        i = min(int(x), n - 2)
        j = min(int(y), n - 2)
        fx = x - i
        fy = y - j
        return ((1 - fx) * (1 - fy) * table[i, j] + fx * (1 - fy) * table[i + 1, j]
                + (1 - fx) * fy * table[i, j + 1] + fx * fy * table[i + 1, j + 1])