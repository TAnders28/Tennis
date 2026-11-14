"""Markov prop projections: set scores, game totals, handicaps, and serve-point counts.

Aces and double faults are Poisson in expected service points, using empirical per-point
rates. They are not implied by the point-win model alone.
"""
from __future__ import annotations

import math
from functools import lru_cache

from Backtest.Markov import GameFrom, MatchFrom, SetFrom, TiebreakFrom, TiebreakServer, _Other, _TbAt

P_ROUND = 6


def _P(p):
    return round(float(p), P_ROUND)


@lru_cache(maxsize=None)
def GamePointsFrom(p, x, y):
    """Expected remaining points in a game from server-points x, receiver-points y."""
    p = _P(p)
    if x >= 4 and x - y >= 2:
        return 0.0
    if y >= 4 and y - x >= 2:
        return 0.0
    if x >= 3 and y >= 3:
        # from deuce-like: each pair of points ends the game with prob p^2+(1-p)^2, else return
        # remaining points from (3,3): 2 / (p^2 + (1-p)^2)
        # from ad-in (x>y):  p*1 + (1-p)*(1 + E_deuce)
        deuce_pts = 2.0 / (p * p + (1.0 - p) * (1.0 - p))
        if x == y:
            return deuce_pts
        if x > y:
            return 1.0 + (1.0 - p) * deuce_pts
        return 1.0 + p * deuce_pts
    return 1.0 + p * GamePointsFrom(p, x + 1, y) + (1.0 - p) * GamePointsFrom(p, x, y + 1)



def _PWinPoint(p_a, p_b, server):
    return p_a if server == 'a' else (1.0 - p_b)


def _TbDeuceServePoints(p_a, p_b, first_server, n_played, for_player):
    """Expected remaining serve points for for_player from a TB deuce (win by 2)."""

    def step(n):
        s0, s1 = TiebreakServer(first_server, n), TiebreakServer(first_server, n + 1)
        p0, p1 = _PWinPoint(p_a, p_b, s0), _PWinPoint(p_a, p_b, s1)
        add = (1.0 if s0 == for_player else 0.0) + (1.0 if s1 == for_player else 0.0)
        p_split = 1.0 - p0 * p1 - (1.0 - p0) * (1.0 - p1)
        return add, p_split

    c0, ps0 = step(n_played)
    c2, ps2 = step(n_played + 2)
    denom = 1.0 - ps0 * ps2
    if denom <= 1e-15:
        return c0 + c2
    return (c0 + ps0 * c2) / denom


@lru_cache(maxsize=None)
def TbServePointsFrom(p_a, p_b, a, b, first_server, target, for_player):
    """Expected remaining serve points for for_player ('a' or 'b') in a tiebreak."""
    p_a, p_b = _P(p_a), _P(p_b)
    if (a >= target and a - b >= 2) or (b >= target and b - a >= 2):
        return 0.0
    if a == b and a >= target - 1:
        return _TbDeuceServePoints(p_a, p_b, first_server, a + b, for_player)
    server = TiebreakServer(first_server, a + b)
    q = p_a if server == 'a' else p_b
    add = 1.0 if server == for_player else 0.0
    return add + q * TbServePointsFrom(p_a, p_b, a + 1, b, first_server, target, for_player) + (
        1.0 - q) * TbServePointsFrom(p_a, p_b, a, b + 1, first_server, target, for_player)


@lru_cache(maxsize=None)
def SetOutcomeDist(p_a, p_b, ga, gb, server, rule):
    """Frozen dict (games_a, games_b, next_server, tiebreak) -> probability.

    Evaluated at the start of a game. Matches SetFrom terminals for tb7/tb10/tb7_12.
    Advantage sets recurse until a 20-game lead cap, then park the remainder on a 2-game win.
    """
    p_a, p_b = _P(p_a), _P(p_b)
    tb_at, tb_target = _TbAt(rule)
    if tb_at is not None and (ga > tb_at + 1 or gb > tb_at + 1 or (ga == tb_at + 1 and gb == tb_at + 1)):
        raise ValueError(f"invalid game score {ga}-{gb} for rule {rule}")
    if ga >= 6 and ga - gb >= 2:
        nxt = server
        return (((ga, gb, nxt, False), 1.0),)
    if gb >= 6 and gb - ga >= 2:
        nxt = server
        return (((ga, gb, nxt, False), 1.0),)
    if tb_at is not None and ga == tb_at and gb == tb_at:
        t = TiebreakFrom(p_a, p_b, 0, 0, server, tb_target)
        nxt = _Other(server)
        return (((ga + 1, gb, nxt, True), t), ((ga, gb + 1, nxt, True), 1.0 - t))
    if rule == "adv" and ga == gb and ga >= 5:
        # expected remaining games is even; stop with a 2-game margin
        hold_a = GameFrom(p_a, 0, 0)
        break_a = 1.0 - GameFrom(p_b, 0, 0)
        q = hold_a * break_a
        r = (1.0 - hold_a) * (1.0 - break_a)
        denom = q + r
        win = q / denom if denom else 0.5
        # skip the extra-game count (hold/break pairs); record as ga+2, gb or ga, gb+2
        nxt = server
        return (((ga + 2, gb, nxt, False), win), ((ga, gb + 2, nxt, False), 1.0 - win))
    if rule == "adv" and (ga >= 20 or gb >= 20):
        nxt = server
        if ga > gb:
            return (((ga, gb, nxt, False), 1.0),)
        if gb > ga:
            return (((ga, gb, nxt, False), 1.0),)
        return (((ga + 2, gb, nxt, False), 0.5), ((ga, gb + 2, nxt, False), 0.5))
    g = GameFrom(p_a, 0, 0) if server == "a" else (1.0 - GameFrom(p_b, 0, 0))
    nxt = _Other(server)
    out = {}
    for prob, nga, ngb in ((g, ga + 1, gb), (1.0 - g, ga, gb + 1)):
        if prob <= 0.0:
            continue
        for key, p in SetOutcomeDist(p_a, p_b, nga, ngb, nxt, rule):
            out[key] = out.get(key, 0.0) + prob * p
    return tuple(sorted(out.items()))


def _ServiceGames(games_a, games_b, first_server, tiebreak):
    n = games_a + games_b
    first_n = (n + 1) // 2
    second_n = n // 2
    if first_server == "a":
        return first_n, second_n
    return second_n, first_n


def SetProjection(p_a, p_b, server="a", rule="tb7"):
    p_a, p_b = _P(p_a), _P(p_b)
    dist = SetOutcomeDist(p_a, p_b, 0, 0, server, rule)
    p_win = 0.0
    p_tb = 0.0
    e_games_a = e_games_b = 0.0
    e_svg_a = e_svg_b = 0.0
    e_svpt_a = e_svpt_b = 0.0
    pts_game_a = GamePointsFrom(p_a, 0, 0)
    pts_game_b = GamePointsFrom(p_b, 0, 0)
    tb_at, tb_target = _TbAt(rule)
    scores = {}
    for (ga, gb, nxt, tb), prob in dist:
        scores[(ga, gb)] = scores.get((ga, gb), 0.0) + prob
        if ga > gb:
            p_win += prob
        if tb:
            p_tb += prob
        e_games_a += prob * ga
        e_games_b += prob * gb
        sa, sb = _ServiceGames(ga, gb, server, tb)
        if tb:
            # the last scored game is the TB; subtract it from service-game counts for point rates
            if server == "a":
                sa -= 1
            else:
                sb -= 1
            e_svpt_a += prob * TbServePointsFrom(p_a, p_b, 0, 0, server, tb_target or 7, "a")
            e_svpt_b += prob * TbServePointsFrom(p_a, p_b, 0, 0, server, tb_target or 7, "b")
        e_svg_a += prob * sa
        e_svg_b += prob * sb
        e_svpt_a += prob * sa * pts_game_a
        e_svpt_b += prob * sb * pts_game_b
    return {
        "p_win": p_win,
        "p_tb": p_tb,
        "scores": scores,
        "e_games_a": e_games_a,
        "e_games_b": e_games_b,
        "e_total_games": e_games_a + e_games_b,
        "e_service_games_a": e_svg_a,
        "e_service_games_b": e_svg_b,
        "e_svpt_a": e_svpt_a,
        "e_svpt_b": e_svpt_b,
    }


def _MixSet(p_a, p_b, server, rule):
    return SetProjection(p_a, p_b, server, rule)


def _ConvolveMatch(p_a, p_b, best_of, final_rule, normal_rule, first_server):
    need = best_of // 2 + 1
    # state: (sets_a, sets_b, server) -> list of (prob, games_a, games_b, sets_a, sets_b, svpt_a, svpt_b, n_tb)
    # Use successive convolution; cap totals in dicts with rounded games (ints already).
    states = {(0, 0, first_server): [(1.0, 0, 0, 0, 0, 0.0, 0.0, 0)]}
    terminals = []
    safety = 0
    while states and safety < 16:
        safety += 1
        nxt_states = {}
        for (sa, sb, server), paths in states.items():
            if sa >= need or sb >= need:
                terminals.extend(paths)
                continue
            rule = final_rule if (sa == need - 1 and sb == need - 1) else normal_rule
            proj = _MixSet(p_a, p_b, server, rule)
            for (ga, gb), p_set in proj["scores"].items():
                tb = (ga >= 6 and gb >= 6)
                # expected svpt for this exact score: use share of SetProjection not split per score;
                # approximate by scaling set expectations is wrong. Recompute from SetOutcomeDist later
                # if needed; for now use score-conditional service games * average points/game,
                # plus TB if applicable.
                sa2, sb2 = sa + (1 if ga > gb else 0), sb + (1 if gb > ga else 0)
                # next server from a representative outcome with this score
                nxt_server = None
                svpt_a = svpt_b = 0.0
                mass = 0.0
                for (xga, xgb, nxt, is_tb), pr in SetOutcomeDist(_P(p_a), _P(p_b), 0, 0, server, rule):
                    if xga == ga and xgb == gb:
                        nxt_server = nxt
                        mass += pr
                if nxt_server is None:
                    nxt_server = _Other(server)
                sga, sgb = _ServiceGames(ga, gb, server, tb)
                pts_a, pts_b = GamePointsFrom(_P(p_a), 0, 0), GamePointsFrom(_P(p_b), 0, 0)
                add_a, add_b = sga * pts_a, sgb * pts_b
                if tb:
                    _, tb_target = _TbAt(rule)
                    if server == "a":
                        add_a -= pts_a
                    else:
                        add_b -= pts_b
                    add_a += TbServePointsFrom(_P(p_a), _P(p_b), 0, 0, server, tb_target or 7, "a")
                    add_b += TbServePointsFrom(_P(p_a), _P(p_b), 0, 0, server, tb_target or 7, "b")
                for prob, tga, tgb, _, _, sva, svb, ntb in paths:
                    rec = (prob * p_set, tga + ga, tgb + gb, sa2, sb2, sva + add_a, svb + add_b, ntb + int(tb))
                    if sa2 >= need or sb2 >= need:
                        terminals.append(rec)
                    else:
                        nxt_states.setdefault((sa2, sb2, nxt_server), []).append(rec)
        states = nxt_states
    return terminals


def MatchProjection(p_a, p_b, best_of=3, final_rule="tb7", normal_rule="tb7"):
    """Pre-match prop distribution, averaging a coin toss for first serve."""
    p_a, p_b = _P(p_a), _P(p_b)
    terms = _ConvolveMatch(p_a, p_b, best_of, final_rule, normal_rule, "a") + _ConvolveMatch(
        p_a, p_b, best_of, final_rule, normal_rule, "b")
    # each branch already sums to 1; together they sum to 2
    set_scores = {}
    totals = {}
    diffs = {}
    set_diffs = {}
    p_tb = 0.0
    e_games = e_diff = e_svpt_a = e_svpt_b = 0.0
    p_win = 0.0
    for prob, ga, gb, sa, sb, sva, svb, ntb in terms:
        prob = prob * 0.5
        set_scores[(sa, sb)] = set_scores.get((sa, sb), 0.0) + prob
        totals[ga + gb] = totals.get(ga + gb, 0.0) + prob
        diffs[ga - gb] = diffs.get(ga - gb, 0.0) + prob
        set_diffs[sa - sb] = set_diffs.get((sa - sb), 0.0) + prob
        if ntb:
            p_tb += prob
        e_games += prob * (ga + gb)
        e_diff += prob * (ga - gb)
        e_svpt_a += prob * sva
        e_svpt_b += prob * svb
        if sa > sb:
            p_win += prob
    return {
        "p_win": p_win,
        "set_scores": set_scores,
        "total_games": totals,
        "game_diff": diffs,
        "set_diff": set_diffs,
        "p_at_least_one_tb": p_tb,
        "p_sweep": set_scores.get((best_of // 2 + 1, 0), 0.0),
        "e_total_games": e_games,
        "e_game_diff": e_diff,
        "e_svpt_a": e_svpt_a,
        "e_svpt_b": e_svpt_b,
    }


def ProbTotalOver(proj, line):
    """P(total games > line). A line of 21.5 is the usual over/under."""
    return sum(p for g, p in proj["total_games"].items() if g > line)


def ProbGameHandicap(proj, line, side="a"):
    """P(games_side - games_other > line). side 'a' or 'b'. line is the spread on that side, e.g. -3.5."""
    # "A -3.5 games" wins if A wins by at least 4 games: games_a - games_b > 3.5, equivalent to covering -3.5
    # Standard: side A with handicap h (h negative if favorite) covers if (game_diff_a) + h > 0? 
    # Kalshi: "Fritz -6.5 games" = Fritz wins by at least 6.5 games, i.e. game_diff_fritz > 6.5.
    if side == "a":
        return sum(p for d, p in proj["game_diff"].items() if d > line)
    return sum(p for d, p in proj["game_diff"].items() if -d > line)


def ProbSetHandicap(proj, line, side="a"):
    if side == "a":
        return sum(p for d, p in proj["set_diff"].items() if d > line)
    return sum(p for d, p in proj["set_diff"].items() if -d > line)


def ProbExactSets(proj, sets_a, sets_b):
    return float(proj["set_scores"].get((sets_a, sets_b), 0.0))


def ProbSetWinner(p_a, p_b, set_no, best_of=3, final_rule="tb7", normal_rule="tb7"):
    """P(A wins set number `set_no` (1-based)) and P(that set is played)."""
    p_a, p_b = _P(p_a), _P(p_b)
    need = best_of // 2 + 1
    # average first server; track P(reach set k with server s and sets sa,sb)
    reach = {("a", 0, 0): 0.5, ("b", 0, 0): 0.5}
    p_win_set = 0.0
    p_played = 0.0
    for k in range(1, best_of + 1):
        nxt = {}
        for (server, sa, sb), prob in reach.items():
            if sa >= need or sb >= need or prob <= 0:
                continue
            if k == set_no:
                p_played += prob
            rule = final_rule if (sa == need - 1 and sb == need - 1) else normal_rule
            proj = SetProjection(p_a, p_b, server, rule)
            if k == set_no:
                p_win_set += prob * proj["p_win"]
            for (ga, gb, nxt_s, _tb), pr in SetOutcomeDist(p_a, p_b, 0, 0, server, rule):
                sa2 = sa + (1 if ga > gb else 0)
                sb2 = sb + (1 if gb > ga else 0)
                key = (nxt_s, sa2, sb2)
                nxt[key] = nxt.get(key, 0.0) + prob * pr
        reach = nxt
    return p_win_set, p_played


def PoissonCdf(k, lam):
    if lam <= 0:
        return 1.0 if k >= 0 else 0.0
    # regularized gamma / recursive Poisson
    if k < 0:
        return 0.0
    term = math.exp(-lam)
    s = term
    for i in range(1, int(k) + 1):
        term *= lam / i
        s += term
        if s >= 1.0:
            return 1.0
    return min(1.0, s)


def PoissonTail(k, lam):
    """P(X >= k) for X ~ Poisson(lam)."""
    if k <= 0:
        return 1.0
    return 1.0 - PoissonCdf(k - 1, lam)


def AceOrDfProjection(e_svpt, rate_per_point, line, is_over=True):
    """P(count > line) or P(count >= threshold) for an integer 'at least k' line.

    If `line` is like 18.5, this is P(X > 18.5) = P(X >= 19).
    If `line` is 19 with is_over False and the contract is 'at least 19', pass line=18.5 or use AtLeast.
    """
    lam = max(0.0, float(e_svpt) * float(rate_per_point))
    threshold = math.floor(line) + 1 if line != int(line) else (int(line) + 1 if is_over else int(line))
    return lam, PoissonTail(threshold, lam)


def AtLeastCount(e_svpt, rate_per_point, k):
    lam = max(0.0, float(e_svpt) * float(rate_per_point))
    return lam, PoissonTail(int(k), lam)