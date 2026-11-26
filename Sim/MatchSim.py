r"""Vectorised Monte Carlo of tennis matches from serve-point win probabilities (research, tested:false).

Same i.i.d. point model as Backtest.Markov: player A wins a point on serve with p_a, B with p_b. Regular games are
drawn from the exact hold probability implied by the point probability (so no per-point loop is needed there);
tiebreaks are played point by point with the standard serve rotation. Sets: first to 6 games, win by 2, tiebreak at
6-6 (to 7, or to 10 in a final set when final_tb=10). Coin-toss first server.

SimulateMatches(p_a, p_b, best_of, final_tb, rng) takes one entry per simulated match (arrays broadcast) and returns
a dict of arrays: a_won, sets_a, sets_b, games_a, games_b, total_games, tiebreaks, straight (winner lost no set).
Synthetic matches only: never real outcomes, never proof of an edge.
"""
from __future__ import annotations

import numpy as np


def HoldProb(p):
    """P(server wins a game) for point-win probability p (exact closed form)."""
    p = np.asarray(p, dtype=float)
    q = 1.0 - p
    deuce = p * p / (p * p + q * q)
    return p ** 4 * (1 + 4 * q + 10 * q * q) + 20 * p ** 3 * q ** 3 * deuce


def _Tiebreak(p_a, p_b, first_a, target, rng):
    """Point-by-point tiebreaks for k sims. first_a: True if A serves the first point. Returns a_won."""
    k = len(p_a)
    pa = np.zeros(k, dtype=int)
    pb = np.zeros(k, dtype=int)
    done = np.zeros(k, dtype=bool)
    a_won = np.zeros(k, dtype=bool)
    n = 0
    while not done.all():
        a_serving = np.where(((n + 1) // 2) % 2 == 0, first_a, ~first_a)
        q = np.where(a_serving, p_a, 1.0 - p_b)
        win = rng.random(k) < q
        act = ~done
        pa += (win & act)
        pb += (~win & act)
        fin = act & (((pa >= target) | (pb >= target)) & (np.abs(pa - pb) >= 2))
        a_won |= fin & (pa > pb)
        done |= fin
        n += 1
        if n > 400:
            break
    return a_won


def SimulateMatches(p_a, p_b, best_of=3, final_tb=7, rng=None, n=None, sigma=0.0):
    """sigma > 0 adds a match-level 'form' shock: delta ~ N(0, sigma) per simulated match, p_a + delta and
    p_b - delta (clipped to [0.05, 0.98]), i.e. on the day one player is better than the pre-match estimate."""
    rng = rng if rng is not None else np.random.default_rng()
    if n is None:
        n = int(np.broadcast(np.asarray(p_a), np.asarray(p_b), np.asarray(best_of), np.asarray(final_tb)).size)
    p_a = np.broadcast_to(np.asarray(p_a, dtype=float), (n,)).copy()
    p_b = np.broadcast_to(np.asarray(p_b, dtype=float), (n,)).copy()
    if sigma > 0:
        delta = rng.normal(0.0, sigma, n)
        p_a = np.clip(p_a + delta, 0.05, 0.98)
        p_b = np.clip(p_b - delta, 0.05, 0.98)
    best_of = np.broadcast_to(np.asarray(best_of, dtype=int), (n,)).copy()
    final_tb = np.broadcast_to(np.asarray(final_tb, dtype=int), (n,)).copy()
    need = (best_of + 1) // 2
    h_a = HoldProb(p_a)
    h_b = HoldProb(p_b)
    a_serves = rng.random(n) < 0.5
    sets_a = np.zeros(n, dtype=int)
    sets_b = np.zeros(n, dtype=int)
    ga = np.zeros(n, dtype=int)
    gb = np.zeros(n, dtype=int)
    tot_a = np.zeros(n, dtype=int)
    tot_b = np.zeros(n, dtype=int)
    tbs = np.zeros(n, dtype=int)
    done = np.zeros(n, dtype=bool)
    for _ in range(400):
        act = np.flatnonzero(~done)
        if len(act) == 0:
            break
        g_a, g_b = ga[act], gb[act]
        in_tb = (g_a == 6) & (g_b == 6)
        # regular games
        reg = act[~in_tb]
        if len(reg):
            hold = np.where(a_serves[reg], h_a[reg], h_b[reg])
            server_wins = rng.random(len(reg)) < hold
            a_game = np.where(a_serves[reg], server_wins, ~server_wins)
            ga[reg] += a_game
            gb[reg] += ~a_game
            tot_a[reg] += a_game
            tot_b[reg] += ~a_game
            a_serves[reg] = ~a_serves[reg]
        tb = act[in_tb]
        set_won_a = np.zeros(len(act), dtype=bool)
        set_won_b = np.zeros(len(act), dtype=bool)
        if len(tb):
            final = (sets_a[tb] == need[tb] - 1) & (sets_b[tb] == need[tb] - 1)
            target = np.where(final, final_tb[tb], 7)
            first_a = a_serves[tb].copy()
            a_tb = _Tiebreak(p_a[tb], p_b[tb], first_a, target, rng)
            tbs[tb] += 1
            tot_a[tb] += a_tb
            tot_b[tb] += ~a_tb
            ga[tb] += a_tb
            gb[tb] += ~a_tb
            a_serves[tb] = ~first_a
        g_a, g_b = ga[act], gb[act]
        set_won_a = ((g_a >= 6) & (g_a - g_b >= 2)) | (g_a == 7) & (g_b == 6)
        set_won_b = ((g_b >= 6) & (g_b - g_a >= 2)) | (g_b == 7) & (g_a == 6)
        sw = act[set_won_a]
        sl = act[set_won_b]
        sets_a[sw] += 1
        sets_b[sl] += 1
        ended = act[set_won_a | set_won_b]
        ga[ended] = 0
        gb[ended] = 0
        done[act] = (sets_a[act] >= need[act]) | (sets_b[act] >= need[act])
    a_won = sets_a > sets_b
    return {"a_won": a_won, "sets_a": sets_a, "sets_b": sets_b, "games_a": tot_a, "games_b": tot_b,
            "total_games": tot_a + tot_b, "tiebreaks": tbs, "straight": np.where(a_won, sets_b == 0, sets_a == 0)}


def ParseScore(score):
    """'6-7(5) 6-3 6-3' -> list of (games_w, games_l, tiebreak) per set, or None for RET/W/O/DEF/unparseable."""
    if not isinstance(score, str) or not score.strip():
        return None
    up = score.upper()
    if any(t in up for t in ("RET", "W/O", "DEF", "ABD", "UNK", "WALKOVER", "UNFINISHED")):
        return None
    out = []
    for part in score.split():
        base = part.split("(")[0]
        if base.startswith("[") or "-" not in base:
            return None
        try:
            w, l = (int(x) for x in base.split("-"))
        except ValueError:
            return None
        out.append((w, l, int(w + l == 13 or (max(w, l) >= 7 and abs(w - l) == 1 and "(" in part))))
    return out or None


def _TiebreakExact(p_a, p_b, a_first, target):
    """Exact P(A wins a tiebreak to `target`), vectorised. a_first: bool array, A serves the first point."""
    k = len(p_a)
    T = int(target)
    mass = {(0, 0): np.ones(k)}
    win = np.zeros(k)
    q = p_a * (1 - p_b)
    r = (1 - p_a) * p_b
    closure = q / (q + r)
    for n in range(0, 2 * T):
        for a in range(0, n + 1):
            b = n - a
            m = mass.pop((a, b), None)
            if m is None:
                continue
            if a == T - 1 and b == T - 1:
                win += m * closure
                continue
            a_srv = np.where(((n + 1) // 2) % 2 == 0, a_first, ~a_first)
            pt = np.where(a_srv, p_a, 1 - p_b)
            for na, nb, pr in ((a + 1, b, pt), (a, b + 1, 1 - pt)):
                if na >= T and na - nb >= 2:
                    win += m * pr
                elif nb >= T and nb - na >= 2:
                    pass
                else:
                    mass[(na, nb)] = mass.get((na, nb), 0) + m * pr
    return win


def _SetExact(p_a, p_b, a_first, tb_target):
    """-> (P(A wins set), P(A serves first in the next set | A wins), same | B wins) as arrays."""
    h_a, h_b = HoldProb(p_a), HoldProb(p_b)
    k = len(p_a)
    mass = {(0, 0): np.ones(k)}
    out = {"a_nextA": np.zeros(k), "a_nextB": np.zeros(k), "b_nextA": np.zeros(k), "b_nextB": np.zeros(k)}
    for n in range(0, 13):
        for ga in range(0, n + 1):
            gb = n - ga
            m = mass.pop((ga, gb), None)
            if m is None:
                continue
            a_srv = a_first if n % 2 == 0 else ~a_first
            if ga == 6 and gb == 6:
                pw = _TiebreakExact(p_a, p_b, a_srv, tb_target)
                nxt_a = ~a_srv  # receiver of the first tiebreak point serves next set
                out["a_nextA"] += np.where(nxt_a, m * pw, 0)
                out["a_nextB"] += np.where(~nxt_a, m * pw, 0)
                out["b_nextA"] += np.where(nxt_a, m * (1 - pw), 0)
                out["b_nextB"] += np.where(~nxt_a, m * (1 - pw), 0)
                continue
            pg = np.where(a_srv, h_a, 1 - h_b)
            nxt_a = ~a_srv
            for na, nb, pr in ((ga + 1, gb, pg), (ga, gb + 1, 1 - pg)):
                if (na >= 6 and na - nb >= 2) or (na == 7):
                    out["a_nextA"] += np.where(nxt_a, m * pr, 0)
                    out["a_nextB"] += np.where(~nxt_a, m * pr, 0)
                elif (nb >= 6 and nb - na >= 2) or (nb == 7):
                    out["b_nextA"] += np.where(nxt_a, m * pr, 0)
                    out["b_nextB"] += np.where(~nxt_a, m * pr, 0)
                else:
                    mass[(na, nb)] = mass.get((na, nb), 0) + m * pr
    return out


def ExactMatchProb(p_a, p_b, best_of=3, final_tb=7):
    """Exact P(A wins) under the same i.i.d. point model, vectorised over matches (coin-toss first server).
    best_of / final_tb may be arrays; computed per distinct (best_of, final_tb) group."""
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    k = len(p_a)
    bo = np.broadcast_to(np.asarray(best_of, dtype=int), (k,))
    ft = np.broadcast_to(np.asarray(final_tb, dtype=int), (k,))
    res = np.zeros(k)
    for b_o in np.unique(bo):
        for f_t in np.unique(ft[bo == b_o]):
            m = (bo == b_o) & (ft == f_t)
            pa, pb = p_a[m], p_b[m]
            need = (b_o + 1) // 2
            total = np.zeros(m.sum())
            for a_first0 in (True, False):
                mass = {(0, 0, a_first0): np.ones(m.sum())}
                win = np.zeros(m.sum())
                for n in range(0, 2 * need - 1):
                    for sa in range(0, n + 1):
                        sb = n - sa
                        for af in (True, False):
                            mm = mass.pop((sa, sb, af), None)
                            if mm is None:
                                continue
                            final = (sa == need - 1) and (sb == need - 1)
                            o = _SetExact(pa, pb, np.full(len(pa), af), f_t if final else 7)
                            for key, (da, db, naf) in {"a_nextA": (1, 0, True), "a_nextB": (1, 0, False),
                                                       "b_nextA": (0, 1, True), "b_nextB": (0, 1, False)}.items():
                                nsa, nsb = sa + da, sb + db
                                if nsa == need:
                                    win += mm * o[key]
                                elif nsb == need:
                                    pass
                                else:
                                    mass[(nsa, nsb, naf)] = mass.get((nsa, nsb, naf), 0) + mm * o[key]
                total += 0.5 * win
            res[m] = total
    return res


def ExactMatchProbForm(p_a, p_b, best_of=3, final_tb=7, sigma=0.0, nodes=9):
    """Exact P(A wins) averaged over the N(0, sigma) form shock (Gauss-Hermite quadrature)."""
    if sigma <= 0:
        return ExactMatchProb(p_a, p_b, best_of, final_tb)
    x, w = np.polynomial.hermite_e.hermegauss(nodes)
    w = w / w.sum()
    p_a = np.asarray(p_a, dtype=float)
    p_b = np.asarray(p_b, dtype=float)
    out = np.zeros(len(p_a))
    for xi, wi in zip(x, w):
        d = sigma * xi
        out += wi * ExactMatchProb(np.clip(p_a + d, 0.05, 0.98), np.clip(p_b - d, 0.05, 0.98), best_of, final_tb)
    return out