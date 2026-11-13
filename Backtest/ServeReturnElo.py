r"""Opponent-adjusted Elo-style serve-points-won and return-points-won ratings.

Prototype alternative to Backtest.ServeReturn (which uses residual shrinkage, not a
K-factor). Each player has a serve rating and a return rating, overall and per
surface. Ratings live on a 400-point Elo scale and start at 0 (tour average).

    logit p_A_serve = logit(mu_s) + log(10)/400 * (r_serve(A) - r_return(B))

mu_s is a running points-weighted tour serve-point rate on that surface. After the
match (leak-free), both players are updated with a fractional-Elo step on the
observed serve/return *rates*:

    r_serve(A) += K_A * (actual_spw_A - expected)
    r_return(B) += K_B * (expected - actual_spw_A)

K = k_a / (n + k_b)^k_c, using each player's count of matches that had point stats,
with an optional lower-level multiplier. Defaults follow the surface-Elo K schedule
and are NOT tuned on the holdout.

Usage (walk + holdout comparison vs the current frozen model):
  .venv\Scripts\python.exe -m Backtest.RunServeReturnElo --out-dir E:\AgentDocs\Tennibot\2026-11-13_market-blend
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from Backtest.ServeReturn import DEFAULT_MU, SURFACES


@dataclass(frozen=True)
class ServeReturnEloParams:
    k_a: float = 250.0
    k_b: float = 5.0
    k_c: float = 0.4
    surface_weight: float = 0.5
    lower_k_mult: float = 1.0
    init: float = 0.0
    scale: float = 400.0
    mu_window: float = 20000.0

    def Key(self):
        return "srelo_" + "_".join(f"{k}{v:g}" for k, v in asdict(self).items() if k != "init")


class ServeReturnElo:
    def __init__(self, params: ServeReturnEloParams | None = None, tour="atp"):
        self.params = params or ServeReturnEloParams()
        default = DEFAULT_MU.get(tour, 0.6)
        self.mu = {s: [default * 5000.0, 5000.0] for s in SURFACES + ("Unknown",)}
        self.mu_decay = 1.0 - 1.0 / self.params.mu_window
        self.serve = {}
        self.ret = {}
        self.count = {}
        self.serve_s = {}
        self.ret_s = {}
        self.count_s = {}

    def Mu(self, surface):
        won, played = self.mu.get(surface, self.mu["Unknown"])
        return won / played

    def _Combined(self, store, store_s, pid, surface):
        init = self.params.init
        overall = store.get(pid, init)
        if surface in SURFACES:
            local = store_s.get((pid, surface), init)
            w = self.params.surface_weight
            return (1.0 - w) * overall + w * local
        return overall

    def _K(self, n, level_tag):
        p = self.params
        k = p.k_a / (n + p.k_b) ** p.k_c
        if level_tag != "tour":
            k *= p.lower_k_mult
        return k

    def _Rate(self, mu, diff):
        # logit(p) = logit(mu) + log(10)/scale * diff
        mu = min(max(mu, 1e-6), 1.0 - 1e-6)
        z = math.log(mu / (1.0 - mu)) + math.log(10.0) / self.params.scale * diff
        z = min(max(z, -20.0), 20.0)
        p = 1.0 / (1.0 + math.exp(-z))
        return min(max(p, 0.30), 0.95)

    def PServe(self, a, b, surface):
        mu = self.Mu(surface)
        diff_a = self._Combined(self.serve, self.serve_s, a, surface) - self._Combined(self.ret, self.ret_s, b, surface)
        diff_b = self._Combined(self.serve, self.serve_s, b, surface) - self._Combined(self.ret, self.ret_s, a, surface)
        return self._Rate(mu, diff_a), self._Rate(mu, diff_b)

    def _Apply(self, store, store_s, pid, surface, delta, k, ks):
        init = self.params.init
        store[pid] = store.get(pid, init) + k * delta
        if surface in SURFACES:
            key = (pid, surface)
            store_s[key] = store_s.get(key, init) + ks * delta

    def Update(self, winner, loser, surface, w_svpt, w_won, l_svpt, l_won, level_tag="tour"):
        values = (w_svpt, w_won, l_svpt, l_won)
        if not all(math.isfinite(v) for v in values) or not (w_svpt > 0 and l_svpt > 0) or w_won > w_svpt or l_won > l_svpt:
            return
        # K from the pre-match match count (one increment per player per match, shared by serve and return).
        p_w, p_l = self.PServe(winner, loser, surface)
        nw, nl = self.count.get(winner, 0), self.count.get(loser, 0)
        kw, kl = self._K(nw, level_tag), self._K(nl, level_tag)
        if surface in SURFACES:
            kws = self._K(self.count_s.get((winner, surface), 0), level_tag)
            kls = self._K(self.count_s.get((loser, surface), 0), level_tag)
        else:
            kws = kls = 0.0
        self._Apply(self.serve, self.serve_s, winner, surface, (w_won / w_svpt) - p_w, kw, kws)
        self._Apply(self.ret, self.ret_s, loser, surface, p_w - (w_won / w_svpt), kl, kls)
        self._Apply(self.serve, self.serve_s, loser, surface, (l_won / l_svpt) - p_l, kl, kls)
        self._Apply(self.ret, self.ret_s, winner, surface, p_l - (l_won / l_svpt), kw, kws)
        self.count[winner] = nw + 1
        self.count[loser] = nl + 1
        if surface in SURFACES:
            self.count_s[(winner, surface)] = self.count_s.get((winner, surface), 0) + 1
            self.count_s[(loser, surface)] = self.count_s.get((loser, surface), 0) + 1
        bucket = self.mu.get(surface, self.mu["Unknown"])
        bucket[0] = bucket[0] * self.mu_decay + w_won + l_won
        bucket[1] = bucket[1] * self.mu_decay + w_svpt + l_svpt


def RunServeReturnElo(frame, params: ServeReturnEloParams | None = None, tour="atp"):
    """Leak-free group walk, same contract as Backtest.Engine: predict a (date, stage)
    group from the pre-group snapshot, then apply that group's results."""
    import numpy as np

    from Backtest.Engine import SharedTable, _Columns

    cols = _Columns(frame)
    n = len(frame)
    model = ServeReturnElo(params, tour)
    table = SharedTable()
    p_match = np.full(n, np.nan)
    p_serve_w = np.full(n, np.nan)
    p_serve_l = np.full(n, np.nan)
    group, winner, loser, surface, best_of, level = (cols[k] for k in
                                                     ("group", "winner", "loser", "surface", "best_of", "level"))
    w_svpt, w_won, l_svpt, l_won = (cols[k] for k in ("w_svpt", "w_won", "l_svpt", "l_won"))
    start = 0
    while start < n:
        g = group[start]
        end = start
        while end < n and group[end] == g:
            end += 1
        for i in range(start, end):
            pa, pb = model.PServe(winner[i], loser[i], surface[i])
            p_serve_w[i] = pa
            p_serve_l[i] = pb
            p_match[i] = table.Prob(pa, pb, best_of[i])
        for i in range(start, end):
            if w_svpt[i] > 0 and l_svpt[i] > 0 and w_won[i] == w_won[i] and l_won[i] == l_won[i]:
                model.Update(winner[i], loser[i], surface[i], w_svpt[i], w_won[i], l_svpt[i], l_won[i], level[i])
        start = end
    return {"p_srelo": p_match, "p_serve_w": p_serve_w, "p_serve_l": p_serve_l, "model": model}