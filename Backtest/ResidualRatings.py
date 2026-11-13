r"""Surface Elo plus time decay, Glicko-style RD, and a fatigue/schedule logit tweak.

Used by the residual-on-market search. The walk is leak-free: each (tourney_date, stage)
group is predicted from the pre-group snapshot, then results are applied.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

SURFACES = ("Hard", "Clay", "Grass")
Q = math.log(10.0) / 400.0
G_SCALE = 3.0 * Q * Q / (math.pi * math.pi)


def Expected(diff):
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


def GlickoG(rd):
    return 1.0 / math.sqrt(1.0 + G_SCALE * rd * rd)


def Logit(p):
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def Sigmoid(x):
    if x > 30:
        return 1.0
    if x < -30:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


@dataclass(frozen=True)
class ResidualParams:
    k_a: float = 220.0
    k_b: float = 5.0
    k_c: float = 0.4
    surface_weight: float = 0.5
    bo5_mult: float = 1.15
    lower_k_mult: float = 1.0
    decay_halflife_days: float = 0.0  # 0 = off
    glicko: int = 0  # 0/1
    fatigue_coef: float = 0.0  # logit per extra match in the last 7 days
    sr_weight: float = 0.0
    init: float = 1500.0
    rd_init: float = 350.0
    rd_min: float = 50.0
    rd_tau: float = 8.0

    def Key(self):
        return "res_" + "_".join(f"{k}{v:g}" for k, v in asdict(self).items() if k not in ("init", "rd_init", "rd_min", "rd_tau", "k_b"))


def BuildGrid():
    """Pre-registered 3,888-config grid. Holdout year is not used to build this."""
    configs = []
    for k_a in (150.0, 220.0, 300.0):
        for k_c in (0.30, 0.40, 0.50):
            for surface_weight in (0.25, 0.50, 0.75):
                for bo5_mult in (1.00, 1.15, 1.30):
                    for lower_k_mult in (0.70, 1.00):
                        for decay_halflife_days in (0.0, 365.0, 730.0):
                            for glicko in (0, 1):
                                for fatigue_coef in (0.0, 0.04):
                                    for sr_weight in (0.0, 0.50):
                                        configs.append(ResidualParams(
                                            k_a=k_a, k_c=k_c, surface_weight=surface_weight,
                                            bo5_mult=bo5_mult, lower_k_mult=lower_k_mult,
                                            decay_halflife_days=decay_halflife_days, glicko=glicko,
                                            fatigue_coef=fatigue_coef, sr_weight=sr_weight))
    return configs


class ResidualElo:
    def __init__(self, params: ResidualParams):
        self.params = params
        self.rating = {}
        self.count = {}
        self.surface_rating = {}
        self.surface_count = {}
        self.rd = {}
        self.surface_rd = {}
        self.last_day = {}
        self.surface_last = {}
        self.recent = {}  # pid -> list of match days (pruned to 8 days)

    def _Age(self, rating, rd, last, day):
        p = self.params
        if last is None:
            return rating, (p.rd_init if p.glicko else rd)
        delta = max(int(day) - int(last), 0)
        if delta and p.decay_halflife_days > 0:
            rating = p.init + (rating - p.init) * (0.5 ** (delta / p.decay_halflife_days))
        if p.glicko:
            rd = min(p.rd_init, math.sqrt(rd * rd + p.rd_tau * p.rd_tau * delta))
        return rating, rd

    def _Read(self, pid, surface, day):
        p = self.params
        overall, rd_o = self._Age(self.rating.get(pid, p.init), self.rd.get(pid, p.rd_init), self.last_day.get(pid), day)
        if surface in SURFACES:
            sr, rd_s = self._Age(self.surface_rating.get((pid, surface), p.init),
                                 self.surface_rd.get((pid, surface), p.rd_init),
                                 self.surface_last.get((pid, surface)), day)
        else:
            sr, rd_s = overall, rd_o
        return overall, sr, rd_o, rd_s

    def _Fatigue(self, pid, day):
        xs = self.recent.get(pid)
        if not xs:
            return 0, 14
        cutoff = int(day) - 7
        n = 0
        for d in reversed(xs):
            if d < cutoff:
                break
            if d <= int(day):
                n += 1
        rest = min(int(day) - xs[-1], 14) if xs[-1] <= int(day) else 14
        return n, rest

    def Predict(self, a, b, surface, best_of, day):
        p = self.params
        ra_o, ra_s, rd_ao, rd_as = self._Read(a, surface, day)
        rb_o, rb_s, rd_bo, rd_bs = self._Read(b, surface, day)
        w = p.surface_weight if surface in SURFACES else 0.0
        diff = (1.0 - w) * (ra_o - rb_o) + w * (ra_s - rb_s)
        if best_of == 5:
            diff *= p.bo5_mult
        if p.glicko:
            rd_opp = (1.0 - w) * rd_bo + w * rd_bs
            diff *= GlickoG(rd_opp)
        prob = Expected(diff)
        if p.fatigue_coef:
            fa, ra = self._Fatigue(a, day)
            fb, rb = self._Fatigue(b, day)
            # more matches in the last 7 days, or less rest, hurts that player
            adj = p.fatigue_coef * ((fb - fa) + 0.5 * math.tanh((ra - rb) / 4.0))
            prob = Sigmoid(Logit(prob) + adj)
        return prob

    def _K(self, n, level_tag, rd):
        p = self.params
        k = p.k_a / (n + p.k_b) ** p.k_c
        if level_tag != "tour":
            k *= p.lower_k_mult
        if p.glicko:
            k *= min(max(rd / 150.0, 0.5), 2.0)
        return k

    def _Write(self, pid, surface, day, overall, sr, rd_o, rd_s, n_o, n_s):
        p = self.params
        self.rating[pid] = overall
        self.count[pid] = n_o
        self.last_day[pid] = int(day)
        if p.glicko:
            self.rd[pid] = max(p.rd_min, rd_o * 0.95)
        if surface in SURFACES:
            self.surface_rating[(pid, surface)] = sr
            self.surface_count[(pid, surface)] = n_s
            self.surface_last[(pid, surface)] = int(day)
            if p.glicko:
                self.surface_rd[(pid, surface)] = max(p.rd_min, rd_s * 0.95)
        rec = self.recent.setdefault(pid, [])
        rec.append(int(day))
        cutoff = int(day) - 8
        if rec[0] < cutoff:
            self.recent[pid] = [d for d in rec if d >= cutoff]

    def Update(self, winner, loser, surface, level_tag, day):
        p = self.params
        rw_o, rw_s, rd_wo, rd_ws = self._Read(winner, surface, day)
        rl_o, rl_s, rd_lo, rd_ls = self._Read(loser, surface, day)
        nw, nl = self.count.get(winner, 0), self.count.get(loser, 0)
        surprise_o = 1.0 - Expected(rw_o - rl_o)
        rw_o = rw_o + self._K(nw, level_tag, rd_wo) * surprise_o
        rl_o = rl_o - self._K(nl, level_tag, rd_lo) * surprise_o
        if surface in SURFACES:
            nsw = self.surface_count.get((winner, surface), 0)
            nsl = self.surface_count.get((loser, surface), 0)
            surprise_s = 1.0 - Expected(rw_s - rl_s)
            rw_s = rw_s + self._K(nsw, level_tag, rd_ws) * surprise_s
            rl_s = rl_s - self._K(nsl, level_tag, rd_ls) * surprise_s
            self._Write(winner, surface, day, rw_o, rw_s, rd_wo, rd_ws, nw + 1, nsw + 1)
            self._Write(loser, surface, day, rl_o, rl_s, rd_lo, rd_ls, nl + 1, nsl + 1)
        else:
            self._Write(winner, surface, day, rw_o, rw_o, rd_wo, rd_wo, nw + 1, nw + 1)
            self._Write(loser, surface, day, rl_o, rl_o, rd_lo, rd_lo, nl + 1, nl + 1)


def Walk(group, winner, loser, surface, best_of, level, days, params: ResidualParams,
         predict_index=None, p_sr=None):
    """Leak-free walk. Returns p_winner (optionally only filled at predict_index)."""
    n = len(winner)
    p_out = np.full(n, np.nan)
    want = None if predict_index is None else set(int(i) for i in predict_index)
    elo = ResidualElo(params)
    w_sr = params.sr_weight
    start = 0
    while start < n:
        g = group[start]
        end = start
        while end < n and group[end] == g:
            end += 1
        for i in range(start, end):
            if want is not None and i not in want:
                continue
            p = elo.Predict(winner[i], loser[i], surface[i], int(best_of[i]), int(days[i]))
            if w_sr and p_sr is not None and np.isfinite(p_sr[i]):
                p = Sigmoid((1.0 - w_sr) * Logit(p) + w_sr * Logit(float(p_sr[i])))
            p_out[i] = p
        for i in range(start, end):
            elo.Update(winner[i], loser[i], surface[i], level[i], int(days[i]))
        start = end
    return p_out
