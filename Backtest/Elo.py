r"""Surface-aware Elo (overall + Hard/Clay/Grass ratings, blended).

K-factor decays with matches played, FiveThirtyEight style: K = a / (n + b) ** c,
using each player's own count (overall count for the overall rating, surface
count for the surface rating). Lower levels (challenger / ITF / qualifying) can
use a smaller K multiplier. Best-of-5 matches stretch the rating gap by bo5_mult
(a longer match favours the better player).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

SURFACES = ("Hard", "Clay", "Grass")


@dataclass(frozen=True)
class EloParams:
    k_a: float = 250.0
    k_b: float = 5.0
    k_c: float = 0.4
    surface_weight: float = 0.5
    bo5_mult: float = 1.0
    lower_k_mult: float = 1.0
    init: float = 1500.0

    def Key(self):
        return "elo_" + "_".join(f"{k}{v:g}" for k, v in asdict(self).items() if k != "init")


def Expected(diff):
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))


class SurfaceElo:
    def __init__(self, params: EloParams):
        self.params = params
        self.rating = {}       # pid -> overall rating
        self.count = {}        # pid -> overall matches
        self.surface_rating = {}  # (pid, surface) -> rating
        self.surface_count = {}

    def Ratings(self, pid, surface):
        init = self.params.init
        overall = self.rating.get(pid, init)
        if surface in SURFACES:
            return overall, self.surface_rating.get((pid, surface), init)
        return overall, overall

    def Diff(self, a, b, surface, best_of=3):
        ra_o, ra_s = self.Ratings(a, surface)
        rb_o, rb_s = self.Ratings(b, surface)
        w = self.params.surface_weight if surface in SURFACES else 0.0
        diff = (1.0 - w) * (ra_o - rb_o) + w * (ra_s - rb_s)
        if best_of == 5:
            diff *= self.params.bo5_mult
        return diff

    def Predict(self, a, b, surface, best_of=3):
        return Expected(self.Diff(a, b, surface, best_of))

    def _K(self, n, level_tag):
        p = self.params
        k = p.k_a / (n + p.k_b) ** p.k_c
        if level_tag != "tour":
            k *= p.lower_k_mult
        return k

    def Update(self, winner, loser, surface, level_tag="tour"):
        init = self.params.init
        rw = self.rating.get(winner, init)
        rl = self.rating.get(loser, init)
        nw = self.count.get(winner, 0)
        nl = self.count.get(loser, 0)
        surprise = 1.0 - Expected(rw - rl)
        self.rating[winner] = rw + self._K(nw, level_tag) * surprise
        self.rating[loser] = rl - self._K(nl, level_tag) * surprise
        self.count[winner] = nw + 1
        self.count[loser] = nl + 1
        if surface in SURFACES:
            kw, kl = (winner, surface), (loser, surface)
            rw = self.surface_rating.get(kw, init)
            rl = self.surface_rating.get(kl, init)
            nw = self.surface_count.get(kw, 0)
            nl = self.surface_count.get(kl, 0)
            surprise = 1.0 - Expected(rw - rl)
            self.surface_rating[kw] = rw + self._K(nw, level_tag) * surprise
            self.surface_rating[kl] = rl - self._K(nl, level_tag) * surprise
            self.surface_count[kw] = nw + 1
            self.surface_count[kl] = nl + 1