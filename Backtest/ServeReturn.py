r"""Serve/return point model: per-player serve-points-won (SPW) and return-points-won
(RPW) estimates, surface-aware, opponent-adjusted and shrunk toward the tour mean.

For a matchup A vs B on surface s (Barnett-Clarke style, in deviations):
    p_A_serve = mu_s + spw_dev(A, s) - rpw_dev(B, s)
    p_B_serve = mu_s + spw_dev(B, s) - rpw_dev(A, s)
where mu_s is the running tour-wide serve-point win rate on that surface.

Updates (after the match, leak-free) use opponent-adjusted residuals:
    serve residual of A = A serve points won - A serve points * (mu_s - rpw_dev(B, s) pre-match)
Each player's history decays geometrically per match (half-life in matches).
Estimate = (surface residual + overall_weight * all-surface residual) /
           (surface points + overall_weight * all points + shrink_k).
Matches without point stats (most ITF rows) do not update this model.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

SURFACES = ("Hard", "Clay", "Grass")
DEFAULT_MU = {"atp": 0.635, "wta": 0.565}


@dataclass(frozen=True)
class ServeReturnParams:
    shrink_k: float = 800.0
    half_life: float = 150.0
    overall_weight: float = 0.5
    mu_window: float = 20000.0  # points-weighted memory of the tour mean (in matches)

    def Key(self):
        return "sr_" + "_".join(f"{k}{v:g}" for k, v in asdict(self).items())


class ServeReturn:
    def __init__(self, params: ServeReturnParams, tour="atp"):
        self.params = params
        self.decay = 0.5 ** (1.0 / params.half_life) if params.half_life < 1e8 else 1.0
        self.mu_decay = 1.0 - 1.0 / params.mu_window
        default = DEFAULT_MU.get(tour, 0.6)
        self.mu = {s: [default * 5000.0, 5000.0] for s in SURFACES + ("Unknown",)}
        # pid -> {bucket: [serve_resid, serve_pts, return_resid, return_pts]} ; bucket in SURFACES + ("all",)
        self.stats = {}

    def Mu(self, surface):
        won, played = self.mu.get(surface, self.mu["Unknown"])
        return won / played

    def Dev(self, pid, surface):
        player = self.stats.get(pid)
        if player is None:
            return 0.0, 0.0
        p = self.params
        overall = player.get("all", (0.0, 0.0, 0.0, 0.0))
        local = player.get(surface, (0.0, 0.0, 0.0, 0.0)) if surface in SURFACES else (0.0, 0.0, 0.0, 0.0)
        w = p.overall_weight
        spw = (local[0] + w * overall[0]) / (local[1] + w * overall[1] + p.shrink_k)
        rpw = (local[2] + w * overall[2]) / (local[3] + w * overall[3] + p.shrink_k)
        return spw, rpw

    def PServe(self, a, b, surface):
        mu = self.Mu(surface)
        a_spw, a_rpw = self.Dev(a, surface)
        b_spw, b_rpw = self.Dev(b, surface)
        p_a = min(max(mu + a_spw - b_rpw, 0.30), 0.95)
        p_b = min(max(mu + b_spw - a_rpw, 0.30), 0.95)
        return p_a, p_b

    def _Add(self, pid, surface, serve_resid, serve_pts, return_resid, return_pts):
        player = self.stats.setdefault(pid, {})
        buckets = ("all", surface) if surface in SURFACES else ("all",)
        d = self.decay
        for bucket in player:
            v = player[bucket]
            player[bucket] = [v[0] * d, v[1] * d, v[2] * d, v[3] * d]
        for bucket in buckets:
            v = player.setdefault(bucket, [0.0, 0.0, 0.0, 0.0])
            v[0] += serve_resid
            v[1] += serve_pts
            v[2] += return_resid
            v[3] += return_pts

    def Update(self, winner, loser, surface, w_svpt, w_won, l_svpt, l_won):
        """w_won / l_won = serve points won by winner / loser."""
        values = (w_svpt, w_won, l_svpt, l_won)
        if not all(math.isfinite(v) for v in values) or not (w_svpt > 0 and l_svpt > 0) or w_won > w_svpt or l_won > l_svpt:
            return
        mu = self.Mu(surface)
        w_spw, w_rpw = self.Dev(winner, surface)
        l_spw, l_rpw = self.Dev(loser, surface)
        # serve residuals vs. what the opponent's return level predicts
        w_serve_resid = w_won - w_svpt * (mu - l_rpw)
        l_serve_resid = l_won - l_svpt * (mu - w_rpw)
        # return residuals vs. what the opponent's serve level predicts
        w_return_resid = (l_svpt - l_won) - l_svpt * (1.0 - mu - l_spw)
        l_return_resid = (w_svpt - w_won) - w_svpt * (1.0 - mu - w_spw)
        self._Add(winner, surface, w_serve_resid, w_svpt, w_return_resid, l_svpt)
        self._Add(loser, surface, l_serve_resid, l_svpt, l_return_resid, w_svpt)
        bucket = self.mu.get(surface, self.mu["Unknown"])
        bucket[0] = bucket[0] * self.mu_decay + w_won + l_won
        bucket[1] = bucket[1] * self.mu_decay + w_svpt + l_svpt