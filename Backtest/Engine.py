r"""Leak-free chronological walk over matches (see Backtest.LoadMatches for the ordering).

For each group (same tourney_date + round stage):
  1. every match in the group is predicted from the frozen pre-group state;
  2. only then are the group's results applied to the models.
The engine never reads a match's result before its prediction is stored.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from Backtest.Elo import EloParams, SurfaceElo
from Backtest.Markov import MatchProbTable
from Backtest.ServeReturn import ServeReturn, ServeReturnParams

_TABLE = None


def SharedTable():
    global _TABLE
    if _TABLE is None:
        _TABLE = MatchProbTable()
    return _TABLE


def Logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def Sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def Blend(p_elo, p_sr, weight_sr):
    return Sigmoid((1 - weight_sr) * Logit(p_elo) + weight_sr * Logit(p_sr))


def _Columns(frame):
    def Col(name, fill=np.nan):
        return frame[name].to_numpy(dtype=float, na_value=fill)
    w_won = Col("w_1stWon") + Col("w_2ndWon")
    l_won = Col("l_1stWon") + Col("l_2ndWon")
    return {
        "group": frame["group_id"].to_numpy(),
        "winner": frame["winner_id"].to_numpy(dtype="int64"),
        "loser": frame["loser_id"].to_numpy(dtype="int64"),
        "surface": frame["surface_norm"].astype(str).to_numpy(),
        "best_of": frame["best_of"].to_numpy(dtype="int64"),
        "level": frame["level_tag"].astype(str).to_numpy(),
        "w_svpt": Col("w_svpt"), "w_won": w_won, "l_svpt": Col("l_svpt"), "l_won": l_won,
    }


def RunModels(frame: pd.DataFrame, elo_params: EloParams | None = None, sr_params: ServeReturnParams | None = None,
              tour="atp", record_state=False, stop_before_group=None):
    """Return a dict of numpy arrays aligned with `frame` rows: p_elo, p_sr (winner's pre-match probability),
    p_serve_w / p_serve_l (SR model), and optionally pre-match Elo ratings. Also returns the final model objects.
    `stop_before_group`: stop (without applying) at this group id - used by PredictDay and the leak test."""
    cols = _Columns(frame)
    n = len(frame)
    elo = SurfaceElo(elo_params) if elo_params else None
    sr = ServeReturn(sr_params, tour) if sr_params else None
    table = SharedTable() if sr else None
    p_elo = np.full(n, np.nan)
    p_sr = np.full(n, np.nan)
    p_serve_w = np.full(n, np.nan)
    p_serve_l = np.full(n, np.nan)
    elo_w = np.full(n, np.nan) if record_state else None
    elo_l = np.full(n, np.nan) if record_state else None
    group, winner, loser, surface, best_of, level = (cols[k] for k in ("group", "winner", "loser", "surface", "best_of", "level"))
    w_svpt, w_won, l_svpt, l_won = (cols[k] for k in ("w_svpt", "w_won", "l_svpt", "l_won"))
    start = 0
    while start < n:
        g = group[start]
        if stop_before_group is not None and g >= stop_before_group:
            break
        end = start
        while end < n and group[end] == g:
            end += 1
        # 1) predict the whole group from the pre-group state
        for i in range(start, end):
            a, b, s, bo = winner[i], loser[i], surface[i], best_of[i]
            if elo:
                p_elo[i] = elo.Predict(a, b, s, bo)
                if record_state:
                    elo_w[i] = elo.Diff(a, b, s, 3)
            if sr:
                pa, pb = sr.PServe(a, b, s)
                p_serve_w[i] = pa
                p_serve_l[i] = pb
                p_sr[i] = table.Prob(pa, pb, bo)
        # 2) then apply the group's results
        for i in range(start, end):
            a, b, s = winner[i], loser[i], surface[i]
            if elo:
                elo.Update(a, b, s, level[i])
            if sr and w_svpt[i] > 0 and l_svpt[i] > 0 and w_won[i] == w_won[i] and l_won[i] == l_won[i]:
                sr.Update(a, b, s, w_svpt[i], w_won[i], l_svpt[i], l_won[i])
        start = end
    out = {"p_elo": p_elo, "p_sr": p_sr, "p_serve_w": p_serve_w, "p_serve_l": p_serve_l, "elo": elo, "sr": sr}
    if record_state:
        out["elo_diff"] = elo_w
    return out