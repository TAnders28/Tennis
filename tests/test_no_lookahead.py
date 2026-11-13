"""Leak tests for the chronological engine.

1. Future invariance: rewriting or shuffling every result after a cutoff group leaves all
   predictions up to and including the cutoff group unchanged.
2. Same-group isolation: flipping a result inside group g changes no prediction in group g.
3. Snapshot check: each prediction equals a fresh model fed ONLY the groups before it.
"""
import unittest

import numpy as np
import pandas as pd

from Backtest.Elo import EloParams, SurfaceElo
from Backtest.Engine import RunModels
from Backtest.ServeReturn import ServeReturnParams


def SyntheticFrame(n_groups=60, per_group=6, n_players=30, seed=3):
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, 1, n_players)
    rows = []
    for g in range(n_groups):
        players = rng.permutation(n_players)[: per_group * 2]
        for k in range(per_group):
            a, b = int(players[2 * k]), int(players[2 * k + 1])
            p_a = 1 / (1 + np.exp(-(skill[a] - skill[b])))
            winner, loser = (a, b) if rng.random() < p_a else (b, a)
            w_svpt, l_svpt = int(rng.integers(50, 90)), int(rng.integers(50, 90))
            rows.append({"group_id": g, "winner_id": winner, "loser_id": loser,
                         "surface_norm": ["Hard", "Clay", "Grass"][g % 3], "best_of": 3 if g % 4 else 5,
                         "level_tag": "tour" if g % 5 else "challenger",
                         "w_svpt": w_svpt, "w_1stWon": int(w_svpt * 0.45), "w_2ndWon": int(w_svpt * 0.2),
                         "l_svpt": l_svpt, "l_1stWon": int(l_svpt * 0.4), "l_2ndWon": int(l_svpt * 0.18)})
    return pd.DataFrame(rows)


ELO = EloParams(k_a=250, k_b=5, k_c=0.4, surface_weight=0.5, bo5_mult=1.1, lower_k_mult=0.8)
SR = ServeReturnParams(shrink_k=100, half_life=30, overall_weight=0.5)


def Run(frame):
    out = RunModels(frame, ELO, SR, "atp")
    return np.column_stack([out["p_elo"], out["p_sr"], out["p_serve_w"], out["p_serve_l"]])


def Flip(frame, mask):
    frame = frame.copy()
    w = frame.loc[mask, "winner_id"].copy()
    frame.loc[mask, "winner_id"] = frame.loc[mask, "loser_id"]
    frame.loc[mask, "loser_id"] = w
    return frame


class NoLookaheadTest(unittest.TestCase):
    def test_future_invariance(self):
        frame = SyntheticFrame()
        base = Run(frame)
        cutoff = 30
        future = frame["group_id"] > cutoff
        flipped = Run(Flip(frame, future))
        past = (frame["group_id"] <= cutoff).to_numpy()
        # orientation of rows flips for future rows only; past rows must be bit-identical
        np.testing.assert_array_equal(base[past], flipped[past])
        # and shuffling the future rows' order (within their groups) changes nothing in the past either
        shuffled = pd.concat([frame[~future], frame[future].sample(frac=1, random_state=1).sort_values("group_id", kind="stable")])
        np.testing.assert_array_equal(base[past], Run(shuffled.reset_index(drop=True))[past])

    def test_same_group_isolation(self):
        frame = SyntheticFrame()
        base = Run(frame)
        g = 25
        in_group = (frame["group_id"] == g).to_numpy()
        first = np.flatnonzero(in_group)[0]
        mask = pd.Series(False, index=frame.index)
        mask.iloc[first] = True
        changed = Run(Flip(frame, mask))
        others = in_group.copy()
        others[first] = False
        np.testing.assert_array_equal(base[others], changed[others])
        later = (frame["group_id"] > g).to_numpy()
        self.assertFalse(np.allclose(base[later, 0], changed[later, 0]), "a result should affect later groups")

    def test_snapshot_excludes_scored_match(self):
        frame = SyntheticFrame(n_groups=20)
        out = RunModels(frame, ELO, None, "atp")
        for i in range(0, len(frame), 7):
            g = frame["group_id"].iat[i]
            elo = SurfaceElo(ELO)
            for row in frame[frame["group_id"] < g].itertuples():
                elo.Update(row.winner_id, row.loser_id, row.surface_norm, row.level_tag)
            expected = elo.Predict(frame["winner_id"].iat[i], frame["loser_id"].iat[i], frame["surface_norm"].iat[i],
                                   frame["best_of"].iat[i])
            self.assertAlmostEqual(out["p_elo"][i], expected, places=12)

    def test_stop_before_group(self):
        frame = SyntheticFrame(n_groups=20)
        out = RunModels(frame, ELO, None, "atp", stop_before_group=10)
        self.assertTrue(np.isnan(out["p_elo"][(frame["group_id"] >= 10).to_numpy()]).all())


if __name__ == "__main__":
    unittest.main()