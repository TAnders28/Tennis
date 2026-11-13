"""Leak tests for the Elo-style serve/return prototype (synthetic data only)."""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from Backtest.ServeReturnElo import RunServeReturnElo, ServeReturnElo, ServeReturnEloParams
from tests.test_no_lookahead import Flip, SyntheticFrame


def Run(frame):
    out = RunServeReturnElo(frame, ServeReturnEloParams(), "atp")
    return np.column_stack([out["p_srelo"], out["p_serve_w"], out["p_serve_l"]])


class ServeReturnEloLeakTest(unittest.TestCase):
    def test_future_invariance(self):
        frame = SyntheticFrame()
        base = Run(frame)
        cutoff = 30
        future = frame["group_id"] > cutoff
        flipped = Run(Flip(frame, future))
        past = (frame["group_id"] <= cutoff).to_numpy()
        np.testing.assert_array_equal(base[past], flipped[past])

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

    def test_count_increments_once_per_match(self):
        model = ServeReturnElo(ServeReturnEloParams(), "atp")
        model.Update(1, 2, "Hard", 80, 64, 70, 40, "tour")
        self.assertEqual(model.count[1], 1)
        self.assertEqual(model.count[2], 1)
        self.assertEqual(model.count_s[(1, "Hard")], 1)
        # ratings moved in the expected direction: winner served above tour mu, so serve rating rises
        self.assertGreater(model.serve[1], 0.0)


if __name__ == "__main__":
    unittest.main()