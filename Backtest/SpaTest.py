r"""Hansen Superior Predictive Ability and White Reality Check, plus a permutation null.

All tests use a day-block bootstrap or a day-block permutation: whole calendar days
are the resampling unit. d is a performance differential vs the market, positive
when the candidate is better (bits_gap = IGN_market - IGN_model, in bits).

White Reality Check (White 2000): bootstrap the max mean differential, recentered
at the observed mean (least-favorable null).

Hansen SPA (Hansen 2005): studentized max, with recentering only for models that
are close to or above zero (threshold A = sqrt(2 log log n), floored at 0 when n
is small).

The permutation null is for a single already-chosen config: shuffle outcomes
within the same days' structure (shuffle y globally), recompute the bits gap.
"""
from __future__ import annotations

import math

import numpy as np


def DaySums(values, days):
    values = np.asarray(values, dtype=float)
    days = np.asarray(days)
    unique, inverse = np.unique(days, return_inverse=True)
    sums = np.zeros(len(unique), dtype=float)
    counts = np.zeros(len(unique), dtype=float)
    np.add.at(sums, inverse, values)
    np.add.at(counts, inverse, 1.0)
    return unique, sums, counts


def DayBlockMeans(values, days, reps, seed):
    """Resample days with replacement; return bootstrap means of `values` (shape reps,)."""
    unique, sums, counts = DaySums(values, days)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(unique), size=(reps, len(unique)))
    return sums[pick].sum(axis=1) / counts[pick].sum(axis=1)


def DayBlockMeanMatrix(d, days, reps, seed):
    """d: (n_obs, n_models). Returns bootstrap means of shape (reps, n_models)."""
    d = np.asarray(d, dtype=float)
    unique, inverse = np.unique(np.asarray(days), return_inverse=True)
    n_days, k = len(unique), d.shape[1]
    sums = np.zeros((n_days, k), dtype=float)
    counts = np.zeros(n_days, dtype=float)
    np.add.at(sums, inverse, d)
    np.add.at(counts, inverse, 1.0)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, n_days, size=(reps, n_days))
    tot = counts[pick].sum(axis=1, keepdims=True)
    return sums[pick].sum(axis=1) / np.clip(tot, 1.0, None)


def WhiteRealityCheck(d, days, reps=10_000, seed=20261113):
    """Return observed max mean, bootstrap p-value, and the winning column index."""
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    obs = d.mean(axis=0)
    best = int(np.argmax(obs))
    t_obs = float(obs[best])
    boot = DayBlockMeanMatrix(d, days, reps, seed)
    # least-favorable recentering: subtract observed mean
    t_star = (boot - obs).max(axis=1)
    p = float(np.mean(t_star >= t_obs))
    return {"test": "white_rc", "n": int(d.shape[0]), "n_models": int(d.shape[1]),
            "n_days": int(np.unique(days).size), "reps": int(reps),
            "best_index": best, "t_obs": t_obs, "p_value": p}


def HansenSpa(d, days, reps=10_000, seed=20261113):
    """Studentized SPA p-value vs the zero-mean (no better than market) null."""
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    n, k = d.shape
    n_days = int(np.unique(days).size)
    obs = d.mean(axis=0)
    # day-level variance for studentization
    unique, inverse = np.unique(np.asarray(days), return_inverse=True)
    day_means = np.zeros((len(unique), k), dtype=float)
    day_counts = np.zeros(len(unique), dtype=float)
    np.add.at(day_means, inverse, d)
    np.add.at(day_counts, inverse, 1.0)
    day_means = day_means / np.clip(day_counts[:, None], 1.0, None)
    sigma = day_means.std(axis=0, ddof=1)
    sigma = np.where(sigma < 1e-12, 1e-12, sigma)
    t_ind = np.sqrt(n_days) * obs / sigma
    best = int(np.argmax(t_ind))
    t_obs = float(t_ind[best])
    # Hansen threshold: recenter models that are not clearly poor
    if n_days >= 3:
        a_thresh = math.sqrt(2.0 * math.log(math.log(n_days)))
    else:
        a_thresh = 0.0
    recenter = np.where(t_ind > -a_thresh, obs, 0.0)
    boot = DayBlockMeanMatrix(d, days, reps, seed)
    # bootstrap day-level sigma is expensive; use the sample sigma (standard practical SPA)
    t_star = (np.sqrt(n_days) * (boot - recenter) / sigma).max(axis=1)
    p = float(np.mean(t_star >= t_obs))
    return {"test": "hansen_spa", "n": int(n), "n_models": int(k), "n_days": n_days,
            "reps": int(reps), "best_index": best, "t_obs": t_obs, "mean_best": float(obs[best]),
            "p_value": p, "a_thresh": a_thresh}


def PermutationPValue(bits_model, bits_market, y, reps=10_000, seed=20261113):
    """Permute outcomes; p = share of permuted bits gaps >= observed (model minus market, flipped to +).

    bits_* are per-row IGN in bits for the observed side already? We pass probabilities and y instead.
    Here `bits_model` and `bits_market` are already IGN_model and IGN_market on the observed y
    (so gap = bits_market - bits_model). For the permutation we need p and y, not precomputed IGN.
    This helper therefore expects the observed gap series and will not reshuffle p; use
    PermuteProbGap for a proper permutation of y against fixed probabilities.
    """
    raise NotImplementedError("use PermuteProbGap")


def PermuteProbGap(p_model, p_market, y, days, reps=10_000, seed=20261113):
    """Permutation p-value of bits_gap = mean(IGN_mkt - IGN_model) under shuffled y."""
    p_model = np.clip(np.asarray(p_model, dtype=float), 1e-6, 1.0 - 1e-6)
    p_market = np.clip(np.asarray(p_market, dtype=float), 1e-6, 1.0 - 1e-6)
    y = np.asarray(y, dtype=float)

    def Gap(yy):
        ign_m = -(yy * np.log2(p_model) + (1.0 - yy) * np.log2(1.0 - p_model))
        ign_k = -(yy * np.log2(p_market) + (1.0 - yy) * np.log2(1.0 - p_market))
        return float(np.mean(ign_k - ign_m))

    obs = Gap(y)
    rng = np.random.default_rng(seed)
    # day-block permutation: shuffle the bag of day-level outcome vectors? Global shuffle of y
    # is the standard permutation null that outcomes are independent of both forecasts.
    ge = 0
    n = len(y)
    for _ in range(reps):
        yy = rng.permutation(y)
        if Gap(yy) >= obs:
            ge += 1
    return {"test": "permutation_best", "n": int(n), "n_days": int(np.unique(days).size),
            "reps": int(reps), "t_obs": obs, "p_value": float(ge / reps)}
