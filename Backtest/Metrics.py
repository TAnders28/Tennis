"""Scoring helpers: log loss, Brier, accuracy, reliability tables, match-level bootstrap."""
from __future__ import annotations

import numpy as np

EPS = 1e-6


def Score(p_winner):
    """Metrics from the winner's pre-match probability (orientation-free)."""
    p = np.clip(np.asarray(p_winner, dtype=float), EPS, 1 - EPS)
    p = p[~np.isnan(p)]
    if len(p) == 0:
        return {"n": 0, "accuracy": np.nan, "log_loss": np.nan, "brier": np.nan}
    accuracy = np.mean(np.where(p > 0.5, 1.0, np.where(p == 0.5, 0.5, 0.0)))
    return {"n": int(len(p)), "accuracy": float(accuracy), "log_loss": float(-np.mean(np.log(p))),
            "brier": float(np.mean((1 - p) ** 2))}


def Orient(p_winner, keys):
    """Deterministically flip half the rows so calibration is not all 'winner side'. Returns (p_side, y_side)."""
    p = np.asarray(p_winner, dtype=float)
    flip = np.array([hash_ % 2 == 1 for hash_ in (Fnv(str(k)) for k in keys)])
    p_side = np.where(flip, 1 - p, p)
    y_side = np.where(flip, 0.0, 1.0)
    return p_side, y_side


def Fnv(text):
    h = 2166136261
    for ch in text.encode("utf-8"):
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


def Reliability(p, y, bins=10):
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        rows.append({"bin_lo": round(lo, 2), "bin_hi": round(hi, 2), "n": int(mask.sum()),
                     "mean_pred": float(p[mask].mean()), "observed": float(y[mask].mean())})
    return rows


def Ece(p, y, bins=10):
    rows = Reliability(p, y, bins)
    total = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["mean_pred"] - r["observed"]) for r in rows) / total) if total else float("nan")


def ClusterBootstrap(values, clusters, stat=np.mean, reps=1000, seed=7, percentiles=(2.5, 97.5)):
    """Match-level (cluster) bootstrap CI: resample whole matches, not individual points.

    `percentiles` defaults to (2.5, 97.5) — a 95% interval — so existing callers stay valid.
    Pass (5, 95) for a 90% interval.
    """
    values = np.asarray(values, dtype=float)
    clusters = np.asarray(clusters)
    unique, inverse = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inverse, weights=values)
    counts = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(reps):
        pick = rng.integers(0, len(unique), len(unique))
        out.append(sums[pick].sum() / counts[pick].sum())
    return tuple(float(np.percentile(out, p)) for p in percentiles)


def NaiveCi(values):
    values = np.asarray(values, dtype=float)
    se = values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean() - 1.96 * se), float(values.mean() + 1.96 * se)


def LogLossBits(p, y):
    """Per-row Ignorance in bits: -log2 of the probability on the observed side."""
    p = np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log2(p) + (1.0 - y) * np.log2(1.0 - p))


def MeanCi(values, reps=2000, seed=7, ci=90):
    """IID bootstrap CI for the mean. `ci` is the central interval (90 -> 5th/95th)."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    lo_p = (100.0 - ci) / 2.0
    hi_p = 100.0 - lo_p
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, len(values), size=(reps, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, lo_p)), float(np.percentile(means, hi_p))