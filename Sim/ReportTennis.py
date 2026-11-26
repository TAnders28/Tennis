r"""Score the tennis simulator output (Sim.ValidateTennis). Research, tested:false.

Pre-declared pass rules for 'usable as a variance / stress-test engine' (per split, holdout decides):
  V1 total-games interval coverage at 50/80/90% within +-5 pts of nominal;
  V2 pre-match win-probability ECE (10 bins) <= 0.03;
  V3 straight-sets rate and any-tiebreak rate within +-3 pts of reality (best-of-3 and best-of-5 separately);
  V4 upset rate (sim favourite loses) within +-2 pts of the sim's own expectation.
'Usable for pricing' additionally needs Brier <= Kalshi T-60 on the same matches (not expected).
Near-miss ('same match replayed from its own serve stats') is usable for variance reduction only if mean(y - p_nm)
has a CI covering 0 and the slope of (y - p_nm) on (p_kalshi - 0.5) has a CI covering 0, on the holdout.

Usage: .venv\Scripts\python.exe -m Sim.ReportTennis --matches <csv> --out <json>
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd

NM = ["p_nm_k0", "p_nm_k30", "p_nm_k100"]


def Ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum((b == k).mean() * abs(p[b == k].mean() - y[b == k].mean()) for k in range(bins) if (b == k).any()))


def Ci(x, n=1000, seed=3, groups=None):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, float)
    if groups is None:
        m = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)])
    else:
        g = pd.Series(x).groupby(np.asarray(groups))
        s, c = g.sum().to_numpy(), g.count().to_numpy()
        m = []
        for _ in range(n):
            i = rng.integers(0, len(s), len(s))
            m.append(s[i].sum() / c[i].sum())
        m = np.array(m)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def Slope(x, r):
    x, r = np.asarray(x, float), np.asarray(r, float)
    xc = x - x.mean()
    return float((xc * (r - r.mean())).sum() / (xc * xc).sum())


def SlopeCi(x, r, n=1000, seed=5):
    rng = np.random.default_rng(seed)
    x, r = np.asarray(x, float), np.asarray(r, float)
    v = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        v.append(Slope(x[i], r[i]))
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def KalshiFee(p):
    return np.ceil(0.07 * 100 * p * (1 - p) - 1e-9) / 100.0


def Block(d, sfx=""):
    d = d.copy()
    if sfx:
        for c in ("p_sim", "sim_p_a_cell", "sim_p_straight", "sim_p_tb", "sim_total_mean", "pit_total"):
            d[c] = d[c + sfx]
    y, p = d["y"].to_numpy(), d["p_sim"].to_numpy()
    out = {"n": int(len(d)), "brier_sim": float(np.mean((p - y) ** 2)), "ece_sim": Ece(p, y),
           "brier_blend": float(np.mean((d["p_blend_a"] - y) ** 2)), "mc_vs_exact_mad": float(np.mean(np.abs(d["sim_p_a_cell"] - p)))}
    pit = d["pit_total"].to_numpy()
    for c in (0.5, 0.8, 0.9):
        out[f"cover_total_{int(c*100)}"] = float((np.abs(pit - 0.5) <= c / 2).mean())
    out["real_total_mean"] = float(d["real_total_games"].mean())
    out["sim_total_mean"] = float(d["sim_total_mean"].mean())
    for bo in (3, 5):
        s = d[d["best_of"] == bo]
        if len(s):
            out[f"bo{bo}_n"] = int(len(s))
            out[f"bo{bo}_real_straight"] = float(s["real_straight"].mean())
            out[f"bo{bo}_sim_straight"] = float(s["sim_p_straight"].mean())
            out[f"bo{bo}_real_tb"] = float(s["real_tb"].mean())
            out[f"bo{bo}_sim_tb"] = float(s["sim_p_tb"].mean())
    fav_a = p >= 0.5
    out["real_upset"] = float(np.where(fav_a, y < 0.5, y > 0.5).mean())
    out["sim_expected_upset"] = float(np.mean(np.minimum(p, 1 - p)))
    k = d[d["p_kalshi_a"].notna()]
    if len(k) > 50:
        yk, pk, ps = k["y"].to_numpy(), k["p_kalshi_a"].to_numpy(), k["p_sim"].to_numpy()
        diff = (ps - yk) ** 2 - (pk - yk) ** 2
        out.update({"n_kalshi": int(len(k)), "brier_kalshi": float(np.mean((pk - yk) ** 2)), "brier_sim_same": float(np.mean((ps - yk) ** 2)),
                    "brier_sim_minus_kalshi": float(diff.mean()), "brier_sim_minus_kalshi_ci": Ci(diff, groups=k["date"].to_numpy())})
    nm = {}
    for c in NM:
        s = d[d[c].notna()]
        r = s["y"].to_numpy() - s[c].to_numpy()
        e = {"n": int(len(s)), "mean_y_minus_p": float(r.mean()), "ci": Ci(r),
             "slope_on_pre": Slope(s["p_sim"].to_numpy() - 0.5, r), "slope_on_pre_ci": SlopeCi(s["p_sim"].to_numpy() - 0.5, r),
             "flip_rate": float(np.mean(np.abs(s[c].to_numpy() - s["y"].to_numpy()) > 0.5)),
             "var_y_minus_pre": float(np.var(s["y"] - s["p_sim"])), "var_nm_minus_pre": float(np.var(s[c] - s["p_sim"]))}
        sk = s[s["p_kalshi_a"].notna()]
        if len(sk) > 50:
            rk = sk["y"].to_numpy() - sk[c].to_numpy()
            x = sk["p_kalshi_a"].to_numpy() - 0.5
            e.update({"n_kalshi": int(len(sk)), "slope_on_kalshi": Slope(x, rk), "slope_on_kalshi_ci": SlopeCi(x, rk),
                      "mean_y_minus_p_kalshi_rows": float(rk.mean()), "ci_kalshi_rows": Ci(rk),
                      "var_y_minus_kalshi": float(np.var(sk["y"] - sk["p_kalshi_a"])), "var_nm_minus_kalshi": float(np.var(sk[c] - sk["p_kalshi_a"]))})
            e["n_multiplier"] = e["var_y_minus_kalshi"] / max(e["var_nm_minus_kalshi"], 1e-9)
        nm[c] = e
    out["near_miss"] = nm
    return out


def Fav90(d, sfx=""):
    """Market favourite at >= 90c T-60 mid: outcomes vs mid vs simulator vs near-miss, taker P&L at the ask."""
    k = d[d["p_kalshi_a"].notna()].copy()
    fav_a = k["p_kalshi_a"] >= 0.5
    k["fav_mid"] = np.where(fav_a, k["p_kalshi_a"], 1 - k["p_kalshi_a"])
    k = k[k["fav_mid"] >= 0.90]
    fa = (k["p_kalshi_a"] >= 0.5).to_numpy()
    yf = np.where(fa, k["y"], 1 - k["y"])
    ask = np.where(fa, k["k_ask_a"], k["k_ask_b"])
    ps = np.where(fa, k["p_sim" + sfx], 1 - k["p_sim" + sfx])
    out = {"n": int(len(k)), "fav_win": float(yf.mean()), "mean_mid": float(k["fav_mid"].mean()), "mean_ask": float(np.nanmean(ask)),
           "sim_p_fav": float(ps.mean()), "sim_says_fav_underpriced_share": float((ps > k["fav_mid"].to_numpy()).mean())}
    fee = KalshiFee(ask)
    pnl = yf - ask - fee
    out["pnl_raw_mean"] = float(np.nanmean(pnl))
    out["pnl_raw_sd"] = float(np.nanstd(pnl))
    out["pnl_raw_ci"] = Ci(pnl[np.isfinite(pnl)], groups=k["date"].to_numpy()[np.isfinite(pnl)])
    for c in NM:
        pn = np.where(fa, k[c], 1 - k[c])
        m = np.isfinite(pn) & np.isfinite(pnl)
        la = pn[m] - ask[m] - fee[m]
        out[f"{c}_n"] = int(m.sum())
        out[f"{c}_pnl_mean"] = float(la.mean())
        out[f"{c}_pnl_sd"] = float(la.std())
        out[f"{c}_pnl_raw_same_rows"] = float(pnl[m].mean())
    by = {}
    for lvl, g in k.groupby("kalshi_level"):
        fa2 = (g["p_kalshi_a"] >= 0.5).to_numpy()
        by[str(lvl)] = {"n": int(len(g)), "fav_win": float(np.where(fa2, g["y"], 1 - g["y"]).mean()), "mean_mid": float(np.where(fa2, g["p_kalshi_a"], 1 - g["p_kalshi_a"]).mean()),
                        "sim_p_fav": float(np.where(fa2, g["p_sim" + sfx], 1 - g["p_sim" + sfx]).mean())}
    out["by_level"] = by
    return out


def Stress(d, seed=9):
    """Tilted matches: how far does P(A wins) move if A's true serve rate is 1 or 2 points off our estimate, and how
    big is our real serve-rate error (observed minus pre-match, net of binomial noise)."""
    from Sim.MatchSim import ExactMatchProb
    h = d[(d["split"] != "tune")].sample(min(20000, len(d)), random_state=seed)
    pa, pb, bo, ft = h["p_a_serve"].to_numpy(), h["p_b_serve"].to_numpy(), h["best_of"].to_numpy(), h["final_tb"].to_numpy()
    base = ExactMatchProb(pa, pb, bo, ft)
    out = {"n": int(len(h))}
    for t in (0.01, 0.02):
        up = ExactMatchProb(np.clip(pa + t, 0.05, 0.98), pb, bo, ft)
        out[f"mean_abs_dp_serve_plus_{int(t*100)}pt"] = float(np.mean(np.abs(up - base)))
        out[f"flip_fav_share_{int(t*100)}pt"] = float(np.mean((up >= 0.5) != (base >= 0.5)))
    m = h[(h["obs_a_pts"] > 20) & h["obs_a_won"].notna() & (h["obs_a_won"] <= h["obs_a_pts"])]
    r = (m["obs_a_won"] / m["obs_a_pts"]).to_numpy()
    e = r - m["p_a_serve"].to_numpy()
    binom = np.mean(r * (1 - r) / m["obs_a_pts"].to_numpy())
    out["serve_err_sd_raw"] = float(e.std())
    out["serve_err_sd_net_of_binomial"] = float(np.sqrt(max(e.var() - binom, 0.0)))
    out["serve_err_mean"] = float(e.mean())
    return out


def Verdict(h):
    v1 = all(abs(h[f"cover_total_{c}"] - c / 100) <= 0.05 for c in (50, 80, 90))
    v2 = h["ece_sim"] <= 0.03
    v3 = all(abs(h[f"bo{b}_real_{m}"] - h[f"bo{b}_sim_{m}"]) <= 0.03 for b in (3, 5) if f"bo{b}_n" in h for m in ("straight", "tb"))
    v4 = abs(h["real_upset"] - h["sim_expected_upset"]) <= 0.02
    return {"V1_total_coverage": v1, "V2_calibration": v2, "V3_sets_tiebreaks": v3, "V4_upsets": v4,
            "usable_variance_engine": bool(v1 and v2 and v3 and v4), "usable_for_pricing": bool(h.get("brier_sim_minus_kalshi", 1) <= 0)}


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = pd.read_csv(a.matches, low_memory=False)
    res = {}
    res["sigma_form"] = float(d["sigma_form"].iloc[0])
    res["lam_form"] = float(d["lam_form"].iloc[0]) if "lam_form" in d else 1.0
    res["stress"] = Stress(d)
    for sfx, name in (("", "iid"), ("_form", "form")):
        for split in ("tune", "holdout", "forward"):
            res[f"{name}_{split}"] = Block(d[d["split"] == split], sfx)
            res[f"{name}_verdict_{split}"] = Verdict(res[f"{name}_{split}"])
        ho = d[d["split"] != "tune"]
        res[f"{name}_holdout_by_level"] = {f"{t}_{l}": Block(g, sfx) for (t, l), g in ho.groupby(["tour", "level_tag"])}
        res[f"{name}_fav90_kalshi_matched"] = Fav90(d, sfx)
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)
    keys = ["sigma_form", "lam_form", "stress", "form_holdout", "form_forward", "iid_verdict_holdout", "form_verdict_tune", "form_verdict_holdout", "form_verdict_forward", "form_fav90_kalshi_matched"]
    print(json.dumps({k: res[k] for k in keys}, indent=1))


if __name__ == "__main__":
    Main()
