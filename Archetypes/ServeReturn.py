"""Tennis serve-style x return-style archetypes and leak-free matchup tests.

<py> -m Archetypes.ServeReturn --tour atp --out E:\\AgentDocs\\Sportsbot\\2026-11-27_matchup-archetypes [--boot 200] [--k 0]
Data on disk: data/raw/sackmann/<tour>/<tour>_matches_<Y>.csv (+ qual/chall or qual/itf files), match-level serve stats.
Profiles per player-season (min 300 serve and 300 return points): serve = ace%, DF%, 1st-in%, 1st-won%, 2nd-won%,
BP-faced per service game; return = opponent ace% conceded, return-won% vs 1st and vs 2nd serve, BP-earned per return
game; plus clay/grass share of points (surface diet).
Leak-free: archetypes fit on 2017-2019 profiles; a match in season T uses season T-1 profiles; interactions are fit on
2019-2023 matches with BH-FDR; 2024-2025 is the holdout; 2026 a forward look.
Model per outcome (one row per server-in-match, weighted by serve points): ridge with server-season + returner-season
effects, surface, level, server-hand x returner-hand, and server-archetype x returner-archetype cells (double-centred).
Game-cluster bootstrap = resample matches.
"""
import argparse, glob, json, os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

RAW = os.path.join("data", "raw", "sackmann")
YEARS = list(range(2017, 2027))
FIT_ARCH, FIT, HOLD, FWD = [2017, 2018, 2019], [2019, 2020, 2021, 2022, 2023], [2024, 2025], [2026]
SEED = 3
OUTCOMES = {"spw": 0.025, "ace_rate": 0.02, "first_won": 0.03, "second_won": 0.03}
SERVE_F = ["s_ace", "s_df", "s_1in", "s_1won", "s_2won", "s_bpf_g"]
RET_F = ["r_ace_against", "r_1won", "r_2won", "r_bpe_g"]


def LoadMatches(tour):
    fs = []
    for y in YEARS:
        fs += glob.glob(os.path.join(RAW, tour, f"{tour}_matches_{y}.csv")) + glob.glob(os.path.join(RAW, tour, f"{tour}_matches_qual_*_{y}.csv"))
    cols = ["tourney_id", "surface", "tourney_level", "tourney_date", "match_num", "winner_id", "winner_hand", "loser_id", "loser_hand",
            "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon", "w_SvGms", "w_bpSaved", "w_bpFaced",
            "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon", "l_SvGms", "l_bpSaved", "l_bpFaced"]
    d = pd.concat([pd.read_csv(f, usecols=lambda c: c in cols, low_memory=False) for f in fs], ignore_index=True)
    d = d.dropna(subset=["w_svpt", "l_svpt", "w_1stIn", "l_1stIn"])
    d = d[(d["w_svpt"] >= 20) & (d["l_svpt"] >= 20)]
    d["season"] = d["tourney_date"].astype(int) // 10000
    d["match"] = d["tourney_id"].astype(str) + "_" + d["match_num"].astype(str) + "_" + d["winner_id"].astype(str)
    d = d.drop_duplicates("match")
    rows = []
    for a, b in [("w", "l"), ("l", "w")]:
        r = pd.DataFrame({"season": d["season"], "match": d["match"], "surface": d["surface"].fillna("Hard"), "level": d["tourney_level"].astype(str),
                          "server": d[f"{'winner' if a == 'w' else 'loser'}_id"], "returner": d[f"{'winner' if b == 'w' else 'loser'}_id"],
                          "s_hand": d[f"{'winner' if a == 'w' else 'loser'}_hand"].fillna("U"), "r_hand": d[f"{'winner' if b == 'w' else 'loser'}_hand"].fillna("U"),
                          "svpt": d[f"{a}_svpt"], "ace": d[f"{a}_ace"], "df": d[f"{a}_df"], "in1": d[f"{a}_1stIn"], "won1": d[f"{a}_1stWon"],
                          "won2": d[f"{a}_2ndWon"], "svgms": d[f"{a}_SvGms"], "bpf": d[f"{a}_bpFaced"]})
        rows.append(r)
    s = pd.concat(rows, ignore_index=True)
    s = s[(s["svpt"] > 0) & (s["in1"] > 0) & (s["svpt"] > s["in1"])]
    s["won12"] = s["won1"] + s["won2"]; s["in2"] = s["svpt"] - s["in1"]
    s["spw"] = (s["won1"] + s["won2"]) / s["svpt"]
    s["ace_rate"] = s["ace"] / s["svpt"]
    s["first_won"] = s["won1"] / s["in1"]
    s["second_won"] = s["won2"] / (s["svpt"] - s["in1"])
    return s.replace([np.inf, -np.inf], np.nan).dropna(subset=["spw", "first_won", "second_won"])


def Profiles(s, min_pts=300):
    g = s.groupby(["season", "server"]).agg(svpt=("svpt", "sum"), ace=("ace", "sum"), df=("df", "sum"), in1=("in1", "sum"), won1=("won1", "sum"),
                                            won2=("won2", "sum"), svgms=("svgms", "sum"), bpf=("bpf", "sum"))
    sv = pd.DataFrame({"s_ace": g.ace / g.svpt, "s_df": g.df / g.svpt, "s_1in": g.in1 / g.svpt, "s_1won": g.won1 / g.in1,
                       "s_2won": g.won2 / (g.svpt - g.in1), "s_bpf_g": g.bpf / g.svgms.clip(lower=1), "s_pts": g.svpt})
    r = s.groupby(["season", "returner"]).agg(svpt=("svpt", "sum"), ace=("ace", "sum"), in1=("in1", "sum"), won1=("won1", "sum"),
                                              won2=("won2", "sum"), svgms=("svgms", "sum"), bpf=("bpf", "sum"))
    rt = pd.DataFrame({"r_ace_against": r.ace / r.svpt, "r_1won": 1 - r.won1 / r.in1, "r_2won": 1 - r.won2 / (r.svpt - r.in1),
                       "r_bpe_g": r.bpf / r.svgms.clip(lower=1), "r_pts": r.svpt})
    sv.index.names = rt.index.names = ["season", "player"]
    p = sv.join(rt, how="inner").reset_index()
    return p[(p["s_pts"] >= min_pts) & (p["r_pts"] >= min_pts)].replace([np.inf, -np.inf], np.nan)


def OneHot(codes, n, scale):
    return sp.csr_matrix((np.full(len(codes), scale), (np.arange(len(codes)), codes)), shape=(len(codes), n))


def DoubleCentre(g, w, iters=30):
    g = g.copy()
    for _ in range(iters):
        g -= ((g * w).sum(1) / np.maximum(w.sum(1), 1e-9))[:, None]
        g -= ((g * w).sum(0) / np.maximum(w.sum(0), 1e-9))[None, :]
    return g


LINK = {"mode": "linear", "oc": None}
COUNTS = {"spw": ("won12", "svpt"), "ace_rate": ("ace", "svpt"), "first_won": ("won1", "in1"), "second_won": ("won2", "in2")}


def Fit(d, y, wts, ks, kr, alpha, rows=None):
    dd = d if rows is None else d.iloc[rows]
    yy = y if rows is None else y[rows]
    ww = wts if rows is None else wts[rows]
    bl = []
    for col, sc in [("skey", 1.0), ("rkey", 1.0), ("surface", 10.0), ("level", 10.0), ("hands", 10.0), ("season", 10.0)]:
        c = pd.Categorical(dd[col]); bl.append(OneHot(c.codes, len(c.categories), sc))
    cell = (dd["s_arch"].to_numpy() * kr + dd["r_arch"].to_numpy()).astype(int)
    bl.append(OneHot(cell, ks * kr, 3.0))
    X = sp.hstack(bl).tocsr()
    if LINK["mode"] == "logit":
        from sklearn.linear_model import LogisticRegression
        sc_, tr_ = COUNTS[LINK["oc"]]
        succ, tri = dd[sc_].to_numpy(float), dd[tr_].to_numpy(float)
        XX = sp.vstack([X, X]).tocsr(); YY = np.r_[np.ones(len(dd)), np.zeros(len(dd))]; WW = np.r_[succ, tri - succ]
        m = LogisticRegression(C=0.3 ** 2, solver="lbfgs", max_iter=1000, tol=1e-6).fit(XX, YY, sample_weight=WW)
        g = m.coef_[0][-ks * kr:].reshape(ks, kr) * 3.0
    else:
        m = Ridge(alpha=alpha, solver="sparse_cg", tol=1e-7, max_iter=5000).fit(X, yy, sample_weight=ww)
        g = m.coef_[-ks * kr:].reshape(ks, kr) * 3.0
    w = np.bincount(cell, weights=ww, minlength=ks * kr).reshape(ks, kr)
    return DoubleCentre(g, w), np.bincount(cell, minlength=ks * kr).reshape(ks, kr)


def Boot(d, B, rng):
    ug, inv = np.unique(d["match"].to_numpy(), return_inverse=True)
    idx = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    for _ in range(B):
        yield np.concatenate([idx[i] for i in rng.integers(0, len(ug), len(ug))])


def BH(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.minimum(q, 1)


def Arch(prof, feats, k, kmin=3, kmax=8, max_pc=4):
    fit = prof[prof["season"].isin(FIT_ARCH)]
    med = fit[feats].median()
    sc = StandardScaler().fit(fit[feats].fillna(med))
    pca = PCA(min(max_pc, len(feats)), random_state=SEED).fit(sc.transform(fit[feats].fillna(med)))
    S = pca.transform(sc.transform(fit[feats].fillna(med)))
    bic, best = {}, None
    for kk in ([k] if k else range(kmin, kmax + 1)):
        gm = GaussianMixture(kk, covariance_type="diag", n_init=5, random_state=SEED).fit(S)
        bic[kk] = float(gm.bic(S))
        if best is None or bic[kk] < best[0]:
            best = (bic[kk], kk, gm)
    _, kk, gm = best
    order = np.argsort(-np.bincount(gm.predict(S), minlength=kk)); remap = {int(o): i for i, o in enumerate(order)}
    Sall = pca.transform(sc.transform(prof[feats].fillna(med)))
    lab = np.array([remap[int(i)] for i in gm.predict(Sall)])
    Zfit = pd.DataFrame(sc.transform(fit[feats].fillna(med)), columns=feats).assign(a=lab[prof["season"].isin(FIT_ARCH).to_numpy()]).groupby("a").mean()
    names = {int(a): " / ".join(f"{'high' if r[f] > 0 else 'low'} {f}" for f in r.abs().sort_values(ascending=False).index[:3]) for a, r in Zfit.iterrows()}
    return lab, kk, bic, names, Zfit, np.round(pca.explained_variance_ratio_, 3).tolist(), Sall


def Main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tour", default="atp")
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--link", default="linear", choices=["linear", "logit"])
    a = ap.parse_args()
    LINK["mode"] = a.link
    pre = f"tennis_{a.tour}" + ("" if a.link == "linear" else "_logit")
    rng = np.random.default_rng(SEED)
    s = LoadMatches(a.tour)
    prof = Profiles(s)
    rep = {"args": vars(a), "server_rows_by_season": s.groupby("season").size().to_dict(), "profiles_by_season": prof.groupby("season").size().to_dict()}
    for kind, feats in [("serve", SERVE_F), ("return", RET_F)]:
        lab, k, bic, names, Z, var, S = Arch(prof, feats, a.k)
        prof[f"{kind}_arch"] = lab
        prof[f"{kind}_pc1"], prof[f"{kind}_pc2"] = S[:, 0], S[:, 1]
        pv = prof.pivot_table(index="player", columns="season", values=f"{kind}_arch")
        pers = {f"{y-1}->{y}": float((pv[y - 1] == pv[y])[pv[[y - 1, y]].notna().all(1)].mean()) for y in YEARS[1:] if y in pv and y - 1 in pv}
        rep[f"{kind}_archetypes"] = {"k": k, "bic": bic, "pca_var": var, "names": names, "persistence": pers}
        Z.round(2).to_csv(os.path.join(a.out, f"{pre}_{kind}_archetype_centroids_z.csv"))
    prof.to_csv(os.path.join(a.out, f"{pre}_profiles.csv"), index=False)
    ks, kr = rep["serve_archetypes"]["k"], rep["return_archetypes"]["k"]
    L = prof[["season", "player", "serve_arch", "return_arch"]].copy(); L["season"] += 1
    d = s.merge(L[["season", "player", "serve_arch"]].rename(columns={"player": "server", "serve_arch": "s_arch"}), on=["season", "server"]) \
         .merge(L[["season", "player", "return_arch"]].rename(columns={"player": "returner", "return_arch": "r_arch"}), on=["season", "returner"])
    d["skey"] = d["server"].astype(str) + "_" + d["season"].astype(str)
    d["rkey"] = d["returner"].astype(str) + "_" + d["season"].astype(str)
    d["hands"] = d["s_hand"].astype(str) + d["r_hand"].astype(str)
    rep["labelled_rows_by_season"] = d.groupby("season").size().to_dict()
    rows = []
    for oc, tau in OUTCOMES.items():
        LINK["oc"] = oc
        res = {}
        for tag, ss in [("fit", FIT), ("hold", HOLD), ("fwd", FWD)]:
            dd = d[d["season"].isin(ss)].reset_index(drop=True)
            if len(dd) < 200:
                continue
            y = dd[oc].to_numpy(float)
            wts = dd["svpt"].to_numpy(float) / dd["svpt"].mean()
            alpha = float(np.average((y - np.average(y, weights=wts)) ** 2, weights=wts) / tau ** 2)
            g, n = Fit(dd, y, wts, ks, kr, alpha)
            dr = np.array([Fit(dd, y, wts, ks, kr, alpha, rows=i)[0] for i in Boot(dd, a.boot if tag != "fwd" else 0, rng)]) if tag != "fwd" else None
            res[tag] = (g, n, dr)
        gf, nf, df_ = res["fit"]; gh, nh, dh = res["hold"]
        sef, seh = df_.std(0, ddof=1), dh.std(0, ddof=1)
        pf = 2 * norm.sf(np.abs(gf / sef)); qf = BH(pf.ravel()).reshape(pf.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            ph = norm.sf(np.sign(gf) * gh / seh)
        sig = qf < 0.10; qh = np.full(gf.shape, np.nan)
        if sig.any():
            qh[sig] = BH(np.nan_to_num(ph[sig], nan=1.0))
        ww = np.sqrt(nf * nh).ravel().astype(float); x, z = gf.ravel(), gh.ravel()
        mx, mz = np.average(x, weights=ww), np.average(z, weights=ww)
        corr = float(np.average((x - mx) * (z - mz), weights=ww) / np.sqrt(np.average((x - mx) ** 2, weights=ww) * np.average((z - mz) ** 2, weights=ww)))
        rep.setdefault("replication", {})[oc] = {"corr_fit_hold": round(corr, 3), "n_sig_fit": int(sig.sum()), "n_confirmed": int(np.nansum(qh < 0.10)),
                                                "cells": ks * kr, "sd_eff_fit": round(float(np.sqrt(np.average(gf ** 2, weights=nf))), 4),
                                                "median_se_fit": round(float(np.median(sef)), 4)}
        print(a.tour, oc, rep["replication"][oc], flush=True)
        for i in range(ks):
            for j in range(kr):
                r = {"outcome": oc, "b_arch": i, "p_arch": j, "serve_name": rep["serve_archetypes"]["names"].get(i), "return_name": rep["return_archetypes"]["names"].get(j),
                     "n_fit": int(nf[i, j]), "eff_fit": gf[i, j], "lo90_fit": np.percentile(df_[:, i, j], 5), "hi90_fit": np.percentile(df_[:, i, j], 95),
                     "q_fit": qf[i, j], "n_hold": int(nh[i, j]), "eff_hold": gh[i, j], "lo90_hold": np.percentile(dh[:, i, j], 5),
                     "hi90_hold": np.percentile(dh[:, i, j], 95), "p_hold_same_dir": ph[i, j], "q_hold": qh[i, j], "confirmed": bool(sig[i, j] and qh[i, j] < 0.10)}
                if "fwd" in res:
                    r.update(n_fwd=int(res["fwd"][1][i, j]), eff_fwd=res["fwd"][0][i, j])
                rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"{pre}_interactions.csv"), index=False)
    json.dump(rep, open(os.path.join(a.out, f"{pre}_report.json"), "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print("DONE", rep["labelled_rows_by_season"], flush=True)


if __name__ == "__main__":
    Main()


