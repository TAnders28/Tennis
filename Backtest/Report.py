r"""Build the AgentDocs deliverable from finished backtest outputs (no model runs here).

  .venv\Scripts\python.exe -m Backtest.Report --out-dir E:\AgentDocs\Tennibot\2026-11-13_tennis-model --base 2026-11-13_tennis-model --findings findings.md

Files (same base name): <base>.md (findings + tables + caveats), <base>.csv (tidy per-season metrics:
date,series,n,accuracy,log_loss,brier), <base>_kalshi.csv (Kalshi-window market comparison + bet results),
<base>_equity.csv (date,series,equity,drawdown_pct for Kalshi-window bets), <base>.json (summary).
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
BT = REPO_ROOT / "data" / "backtest"


def Fmt(frame, floats=3):
    frame = frame.copy()
    for c in frame.columns:
        if frame[c].dtype.kind == "f":
            frame[c] = frame[c].map(lambda v: "" if pd.isna(v) else f"{v:.{floats}f}")
    header = "| " + " | ".join(map(str, frame.columns)) + " |"
    sep = "|" + "---|" * len(frame.columns)
    rows = ["| " + " | ".join(map(str, r)) + " |" for r in frame.itertuples(index=False)]
    return "\n".join([header, sep] + rows)


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--findings", default=None, help="markdown file with the written findings (goes on top)")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=REPO_ROOT).stdout.strip()

    metrics = pd.concat([pd.read_csv(BT / t / "metrics.csv").assign(tour=t) for t in ("atp", "wta")], ignore_index=True)
    tidy = metrics.assign(date=metrics["season"].astype(str) + "-12-31",
                          series=metrics["tour"] + "_" + metrics["level"] + "_" + metrics["model"])
    tidy = tidy[tidy["season"] >= 2010][["date", "series", "split", "n", "accuracy", "log_loss", "brier"]]
    tidy.to_csv(out / f"{args.base}.csv", index=False)

    frozen = {t: json.loads((BT / t / "frozen.json").read_text()) for t in ("atp", "wta")}
    frozen_r1 = {t: json.loads((BT / t / "frozen_r1.json").read_text()) for t in ("atp", "wta") if (BT / t / "frozen_r1.json").exists()}
    markov = json.loads((BT / "markov" / "summary.json").read_text()) if (BT / "markov" / "summary.json").exists() else None
    kr_dir = BT / "kalshi_reality"
    market = pd.read_csv(kr_dir / "market_compare.csv") if (kr_dir / "market_compare.csv").exists() else pd.DataFrame()
    bets = pd.read_csv(kr_dir / "summary.csv") if (kr_dir / "summary.csv").exists() else pd.DataFrame()
    equity = pd.read_csv(kr_dir / "equity.csv") if (kr_dir / "equity.csv").exists() else pd.DataFrame()
    if len(equity):
        equity.to_csv(out / f"{args.base}_equity.csv", index=False)
    if len(market) or len(bets):
        pd.concat([market.assign(table="market_compare"), bets.assign(table="bets")], ignore_index=True).to_csv(out / f"{args.base}_kalshi.csv", index=False)

    md = []
    if args.findings:
        md.append(Path(args.findings).read_text(encoding="utf-8"))
    md += ["", "## Frozen parameters and search", ""]
    for t in ("atp", "wta"):
        f = frozen[t]
        r1 = frozen_r1.get(t, {}).get("configs_tried", {})
        r2 = f.get("configs_tried", {})
        total = sum(r1.values()) + sum(r2.values())
        md.append(f"- **{t.upper()}** Elo {f['elo']}; serve/return {f['sr']}; blend weight on serve/return {f['blend_weight_sr']}. "
                  f"Configs tried: round 1 {r1}, round 2 {r2} (total {total}). Frozen {f['frozen_at']}; data through {f['data_last_date']}.")
    md += ["", f"Selection: {frozen['atp']['selection']}.", ""]
    md += ["## Per-season metrics, tour-level main draw (blend = headline model)", ""]
    view = metrics[(metrics.level == "tour") & (metrics.season >= 2010)]
    pivot = view.pivot_table(index=["tour", "season", "split", "n"], columns="model", values=["accuracy", "log_loss", "brier"]).reset_index()
    pivot.columns = ["_".join(c).strip("_") for c in pivot.columns]
    md += [Fmt(pivot[["tour", "season", "split", "n", "accuracy_blend", "log_loss_blend", "brier_blend", "accuracy_elo", "log_loss_elo",
                      "accuracy_sr", "log_loss_sr"]]), ""]
    md += ["## Holdout (2023-2025) and forward (2026 Jan-Jun) by level, blend model", ""]
    lv = metrics[(metrics.model == "blend") & metrics.split.isin(["holdout", "forward"])]
    lv = lv.groupby(["tour", "level", "split"]).apply(lambda g: pd.Series({
        "n": int(g["n"].sum()), "accuracy": (g["accuracy"] * g["n"]).sum() / g["n"].sum(),
        "log_loss": (g["log_loss"] * g["n"]).sum() / g["n"].sum(), "brier": (g["brier"] * g["n"]).sum() / g["n"].sum()}),
        include_groups=False).reset_index()
    md += [Fmt(lv), ""]
    if len(market):
        md += ["## Model vs Kalshi market (same matches; Kalshi mid at T-60 and T-5 of scheduled start)", ""]
        cols = ["tour", "level", "split", "n", "market_t60_log_loss", "blend_log_loss", "elo_log_loss", "sr_log_loss",
                "market_t60_fav_acc", "blend_acc", "market_t5_n", "market_t5_log_loss", "market_t5_fav_acc"]
        md += [Fmt(market[[c for c in cols if c in market.columns]]), ""]
    if len(bets):
        md += ["## Kalshi betting simulation (blend, 1 contract per bet, threshold 0, one side per match)", ""]
        b = bets[(bets.model == "blend") & (bets.threshold == 0.0)]
        md += [Fmt(b[["tour", "level", "split", "entry", "fill", "fee_basis", "n_bets", "wins", "roi", "avg_edge", "avg_price", "clv_mean", "clv_pos_share"]]), ""]
        md += ["### Threshold sensitivity (blend, t60, ask fill, Tim's fee; thresholds fixed in advance, not tuned)", ""]
        s = bets[(bets.model == "blend") & (bets.entry == "t60") & (bets.fill == "ask") & (bets.fee_basis == "tim_est_0.02")]
        md += [Fmt(s[["tour", "level", "split", "threshold", "n_bets", "roi", "clv_mean"]]), ""]
        md += ["### Elo-only and serve/return-only (t60, ask, Tim's fee, threshold 0)", ""]
        s = bets[bets.model.isin(["elo", "sr"]) & (bets.entry == "t60") & (bets.fill == "ask") & (bets.fee_basis == "tim_est_0.02") & (bets.threshold == 0.0)]
        md += [Fmt(s[["tour", "level", "split", "model", "n_bets", "roi", "clv_mean"]]), ""]
    if markov:
        md += ["## In-play Markov model validation (Grand Slam point-by-point)", "",
               f"Chosen in-match learning column: {markov['chosen_column']} (tuned on 2011-2019 by log loss: {markov['tune_log_loss_by_m']}).",
               f"Validated on holdout: **{markov['validated']}** (criteria: {markov['criteria']}).",
               f"Match status counts (all years): {markov['match_status_counts']}.", ""]
        rows = [{"split": k, **{kk: vv for kk, vv in v.items() if not isinstance(vv, list)},
                 "ll_ci_naive": "{:.3f}-{:.3f}".format(*v["log_loss_ci_naive"]),
                 "ll_ci_match_bootstrap": "{:.3f}-{:.3f}".format(*v["log_loss_ci_match_bootstrap"])} for k, v in markov["summary"].items()]
        md += [Fmt(pd.DataFrame(rows)), "", "Holdout reliability (point-level; CIs: naive vs match-level bootstrap):", ""]
        rel = pd.DataFrame(markov["holdout_reliability"])
        rel["ci_naive"] = rel["ci_naive"].map(lambda c: f"{c[0]:.3f}-{c[1]:.3f}")
        rel["ci_match_bootstrap"] = rel["ci_match_bootstrap"].map(lambda c: f"{c[0]:.3f}-{c[1]:.3f}")
        md += [Fmt(rel), "", "By phase (holdout):", ""]
        ph = pd.DataFrame(markov["phases"])
        md += [Fmt(ph[ph.split == "holdout"]), ""]
    md += ["", f"_Generated by Backtest.Report from repo E:\\Tennis\\tennis at git {head}._"]
    (out / f"{args.base}.md").write_text("\n".join(md), encoding="utf-8")
    summary = {"git": head, "frozen": frozen, "markov_validated": markov["validated"] if markov else None,
               "markov_holdout": markov["summary"].get("holdout") if markov else None,
               "holdout_by_level": lv.to_dict(orient="records"),
               "kalshi_market_compare": market.to_dict(orient="records") if len(market) else [],
               "kalshi_bets_primary": bets[(bets.model == "blend") & (bets.threshold == 0.0) & (bets.entry == "t60")].to_dict(orient="records") if len(bets) else []}
    (out / f"{args.base}.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    print("wrote", sorted(p.name for p in out.iterdir()))


if __name__ == "__main__":
    Main()