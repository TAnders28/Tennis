r"""Write the residual-search AgentDocs markdown from a finished ResidualSearch run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from Backtest.ResidualSearch import DEFAULT_BASE, DEFAULT_OUT, HOLDOUT_START, Main as RunSearch


def FmtCi(mean, lo, hi, n, digits=4):
    if mean is None or (isinstance(mean, float) and mean != mean):
        return f"n/a (N={n})"
    return f"{mean:+.{digits}f} (95% day-block CI {lo:+.{digits}f} to {hi:+.{digits}f}, N={n})"


def WriteReport(out_dir: Path, base: str):
    out_dir = Path(out_dir)
    best = json.loads((out_dir / f"{base}_best.json").read_text(encoding="utf-8"))
    spa = pd.read_csv(out_dir / f"{base}_spa.csv")
    stats_path = out_dir / f"{base}_bankroll_stats.csv"
    stats = pd.read_csv(stats_path) if stats_path.exists() else pd.DataFrame()
    grid_path = out_dir / f"{base}_grid.csv"
    n_configs = best.get("n_configs")
    params = best["best_params"]
    hold = best["holdout"]
    fit = best["fit"]
    spa_d = best["spa"]
    perm = best["permutation_holdout"]
    rc = best["white_rc"]

    def ParamLine():
        keys = ["k_a", "k_c", "surface_weight", "bo5_mult", "lower_k_mult",
                "decay_halflife_days", "glicko", "fatigue_coef", "sr_weight"]
        return ", ".join(f"{k}={params[k]}" for k in keys)

    bank_lines = ["No bankroll rows."]
    if len(stats):
        view = stats[(stats["level"] == "all") & (stats["fee_basis"] == "tim_est_0.02")]
        bits = []
        for r in view.itertuples():
            bits.append(
                f"- {r.tour} {r.strategy}: final ${r.final_equity:,.0f}, ROI {r.roi:+.1%} "
                f"(N={int(r.n_bets)}); 10,000-path MC equity 5/50/95 "
                f"${r.mc_equity_p05:,.0f} / ${r.mc_equity_p50:,.0f} / ${r.mc_equity_p95:,.0f}."
            )
        extra = stats[(stats["level"] == "all") & (stats["fee_basis"] == "kalshi_verified")
                      & (stats["strategy"] == "flat100")]
        if len(extra):
            bits.append("Kalshi verified fee (0.07 p (1-p)) flat100, same 3c line: " +
                        ", ".join(f"{r.tour} ROI {r.roi:+.1%} (N={int(r.n_bets)})" for r in extra.itertuples()) + ".")
        if bits:
            bank_lines = bits

    pinnacle_line = (
        "Pinnacle closing odds from tennis-data.co.uk were **not** available locally "
        f"(no files under `data/raw/tennis_data` and no `data/processed/tennis_data_*.parquet`). "
        "The site forbids bots (HTTP 403), so this run did not download them. Tim can drop the yearly "
        "xlsx/csv files into `data/raw/tennis_data/atp` and `.../wta` and re-run after `PullTennisData --ingest`. "
        "The residual search used Kalshi T-60 de-vigged asks where they overlap Sackmann matches "
        "(June 2025 through early June 2026)."
        if not best.get("pinnacle_tennis_data_present") else
        "Pinnacle files were present locally; see the best.json market field."
    )

    lines = [
        f"# Tennis residual-on-market search ({base})",
        "",
        f"**Verdict: {best['verdict']}.** {best['verdict_reason']}",
        "",
        "## Pre-registered split",
        "",
        "Chosen from Kalshi date coverage only, before any model scores were computed:",
        "",
        f"- Fit / config selection: market matches with `date_et` before {HOLDOUT_START.isoformat()} (2025 H2).",
        f"- Frozen holdout: `date_et` on or after {HOLDOUT_START.isoformat()} (2026, through the Sackmann snapshot around 2026-06-01).",
        f"- {full_grid_note(n_configs)} rating configs (surface Elo K, surface blend, best-of-5 multiplier, lower-level K, time decay, Glicko-style RD, fatigue/schedule, serve/return blend). Residual weight `w` is fit walk-forward by month on earlier fit matches (default 0.5 until 200 training matches) and frozen from all of 2025 for the holdout.",
        "- Offset = logit of the de-vigged Kalshi T-60 ask. `logit(p) = logit(p_mkt) + w * (logit(p_model) - logit(p_mkt))`.",
        "",
        "## Data blocker",
        "",
        pinnacle_line,
        "",
        "Sackmann rating data (Aneeshers mirror) ends 2026-06-01. Ratings still walk the full leak-free history from 2000; only matches with a usable T-60 quote are scored.",
        "",
        "## Best config (selected on 2025 walk-forward bits gap)",
        "",
        ParamLine() + f", frozen w={best['w_frozen']:.3f}.",
        "",
        f"- Fit bits gap vs T-60: {FmtCi(fit.get('bits_gap'), fit.get('bits_gap_ci95_lo'), fit.get('bits_gap_ci95_hi'), fit.get('n'))}. Positive means the residual beats the market.",
        f"- Fit CLV (early vs T-5, signed by the residual vs early): {FmtCi(fit.get('clv'), fit.get('clv_ci95_lo'), fit.get('clv_ci95_hi'), fit.get('clv_n'))}.",
        "",
        "## Holdout 2026 (frozen)",
        "",
        f"- Bits gap vs T-60: {FmtCi(hold.get('bits_gap'), hold.get('bits_gap_ci95_lo'), hold.get('bits_gap_ci95_hi'), hold.get('n'))}.",
        f"- CLV vs the T-5 close: {FmtCi(hold.get('clv'), hold.get('clv_ci95_lo'), hold.get('clv_ci95_hi'), hold.get('clv_n'))}.",
        "",
        "## SPA / Reality Check / permutation",
        "",
        f"- Hansen SPA on the 2025 walk-forward sample ({spa_d.get('n_models')} models, {spa_d.get('n_days')} days, {spa_d.get('reps')} day-block bootstraps): p={spa_d.get('p_value'):.4f}.",
        f"- White Reality Check (same sample): p={rc.get('p_value'):.4f}.",
        f"- Permutation null on the frozen best config, holdout outcomes shuffled {perm.get('reps')} times: p={perm.get('p_value'):.4f}.",
        "",
        "## $10k bankroll on the frozen holdout",
        "",
        "Fills at the T-60 ask plus fee, 3-cent value line. `flat100` is $100 a match, no compounding, stop below $100. `qkelly` is 0.25 x (p-cost)/(1-cost) of start-of-day equity, capped at 2%. Bankrolls do not reset mid-holdout. 10,000-path Monte Carlo resamples holdout days with replacement. Fee basis `tim_est_0.02` is Tim's estimate; Kalshi's verified taker fee 0.07 p (1-p) is reported beside it.",
        "",
        *bank_lines,
        "",
        "## In-play Monte Carlo grid",
        "",
        "A separate 1,000,000-match point-by-point Monte Carlo per serve-probability cell (p=0.50 to 0.80 step 0.01, best-of-3 and best-of-5) writes `_inplay-pregame.csv`, `_inplay-grid.csv`, and `_inplay-summary.csv`. Those files are a fair-price table for a later live model. **No betting claims are made from the in-play grid.**",
        "",
        "## Caveats",
        "",
        "- Default is PASS. This is a recommendation only; Tim places any bets.",
        "- Kalshi T-60 is a short market window, not a 2005-2026 Pinnacle close. CLV uses the Kalshi early quote vs T-5, not a sportsbook open-to-close.",
        "- The 2025 selection sample is only about half a year of Kalshi matches; 3,888 configs were tried, so a lucky fit is possible even with SPA.",
        "- Possible in-play flags (T-60 to T-5 mid jump > 0.25) are dropped. Quotes have no size.",
        "- Fills assume the ask is available (unlimited size). Fee label `tim_est_0.02` is an estimate.",
        "- Glicko-style RD and fatigue/schedule are simple add-ons, not a full Glicko-2 implementation.",
        "",
        "## Files",
        "",
        f"`{base}.csv` (daily holdout bits gap), `{base}_grid.csv`, `{base}_spa.csv`, `{base}_bankroll.csv`, `{base}_bankroll_stats.csv`, `{base}_best.json`, plus the in-play MC files if that job has finished.",
        "",
    ]
    path = out_dir / f"{base}.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", path, flush=True)
    return path


def full_grid_note(n):
    return f"{n:,}" if n is not None else "the"


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if not args.report_only:
        RunSearch(["--out-dir", str(args.out_dir), "--base", args.base, "--workers", str(args.workers)]
                  + (["--smoke"] if args.smoke else []))
    WriteReport(args.out_dir, args.base)


if __name__ == "__main__":
    Main()
