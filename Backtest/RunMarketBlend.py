r"""Run the LSE-style market-blend prototype (tasks 1-5) and write the AgentDocs folder.

Default is local data only: no Kalshi API calls, no orders. Pass --refresh-liquidity
to survey volume/open interest from the public API (read-only). Run at BelowNormal
via RunJob.ps1; this module does not set process priority itself.

Usage:
  powershell -ExecutionPolicy Bypass -File .\RunJob.ps1 -Name market_blend -Module Backtest.RunMarketBlend -ModuleArgs "--out-dir E:\AgentDocs\Tennibot\2026-11-13_market-blend"
  .venv\Scripts\python.exe -m Backtest.RunMarketBlend --out-dir E:\AgentDocs\Tennibot\2026-11-13_market-blend
  .venv\Scripts\python.exe -m Backtest.RunMarketBlend --refresh-liquidity --out-dir E:\AgentDocs\Tennibot\2026-11-13_market-blend
"""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from Backtest.FrozenParams import MODEL_VERSION

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(r"E:\AgentDocs\Tennibot\2026-11-13_market-blend")
ET = ZoneInfo("America/New_York")


def GitHead():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def Load(path):
    return pd.read_csv(path) if path.exists() else None


def Fmt(frame, columns=None, float_fmt="{:.4f}"):
    if frame is None or len(frame) == 0:
        return "_not produced in this run._"
    view = frame[columns] if columns else frame
    # to_markdown needs tabulate, which is not a repo dependency; emit a simple pipe table.
    cols = list(view.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for row in view.itertuples(index=False):
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("" if pd.isna(value) else float_fmt.format(value))
            else:
                cells.append("" if pd.isna(value) else str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def Headline(gap, tour, split, level):
    if gap is None:
        return "n/a"
    part = gap[(gap["tour"] == tour) & (gap["split"] == split) & (gap["level"] == level)]
    if len(part) == 0:
        return "n/a"
    r = part.iloc[0]
    return (f"{r['ign_rel_bits']:+.4f} bits (90% CI {r['ci90_lo']:+.4f} to {r['ci90_hi']:+.4f}, N={int(r['n'])})")


def WriteReport(out_dir, base, refresh_liquidity, ran):
    gap = Load(out_dir / f"{base}_gap.csv")
    weights = Load(out_dir / f"{base}_weights.csv")
    oos = Load(out_dir / f"{base}_blend_oos.csv")
    bank = Load(out_dir / f"{base}_bankroll_stats.csv")
    srelo = Load(out_dir / f"{base}_serve-return-elo.csv")
    liq = Load(out_dir / f"{base}_liquidity.csv")
    now = dt.datetime.now(ET)
    hour = now.strftime("%I").lstrip("0") or "12"
    written = f"{now.strftime('%Y-%m-%d')} {hour}:{now.strftime('%M %p')} ET"

    weight_summary = "_weights file not produced._"
    if weights is not None and len(weights):
        fitted = weights[weights["fitted"].astype(str).isin(["True", "true", "1"])]
        lines = []
        for tour, part in fitted.groupby("tour"):
            lines.append(
                f"- **{tour}**: last-week a={part['a'].iloc[-1]:.3f}, b={part['b'].iloc[-1]:.3f}, "
                f"c={part['c'].iloc[-1]:.3f} (mean c={part['c'].mean():.3f}, "
                f"median linear-pool w={part['linear_pool_w'].median():.3f}, weeks fitted={len(part)})."
            )
        weight_summary = "\n".join(lines) if lines else "No week reached the minimum training size; no blend was fitted."

    bank_lines = "_bankroll stats not produced._"
    if bank is not None and len(bank):
        bank_lines = Fmt(bank, ["tour", "level", "strategy", "fee_basis", "n_bets", "final_equity",
                                "staked", "roi", "roi_ci90_lo", "roi_ci90_hi", "max_drawdown_pct", "ruined"])

    srelo_lines = "_serve/return Elo comparison not produced._"
    if srelo is not None and len(srelo):
        view = srelo[(srelo["split"] == "holdout") & (srelo["level"] == "all")]
        srelo_lines = Fmt(view, ["tour", "split", "level", "model", "n", "log_loss", "log_loss_bits", "accuracy", "brier"])

    liq_lines = "_liquidity survey not produced._"
    if liq is not None and len(liq):
        liq_lines = Fmt(liq, [c for c in liq.columns if c not in ("fetched_at_utc",)][:18])

    oos_lines = "_blend OOS file not produced._"
    if oos is not None and len(oos):
        oos_lines = Fmt(oos, ["tour", "split", "level", "n", "ign_blend_bits", "ign_market_bits",
                              "ign_rel_bits", "ci90_lo", "ci90_hi", "growth_per_event"])

    gap_lines = "_gap file not produced._"
    if gap is not None and len(gap):
        gap_lines = Fmt(gap, ["tour", "split", "level", "n", "ign_model_bits", "ign_market_bits",
                              "ign_rel_bits", "ci90_lo", "ci90_hi", "growth_per_event"])

    md = f"""# Tennis market blend vs Kalshi (2026-11-13)

Author: Tennibot. Written {written} (America/New_York). Repo `E:\\Tennis\\tennis` at HEAD `{GitHead()}`. Model `{MODEL_VERSION}`.
Recommendations only. Simulation; no orders. Default remains **PASS** unless the numbers below clear the gate.

## Headline

Relative Ignorance `ign_rel_bits = mean(IGN_model - IGN_market)` in bits per match. Negative means the model beats de-vigged Kalshi T-60. 90% match-level bootstrap CI.

- ATP holdout, tour: {Headline(gap, "atp", "holdout", "tour")}
- ATP forward, tour: {Headline(gap, "atp", "forward", "tour")}
- ATP holdout+forward, tour: {Headline(gap, "atp", "holdout+forward", "tour")}
- WTA holdout, tour: {Headline(gap, "wta", "holdout", "tour")}
- WTA forward, tour: {Headline(gap, "wta", "forward", "tour")}
- WTA holdout+forward, tour: {Headline(gap, "wta", "holdout+forward", "tour")}

Gate (LSE R1/R2): a BET requires pooled out-of-sample `ign_rel_bits` below 0 with the CI excluding 0, **or** a clearly non-zero walk-forward blend weight `c` on the model. Otherwise stay PASS.

## 1. Log-loss gap vs de-vigged Kalshi T-60

{gap_lines}

`growth_per_event = 2^(-ign_rel_bits)` is the wealth-growth factor of a proportional bettor who trusts the model against a bettor who trusts the market (Roulston & Smith / Hagedorn & Smith). A number below 1 means the model loses to the market.

## 2. Walk-forward logistic blend

`logit p_blend = a + b * logit p_mkt + c * logit p_model`, refit each ISO week on all earlier weeks (minimum 200 matches). Applied out of sample to the current week. A linear pool `w * p_model + (1-w) * p_mkt` is fitted as a check (`linear_pool_w`).

{weight_summary}

Out-of-sample log loss of the blend vs Kalshi alone (weeks with a fitted blend only):

{oos_lines}

If `c` sits near 0 and the OOS `ign_rel_bits` CI includes 0 or is positive, the model adds nothing beyond Kalshi and the blend should not be used to bet.

## 3. $10k bankrolls off the blend

Edges from **p_blend**, never raw p_model. Fill = the bet side's ask. Qualifying edge: `p_blend - ask - fee >= 0.03`. Starting bank $10,000, per tour, tour-level matches unless noted.

- `flat100`: $100 per qualifying match, no compounding, stop if equity < $100.
- `qkelly`: 0.25 x (p - cost)/(1 - cost) of start-of-day equity, capped at 2%.
- Fee bases: `tim_est_0.02` (Tim's estimate, labelled) and `kalshi_verified` (`0.07*p*(1-p)`), run as separate series, never substituted silently.

{bank_lines}

## 4. Elo-style serve/return prototype vs the current model

Opponent-adjusted serve-points-won and return-points-won ratings with a decaying K-factor (defaults match the surface-Elo K schedule; **not tuned on the holdout**). Leak-free group walk, Sackmann data through 2026-06-01. Holdout = 2023-2025. Log loss in nats (repo convention) and bits. `srelo` is the prototype; `sr` / `elo` / `blend` are the current frozen model on the same matches.

{srelo_lines}

## 5. Kalshi liquidity by level

WTA 125 is Kalshi `details.tour = WTA Challenger` or a tournament name containing "WTA 125" / "125K", plus the `KXWTACHALLENGERMATCH` series. Compared with ATP/WTA tour and ATP Challenger.

{liq_lines}

Refresh used: {"yes -- public API, read-only" if refresh_liquidity else "no — local dumps only. Open interest is likely missing until you rerun with --refresh-liquidity."}

## Files

All under `{out_dir}`:

- `{base}.md` (this report)
- `{base}_gap.csv`
- `{base}_weights.csv`
- `{base}_blend_oos.csv`
- `{base}_bankroll.csv` (daily equity; `equity`, `drawdown_pct`)
- `{base}_bankroll_stats.csv`
- `{base}_serve-return-elo.csv`
- `{base}_liquidity.csv`

Steps that ran this time: {", ".join(ran) if ran else "none"}.

## Caveats

- **Default stays PASS** unless the R1/R2 gate above is met and the 3-cent bucket's out-of-sample ROI CI is above 0. A point-estimate edge on paper is not a betting recommendation.
- Fills are assumed at the T-60 ask with **unlimited size**. Kalshi candles have no size, so this is optimistic.
- Fee `tim_est_0.02` is Tim's working estimate, not a verified exchange fee; it is always labelled. Kalshi's verified taker fee `0.07*p*(1-p)` is reported beside it (centicent rounding ignored).
- De-vig is the **proportional** split of the two executable asks (yes ask and no ask = 1 - yes bid). This is not the same as the mid `(bid+ask)/2` used in some earlier tennis exports; mid is noted only as a different convention.
- T-60 is the last 1-min candle at or before scheduled start minus 60 minutes, age <= 30 minutes. Tennis start times slip, so T-60 can be hours before the first ball, or in-play if the match started early. Events whose mid moved more than 0.25 between T-60 and T-5 are flagged `possible_inplay` and dropped.
- Kalshi quotes have no source timestamp; candle `end_ts` is the price time used for T-60.
- The scored market window is Kalshi-era only: holdout here is Jun-Dec 2025, forward is Jan-Jun 2026. Tune (2010-19) and validation (2020-22) have no Kalshi T-60 prices and are not in the gap, blend, or bankroll.
- Sackmann rating data ends 2026-06-01. The Elo-only `_ext-2026H2` files from the earlier calibration/bankroll folders are not used here.
- The walk-forward blend is fit only on past weeks. The first weeks (until 200 matches have accumulated) fall back to the market probability and are excluded from the OOS blend score.
- Bankrolls start at $10k and do **not** reset each season. They use p_blend on weeks with a fitted blend only. Qualifying is edge vs **fill** (ask + fee), not vs the mid.
- The serve/return Elo prototype uses frozen defaults (K schedule of surface Elo). Those defaults were not searched on the holdout. It is a prototype, not a replacement until the holdout comparison is read.
- Liquidity: existing `data/raw/kalshi/markets/*.jsonl` dumps have `volume_fp` but typically **no open interest and no standing bid/ask**. Historical spreads come from T-60 candles when those files exist. Open interest needs `--refresh-liquidity`. Volume on complementary yes/no tickers in the same event is summed as separate markets, not de-duplicated.
- WTA 125 classification depends on Kalshi milestone metadata. If a WTA 125 event is missing a milestone, it may be tagged only by series (`KXWTACHALLENGERMATCH`).
- Simulation only. Tim places any wager himself.
"""
    path = out_dir / f"{base}.md"
    path.write_text(md, encoding="utf-8")
    print(f"wrote {path}")
    return path


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base", default="2026-11-13_market-blend")
    parser.add_argument("--refresh-liquidity", action="store_true",
                        help="Opt-in public API pull for the liquidity survey (read-only). Off by default.")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["gap", "blend", "bankroll", "srelo", "liquidity", "report"])
    parser.add_argument("--min-train", type=int, default=200)
    parser.add_argument("--l2-c", type=float, default=0.0)
    parser.add_argument("--levels", nargs="+", default=["tour"])
    args = parser.parse_args(argv)
    skip = set(args.skip)
    ran = []
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if not ({"gap", "blend", "bankroll"} & skip):
        from Backtest.MarketBlend import Main as BlendMain
        BlendMain(["--out-dir", str(args.out_dir), "--base", args.base,
                   "--min-train", str(args.min_train), "--l2-c", str(args.l2_c),
                   "--levels", *args.levels])
        ran.extend(["gap", "blend", "bankroll"])
    else:
        if "gap" not in skip or "blend" not in skip or "bankroll" not in skip:
            raise SystemExit("gap, blend and bankroll are produced together by MarketBlend; skip all three or none.")

    if "srelo" not in skip:
        from Backtest.RunServeReturnElo import Main as SrMain
        SrMain(["--out-dir", str(args.out_dir), "--base", args.base])
        ran.append("srelo")

    if "liquidity" not in skip:
        from Backtest.KalshiLiquidity import Main as LiqMain
        liq_args = ["--out-dir", str(args.out_dir), "--base", args.base]
        if args.refresh_liquidity:
            liq_args.append("--refresh")
        LiqMain(liq_args)
        ran.append("liquidity" + ("+refresh" if args.refresh_liquidity else ""))

    if "report" not in skip:
        WriteReport(args.out_dir, args.base, args.refresh_liquidity, ran)
        ran.append("report")
    print("done:", ", ".join(ran))


if __name__ == "__main__":
    Main()