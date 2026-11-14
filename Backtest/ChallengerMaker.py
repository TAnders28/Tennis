r"""Pre-registered ATP Challenger maker test (KXATPCHALLENGERMATCH only).

Primary analysis is model-free: back the de-vigged T-60 favourite (`market_fav`),
and compare taker-at-ask with maker-at-bid. The Part A walk-forward blend is
secondary and does not choose the side.

Holdout (frozen 2026-11-13 5:34 PM ET, before any results): scheduled start on or
after 2026-08-13 00:00 America/New_York. Official scores are holdout only.

Maker fill: a resting buy at the T-60 bid counts as filled only if a later 1-min
candle, after that T-60 quote and strictly before scheduled start, trades at or
through the bid (yes last <= bid, or yes ask <= bid; the no side is mirrored).

Edge = prob - (fill + 0.02). The $0.02 is Tim's estimate, labelled. Kalshi taker
fee 0.07*P*(1-P) is reported beside it. Challenger series fee_type is `quadratic`
(maker fee 0 on the published schedule); 0.0175*P*(1-P) is a sensitivity.

Usage (after candles through the holdout exist):
  .venv\Scripts\python.exe -m Backtest.ChallengerMaker --out-dir E:\AgentDocs\Tennibot\2026-11-13_challenger-maker
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from Backtest.KalshiJoin import INPLAY_JUMP, STALE_S, LoadJsonl, QuoteAt
from Backtest.MarketBlend import ApplyBlend, DevigAsk
from Backtest.Metrics import ClusterBootstrap
from Backtest.Odds import KalshiFee, TIM_FEE
from DataPull.Http import RAW_DIR

ET = ZoneInfo("America/New_York")
SERIES = "KXATPCHALLENGERMATCH"
HOLDOUT_START = dt.datetime(2026, 8, 13, 0, 0, 0, tzinfo=ET)  # frozen; do not change
DEFAULT_OUT = Path(r"E:\AgentDocs\Tennibot\2026-11-13_challenger-maker")
BLEND_PREDS = Path(r"E:\AgentDocs\Tennibot\2026-11-13_market-blend\2026-11-13_market-blend_blend_preds.parquet")
BLEND_WEIGHTS = Path(r"E:\AgentDocs\Tennibot\2026-11-13_market-blend\2026-11-13_market-blend_weights.csv")
QUAL = 0.03
START_BANK = 10_000.0
FLAT = 100.0
KELLY_FRAC = 0.25
KELLY_CAP = 0.02
BOOT_REPS = 2000
BOOT_SEED = 20261113


def MakerFeeZero(price):
    return 0.0


def MakerFeeSens(price):
    return 0.0175 * price * (1.0 - price)


FEE_FNS = {
    "tim_est_0.02": lambda p: TIM_FEE,
    "kalshi_taker_0.07p(1-p)": KalshiFee,
    "kalshi_maker_0": MakerFeeZero,
    "kalshi_maker_0.0175p(1-p)": MakerFeeSens,
}


def MakerFilled(candles, quote_end_ts, start_ts, side, t60_bid, t60_ask):
    """Return (filled, reason). side 'a' buys yes at t60_bid; side 'b' buys no at 1-t60_ask."""
    for candle in candles:
        end = candle.get("end_ts")
        if end is None or end <= quote_end_ts or end >= start_ts:
            continue
        last, ask, bid = candle.get("last"), candle.get("yes_ask"), candle.get("yes_bid")
        if side == "a":
            if last is not None and last <= t60_bid + 1e-12:
                return True, "last_through_bid"
            if ask is not None and ask <= t60_bid + 1e-12:
                return True, "ask_through_bid"
        else:
            no_bid = 1.0 - t60_ask
            if last is not None and (1.0 - last) <= no_bid + 1e-12:
                return True, "last_through_bid"
            if bid is not None and bid >= t60_ask - 1e-12:
                return True, "ask_through_bid"
    return False, "unfilled"


def LoadStarts():
    starts = {}
    for row in LoadJsonl(RAW_DIR / "kalshi" / "milestones_tennis.jsonl"):
        for ticker in row.get("event_tickers") or []:
            starts[ticker] = row
    return starts


def LoadMarkets():
    by_event = {}
    for row in LoadJsonl(RAW_DIR / "kalshi" / "markets" / f"{SERIES}.jsonl"):
        by_event.setdefault(row["event_ticker"], []).append(row)
    return by_event


def FrozenBlendWeights():
    """Last Part A ATP weights fitted on a week starting before the holdout. Secondary only."""
    if not BLEND_WEIGHTS.exists():
        return None
    weights = pd.read_csv(BLEND_WEIGHTS)
    fitted = weights[(weights["tour"] == "atp") & (weights["fitted"].astype(str).isin(["True", "true", "1"]))]
    fitted = fitted[pd.to_datetime(fitted["date"]) < HOLDOUT_START.replace(tzinfo=None)]
    if len(fitted) == 0:
        return None
    row = fitted.sort_values("date").iloc[-1]
    return {"a": float(row["a"]), "b": float(row["b"]), "c": float(row["c"]), "as_of": str(row["date"])}


def BuildEvents():
    starts = LoadStarts()
    markets = LoadMarkets()
    candle_dir = RAW_DIR / "kalshi" / "candles" / SERIES
    rows = []
    for path in sorted(candle_dir.glob("*.json")) if candle_dir.exists() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        event = payload.get("event_ticker") or path.stem
        raw_start = payload.get("scheduled_start")
        if not raw_start:
            continue
        start = dt.datetime.fromisoformat(raw_start.replace("Z", "+00:00"))
        if start.tzinfo is None:
            start = start.replace(tzinfo=dt.timezone.utc)
        start_et = start.astimezone(ET)
        start_ts = start.timestamp()
        minute = payload.get("minute") or []
        q60 = QuoteAt(minute, start_ts - 3600, stale_s=STALE_S)
        q5 = QuoteAt(minute, start_ts - 300, stale_s=STALE_S)
        if not q60:
            continue
        bid, ask = q60["yes_bid"], q60["yes_ask"]
        if not (bid is not None and ask is not None and 0 < bid < ask < 1):
            continue
        mid60 = (bid + ask) / 2
        mid5 = np.nan
        if q5 and q5["yes_bid"] is not None and q5["yes_ask"] is not None:
            mid5 = (q5["yes_bid"] + q5["yes_ask"]) / 2
        possible_inplay = bool(np.isfinite(mid5) and abs(mid5 - mid60) > INPLAY_JUMP)
        p_mkt = float(DevigAsk([bid], [ask])[0])
        if not np.isfinite(p_mkt):
            continue
        fav = "a" if p_mkt >= 0.5 else "b"
        p_fav = p_mkt if fav == "a" else 1.0 - p_mkt
        ask_fav = ask if fav == "a" else 1.0 - bid
        bid_fav = bid if fav == "a" else 1.0 - ask
        quote_end = q60["end_ts"]
        filled, fill_reason = MakerFilled(minute, quote_end, start_ts, fav, bid, ask)
        books = markets.get(event) or []
        ticker_a = payload.get("ticker")
        market_a = next((m for m in books if m.get("ticker") == ticker_a), books[0] if books else {})
        result = market_a.get("result")
        y_a = {"yes": 1.0, "no": 0.0}.get(result, np.nan)
        y_fav = y_a if fav == "a" else (1.0 - y_a if np.isfinite(y_a) else np.nan)
        details = (starts.get(event) or {}).get("details") or {}
        rows.append({
            "event_ticker": event,
            "ticker_a": ticker_a,
            "series": SERIES,
            "scheduled_start": start.isoformat(),
            "date_et": start_et.date().isoformat(),
            "start_et": start_et.isoformat(),
            "split": "holdout" if start_et >= HOLDOUT_START else "tune",
            "tournament": details.get("tournament_name"),
            "t60_bid": bid, "t60_ask": ask, "t60_spread": ask - bid,
            "t60_age_s": (start_ts - 3600 - quote_end),
            "p_mkt": p_mkt, "fav_side": fav, "p_fav": p_fav,
            "ask_fav": ask_fav, "bid_fav": bid_fav,
            "maker_filled": filled, "fill_reason": fill_reason,
            "possible_inplay": possible_inplay,
            "y_a": y_a, "y_fav": y_fav, "result_a": result,
        })
    frame = pd.DataFrame(rows)
    if len(frame) == 0:
        return frame
    # secondary blend probs, when Part A produced them
    if BLEND_PREDS.exists():
        preds = pd.read_parquet(BLEND_PREDS)
        if "event_ticker" in preds.columns:
            atp = preds[preds["tour"] == "atp"][["event_ticker", "p_model", "p_mkt", "p_blend_mkt", "blend_fitted", "y"]].drop_duplicates("event_ticker")
            atp = atp.rename(columns={"p_mkt": "p_mkt_parta", "y": "y_parta"})
            frame = frame.merge(atp, on="event_ticker", how="left")
    weights = FrozenBlendWeights()
    if weights and len(frame):
        frame["blend_weights_as_of"] = weights["as_of"]
        if "p_model" in frame.columns:
            has = frame["p_model"].notna() & frame["p_mkt"].notna()
            applied = np.full(len(frame), np.nan)
            applied[has.to_numpy()] = ApplyBlend(frame.loc[has, "p_model"], frame.loc[has, "p_mkt"],
                                                 weights["a"], weights["b"], weights["c"])
            # prefer true walk-forward p_blend_mkt when Part A already fitted that week
            if "p_blend_mkt" in frame.columns:
                use_wf = frame.get("blend_fitted", pd.Series(False, index=frame.index)).astype(str).isin(["True", "true", "1"])
                frame["p_blend_sec"] = np.where(use_wf & frame["p_blend_mkt"].notna(), frame["p_blend_mkt"], applied)
            else:
                frame["p_blend_sec"] = applied
            frame["p_blend_fav"] = np.where(frame["fav_side"] == "a", frame["p_blend_sec"], 1.0 - frame["p_blend_sec"])
    return frame.sort_values(["date_et", "scheduled_start", "event_ticker"]).reset_index(drop=True)


def Edges(frame, path, prob_col="p_fav"):
    """Add edge and P&L columns for one execution path. path is 'taker' or 'maker'."""
    out = frame.copy()
    out["path"] = path
    out["prob_source"] = "market_fav" if prob_col == "p_fav" else prob_col
    price = out["ask_fav"] if path == "taker" else out["bid_fav"]
    out["fill_price"] = price
    out["spread"] = out["t60_spread"]
    if path == "maker":
        out["filled"] = out["maker_filled"].astype(bool)
    else:
        out["filled"] = True
        out["fill_reason"] = "taker_at_ask"
    p = out[prob_col] if prob_col in out.columns else out["p_fav"]
    out["fair_value"] = p
    out["edge_tim_0.02"] = p - price - TIM_FEE
    out["fee_taker"] = price.map(KalshiFee)
    out["fee_maker_0"] = 0.0
    out["fee_maker_sens"] = price.map(MakerFeeSens)
    out["edge_taker_fee"] = p - price - out["fee_taker"]
    out["edge_maker_0"] = p - price - 0.0
    out["edge_maker_sens"] = p - price - out["fee_maker_sens"]
    out["qualifies"] = out["edge_tim_0.02"] >= QUAL - 1e-12
    won = out["y_fav"]
    for name, fee_col, suffix in (
        ("tim_est_0.02", None, "tim"),
        ("kalshi_taker_0.07p(1-p)", "fee_taker", "takerfee"),
        ("kalshi_maker_0", "fee_maker_0", "maker0"),
        ("kalshi_maker_0.0175p(1-p)", "fee_maker_sens", "makersens"),
    ):
        fee = TIM_FEE if fee_col is None else out[fee_col]
        cost = price + fee
        pnl = np.where(won == 1, 1.0 - cost, np.where(won == 0, -cost, np.nan))
        out[f"pnl_{suffix}"] = np.where(out["filled"] & np.isfinite(pnl), pnl, np.nan)
        out[f"cost_{suffix}"] = cost
    return out


def DayBlockMeanCi(values, days, ci=95):
    values = np.asarray(values, dtype=float)
    days = np.asarray(days)
    mask = np.isfinite(values)
    values, days = values[mask], days[mask]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    lo, hi = ClusterBootstrap(values, days, reps=BOOT_REPS, seed=BOOT_SEED,
                              percentiles=((100 - ci) / 2, 100 - (100 - ci) / 2))
    return float(values.mean()), float(lo), float(hi), int(len(values))


def HoldoutRow(part, path, prob_source, fee_label, pnl_col, edge_col, qualify=True):
    use = part.copy()
    if qualify:
        use = use[use["qualifies"]]
    scored = use[use["y_fav"].notna()]
    filled = scored[scored["filled"]]
    unfilled = scored[~scored["filled"]]
    mean_edge, lo, hi, n_edge = DayBlockMeanCi(use["edge_tim_0.02"].to_numpy(), use["date_et"].to_numpy())
    pnl = filled[pnl_col].dropna() if pnl_col in filled.columns else pd.Series(dtype=float)
    staked = filled[pnl_col.replace("pnl_", "cost_")].reindex(pnl.index) if len(pnl) else pd.Series(dtype=float)
    # cost column names: cost_tim, cost_takerfee, ...
    cost_col = "cost_" + pnl_col.split("_", 1)[1]
    staked = filled.loc[pnl.index, cost_col] if len(pnl) and cost_col in filled.columns else pd.Series(dtype=float)
    roi = float(pnl.sum() / staked.sum()) if len(pnl) and staked.sum() else float("nan")
    if len(pnl) > 1:
        rng = np.random.default_rng(BOOT_SEED)
        # day-block ROI
        tmp = filled.loc[pnl.index, ["date_et", pnl_col, cost_col]].dropna()
        _, roi_lo, roi_hi, _ = (float("nan"), float("nan"), float("nan"), 0)
        if tmp["date_et"].nunique() > 1:
            days = tmp["date_et"].unique()
            day_pnl = tmp.groupby("date_et")[pnl_col].sum()
            day_cost = tmp.groupby("date_et")[cost_col].sum()
            boots = []
            for _ in range(BOOT_REPS):
                pick = rng.choice(days, size=len(days), replace=True)
                num, den = day_pnl.loc[pick].sum(), day_cost.loc[pick].sum()
                if den:
                    boots.append(num / den)
            if boots:
                roi_lo, roi_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
        else:
            roi_lo = roi_hi = float("nan")
    else:
        roi_lo = roi_hi = float("nan")
    unfilled_roi = 0.0 if len(unfilled) else float("nan")
    fair = filled["fair_value"] if len(filled) else use["fair_value"]
    return {
        "date": HOLDOUT_START.date().isoformat(),
        "series": f"{path}_{prob_source}_{fee_label}_" + ("qual3c" if qualify else "all"),
        "path": path,
        "prob_source": prob_source,
        "fee_basis": fee_label,
        "value_line": "edge>=0.03_tim_est_0.02" if qualify else "all_fav",
        "split": "holdout",
        "n": int(len(scored)),
        "n_qual": int(use["y_fav"].notna().sum()) if qualify else int(len(scored)),
        "n_filled": int(len(filled)),
        "n_unfilled": int(len(unfilled)),
        "fill_rate": float(len(filled) / len(scored)) if len(scored) else float("nan"),
        "mean_edge_tim_0.02": mean_edge,
        "edge_ci95_lo": lo,
        "edge_ci95_hi": hi,
        "n_edge": n_edge,
        "median_spread": float(use["t60_spread"].median()) if len(use) else float("nan"),
        "mean_spread": float(use["t60_spread"].mean()) if len(use) else float("nan"),
        "filled_roi": roi,
        "filled_roi_ci95_lo": roi_lo,
        "filled_roi_ci95_hi": roi_hi,
        "unfilled_roi": unfilled_roi,
        "filled_pnl": float(pnl.sum()) if len(pnl) else 0.0,
        "filled_staked": float(staked.sum()) if len(staked) else 0.0,
        "fair_value_mean": float(fair.mean()) if len(fair.dropna()) else float("nan"),
        "fair_value_dollar": (f"${fair.mean():.2f}" if len(fair.dropna()) else ""),
    }


def Bankroll(part, strategy, pnl_col, cost_col, qualify=True, start=START_BANK):
    use = part[part["y_fav"].notna()].copy()
    if qualify:
        use = use[use["qualifies"]]
    use = use.sort_values(["date_et", "scheduled_start"])
    eq, peak, ruined = start, start, False
    x, bets_cum, rows, stakes, pnls = 0, 0, [], [], []
    for day, d in use.groupby("date_et", sort=True):
        x += 1
        sod, day_pnl, nb = eq, 0.0, 0
        if ruined or eq < FLAT:
            ruined = True
        else:
            for r in d.itertuples():
                if not bool(r.filled):
                    continue
                price = r.fill_price
                cost = getattr(r, cost_col)
                p = r.fair_value
                if strategy == "flat100":
                    stake = FLAT
                else:
                    frac = KELLY_FRAC * (p - cost) / (1.0 - cost) if cost < 1 else 0.0
                    if frac <= 0:
                        continue
                    stake = min(frac, KELLY_CAP) * sod
                won = r.y_fav
                pnl = stake / cost * (1.0 if won == 1 else 0.0) - stake
                day_pnl += pnl
                nb += 1
                stakes.append(stake)
                pnls.append(pnl)
        eq += day_pnl
        bets_cum += nb
        peak = max(peak, eq)
        rows.append({"date": day, "series": f"challenger_{part['path'].iloc[0]}_{strategy}_{pnl_col}",
                     "strategy": strategy, "path": part["path"].iloc[0],
                     "x": x, "bets_day": nb, "bets_cum": bets_cum,
                     "equity": round(eq, 2), "drawdown_pct": round(eq / peak - 1, 6)})
    stakes, pnls = np.asarray(stakes, float), np.asarray(pnls, float)
    roi = float(pnls.sum() / stakes.sum()) if len(stakes) else float("nan")
    return pd.DataFrame(rows), {
        "strategy": strategy, "path": part["path"].iloc[0] if len(part) else "",
        "pnl_col": pnl_col, "n_bets": int(len(stakes)),
        "final_equity": round(eq, 2), "staked": round(float(stakes.sum()), 2) if len(stakes) else 0.0,
        "pnl": round(float(pnls.sum()), 2) if len(stakes) else 0.0, "roi": roi,
        "max_drawdown_pct": min((r["drawdown_pct"] for r in rows), default=0.0),
        "ruined": ruined, "start_bank": start, "qualify": qualify,
    }


def WriteReport(out_dir, base, summary, bank_stats, coverage, fee_note, weights):
    def Table(frame, cols):
        if frame is None or len(frame) == 0:
            return "_none._"
        view = frame[cols]
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
        for row in view.itertuples(index=False):
            cells = []
            for value in row:
                if isinstance(value, float):
                    cells.append("" if pd.isna(value) else f"{value:.4f}")
                else:
                    cells.append("" if pd.isna(value) else str(value))
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines)

    primary = summary[(summary["prob_source"] == "market_fav") & (summary["fee_basis"] == "tim_est_0.02")]
    head = []
    for _, r in primary.iterrows():
        head.append(
            f"- **{r['path']}** {r['value_line']}: N={int(r['n'])}, filled={int(r['n_filled'])}, "
            f"unfilled={int(r['n_unfilled'])}, mean edge after $0.02 = {r['mean_edge_tim_0.02']:+.4f} "
            f"(95% day-block CI {r['edge_ci95_lo']:+.4f} to {r['edge_ci95_hi']:+.4f}), "
            f"filled ROI {r['filled_roi'] if pd.notna(r['filled_roi']) else float('nan'):+.1%} "
            f"(unfilled ROI {r['unfilled_roi'] if pd.notna(r['unfilled_roi']) else float('nan')}), "
            f"median spread {r['median_spread']:.3f}, fair value {r['fair_value_dollar']}."
        )
    bank_lines = Table(bank_stats, [c for c in bank_stats.columns if c in
                                    ("strategy", "path", "pnl_col", "qualify", "n_bets", "final_equity",
                                     "staked", "pnl", "roi", "max_drawdown_pct", "ruined")] if bank_stats is not None else [])
    now = dt.datetime.now(ET)
    hour = now.strftime("%I").lstrip("0") or "12"
    written = f"{now.strftime('%Y-%m-%d')} {hour}:{now.strftime('%M %p')} ET"
    md = f"""# ATP Challenger maker test (2026-11-13)

Author: Tennibot. Written {written} (America/New_York). Series **KXATPCHALLENGERMATCH** only.
Holdout frozen before results: scheduled start on or after **2026-08-13 00:00 America/New_York**.
Primary analysis is **model-free** (`market_fav`). Simulation only; no orders. Default stays **PASS**.

## Headline (holdout, market_fav)

{chr(10).join(head) if head else "_no holdout rows._"}

Coverage: {coverage}

## Fees

{fee_note}

## Holdout summary

{Table(summary, ["path", "prob_source", "fee_basis", "value_line", "n", "n_filled", "n_unfilled", "fill_rate",
                 "mean_edge_tim_0.02", "edge_ci95_lo", "edge_ci95_hi", "median_spread", "filled_roi",
                 "filled_roi_ci95_lo", "filled_roi_ci95_hi", "unfilled_roi", "fair_value_dollar"])}

## $10k bankrolls (holdout only, start $10,000, no season reset)

flat100 = $100 per filled qualifying bet. qkelly = 0.25 x (p - cost)/(1 - cost) of start-of-day equity, 2% cap.
Primary qualifying rule: edge after Tim's $0.02 >= 3c, market_fav side.

{bank_lines}

## Secondary: Part A blend

The walk-forward logistic blend from Part A does not choose the side. Frozen ATP weights used only to attach a secondary probability, when one exists (last pre-holdout week {weights}). If the holdout sits after the Sackmann / Part A window, many rows will have no blend probability; that absence is left as a missing value, not a reason to change the market_fav rule.

## Caveats

- **Default stays PASS.** This is a pre-registered holdout test, not a betting recommendation. A BET would still need a fee-adjusted edge whose holdout CI clears 0, plus an executable fill.
- Side rule is the de-vigged T-60 favourite. Ties (p = 0.5) go to player A. De-vig is proportional on the two asks.
- Taker fill is the favourite's T-60 ask, assumed available with unlimited size (candles have no size).
- Maker fill is the favourite's T-60 bid, and only if a later 1-min candle before scheduled start trades at or through that bid (`last` through the bid, or the ask crossing the bid). Unfilled makers have no P&L; ROI is reported filled vs unfilled (unfilled ROI is 0).
- T-60 is last 1-min candle at or before scheduled start minus 60 minutes, age <= 30 minutes. Events with a mid jump > 0.25 between T-60 and T-5 are dropped as possible in-play.
- Edge for the value line is always `prob - (fill + 0.02)`, Tim's estimate, labelled as an estimate. Kalshi's verified taker fee and the maker-fee cases are extra columns, never a silent replacement.
- Official scores use the holdout only. Tune months (before 2026-08-13 ET) were not used to pick the boundary, the side rule, the fill rule, the 3c line, or the fee basis.
- Fair value is the de-vigged favourite probability, written as $0.xx to win $1.
- Simulation only. Tim places any wager himself.

Files under `{out_dir}`: `{base}.md`, `{base}_preregister.md`, `{base}_events.csv`, `{base}_summary.csv`, `{base}_bankroll.csv`, `{base}_bankroll_stats.csv`.
"""
    path = out_dir / f"{base}.md"
    path.write_text(md, encoding="utf-8")
    return path


def CoverageNote(frame):
    if len(frame) == 0:
        return "no events with a qualifying T-60 quote."
    hold = frame[frame["split"] == "holdout"]
    tune = frame[frame["split"] == "tune"]
    settled_h = hold["y_fav"].notna().sum()
    return (f"tune events with T-60: {len(tune):,}; holdout with T-60: {len(hold):,} "
            f"(settled holdout {int(settled_h):,}); date range {frame['date_et'].min()} to {frame['date_et'].max()} ET; "
            f"in-play flagged and dropped: {int(frame['possible_inplay'].sum()):,}.")


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base", default="2026-11-13_challenger-maker")
    args = parser.parse_args(argv)
    frame = BuildEvents()
    usable = frame[~frame["possible_inplay"]].copy() if len(frame) else frame
    hold = usable[(usable["split"] == "holdout") & usable["y_fav"].notna()].copy()
    # primary: market_fav on taker and maker
    taker = Edges(hold, "taker", "p_fav")
    maker = Edges(hold, "maker", "p_fav")
    rows = []
    for part, path in ((taker, "taker"), (maker, "maker")):
        for qualify in (False, True):
            rows.append(HoldoutRow(part, path, "market_fav", "tim_est_0.02", "pnl_tim", "edge_tim_0.02", qualify))
        if path == "taker":
            rows.append(HoldoutRow(part, path, "market_fav", "kalshi_taker_0.07p(1-p)", "pnl_takerfee", "edge_taker_fee", True))
        else:
            rows.append(HoldoutRow(part, path, "market_fav", "kalshi_maker_0", "pnl_maker0", "edge_maker_0", True))
            rows.append(HoldoutRow(part, path, "market_fav", "kalshi_maker_0.0175p(1-p)", "pnl_makersens", "edge_maker_sens", True))
    # secondary blend, only rows with a blend fav prob, does not change market_fav sides
    if "p_blend_fav" in hold.columns and hold["p_blend_fav"].notna().any():
        for path, base in (("taker", taker), ("maker", maker)):
            sub = base[base["p_blend_fav"].notna()].copy()
            if len(sub) == 0:
                continue
            # edges still from market_fav side and prices; only the displayed fair-value/edge uses blend as a side note
            alt = Edges(hold[hold["p_blend_fav"].notna()], path, "p_blend_fav")
            rows.append(HoldoutRow(alt, path, "blend_secondary", "tim_est_0.02", "pnl_tim", "edge_tim_0.02", True))
    summary = pd.DataFrame(rows)
    curves, stats = [], []
    for part, fee_name, pnl_col, cost_col in (
        (taker, "tim_est_0.02", "pnl_tim", "cost_tim"),
        (taker, "kalshi_taker_0.07p(1-p)", "pnl_takerfee", "cost_takerfee"),
        (maker, "tim_est_0.02", "pnl_tim", "cost_tim"),
        (maker, "kalshi_maker_0", "pnl_maker0", "cost_maker0"),
        (maker, "kalshi_maker_0.0175p(1-p)", "pnl_makersens", "cost_makersens"),
    ):
        for strategy in ("flat100", "qkelly"):
            for qualify in (True, False):
                curve, st = Bankroll(part, strategy, pnl_col, cost_col, qualify=qualify)
                st["fee_basis"] = fee_name
                st["prob_source"] = "market_fav"
                if len(curve):
                    curve["fee_basis"] = fee_name
                    curve["qualify"] = qualify
                    curves.append(curve)
                stats.append(st)
    bank_curve = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    bank_stats = pd.DataFrame(stats)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    usable.to_csv(args.out_dir / f"{args.base}_events.csv", index=False)
    summary.to_csv(args.out_dir / f"{args.base}_summary.csv", index=False)
    bank_curve.to_csv(args.out_dir / f"{args.base}_bankroll.csv", index=False)
    bank_stats.to_csv(args.out_dir / f"{args.base}_bankroll_stats.csv", index=False)
    fee_note = (
        "GET /series/KXATPCHALLENGERMATCH (2026-11-13, this run) returned `fee_type=quadratic`, "
        "`fee_multiplier=1`. GET /series/fee_changes?series_ticker=KXATPCHALLENGERMATCH returned an empty "
        "array (no scheduled or historical changes). For comparison, GET /series/KXATPMATCH returned "
        "`fee_type=quadratic_with_maker_fees`, `fee_multiplier=1`. Kalshi's published fee schedule "
        "(https://kalshi.com/fee-schedule and the July 2026 PDF at https://kalshi.com/docs/kalshi-fee-schedule.pdf) "
        "charges takers `round up(M x 0.07 x C x P x (1-P))` and, only on series in the Maker Fees section, "
        "charges makers `round up(M x 0.0175 x C x P x (1-P))` when a resting order fills. The `quadratic` type "
        "is taker-only (makers pay 0). That is the confirmed Challenger case. The 0.0175*P*(1-P) column is a "
        "sensitivity, not the schedule for this series. Tim's $0.02 remains the official edge estimate, labelled as such."
    )
    weights = FrozenBlendWeights()
    weight_note = f"{weights['as_of']} (a={weights['a']:.3f}, b={weights['b']:.3f}, c={weights['c']:.3f})" if weights else "none available"
    WriteReport(args.out_dir, args.base, summary, bank_stats, CoverageNote(usable), fee_note, weight_note)
    print(summary.to_string(index=False))
    print(bank_stats.to_string(index=False))
    print("coverage:", CoverageNote(usable))
    print("wrote", args.out_dir)


if __name__ == "__main__":
    Main()