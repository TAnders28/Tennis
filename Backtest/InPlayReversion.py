"""Pre-registered model-free test of 1-min mid jumps on KXATPCHALLENGERMATCH candles already on disk.

Hypothesis: thin Challenger books over- or underreact to big state changes, visible as 1-min mid
jumps. Official design is frozen in
E:\\AgentDocs\\Tennibot\\2026-11-13_inplay-reversion\\2026-11-13_inplay-reversion_preregister.md.
Do not change thresholds, holdout, fill rules, or the verdict bar after seeing results.
Fetch nothing.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from Backtest.ChallengerMaker import HOLDOUT_START, MakerFilled
from Backtest.Metrics import ClusterBootstrap
from Backtest.Odds import KalshiFee, TIM_FEE
from Backtest.SpaTest import HansenSpa, WhiteRealityCheck
from DataPull.Http import RAW_DIR

ET = ZoneInfo("America/New_York")
SERIES = "KXATPCHALLENGERMATCH"
CANDLE_DIR = RAW_DIR / "kalshi" / "candles" / SERIES
DEFAULT_OUT = Path(r"E:\AgentDocs\Tennibot\2026-11-13_inplay-reversion")
PREREG = DEFAULT_OUT / "2026-11-13_inplay-reversion_preregister.md"
THRESHOLDS = (0.08, 0.12, 0.20)
HORIZONS = (1, 3, 5, 10)
SIDES = ("continuation", "reversion")
FILLS = ("taker", "maker")
PRIMARY_X = 0.08
QUAL = 0.03
START_BANK = 10_000.0
FLAT = 100.0
KELLY_FRAC = 0.25
KELLY_CAP = 0.02
STOP_EQ = 100.0
SPA_REPS = 10_000
CI_REPS = 10_000
MC_PATHS = 10_000
SEED = 20261113
# Ending equity at the 5th percentile below this is a "large loss" for the BET bar.
MC_LARGE_LOSS = 8_000.0


def CellName(x, horizon, side, fill):
    return f"x{x:.2f}_h{horizon}_{side}_{fill}"


def AllCells():
    return [CellName(x, h, s, f) for x in THRESHOLDS for h in HORIZONS for s in SIDES for f in FILLS]


def Sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def ValidBook(bid, ask):
    return bid is not None and ask is not None and 0.0 < bid < ask < 1.0


def Mid(bid, ask):
    return 0.5 * (bid + ask)


def ToEt(ts):
    if ts is None:
        return None
    if isinstance(ts, str):
        raw = dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))
    elif isinstance(ts, (int, float)):
        raw = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    else:
        raw = ts
    if raw.tzinfo is None:
        raw = raw.replace(tzinfo=dt.timezone.utc)
    return raw.astimezone(ET)


def BuyYes(delta, side):
    up = delta > 0
    return (up and side == "continuation") or ((not up) and side == "reversion")


def TakerPnl(buy_yes, jump_bid, jump_ask, exit_bid, exit_ask, fee):
    if buy_yes:
        return exit_bid - jump_ask - fee
    return jump_bid - exit_ask - fee


def MakerPnl(buy_yes, jump_bid, jump_ask, exit_bid, exit_ask, fee):
    if buy_yes:
        return exit_bid - jump_bid - fee
    return jump_ask - exit_ask - fee


def EntryCost(buy_yes, jump_bid, jump_ask, fill, official=True):
    if fill == "taker":
        px = jump_ask if buy_yes else (1.0 - jump_bid)
        return px + TIM_FEE
    px = jump_bid if buy_yes else (1.0 - jump_ask)
    return px + (0.0 if official else TIM_FEE)


def ExitAt(minutes, jump_end, horizon):
    lo = jump_end + 60 * horizon
    hi = lo + 60
    for candle in minutes:
        end = candle.get("end_ts")
        if end is None or end < lo:
            continue
        if end > hi:
            break
        if ValidBook(candle.get("yes_bid"), candle.get("yes_ask")):
            return candle
    return None


def SortMinutes(raw):
    out = []
    seen = {}
    for candle in raw or []:
        end = candle.get("end_ts")
        if end is None:
            continue
        seen[int(end)] = candle
    for end in sorted(seen):
        out.append(seen[end])
    return out
def ExtractJumps(payload):
    """Return (official_jumps, diagnostics) from one on-disk event payload. No I/O."""
    minutes = SortMinutes(payload.get("minute"))
    start_ts = payload.get("scheduled_start")
    start_et = ToEt(start_ts)
    start_unix = int(start_et.timestamp()) if start_et is not None else None
    event = payload.get("event_ticker") or payload.get("ticker")
    ticker = payload.get("ticker")
    split = "holdout" if (start_et is not None and start_et >= HOLDOUT_START) else "tune"
    date_et = start_et.date().isoformat() if start_et is not None else None
    diag = {
        "event_ticker": event,
        "ticker": ticker,
        "n_minutes": len(minutes),
        "n_valid_book": 0,
        "n_empty_or_invalid": 0,
        "n_gaps": 0,
        "n_official_jumps": 0,
        "n_stale_jumps": 0,
        "n_pre_start": 0,
        "n_post_start": 0,
        "split": split,
        "date_et": date_et,
        "scheduled_start": start_ts,
        "has_start": start_et is not None,
    }
    if len(minutes) >= 2:
        span = minutes[-1]["end_ts"] - minutes[0]["end_ts"]
        expected = (span // 60) + 1
        diag["n_missing_slots"] = max(0, int(expected) - len(minutes))
    else:
        diag["n_missing_slots"] = 0
    official = []
    prev_valid = None
    for candle in minutes:
        bid, ask = candle.get("yes_bid"), candle.get("yes_ask")
        if not ValidBook(bid, ask):
            diag["n_empty_or_invalid"] += 1
            prev_valid = None
            continue
        diag["n_valid_book"] += 1
        mid = Mid(bid, ask)
        if prev_valid is not None:
            gap = int(candle["end_ts"]) - int(prev_valid["end_ts"])
            if gap > 60:
                diag["n_gaps"] += 1
            if gap != 60:
                delta = mid - Mid(prev_valid["yes_bid"], prev_valid["yes_ask"])
                if abs(delta) >= PRIMARY_X:
                    diag["n_stale_jumps"] += 1
            else:
                pbid, pask = prev_valid["yes_bid"], prev_valid["yes_ask"]
                pmid = Mid(pbid, pask)
                delta = mid - pmid
                if abs(delta) >= PRIMARY_X:
                    jump = ScoreJump(
                        minutes, candle, bid, ask, mid, pbid, pask, pmid, delta,
                        event, ticker, start_unix, start_et, split, date_et,
                    )
                    official.append(jump)
                    diag["n_official_jumps"] += 1
                    if jump["timing"] == "pre_start":
                        diag["n_pre_start"] += 1
                    else:
                        diag["n_post_start"] += 1
        prev_valid = candle
    return official, diag


def ScoreJump(minutes, candle, bid, ask, mid, pbid, pask, pmid, delta,
              event, ticker, start_unix, start_et, split, date_et):
    end = int(candle["end_ts"])
    if start_unix is None:
        timing = "unknown"
    else:
        timing = "pre_start" if end <= start_unix else "post_start"
    row = {
        "event_ticker": event,
        "ticker": ticker,
        "date_et": date_et,
        "split": split,
        "timing": timing,
        "jump_end": end,
        "mid_prev": pmid,
        "mid": mid,
        "delta": delta,
        "abs_delta": abs(delta),
        "bid": bid,
        "ask": ask,
        "spread": ask - bid,
        "last": candle.get("last"),
        "volume": candle.get("volume"),
        "gap_s": 60,
        "start_et": start_et.isoformat() if start_et is not None else None,
    }
    for horizon in HORIZONS:
        ext = ExitAt(minutes, end, horizon)
        prefix = f"h{horizon}_"
        if ext is None:
            row[prefix + "exit_ok"] = False
            row[prefix + "exit_end"] = None
            row[prefix + "exit_bid"] = None
            row[prefix + "exit_ask"] = None
            row[prefix + "exit_mid"] = None
            continue
        ebid, eask = ext["yes_bid"], ext["yes_ask"]
        row[prefix + "exit_ok"] = True
        row[prefix + "exit_end"] = int(ext["end_ts"])
        row[prefix + "exit_bid"] = ebid
        row[prefix + "exit_ask"] = eask
        row[prefix + "exit_mid"] = Mid(ebid, eask)
        for side in SIDES:
            buy_yes = BuyYes(delta, side)
            fee_k = KalshiFee(ask if buy_yes else (1.0 - bid))
            row[f"{prefix}{side}_taker_pnl"] = TakerPnl(buy_yes, bid, ask, ebid, eask, TIM_FEE)
            row[f"{prefix}{side}_taker_pnl_kalshi"] = TakerPnl(buy_yes, bid, ask, ebid, eask, fee_k)
            row[f"{prefix}{side}_taker_cost"] = EntryCost(buy_yes, bid, ask, "taker", True)
            row[f"{prefix}{side}_taker_fee_kalshi"] = fee_k
            filled, reason = MakerFilled(
                minutes, quote_end_ts=end, start_ts=int(ext["end_ts"]),
                side=("a" if buy_yes else "b"), t60_bid=bid, t60_ask=ask,
            )
            row[f"{prefix}{side}_maker_filled"] = filled
            row[f"{prefix}{side}_maker_reason"] = reason
            if filled:
                row[f"{prefix}{side}_maker_pnl"] = MakerPnl(buy_yes, bid, ask, ebid, eask, 0.0)
                row[f"{prefix}{side}_maker_pnl_tim02"] = MakerPnl(buy_yes, bid, ask, ebid, eask, TIM_FEE)
                row[f"{prefix}{side}_maker_cost"] = EntryCost(buy_yes, bid, ask, "maker", True)
            else:
                row[f"{prefix}{side}_maker_pnl"] = None
                row[f"{prefix}{side}_maker_pnl_tim02"] = None
                row[f"{prefix}{side}_maker_cost"] = EntryCost(buy_yes, bid, ask, "maker", True)
    return row


def ProcessEventPath(path_str):
    try:
        payload = json.loads(Path(path_str).read_text(encoding="utf-8"))
        official, diag = ExtractJumps(payload)
        diag["path"] = path_str
        diag["error"] = None
        return official, diag
    except Exception as exc:  # noqa: BLE001 - keep a worker from killing the pool
        return [], {"path": path_str, "error": f"{type(exc).__name__}: {exc}",
                    "n_minutes": 0, "n_official_jumps": 0, "n_stale_jumps": 0,
                    "n_valid_book": 0, "n_empty_or_invalid": 0, "n_gaps": 0,
                    "n_missing_slots": 0, "n_pre_start": 0, "n_post_start": 0,
                    "has_start": False, "split": None, "date_et": None}


def LoadAllJumps(candle_dir, workers=7):
    paths = sorted(str(p) for p in Path(candle_dir).glob("*.json"))
    official, diags = [], []
    if not paths:
        return official, diags
    workers = max(1, int(workers))
    if workers == 1 or len(paths) < 8:
        for path in paths:
            jumps, diag = ProcessEventPath(path)
            official.extend(jumps)
            diags.append(diag)
        return official, diags
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(ProcessEventPath, path) for path in paths]
        for fut in as_completed(futs):
            jumps, diag = fut.result()
            official.extend(jumps)
            diags.append(diag)
    return official, diags
def JumpsFrame(official):
    if not official:
        return pd.DataFrame()
    return pd.DataFrame(official)


def CellSeries(frame, x, horizon, side, fill):
    """Return scored official P&L (NaN if not scored) for one cell, aligned to frame rows."""
    if frame.empty:
        return pd.Series(dtype=float)
    ok = frame["abs_delta"] >= (x - 1e-15)
    prefix = f"h{horizon}_{side}_"
    if fill == "taker":
        pnl = frame[prefix + "taker_pnl"].astype(float)
        exit_ok = frame[f"h{horizon}_exit_ok"].astype(bool)
        scored = ok & exit_ok & pnl.notna()
        out = pnl.where(scored)
    else:
        pnl = frame[prefix + "maker_pnl"].astype(float)
        filled = frame[prefix + "maker_filled"].fillna(False).astype(bool)
        exit_ok = frame[f"h{horizon}_exit_ok"].astype(bool)
        scored = ok & exit_ok & filled & pnl.notna()
        out = pnl.where(scored)
    return out


def CellCost(frame, x, horizon, side, fill):
    ok = frame["abs_delta"] >= (x - 1e-15)
    prefix = f"h{horizon}_{side}_"
    cost = frame[prefix + f"{fill}_cost"].astype(float)
    return cost.where(ok)


def SummarizeCell(frame, x, horizon, side, fill, split, timing, reps=CI_REPS, seed=SEED):
    sub = frame
    if split:
        sub = sub[sub["split"] == split]
    if timing and timing != "all":
        sub = sub[sub["timing"] == timing]
    name = CellName(x, horizon, side, fill)
    n_jumps = int((sub["abs_delta"] >= (x - 1e-15)).sum()) if not sub.empty else 0
    pnl = CellSeries(sub, x, horizon, side, fill) if not sub.empty else pd.Series(dtype=float)
    scored = pnl.dropna()
    n_scored = int(len(scored))
    prefix = f"h{horizon}_{side}_"
    n_missing_exit = 0
    n_unfilled = 0
    if not sub.empty:
        meet = sub["abs_delta"] >= (x - 1e-15)
        n_missing_exit = int((meet & ~sub[f"h{horizon}_exit_ok"].astype(bool)).sum())
        if fill == "maker":
            n_unfilled = int((meet & sub[f"h{horizon}_exit_ok"].astype(bool) & ~sub[prefix + "maker_filled"].fillna(False).astype(bool)).sum())
    mean = float(scored.mean()) if n_scored else float("nan")
    ci_lo = ci_hi = float("nan")
    n_days = 0
    if n_scored and "date_et" in sub.columns:
        part = sub.loc[scored.index]
        days = part["date_et"].astype(str).to_numpy()
        n_days = int(pd.unique(days).size)
        if n_days >= 2:
            ci_lo, ci_hi = ClusterBootstrap(
                scored.to_numpy(), days, stat=np.mean, reps=reps, seed=seed,
                percentiles=(2.5, 97.5),
            )
    spreads = sub.loc[sub["abs_delta"] >= (x - 1e-15), "spread"] if not sub.empty else pd.Series(dtype=float)
    med_spread = float(spreads.median()) if len(spreads) else float("nan")
    med_jump = float(sub.loc[sub["abs_delta"] >= (x - 1e-15), "abs_delta"].median()) if n_jumps else float("nan")
    return {
        "cell": name, "x": x, "horizon": horizon, "side": side, "fill": fill,
        "split": split or "all", "timing": timing or "all",
        "n_jumps": n_jumps, "n_scored": n_scored, "n_unfilled": n_unfilled,
        "n_missing_exit": n_missing_exit, "n_days": n_days,
        "mean_edge": mean, "ci_lo": ci_lo, "ci_hi": ci_hi,
        "median_spread": med_spread, "median_abs_delta": med_jump,
        "qualify_3c": bool(n_scored > 0 and mean >= QUAL),
    }


def SummaryTable(frame, splits=("tune", "holdout"), timings=("all", "pre_start", "post_start"),
                 reps=CI_REPS, seed=SEED):
    rows = []
    for split in splits:
        for timing in timings:
            for x in THRESHOLDS:
                for horizon in HORIZONS:
                    for side in SIDES:
                        for fill in FILLS:
                            rows.append(SummarizeCell(frame, x, horizon, side, fill, split, timing, reps, seed))
    return pd.DataFrame(rows)


def SpaPanel(frame):
    """Holdout official jumps (|d|>=0.08). Untraded / unfilled / missing exit -> 0 for SPA/RC only."""
    hold = frame[frame["split"] == "holdout"].copy()
    if hold.empty:
        return hold, np.zeros((0, 0)), []
    names = AllCells()
    cols = []
    for name in names:
        # parse back
        _, rest = name.split("_", 1)
        # name is x0.08_h1_continuation_taker
        parts = name.split("_")
        x = float(parts[0][1:])
        horizon = int(parts[1][1:])
        fill = parts[-1]
        side = "continuation" if "continuation" in name else "reversion"
        pnl = CellSeries(hold, x, horizon, side, fill)
        cols.append(pnl.fillna(0.0).to_numpy())
    d = np.column_stack(cols) if cols else np.zeros((len(hold), 0))
    return hold, d, names


def TuneCandidates(summary):
    tune = summary[(summary["split"] == "tune") & (summary["timing"] == "all")]
    return tune[tune["qualify_3c"]].copy()


def KellyFrac(mu, var):
    if not (mu > 0) or not (var > 0) or not math.isfinite(mu) or not math.isfinite(var):
        return 0.0
    return min(KELLY_CAP, max(0.0, KELLY_FRAC * mu / var))


def TuneReturnMoments(frame, x, horizon, side, fill):
    tune = frame[frame["split"] == "tune"]
    pnl = CellSeries(tune, x, horizon, side, fill)
    cost = CellCost(tune, x, horizon, side, fill)
    ok = pnl.notna() & cost.notna() & (cost > 1e-9)
    if not ok.any():
        return float("nan"), float("nan")
    ret = (pnl[ok] / cost[ok]).to_numpy()
    if len(ret) < 2:
        return float(ret.mean()) if len(ret) else float("nan"), float("nan")
    return float(ret.mean()), float(ret.var(ddof=1))


def TradesForCell(frame, x, horizon, side, fill, split="holdout"):
    sub = frame[frame["split"] == split].copy()
    if sub.empty:
        return sub
    pnl = CellSeries(sub, x, horizon, side, fill)
    cost = CellCost(sub, x, horizon, side, fill)
    sub = sub.assign(pnl=pnl, cost=cost)
    sub = sub[sub["pnl"].notna() & sub["cost"].notna() & (sub["cost"] > 1e-9)]
    return sub.sort_values(["date_et", "jump_end", "event_ticker"])


def ReplayPath(day_groups, pick, start, kind, kelly_f):
    eq = start
    peak = start
    trough = start
    n_bets = 0
    ruined = False
    for idx in pick:
        day = day_groups[int(idx)]
        if eq < STOP_EQ:
            ruined = True
            break
        sod = eq
        if kind == "flat100":
            stake = FLAT if eq >= STOP_EQ else 0.0
        else:
            stake = kelly_f * sod
        if stake <= 0:
            continue
        for cost, pnl in day:
            if eq < STOP_EQ:
                ruined = True
                break
            contracts = math.floor((stake / cost) * 1e9) / 1e9  # keep fractional contracts
            if contracts <= 0:
                continue
            eq = eq + contracts * pnl
            n_bets += 1
            if eq > peak:
                peak = eq
            if eq < trough:
                trough = eq
        if ruined:
            break
    if eq < STOP_EQ:
        ruined = True
    return eq, trough, n_bets, ruined


def DayGroups(trades):
    groups, days = [], []
    for day, part in trades.groupby("date_et", sort=True):
        groups.append(list(zip(part["cost"].to_numpy(), part["pnl"].to_numpy())))
        days.append(day)
    return groups, days


def MonteCarlo(trades, kind, kelly_f, paths=MC_PATHS, seed=SEED, start=START_BANK):
    if trades is None or trades.empty:
        return {
            "style": kind, "n_paths": int(paths), "n_days": 0, "n_trades": 0,
            "chrono_end": start, "chrono_trough": start, "chrono_bets": 0,
            "mean_end": start, "p5": start, "p50": start, "p95": start,
            "p_ruin": 0.0, "kelly_f": kelly_f,
        }
    groups, _days = DayGroups(trades)
    n_days = len(groups)
    chrono_pick = np.arange(n_days)
    chrono = ReplayPath(groups, chrono_pick, start, kind, kelly_f)
    rng = np.random.default_rng(seed)
    ends = np.empty(paths)
    ruins = 0
    for i in range(paths):
        pick = rng.integers(0, n_days, size=n_days)
        eq, _trough, _nb, ruined = ReplayPath(groups, pick, start, kind, kelly_f)
        ends[i] = eq
        ruins += int(ruined)
    return {
        "style": kind, "n_paths": int(paths), "n_days": n_days, "n_trades": int(len(trades)),
        "chrono_end": chrono[0], "chrono_trough": chrono[1], "chrono_bets": chrono[2],
        "mean_end": float(ends.mean()), "p5": float(np.percentile(ends, 5)),
        "p50": float(np.percentile(ends, 50)), "p95": float(np.percentile(ends, 95)),
        "p_ruin": ruins / float(paths), "kelly_f": kelly_f,
    }


def ChronoCurve(trades, kind, kelly_f, start=START_BANK):
    rows = []
    if trades is None or trades.empty:
        return pd.DataFrame(rows)
    eq = start
    sod_date = None
    sod = start
    stake = 0.0
    for rec in trades.itertuples(index=False):
        if rec.date_et != sod_date:
            sod_date = rec.date_et
            sod = eq
            stake = FLAT if kind == "flat100" else kelly_f * sod
        if eq < STOP_EQ:
            break
        if stake <= 0:
            continue
        contracts = stake / rec.cost
        eq = eq + contracts * rec.pnl
        rows.append({
            "date_et": rec.date_et, "event_ticker": rec.event_ticker, "jump_end": rec.jump_end,
            "style": kind, "equity": eq, "stake": stake, "cost": rec.cost, "pnl": rec.pnl,
        })
    return pd.DataFrame(rows)
def Coverage(frame, diags):
    d = pd.DataFrame(diags) if diags else pd.DataFrame()
    n_files = len(d)
    n_err = int(d["error"].notna().sum()) if (not d.empty and "error" in d.columns) else 0
    n_min = int((d["n_minutes"] > 0).sum()) if not d.empty else 0
    n_off = int(d["n_official_jumps"].sum()) if not d.empty else 0
    n_stale = int(d["n_stale_jumps"].sum()) if not d.empty else 0
    n_gaps = int(d["n_gaps"].sum()) if not d.empty else 0
    n_missing = int(d["n_missing_slots"].sum()) if not d.empty else 0
    n_empty = int(d["n_empty_or_invalid"].sum()) if not d.empty else 0
    n_valid = int(d["n_valid_book"].sum()) if not d.empty else 0
    if frame is None or frame.empty:
        return (f"candle files {n_files:,}; with minutes {n_min:,}; read errors {n_err:,}; "
                f"official jumps {n_off:,}; stale/gapped |d|>=0.08 excluded {n_stale:,}; "
                f"gap counts {n_gaps:,}; missing 1-min slots {n_missing:,}; "
                f"valid-book minutes {n_valid:,}; empty/invalid minutes {n_empty:,}. No official jumps.")
    n_tune = int((frame["split"] == "tune").sum())
    n_hold = int((frame["split"] == "holdout").sum())
    n_pre = int((frame["timing"] == "pre_start").sum())
    n_post = int((frame["timing"] == "post_start").sum())
    dates = frame["date_et"].dropna()
    med_sp = float(frame["spread"].median()) if len(frame) else float("nan")
    return (f"candle files {n_files:,}; with minutes {n_min:,}; read errors {n_err:,}; "
            f"official gap_s=60 jumps |d|>=0.08: {len(frame):,} "
            f"(tune {n_tune:,}, holdout {n_hold:,}; pre_start {n_pre:,}, post_start {n_post:,}); "
            f"stale/gapped |d|>=0.08 excluded {n_stale:,}; gap counts {n_gaps:,}; "
            f"missing 1-min slots {n_missing:,}; valid-book minutes {n_valid:,}; "
            f"empty/invalid minutes {n_empty:,}; date_et {dates.min()} to {dates.max()}; "
            f"median jump spread ${med_sp:.2f}.")


def FmtEdge(x):
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return "n/a"
    return f"${x:+.4f}"


def FmtMoney(x):
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(x):
        return "n/a"
    return f"${x:,.2f}"


def Verdict(spa, rc, candidates, hold_summary, bank_stats):
    """Frozen BET bar: SPA or RC rejects 0 at 5% AND a 3c cell has holdout CI above 0 AND MC p5 is not a large loss."""
    spa_p = spa.get("p_value") if spa else None
    rc_p = rc.get("p_value") if rc else None
    test_ok = (spa_p is not None and spa_p < 0.05) or (rc_p is not None and rc_p < 0.05)
    ci_ok = False
    ci_cells = []
    if candidates is not None and not candidates.empty and hold_summary is not None:
        hold = hold_summary[(hold_summary["split"] == "holdout") & (hold_summary["timing"] == "all")]
        for cell in candidates["cell"]:
            row = hold[hold["cell"] == cell]
            if row.empty:
                continue
            lo = float(row.iloc[0]["ci_lo"])
            hi = float(row.iloc[0]["ci_hi"])
            if math.isfinite(lo) and lo > 0 and math.isfinite(hi):
                ci_ok = True
                ci_cells.append(cell)
    mc_ok = False
    mc_notes = []
    if bank_stats is not None and not bank_stats.empty and candidates is not None and not candidates.empty:
        for rec in bank_stats.itertuples(index=False):
            if rec.cell not in set(candidates["cell"]):
                continue
            p5 = float(rec.p5)
            large = (not math.isfinite(p5)) or p5 < MC_LARGE_LOSS
            mc_notes.append((rec.cell, rec.style, p5, large))
            if not large:
                mc_ok = True
    # BET requires the test rejection, a qualifying CI, and that at least one 3c-cell MC style is not a large loss.
    # If there are no 3c candidates, BET is impossible.
    has_cand = candidates is not None and not candidates.empty
    if test_ok and ci_ok and mc_ok and has_cand:
        verdict = "BET"
    else:
        verdict = "PASS"
    reasons = []
    reasons.append(f"SPA p={spa_p if spa_p is not None else 'n/a'}, RC p={rc_p if rc_p is not None else 'n/a'}; 5% rejection={'yes' if test_ok else 'no'}.")
    if not has_cand:
        reasons.append("No tune cell reached a mean official edge of 3c or more.")
    else:
        reasons.append("Tune 3c cells: " + ", ".join(candidates["cell"].tolist()) + ".")
        reasons.append("Holdout CI entirely above 0: " + (", ".join(ci_cells) if ci_cells else "none") + ".")
    if mc_notes:
        bits = [f"{c} {s} p5={FmtMoney(p)}{' LARGE LOSS' if lg else ''}" for c, s, p, lg in mc_notes]
        reasons.append("MC 5th percentiles (large loss = p5 < $8,000): " + "; ".join(bits) + ".")
    else:
        reasons.append("No $10k MC paths on a 3c-candidate cell.")
    reasons.append("Default is PASS unless every frozen BET condition holds.")
    return verdict, " ".join(reasons)


def Table(df, cols, n=80):
    if df is None or df.empty:
        return "_none._"
    use = [c for c in cols if c in df.columns]
    show = df[use].copy()
    for col in ("mean_edge", "ci_lo", "ci_hi", "median_spread", "median_abs_delta"):
        if col in show.columns:
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    for col in ("p5", "p50", "p95", "mean_end", "chrono_end", "chrono_trough"):
        if col in show.columns:
            show[col] = show[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    if "p_value" in show.columns:
        show["p_value"] = show["p_value"].map(lambda v: "" if pd.isna(v) else f"{v:.4f}")
    if "p_ruin" in show.columns:
        show["p_ruin"] = show["p_ruin"].map(lambda v: "" if pd.isna(v) else f"{v:.3f}")
    return show.head(n).to_string(index=False)


def WriteReport(out_dir, base, coverage, summary, spa, rc, spa_names, candidates, bank_stats, pre_sha, extra):
    hold = summary[(summary["split"] == "holdout") & (summary["timing"] == "all")].copy()
    hold = hold.sort_values(["mean_edge", "n_scored"], ascending=[False, False])
    tune = summary[(summary["split"] == "tune") & (summary["timing"] == "all")].copy()
    tune = tune.sort_values(["mean_edge", "n_scored"], ascending=[False, False])
    pre = summary[(summary["split"] == "holdout") & (summary["timing"] == "pre_start")].copy()
    post = summary[(summary["split"] == "holdout") & (summary["timing"] == "post_start")].copy()
    verdict, why = Verdict(spa, rc, candidates, summary, bank_stats)
    spa_p_s = f"{spa['p_value']:.4f}" if spa else 'n/a'
    rc_p_s = f"{rc['p_value']:.4f}" if rc else 'n/a'
    spa_mean_s = FmtEdge(spa.get('mean_best') if spa else None)
    spa_t_s = f"{spa['t_obs']:.4f}" if spa else 'n/a'
    rc_t_s = f"{rc['t_obs']:.4f}" if rc else 'n/a'
    spa_best = spa_names[spa["best_index"]] if spa and spa_names else "n/a"
    rc_best = spa_names[rc["best_index"]] if rc and spa_names else "n/a"
    cand_list = ", ".join(candidates["cell"].tolist()) if candidates is not None and not candidates.empty else "none"
    cols = ["cell", "x", "horizon", "side", "fill", "n_jumps", "n_scored", "n_unfilled",
            "n_missing_exit", "n_days", "mean_edge", "ci_lo", "ci_hi", "median_spread", "qualify_3c"]
    bcols = ["cell", "style", "n_trades", "n_days", "n_paths", "kelly_f", "chrono_end",
             "chrono_trough", "mean_end", "p5", "p50", "p95", "p_ruin"]
    text = f"""# In-play mid-jump continuation vs reversion (KXATPCHALLENGERMATCH)

Written {dt.datetime.now(ET).strftime('%Y-%m-%d %I:%M:%S %p')} America/New_York.

Pre-registration: `{PREREG}` (SHA256 `{pre_sha}`). The holdout (scheduled start on or after **2026-08-13 00:00 America/New_York**), thresholds, horizons, sides, fills, fees, and the BET bar were frozen in that file before any jump or P&L numbers were computed.

## Verdict

**{verdict}**

{why}

This is a pre-registered holdout test, not a betting recommendation. No orders or wagers were placed.

## Headline

- Coverage: {coverage}
- Hansen SPA vs 0 (10,000 day-block bootstraps, holdout panel with untraded rows as 0): p={spa_p_s}, best={spa_best}, mean_best={spa_mean_s}, t_obs={spa_t_s}.
- White Reality Check vs 0 (same panel): p={rc_p_s}, best={rc_best}, t_obs={rc_t_s}.
- Tune 3c candidates (mean official edge >= $0.03, timing=all): {cand_list}.

{extra}

## Fees

Official taker P&L is `exit_bid - entry_ask - 0.02` (Tim's $0.02 estimate, labelled as an estimate). Official maker P&L is `exit_bid - entry_bid` with maker fee 0 on this series (`fee_type=quadratic`, confirmed 2026-11-13). Kalshi's verified taker fee `0.07*P*(1-P)` is reported in the jump-level columns (`*_taker_pnl_kalshi`, `*_taker_fee_kalshi`) and is not the official edge. A maker-minus-$0.02 sensitivity is in `*_maker_pnl_tim02`.

## Coverage and stale minutes

{coverage}

Candles exist only for minutes with activity, and a mid change can be a quote jump rather than a trade. Official jumps require a valid book mid exactly 60 seconds earlier; wider gaps are excluded from official scores and counted as stale. The candle window on disk typically ends about T+5 minutes after scheduled start, so most jumps are expected to be pre-start book moves, not mid-match score changes. Official scores pool pre_start and post_start and also split them.

Optional overshoot vs the residual-search in-play MC grid plus a T-60 prior was skipped: the grid is keyed by score state and serve probabilities, and live match state at the jump is not on the candle files.

## Holdout cells (timing = all), sorted by mean official edge

{Table(hold, cols)}

## Tune cells (timing = all; 3c filter is taken only from this block)

{Table(tune, cols)}

## Holdout pre_start vs post_start (top rows by mean official edge)

pre_start:

{Table(pre.sort_values('mean_edge', ascending=False), cols, n=24)}

post_start:

{Table(post.sort_values('mean_edge', ascending=False), cols, n=24)}

## $10k Monte Carlo (holdout, 10,000 day-block paths)

Start $10,000, no season reset, stop if equity < $100. flat100 stakes $100 per trade. quarter-Kelly uses the frozen tune mean/variance of official P&L / entry cost, times 0.25, capped at 2% of start-of-day equity. Paths trade only 3c-candidate cells frozen on tune. A 5th percentile ending equity below $8,000 is a large loss for the BET bar.

{Table(bank_stats, bcols) if bank_stats is not None else '_no 3c-candidate MC._'}

## What would have to be true to change this

A BET would need the holdout SPA (or RC) to reject zero at 5%, a 3c-candidate cell whose holdout 95% day-block CI lies entirely above 0, and a $10k MC 5th percentile that is not a large loss (p5 >= $8,000). Anything short of that stays PASS.
"""
    # fix spa p formatting when spa is None - the f-string above may have failed. write carefully below if needed.
    path = Path(out_dir) / f"{base}.md"
    path.write_text(text, encoding="utf-8")
    return path, verdict, why

def Main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--base", default="2026-11-13_inplay-reversion")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--candle-dir", default=str(CANDLE_DIR))
    parser.add_argument("--spa-reps", type=int, default=SPA_REPS)
    parser.add_argument("--ci-reps", type=int, default=CI_REPS)
    parser.add_argument("--mc-paths", type=int, default=MC_PATHS)
    args = parser.parse_args(argv)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pre_path = out / "2026-11-13_inplay-reversion_preregister.md"
    if not pre_path.exists():
        sys.exit(f"missing preregister {pre_path}; write it before running results")
    pre_sha = Sha256(pre_path)
    print(f"preregister SHA256 {pre_sha}", flush=True)
    print(f"loading candles from {args.candle_dir} workers={args.workers}", flush=True)

    official, diags = LoadAllJumps(args.candle_dir, workers=args.workers)
    frame = JumpsFrame(official)
    if not frame.empty:
        frame = frame.sort_values(["date_et", "jump_end", "event_ticker"]).reset_index(drop=True)
        jump_path = out / f"{args.base}_jumps.csv"
        frame.to_csv(jump_path, index=False)
        print(f"wrote {jump_path} rows={len(frame)}", flush=True)
    pd.DataFrame(diags).to_csv(out / f"{args.base}_events.csv", index=False)
    coverage = Coverage(frame, diags)
    print(coverage, flush=True)

    if frame.empty:
        summary = pd.DataFrame()
        spa = rc = None
        names = []
        candidates = pd.DataFrame()
        bank_stats = pd.DataFrame()
        extra = "No official jumps were found on the candles already on disk."
        path, verdict, why = WriteReport(
            out, args.base, coverage, summary, spa, rc, names, candidates, bank_stats, pre_sha, extra,
        )
        print(f"verdict {verdict}", flush=True)
        print(why, flush=True)
        print(f"wrote {path}", flush=True)
        return 0

    print(f"summarizing {len(AllCells())} cells x splits/timings, ci_reps={args.ci_reps}", flush=True)
    summary = SummaryTable(frame, reps=args.ci_reps, seed=SEED)
    summary.to_csv(out / f"{args.base}_summary.csv", index=False)

    print(f"SPA/RC reps={args.spa_reps}", flush=True)
    hold, d, names = SpaPanel(frame)
    days = hold["date_et"].astype(str).to_numpy()
    if len(hold) and d.size:
        spa = HansenSpa(d, days, reps=args.spa_reps, seed=SEED)
        rc = WhiteRealityCheck(d, days, reps=args.spa_reps, seed=SEED)
        spa_best = names[spa["best_index"]]
        rc_best = names[rc["best_index"]]
        spa_df = pd.DataFrame([
            {**spa, "best_cell": spa_best},
            {**rc, "best_cell": rc_best},
        ])
    else:
        spa = rc = None
        spa_best = rc_best = "n/a"
        spa_df = pd.DataFrame()
    spa_df.to_csv(out / f"{args.base}_spa.csv", index=False)
    print(f"SPA p={spa['p_value'] if spa else 'n/a'} best={spa_best}; RC p={rc['p_value'] if rc else 'n/a'} best={rc_best}", flush=True)

    candidates = TuneCandidates(summary)
    print(f"tune 3c candidates: {0 if candidates.empty else len(candidates)}", flush=True)
    bank_rows = []
    curves = []
    for rec in candidates.itertuples(index=False):
        trades = TradesForCell(frame, rec.x, rec.horizon, rec.side, rec.fill, split="holdout")
        mu, var = TuneReturnMoments(frame, rec.x, rec.horizon, rec.side, rec.fill)
        kelly_f = KellyFrac(mu, var)
        print(f"  MC {rec.cell} holdout trades={len(trades)} tune mu={mu} var={var} kelly_f={kelly_f}", flush=True)
        for kind in ("flat100", "qkelly"):
            stats = MonteCarlo(
                trades, kind, kelly_f if kind == "qkelly" else 0.0,
                paths=args.mc_paths, seed=SEED, start=START_BANK,
            )
            stats["cell"] = rec.cell
            stats["tune_mu"] = mu
            stats["tune_var"] = var
            bank_rows.append(stats)
            curve = ChronoCurve(trades, kind, kelly_f if kind == "qkelly" else 0.0)
            if not curve.empty:
                curve["cell"] = rec.cell
                curves.append(curve)
    bank_stats = pd.DataFrame(bank_rows)
    if not bank_stats.empty:
        bank_stats.to_csv(out / f"{args.base}_bankroll_stats.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(out / f"{args.base}_bankroll.csv", index=False)

    spa_p = f"{spa['p_value']:.4f}" if spa else "n/a"
    rc_p = f"{rc['p_value']:.4f}" if rc else "n/a"
    extra = (
        f"SPA best cell `{spa_best}` (mean {FmtEdge(spa.get('mean_best') if spa else None)}). "
        f"RC best cell `{rc_best}` (t_obs {FmtEdge(rc.get('t_obs') if rc else None)}). "
        f"p-values: SPA {spa_p}, RC {rc_p}."
    )
    path, verdict, why = WriteReport(
        out, args.base, coverage, summary, spa, rc, names, candidates, bank_stats, pre_sha, extra,
    )
    print(f"verdict {verdict}", flush=True)
    print(why, flush=True)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(Main())