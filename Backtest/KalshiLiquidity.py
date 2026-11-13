r"""Survey Kalshi tennis-match liquidity by level: tour vs ATP Challenger vs WTA 125.

Reports market count, bid/ask spread, volume and open interest. Default mode reads
local dumps only (no API). Pass --refresh to pull an expanded field set (volume,
open interest, bid/ask) from the public Kalshi API, read-only, no auth, no orders.
The refresh writes a sidecar under data/raw/kalshi/liquidity/ and does not overwrite
data/raw/kalshi/markets/.

Level tags come from the market series plus Kalshi milestone details.tour /
tournament_name (WTA Challenger and names containing "WTA 125" / "125K" -> wta_125).

Spread for settled historical markets is taken from the T-60 1-min candle when
present (same rule as KalshiJoin: last candle at or before T-60, age <= 30 min).
Open interest is usually missing from the existing market dumps until --refresh.

Usage:
  .venv\Scripts\python.exe -m Backtest.KalshiLiquidity --out-dir E:\AgentDocs\Tennibot\2026-11-13_market-blend
  .venv\Scripts\python.exe -m Backtest.KalshiLiquidity --refresh --out-dir E:\AgentDocs\Tennibot\2026-11-13_market-blend
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

from Backtest.KalshiJoin import STALE_S, QuoteAt
from DataPull.Http import PoliteSession, RAW_DIR
from DataPull.PullKalshi import API, SERIES_LEVEL, SERIES_TOUR

DEFAULT_OUT = Path(r"E:\AgentDocs\Tennibot\2026-11-13_market-blend")
SERIES = ["KXATPMATCH", "KXWTAMATCH", "KXATPCHALLENGERMATCH", "KXWTACHALLENGERMATCH", "KXCHALLENGERMATCH"]
KEEP = ["ticker", "event_ticker", "title", "yes_sub_title", "no_sub_title", "rules_primary", "status", "result",
        "open_time", "close_time", "occurrence_datetime", "expected_expiration_time", "settlement_ts",
        "last_price_dollars", "volume", "volume_fp", "volume_24h", "open_interest", "open_interest_fp",
        "yes_bid_dollars", "yes_ask_dollars", "liquidity", "custom_strike", "series"]


def LoadJsonl(path):
    if not path.exists() or path.stat().st_size == 0:
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def LoadMilestones():
    by_event = {}
    for row in LoadJsonl(RAW_DIR / "kalshi" / "milestones_tennis.jsonl"):
        for ticker in row.get("event_tickers") or []:
            by_event[ticker] = row
    return by_event


def ClassifyLevel(series, details):
    details = details or {}
    tour = str(details.get("tour") or "")
    name = str(details.get("tournament_name") or "")
    blob = f"{tour} {name}".lower()
    if series in ("KXATPCHALLENGERMATCH", "KXCHALLENGERMATCH") or tour == "ATP Challenger":
        return "atp_challenger"
    if series == "KXWTACHALLENGERMATCH" or tour == "WTA Challenger" or "wta 125" in blob or "125k" in blob:
        return "wta_125"
    if series == "KXATPMATCH" or tour in ("ATP", "Davis Cup"):
        return "atp_tour"
    if series == "KXWTAMATCH" or tour == "WTA":
        return "wta_tour"
    if tour == "ITF" or (series or "").startswith("KXITF"):
        return "itf"
    if tour == "EXHIBITION":
        return "exhibition"
    if series in SERIES_LEVEL:
        tour_name = SERIES_TOUR.get(series, "")
        level = SERIES_LEVEL[series]
        if level == "challenger":
            return "atp_challenger" if tour_name == "atp" else "wta_125"
        return f"{tour_name}_tour" if tour_name else "other"
    return "other"


def Num(value):
    if value in (None, ""):
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def RefreshMarkets(series_list, out_path):
    """Read-only public API pull with volume, open interest and bid/ask. Sidecar only."""
    session = PoliteSession(min_interval_s=0.2, max_retries=3, timeout_s=30)
    markets = {}
    for series in series_list:
        for path, extra in (("/historical/markets", {}), ("/markets", {"status": "settled"}),
                            ("/markets", {"status": "open"})):
            cursor = ""
            while True:
                params = {"series_ticker": series, "limit": 1000, **extra}
                if cursor:
                    params["cursor"] = cursor
                response = session.Get(API + path, params=params)
                if response.status_code != 200:
                    break
                payload = response.json()
                for market in payload.get("markets", []) or []:
                    row = {key: market.get(key) for key in KEEP}
                    row["series"] = series
                    row["source_endpoint"] = path
                    markets[row["ticker"] or f"{series}:{len(markets)}"] = row
                cursor = payload.get("cursor") or ""
                if not cursor or not payload.get("markets"):
                    break
        print(f"refresh {series}: running total {len(markets):,}", flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(markets.values(), key=lambda r: r.get("ticker") or "")
    with open(out_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"refresh wrote {len(rows):,} markets -> {out_path}", flush=True)
    return rows


def LoadLocalMarkets(series_list):
    rows = []
    for series in series_list:
        for row in LoadJsonl(RAW_DIR / "kalshi" / "markets" / f"{series}.jsonl"):
            row.setdefault("series", series)
            rows.append(row)
    return rows


def T60Spreads(series_list):
    """event_ticker -> (bid, ask, spread) from the T-60 candle, when it qualifies."""
    starts = {}
    for row in LoadJsonl(RAW_DIR / "kalshi" / "milestones_tennis.jsonl"):
        start = row.get("start_date")
        if not start:
            continue
        try:
            start_ts = dt.datetime.fromisoformat(start.replace("Z", "+00:00")).timestamp()
        except ValueError:
            continue
        for ticker in row.get("event_tickers") or []:
            starts[ticker] = start_ts
    out = {}
    for series in series_list:
        candle_dir = RAW_DIR / "kalshi" / "candles" / series
        if not candle_dir.exists():
            continue
        for path in candle_dir.glob("*.json"):
            event = path.stem
            start_ts = starts.get(event)
            if start_ts is None:
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            quote = QuoteAt(payload.get("minute") or [], start_ts - 3600, stale_s=STALE_S)
            if not quote:
                continue
            bid, ask = quote["yes_bid"], quote["yes_ask"]
            if bid is not None and ask is not None and 0 < bid < ask < 1:
                out[event] = {"t60_bid": bid, "t60_ask": ask, "t60_spread": ask - bid}
    return out


def MarketTable(rows, milestones, spreads):
    recs = []
    for row in rows:
        event = row.get("event_ticker")
        series = row.get("series") or ""
        details = (milestones.get(event) or {}).get("details") or {}
        level = ClassifyLevel(series, details)
        bid = Num(row.get("yes_bid_dollars"))
        ask = Num(row.get("yes_ask_dollars"))
        spread = (ask - bid) if np.isfinite(bid) and np.isfinite(ask) and ask > bid else np.nan
        t60 = spreads.get(event) or {}
        if not np.isfinite(spread) and t60:
            spread = t60["t60_spread"]
            bid = t60["t60_bid"]
            ask = t60["t60_ask"]
        recs.append({
            "ticker": row.get("ticker"),
            "event_ticker": event,
            "series": series,
            "level": level,
            "status": row.get("status"),
            "title": row.get("title"),
            "tournament_name": details.get("tournament_name"),
            "tour_label": details.get("tour"),
            "volume": Num(row.get("volume_fp") if row.get("volume_fp") not in (None, "") else row.get("volume")),
            "open_interest": Num(row.get("open_interest_fp") if row.get("open_interest_fp") not in (None, "")
                                 else row.get("open_interest")),
            "yes_bid": bid,
            "yes_ask": ask,
            "spread": spread,
            "close_time": row.get("close_time") or row.get("settlement_ts") or row.get("occurrence_datetime"),
        })
    return pd.DataFrame(recs)


def Summarise(markets, asof):
    rows = []

    def One(part, level):
        vol = part["volume"].dropna()
        oi = part["open_interest"].dropna()
        sp = part["spread"].dropna()
        rows.append({
            "date": asof,
            "series": level,
            "level": level,
            "n_markets": int(len(part)),
            "n_events": int(part["event_ticker"].nunique()),
            "n_with_volume": int(vol.size),
            "n_with_oi": int(oi.size),
            "n_with_spread": int(sp.size),
            "mean_volume": float(vol.mean()) if len(vol) else np.nan,
            "median_volume": float(vol.median()) if len(vol) else np.nan,
            "total_volume": float(vol.sum()) if len(vol) else np.nan,
            "mean_oi": float(oi.mean()) if len(oi) else np.nan,
            "median_oi": float(oi.median()) if len(oi) else np.nan,
            "total_oi": float(oi.sum()) if len(oi) else np.nan,
            "mean_spread": float(sp.mean()) if len(sp) else np.nan,
            "median_spread": float(sp.median()) if len(sp) else np.nan,
            "p90_spread": float(sp.quantile(0.9)) if len(sp) else np.nan,
        })

    for level, part in markets.groupby("level"):
        One(part, level)
    One(markets, "all")
    # rollups
    if "atp_tour" in set(markets["level"]) or "wta_tour" in set(markets["level"]):
        One(markets[markets["level"].isin(["atp_tour", "wta_tour"])], "tour_all")
    if "atp_challenger" in set(markets["level"]) or "wta_125" in set(markets["level"]):
        One(markets[markets["level"].isin(["atp_challenger", "wta_125"])], "challenger_all")
    return pd.DataFrame(rows)


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base", default="2026-11-13_market-blend")
    parser.add_argument("--refresh", action="store_true",
                        help="Hit the public Kalshi API (read-only) for volume/OI/bid-ask; default is local files only.")
    parser.add_argument("--series", nargs="+", default=SERIES)
    args = parser.parse_args(argv)
    fetched_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    if args.refresh:
        sidecar = RAW_DIR / "kalshi" / "liquidity" / f"markets_{fetched_at[:10]}.jsonl"
        rows = RefreshMarkets(args.series, sidecar)
        source = f"refresh:{sidecar}"
    else:
        rows = LoadLocalMarkets(args.series)
        source = "local:data/raw/kalshi/markets"
    milestones = LoadMilestones()
    spreads = T60Spreads(args.series)
    markets = MarketTable(rows, milestones, spreads)
    asof = fetched_at[:10]
    if markets["close_time"].notna().any():
        try:
            asof = pd.to_datetime(markets["close_time"].dropna(), utc=True, errors="coerce").max().date().isoformat()
        except Exception:
            pass
    summary = Summarise(markets, asof)
    summary["source"] = source
    summary["fetched_at_utc"] = fetched_at
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = args.out_dir / f"{args.base}_liquidity.csv"
    summary.to_csv(out, index=False)
    print(summary.to_string(index=False))
    print(f"wrote {out}  (source={source}; T-60 spreads on {len(spreads):,} events; OI non-null "
          f"{int(markets['open_interest'].notna().sum()):,}/{len(markets):,})")
    return summary


if __name__ == "__main__":
    Main()