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


def ValidBook(bid, ask):
    try:
        bid, ask = float(bid), float(ask)
    except (TypeError, ValueError):
        return False
    return 0.0 < bid < ask < 1.0


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
        api_bid = Num(row.get("yes_bid_dollars"))
        api_ask = Num(row.get("yes_ask_dollars"))
        live_ok = ValidBook(api_bid, api_ask)
        status = str(row.get("status") or "")
        is_open = status in ("open", "active", "unopened")
        t60 = spreads.get(event) or {}
        t60_ok = bool(t60) and ValidBook(t60.get("t60_bid"), t60.get("t60_ask"))
        # Settled books often sit at 0 / 1 (spread 1.00). Use a live book only for open markets;
        # otherwise prefer the T-60 candle.
        if is_open and live_ok:
            bid, ask, spread, spread_basis = api_bid, api_ask, api_ask - api_bid, "live"
        elif t60_ok:
            bid, ask, spread, spread_basis = t60["t60_bid"], t60["t60_ask"], t60["t60_spread"], "t60"
        elif live_ok:
            bid, ask, spread, spread_basis = api_bid, api_ask, api_ask - api_bid, "api_other"
        else:
            bid, ask, spread, spread_basis = api_bid, api_ask, np.nan, "none"
        recs.append({
            "ticker": row.get("ticker"),
            "event_ticker": event,
            "series": series,
            "level": level,
            "status": status,
            "is_open": is_open,
            "title": row.get("title"),
            "tournament_name": details.get("tournament_name"),
            "tour_label": details.get("tour"),
            "volume": Num(row.get("volume_fp") if row.get("volume_fp") not in (None, "") else row.get("volume")),
            "open_interest": Num(row.get("open_interest_fp") if row.get("open_interest_fp") not in (None, "")
                                 else row.get("open_interest")),
            "yes_bid": bid,
            "yes_ask": ask,
            "spread": spread,
            "spread_t60": t60["t60_spread"] if t60_ok else np.nan,
            "spread_live": (api_ask - api_bid) if (is_open and live_ok) else np.nan,
            "spread_basis": spread_basis,
            "close_time": row.get("close_time") or row.get("settlement_ts") or row.get("occurrence_datetime"),
        })
    return pd.DataFrame(recs)


def Summarise(markets, asof):
    rows = []

    def One(part, level):
        vol = part["volume"].dropna()
        opened = part[part["is_open"]] if "is_open" in part.columns else part.iloc[0:0]
        oi_open = opened["open_interest"].dropna() if len(opened) else pd.Series(dtype=float)
        sp = part["spread"].dropna()
        sp_t60 = part["spread_t60"].dropna() if "spread_t60" in part.columns else sp
        sp_live = part["spread_live"].dropna() if "spread_live" in part.columns else pd.Series(dtype=float)
        rows.append({
            "date": asof,
            "series": level,
            "level": level,
            "n_markets": int(len(part)),
            "n_events": int(part["event_ticker"].nunique()),
            "n_open": int(len(opened)),
            "n_with_volume": int(vol.size),
            "n_with_oi_open": int(oi_open.size),
            "n_with_spread": int(sp.size),
            "n_with_spread_t60": int(sp_t60.size),
            "n_with_spread_live": int(sp_live.size),
            "mean_volume": float(vol.mean()) if len(vol) else np.nan,
            "median_volume": float(vol.median()) if len(vol) else np.nan,
            "total_volume": float(vol.sum()) if len(vol) else np.nan,
            "mean_oi_open": float(oi_open.mean()) if len(oi_open) else np.nan,
            "median_oi_open": float(oi_open.median()) if len(oi_open) else np.nan,
            "total_oi_open": float(oi_open.sum()) if len(oi_open) else np.nan,
            "mean_spread": float(sp.mean()) if len(sp) else np.nan,
            "median_spread": float(sp.median()) if len(sp) else np.nan,
            "p90_spread": float(sp.quantile(0.9)) if len(sp) else np.nan,
            "mean_spread_t60": float(sp_t60.mean()) if len(sp_t60) else np.nan,
            "median_spread_t60": float(sp_t60.median()) if len(sp_t60) else np.nan,
            "mean_spread_live": float(sp_live.mean()) if len(sp_live) else np.nan,
            "median_spread_live": float(sp_live.median()) if len(sp_live) else np.nan,
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
    sidecar_dir = RAW_DIR / "kalshi" / "liquidity"
    if args.refresh:
        sidecar = sidecar_dir / f"markets_{fetched_at[:10]}.jsonl"
        rows = RefreshMarkets(args.series, sidecar)
        source = f"refresh:{sidecar}"
    else:
        existing = sorted(sidecar_dir.glob("markets_*.jsonl")) if sidecar_dir.exists() else []
        if existing:
            sidecar = existing[-1]
            rows = LoadJsonl(sidecar)
            source = f"sidecar:{sidecar}"
        else:
            rows = LoadLocalMarkets(args.series)
            source = "local:data/raw/kalshi/markets"
    milestones = LoadMilestones()
    spreads = T60Spreads(args.series)
    markets = MarketTable(rows, milestones, spreads)
    asof = fetched_at[:10]
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