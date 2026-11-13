r"""Live tennis report for Degen: one JSONL row per Kalshi match contract + a latest-status Markdown.

  .venv\Scripts\python.exe -m LiveOdds.Report --date 2026-11-13
  .venv\Scripts\python.exe -m LiveOdds.Report --date 2026-11-13 --out-root data\live_smoke   (testing)

Appends E:\AgentDocs\Tennibot\<date>_live\<date>_live.jsonl and overwrites <date>_live_latest.md.
Read-only Kalshi public data; no orders ever.

Row contract: observed_at, game_id, teams, score, period, clock, state_time, venue, market, side,
bid, ask, size, price_time, quote_qualified, fee_basis, model_prob|null, edge|null, action, reason.
Kalshi's market list has no quote timestamp: price_time is null and observed_at is our fetch time.
quote_qualified = executable yes bid AND ask present (0 < bid < ask < 1) with ask size > 0 for this
exact contract, fetched in this run.

BET requires, in order: quote_qualified; price and game state within the staleness limit
(STALE_S = 45 s); a validated in-play model (data/backtest/markov/summary.json "validated": true);
edge > 0 after Tim's $0.02 fee estimate. There is NO live score feed wired in yet, so game state is
unknown and every row is PASS with reason "stale" (no game state) or "no live model" - by design.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from LiveOdds.KalshiPublic import KalshiPublic, MATCH_SERIES

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTDOCS = Path(r"E:\AgentDocs\Tennibot")
MARKOV_SUMMARY = REPO_ROOT / "data" / "backtest" / "markov" / "summary.json"
STALE_S = 45
ET = ZoneInfo("America/New_York")


def LiveModelValidated():
    if not MARKOV_SUMMARY.exists():
        return False, "no markov validation file"
    summary = json.loads(MARKOV_SUMMARY.read_text())
    return bool(summary.get("validated")), summary.get("chosen_column")


def Assess(market, game_state, validated):
    bid, ask, size = market["yes_bid"], market["yes_ask"], market["yes_ask_size"]
    qualified = bool(bid is not None and ask is not None and 0 < bid < ask < 1 and size and size > 0)
    row = {"bid": bid, "ask": ask, "size": size, "quote_qualified": qualified, "model_prob": None, "edge": None}
    if not qualified:
        return {**row, "action": "PASS", "reason": "quote not qualified"}
    if game_state is None:
        return {**row, "action": "PASS", "reason": "stale (no live game state feed)"}
    if not validated:
        return {**row, "action": "PASS", "reason": "no live model"}
    return {**row, "action": "PASS", "reason": "live model wiring pending (needs score state + pre-match p_serve)"}


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True)
    parser.add_argument("--out-root", default=str(AGENTDOCS))
    args = parser.parse_args(argv)
    validated, model_column = LiveModelValidated()
    client = KalshiPublic()
    events = client.Slate(list(MATCH_SERIES))
    target = dt.date.fromisoformat(args.date)
    out_dir = Path(args.out_root) / f"{args.date}_live"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{args.date}_live.jsonl"
    rows = []
    for ticker, event in sorted(events.items()):
        start = event.get("start")
        if not start or dt.datetime.fromisoformat(start.replace("Z", "+00:00")).astimezone(ET).date() != target:
            continue
        players = " vs ".join(m["player"] or "?" for m in sorted(event["markets"], key=lambda m: m["ticker"]))
        for market in event["markets"]:
            assessed = Assess(market, None, validated)
            rows.append({"observed_at": market["fetched_at"], "game_id": ticker, "teams": players, "score": None, "period": None,
                         "clock": None, "state_time": None, "venue": "kalshi", "market": market["ticker"], "side": market["player"],
                         "price_time": None, "fee_basis": "tim_est_0.02", "scheduled_start": start,
                         "tour": event["tour"], "level": event["level"], **assessed})
    with open(jsonl, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    now_et = dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M %Z")
    lines = [f"# Tennis live {args.date} - updated {now_et}", "",
             f"In-play Markov model validated on slam holdout: {validated} ({model_column}). Live score feed: NOT CONNECTED, so all rows PASS.",
             "Fee basis: Tim's $0.02 estimate. Kalshi gives no quote timestamp (price_time null; observed_at = fetch time, UTC).", "",
             "| start ET | match | contract | bid | ask | size | action | reason |", "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: r["scheduled_start"]):
        start_et = dt.datetime.fromisoformat(r["scheduled_start"].replace("Z", "+00:00")).astimezone(ET).strftime("%H:%M")
        lines.append(f"| {start_et} | {r['teams']} | {r['side']} | {r['bid']} | {r['ask']} | {r['size']} | {r['action']} | {r['reason']} |")
    lines += ["", "Blockers: no live score/state source; in-play model not wired to quotes. No orders are ever placed."]
    (out_dir / f"{args.date}_live_latest.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"{len(rows)} rows -> {jsonl}")


if __name__ == "__main__":
    Main()