r"""Live tennis report for Degen: one JSONL row per Kalshi match contract + a latest-status Markdown.

  .venv\Scripts\python.exe -m LiveOdds.Report --date 2026-11-13 [--event KXATPCHALLENGERMATCH-26NOV13DRACHI]
  .venv\Scripts\python.exe -m LiveOdds.Report --date 2026-11-13 --out-root data\live_smoke   (testing)

Appends E:\AgentDocs\Tennibot\<date>_live\<date>_live.jsonl and overwrites <date>_live_latest.md.
Read-only Kalshi public data; no orders ever.

Row contract: observed_at, game_id, teams, score, period, clock, state_time, venue, market, side,
bid, ask, size, price_time, quote_qualified, fee_basis, model_prob|null, edge|null, action, reason
(+ edge_kalshi_fee, state_fetched_at, pregame_prob, p_serve_a/b).
Prices: Kalshi market list (yes bid/ask/size), no quote timestamp -> price_time null, observed_at = fetch time.
Score: Kalshi live_data (LiveOdds.LiveScore), no source timestamp -> state_time null, state_fetched_at = fetch time.
Model: Markov LiveWinProb from the score and a p_serve pair implied by the pregame probability
(PredictDay ratings: Elo + frozen serve/return blend). The pair is mu +/- d, with mu the tour-mean
serve-point rate and d solved so the pregame match probability is reproduced.

BET requires, in order: quote_qualified; live state parsed; quote and state fetched within STALE_S
(45 s) of each other and of now (fetch-time based, since neither source gives a source timestamp);
the in-play model validated at this level (only Grand Slam holdout exists:
data/backtest/markov/summary.json; no non-slam point-by-point data is available, so other levels PASS
"live model unvalidated at this level"); edge > 0 after Tim's $0.02 fee estimate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

from Backtest.Markov import LiveWinProb, MatchProb
from Backtest.Odds import KalshiFee, TIM_FEE
from LiveOdds.KalshiPublic import KalshiPublic, MATCH_SERIES
from LiveOdds.LiveScore import FetchLive, ParseState

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTDOCS = Path(r"E:\AgentDocs\Tennibot")
MARKOV_SUMMARY = REPO_ROOT / "data" / "backtest" / "markov" / "summary.json"
STALE_S = 45
ET = ZoneInfo("America/New_York")
TOUR_MU = {"atp": 0.64, "wta": 0.56}  # tour-mean serve-point win rate (Sackmann 2015-2025 approx.)
SLAMS = ("australian open", "french open", "roland", "wimbledon", "us open")


def LiveModelValidated():
    if not MARKOV_SUMMARY.exists():
        return False, "no markov validation file"
    summary = json.loads(MARKOV_SUMMARY.read_text())
    return bool(summary.get("validated")), summary.get("chosen_column")


def ImpliedServe(p_pre, tour, best_of):
    mu, lo, hi = TOUR_MU[tour], -0.2, 0.2
    for _ in range(40):
        mid = (lo + hi) / 2
        if MatchProb(mu + mid, mu - mid, best_of) < p_pre:
            lo = mid
        else:
            hi = mid
    d = (lo + hi) / 2
    return mu + d, mu - d


def Age(iso, now):
    return (now - dt.datetime.fromisoformat(iso)).total_seconds()


def Assess(market, side_prob, state, quote_age, state_age, validated, level_ok):
    bid, ask, size = market["yes_bid"], market["yes_ask"], market["yes_ask_size"]
    qualified = bool(bid is not None and ask is not None and 0 < bid < ask < 1 and size and size > 0)
    row = {"bid": bid, "ask": ask, "size": size, "quote_qualified": qualified, "model_prob": side_prob,
           "edge": None, "edge_kalshi_fee": None}
    if side_prob is not None and ask is not None and 0 < ask < 1:
        row["edge"] = round(side_prob - ask - TIM_FEE, 4)
        row["edge_kalshi_fee"] = round(side_prob - ask - KalshiFee(ask), 4)
    if not qualified:
        return {**row, "action": "PASS", "reason": "quote not qualified"}
    if state is None:
        return {**row, "action": "PASS", "reason": "no live game state (not in play or unparsed)"}
    if quote_age > STALE_S or state_age > STALE_S:
        return {**row, "action": "PASS", "reason": f"stale (quote {quote_age:.0f}s, state {state_age:.0f}s > {STALE_S}s)"}
    if side_prob is None:
        return {**row, "action": "PASS", "reason": "no model prob (unknown player)"}
    if not (validated and level_ok):
        return {**row, "action": "PASS", "reason": "live model unvalidated at this level" if validated else "no live model"}
    if row["edge"] <= 0:
        return {**row, "action": "PASS", "reason": "edge <= 0 after Tim's $0.02 fee est"}
    return {**row, "action": "BET", "reason": "validated model, qualified fresh quote, edge > 0"}


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True)
    parser.add_argument("--out-root", default=str(AGENTDOCS))
    parser.add_argument("--event", nargs="*", default=None, help="only these event tickers")
    parser.add_argument("--no-model", action="store_true", help="skip ratings replay (prices + score only)")
    args = parser.parse_args(argv)
    validated, model_column = LiveModelValidated()
    target = dt.date.fromisoformat(args.date)
    client = KalshiPublic()
    events = client.Slate(list(MATCH_SERIES))
    events = {t: e for t, e in events.items() if e.get("start")
              and dt.datetime.fromisoformat(e["start"].replace("Z", "+00:00")).astimezone(ET).date() == target
              and (not args.event or t in args.event)}
    states = {}
    if not args.no_model:
        from Backtest.PredictDay import BuildState, Resolve
        from Backtest.Engine import Blend, SharedTable
        states = {tour: BuildState(tour, target) for tour in {e["tour"] for e in events.values()}}
    out_dir = Path(args.out_root) / f"{args.date}_live"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / f"{args.date}_live.jsonl"
    rows = []
    for ticker, event in sorted(events.items()):
        markets = sorted(event["markets"], key=lambda m: m["ticker"])
        details, state_fetched = (None, None)
        if event.get("milestone_id"):
            details, state_fetched = FetchLive(client.session, event["milestone_id"])
        # re-fetch quotes right after the score so both are fresh together
        fresh_markets = {m["ticker"]: m for m in client.OpenMarkets(event["series"]) if m["event_ticker"] == ticker} if details else {}
        markets = [fresh_markets.get(m["ticker"], m) for m in markets]
        state = ParseState(details)
        title = (event.get("title") or "").lower()
        first = title.split(" vs ")[0].strip() if " vs " in title else ""
        # competitor 1 ('a' in state) = first name in milestone title
        def IsFirst(m):
            return bool(first) and first.split()[-1] in (m["player"] or "").lower()
        comp1 = next((m for m in markets if IsFirst(m)), markets[0])
        comp2 = next((m for m in markets if m is not comp1), None)
        competition = event["details"].get("tournament_name") or ""
        best_of = 5 if (event["tour"] == "atp" and event["level"] == "tour" and any(s in competition.lower() for s in SLAMS)) else 3
        level_ok = event["level"] == "tour" and any(s in competition.lower() for s in SLAMS)
        p_pre = p1 = sa = sb = None
        if states and comp2 is not None:
            st = states[event["tour"]]
            id1, id2 = Resolve(st["index"], comp1["player"], st["last_seen"]), Resolve(st["index"], comp2["player"], st["last_seen"])
            if id1 is not None and id2 is not None:
                surface, _ = st["surfaces"].Get(competition)
                pe = st["elo"].Predict(id1, id2, surface, best_of)
                s1, s2 = st["sr"].PServe(id1, id2, surface)
                p_pre = Blend(pe, SharedTable().Prob(s1, s2, best_of), st["weight"])
                sa, sb = ImpliedServe(p_pre, event["tour"], best_of)
                if state:
                    p1 = LiveWinProb(sa, sb, state["sets_a"], state["sets_b"], state["games_a"], state["games_b"],
                                     state["points_a"], state["points_b"], state["server"], best_of)
                elif not details or details.get("status") != "live":
                    p1 = None
        now = dt.datetime.now(dt.timezone.utc)
        for m in markets:
            side_prob = None if p1 is None else (p1 if m is comp1 else 1 - p1)
            q_age = Age(m["fetched_at"], now)
            s_age = Age(state_fetched, now) if state_fetched else 1e9
            assessed = Assess(m, None if side_prob is None else round(side_prob, 4), state, q_age, s_age, validated, level_ok)
            server_name = None
            if state:
                server_name = (comp1 if state["server"] == "a" else comp2)["player"]
            rows.append({"observed_at": m["fetched_at"], "game_id": ticker,
                         "teams": f"{comp1['player']} vs {comp2['player'] if comp2 else '?'}",
                         "score": state["score"] if state else None, "period": (state or {}).get("period") or (details or {}).get("match_status"),
                         "clock": None, "server": server_name, "state_time": None, "state_fetched_at": state_fetched,
                         "venue": "kalshi", "market": m["ticker"], "side": m["player"], "price_time": None,
                         "fee_basis": "tim_est_0.02 (edge); kalshi_verified 0.07p(1-p) in edge_kalshi_fee",
                         "scheduled_start": event["start"], "tour": event["tour"], "level": event["level"],
                         "competition": competition, "pregame_prob": None if p_pre is None else round(p_pre if m is comp1 else 1 - p_pre, 4),
                         "p_serve_a": sa, "p_serve_b": sb, "model_version": "markov_implied_serve_v1", **assessed})
    with open(jsonl, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")
    now_et = dt.datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S %Z")
    lines = [f"# Tennis live {args.date} - updated {now_et}", "",
             f"Score: Kalshi live_data (no source timestamp; state_time null). Prices: Kalshi market bid/ask/size (no quote timestamp).",
             f"In-play Markov validated on Grand Slam holdout only: {validated} ({model_column}). Non-slam levels PASS 'live model unvalidated at this level'.",
             "Edge = model - ask - fee; Tim's $0.02 estimate and Kalshi verified 0.07p(1-p) side by side. No orders.", "",
             "| start ET | match | level | score (srv) | contract | model fair | bid | ask | size | edge Tim | edge Kalshi | action | reason |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["scheduled_start"], r["market"])):
        start_et = dt.datetime.fromisoformat(r["scheduled_start"].replace("Z", "+00:00")).astimezone(ET).strftime("%H:%M")
        fair = "-" if r["model_prob"] is None else f"${r['model_prob']:.2f}"
        e1 = "-" if r["edge"] is None else f"{r['edge']:+.3f}"
        e2 = "-" if r["edge_kalshi_fee"] is None else f"{r['edge_kalshi_fee']:+.3f}"
        score = (r["score"] + f" ({r['server']})") if r["score"] else (r["period"] or "-")
        lines.append(f"| {start_et} | {r['teams']} | {r['tour']}/{r['level']} | {score} | {r['side']} | {fair} | {r['bid']} | {r['ask']} | {r['size']} | {e1} | {e2} | {r['action']} | {r['reason']} |")
    (out_dir / f"{args.date}_live_latest.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"{len(rows)} rows -> {jsonl}")


if __name__ == "__main__":
    Main()