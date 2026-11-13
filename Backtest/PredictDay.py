r"""PredictDay: pre-match fair values for one ET date's Kalshi tennis slate. Default action PASS.

  .venv\Scripts\python.exe -m Backtest.PredictDay --date 2026-11-13 --dry-run
  .venv\Scripts\python.exe -m Backtest.PredictDay --date 2026-11-13 --out-dir E:\AgentDocs\Tennibot\<folder>

Ratings: frozen params (Backtest.FrozenParams), replayed over every Sackmann match with
tourney_date < --date. Slate + prices: Kalshi public API (read-only; KXATPMATCH, KXWTAMATCH,
challenger series); start times from Kalshi milestones (scheduled, ET shown). Fair value is the
model probability as $0.xx to win $1. Edge = model_prob - yes_ask - 0.02 (Tim's estimate),
with Kalshi's verified taker fee 0.07*p*(1-p) shown alongside.

Action is BET only if ALL hold: quote has ask and size, edge > 0 (Tim's fee), both players
known to the ratings, and the ratings are fresh (last result within 14 days of --date).
Otherwise PASS with the first failed check as the reason. Recommendations only; no orders.
Two tables: "Most likely to win" (ranked by win probability) and "Best value" (ranked by edge).
Without --dry-run, one row per match is appended to data/predictions/<season>.jsonl (only
for matches whose scheduled start is still in the future). TODO: a schedule source beyond
Kalshi (matches Kalshi has not listed yet are invisible here).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from Backtest.Elo import EloParams
from Backtest.Engine import Blend, RunModels, SharedTable
from Backtest.FrozenParams import FROZEN, MODEL_VERSION
from Backtest.LoadMatches import LoadMatches
from Backtest.NameMatch import NameIndex
from Backtest.Odds import KalshiFee, TIM_FEE
from Backtest.ServeReturn import ServeReturnParams
from Backtest.Surfaces import SurfaceLookup

REPO_ROOT = Path(__file__).resolve().parents[1]
PRED_DIR = REPO_ROOT / "data" / "predictions"
ET = ZoneInfo("America/New_York")
FRESH_DAYS = 14


def BuildState(tour, target_date):
    frame = LoadMatches(tour)
    frame = frame[frame["date"] < pd.Timestamp(target_date)].reset_index(drop=True)
    params = FROZEN[tour]
    result = RunModels(frame, EloParams(**params["elo"]), ServeReturnParams(**params["sr"]), tour)
    names = pd.concat([frame[["winner_id", "winner_name", "date"]].set_axis(["i", "n", "d"], axis=1),
                       frame[["loser_id", "loser_name", "date"]].set_axis(["i", "n", "d"], axis=1)])
    last_seen = names.groupby("i")["d"].max()
    index = NameIndex(names.drop_duplicates("i")[["i", "n"]].itertuples(index=False, name=None))
    return {"elo": result["elo"], "sr": result["sr"], "index": index, "last_date": frame["date"].max().date(),
            "surfaces": SurfaceLookup(frame), "weight": params["blend_weight_sr"], "last_seen": last_seen}


def Resolve(index, name):
    ids = index.LookupFull(name)
    return next(iter(ids)) if len(ids) == 1 else None


def Confidence(p, fresh, known):
    if not (fresh and known):
        return "low"
    gap = abs(p - 0.5)
    return "high" if gap >= 0.25 else ("medium" if gap >= 0.12 else "low")


def ScoreEvent(event_ticker, event, states, target_date):
    tour = event["tour"]
    state = states[tour]
    markets = sorted(event["markets"], key=lambda m: m["ticker"])
    if len(markets) != 2:
        return None
    a, b = markets
    start = dt.datetime.fromisoformat(event["start"].replace("Z", "+00:00")) if event.get("start") else None
    if start is None or start.astimezone(ET).date() != target_date:
        return None
    competition = event["details"].get("tournament_name") or a.get("rules") or ""
    surface, surface_source = state["surfaces"].Get(competition)
    best_of = 5 if any(k in competition.lower() for k in ("australian open", "french open", "roland", "wimbledon", "us open")) and tour == "atp" and event["level"] == "tour" else 3
    id_a, id_b = Resolve(state["index"], a["player"]), Resolve(state["index"], b["player"])
    known = id_a is not None and id_b is not None
    fresh = (target_date - state["last_date"]).days <= FRESH_DAYS
    row = {"event_ticker": event_ticker, "tour": tour, "level": event["level"], "start_utc": start.isoformat(),
           "start_et": start.astimezone(ET).strftime("%H:%M"), "start_source": event.get("start_source"),
           "competition": competition, "round": event["details"].get("round"), "surface": surface,
           "surface_source": surface_source, "best_of": best_of, "player_a": a["player"], "player_b": b["player"],
           "ask_a": a["yes_ask"], "ask_b": b["yes_ask"], "ask_size_a": a["yes_ask_size"], "ask_size_b": b["yes_ask_size"],
           "bid_a": a["yes_bid"], "bid_b": b["yes_bid"], "fetched_at": a["fetched_at"], "price_time": None,
           "ticker_a": a["ticker"], "ticker_b": b["ticker"], "ratings_last_date": str(state["last_date"])}
    if known:
        elo = state["elo"]
        ra = elo.Ratings(id_a, surface)
        rb = elo.Ratings(id_b, surface)
        p_elo = elo.Predict(id_a, id_b, surface, best_of)
        sa, sb = state["sr"].PServe(id_a, id_b, surface)
        p_sr = SharedTable().Prob(sa, sb, best_of)
        p_a = Blend(p_elo, p_sr, state["weight"])
        row.update({"elo_a": round(ra[0]), "elo_b": round(rb[0]), "elo_surf_a": round(ra[1]), "elo_surf_b": round(rb[1]),
                    "p_elo_a": p_elo, "p_sr_a": p_sr, "p_a": p_a, "p_serve_a": sa, "p_serve_b": sb})
    else:
        row.update({"p_a": None})
    # pick = more likely winner; value side = larger edge
    checks = []
    if row["p_a"] is None:
        row.update({"fav": None, "fav_prob": None, "value_side": None, "edge": None, "edge_kalshi_fee": None})
        checks.append("unknown player" if not known else "no model prob")
    else:
        p_a = row["p_a"]
        row["fav"], row["fav_prob"] = (a["player"], p_a) if p_a >= 0.5 else (b["player"], 1 - p_a)
        edges = []
        for side, p, ask, size in (("a", p_a, a["yes_ask"], a["yes_ask_size"]), ("b", 1 - p_a, b["yes_ask"], b["yes_ask_size"])):
            if ask is not None and 0 < ask < 1:
                edges.append((p - ask - TIM_FEE, side, p, ask, size))
        if edges:
            edge, side, p, ask, size = max(edges)
            row.update({"value_side": a["player"] if side == "a" else b["player"], "value_prob": p, "value_ask": ask,
                        "value_size": size, "edge": edge, "edge_kalshi_fee": p - ask - KalshiFee(ask)})
            if not size or size <= 0:
                checks.append("no ask size")
            if edge <= 0:
                checks.append("edge <= 0 after Tim's $0.02 fee est")
        else:
            row.update({"value_side": None, "edge": None, "edge_kalshi_fee": None})
            checks.append("no executable ask")
    if start <= dt.datetime.now(dt.timezone.utc):
        checks.insert(0, "already started (scheduled start passed; price may be in-play)")
    if not fresh:
        checks.insert(0, f"ratings stale (results end {state['last_date']})")
    row["action"] = "PASS" if checks else "BET"
    row["reason"] = checks[0] if checks else "edge > 0, quote has size, ratings fresh"
    row["confidence"] = Confidence(row.get("fav_prob") or 0.5, fresh, known)
    return row


def Money(p):
    return "-" if p is None or pd.isna(p) else f"${p:.2f}"


def Markdown(rows, target_date, last_dates):
    lines = [f"# Tennis PredictDay {target_date} (times ET; Kalshi scheduled starts)", "",
             f"Ratings use results through " + ", ".join(f"{k.upper()} {v}" for k, v in last_dates.items()) + f". Model {MODEL_VERSION}. Fair value = model probability as $ to win $1. "
             "Edge uses Tim's $0.02 fee estimate (Kalshi verified fee 0.07*p*(1-p) in the edge_kalshi column). Default PASS. No orders.", ""]
    known = [r for r in rows if r.get("fav_prob") is not None]
    lines += ["## Most likely to win", "", "| start ET | match | tour/level | surface | Elo A/B (surf) | ask A/B | pick | model prob | fair value | confidence |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(known, key=lambda r: -r["fav_prob"]):
        lines.append(f"| {r['start_et']} | {r['player_a']} vs {r['player_b']} | {r['tour']}/{r['level']} | {r['surface']}"
                     f"{'*' if r['surface_source'] == 'guess_default_hard' else ''} | {r['elo_a']}/{r['elo_b']} ({r['elo_surf_a']}/{r['elo_surf_b']}) | "
                     f"{Money(r['ask_a'])}/{Money(r['ask_b'])} | {r['fav']} | {r['fav_prob']:.1%} | {Money(r['fav_prob'])} | {r['confidence']} |")
    lines += ["", "## Best value (ranked by edge; PASS unless every check holds)", "",
              "| start ET | match | value side | model prob | ask | size | edge (Tim $0.02) | edge (Kalshi fee) | action | reason |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: -(r.get("edge") if r.get("edge") is not None else -9)):
        edge = "-" if r.get("edge") is None else f"{r['edge']:+.3f}"
        edge_k = "-" if r.get("edge_kalshi_fee") is None else f"{r['edge_kalshi_fee']:+.3f}"
        vp = "-" if r.get("value_prob") is None else f"{r['value_prob']:.1%}"
        lines.append(f"| {r['start_et']} | {r['player_a']} vs {r['player_b']} | {r.get('value_side') or '-'} | {vp} | "
                     f"{Money(r.get('value_ask'))} | {r.get('value_size') if r.get('value_size') is not None else '-'} | {edge} | {edge_k} | {r['action']} | {r['reason']} |")
    unknown = [r for r in rows if r.get("fav_prob") is None]
    if unknown:
        lines += ["", f"{len(unknown)} match(es) have a player the ratings do not know (PASS)."]
    lines += ["", "\\* surface guessed (default Hard). Kalshi gives no quote timestamp; prices are as fetched (fetched_at, UTC)."]
    return "\n".join(lines)


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True, help="ET date YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true", help="print only; never append the prediction log")
    parser.add_argument("--tours", nargs="+", default=["atp", "wta"])
    parser.add_argument("--levels", nargs="+", default=["tour", "challenger"])
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)
    from LiveOdds.KalshiPublic import KalshiPublic, MATCH_SERIES
    target_date = dt.date.fromisoformat(args.date)
    states = {tour: BuildState(tour, target_date) for tour in args.tours}
    series = [s for s, (t, lvl) in MATCH_SERIES.items() if t in args.tours and lvl in args.levels]
    since = (dt.datetime.combine(target_date, dt.time(0), ET) - dt.timedelta(days=1)).astimezone(dt.timezone.utc)
    events = KalshiPublic().Slate(series, since.strftime("%Y-%m-%dT%H:%M:%SZ"))
    rows = [r for r in (ScoreEvent(t, e, states, target_date) for t, e in sorted(events.items())) if r]
    text = Markdown(rows, target_date, {t: str(s["last_date"]) for t, s in states.items()})
    print(text)
    if args.out_dir:
        out = Path(args.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"picks_{args.date}.md").write_text(text, encoding="utf-8")
        pd.DataFrame(rows).to_csv(out / f"picks_{args.date}.csv", index=False)
    if not args.dry_run:
        now = dt.datetime.now().astimezone()
        PRED_DIR.mkdir(parents=True, exist_ok=True)
        path = PRED_DIR / f"{target_date.year}.jsonl"
        written = 0
        with open(path, "a", encoding="utf-8") as handle:
            for r in rows:
                if dt.datetime.fromisoformat(r["start_utc"]) <= now:
                    continue
                pick = r.get("value_side") if r["action"] == "BET" else r.get("fav")
                record = {"created_at": now.isoformat(timespec="seconds"), "game_id": r["event_ticker"], "start_time": r["start_utc"],
                          "pick": pick, "model_prob": r.get("value_prob") if r["action"] == "BET" else r.get("fav_prob"),
                          "market_price": r.get("value_ask"), "price_time": r["price_time"], "fetched_at": r["fetched_at"],
                          "source": "kalshi_public", "model_version": MODEL_VERSION, "action": r["action"], "reason": r["reason"],
                          "p_a": r.get("p_a"), "player_a": r["player_a"], "player_b": r["player_b"], "fee_basis": "tim_est_0.02"}
                handle.write(json.dumps(record) + "\n")
                written += 1
        print(f"\nappended {written} rows to {path}")


if __name__ == "__main__":
    Main()