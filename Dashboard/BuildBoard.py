r"""Build GutCall tennis board JSON for today's and tomorrow's Kalshi match slate.

Surface-Elo win probabilities only (not the serve/return blend). Edge is in cents:

    (model_prob - yes_ask - 0.02) * 100

The $0.02 is Tim's fee estimate, not a verified exchange fee. Kalshi's verified
taker fee 0.07*p*(1-p) is also reported in additive fields. Unknown values are
null; nothing is invented. Recommendations only; no orders.

Follows E:\AgentWorkspaces\GutCall\gutcall\SCHEMA.md (read after the first draft). Extra fee fields are additive: sport, generated_at (ISO with offset), games[] with
game_id, start_time, home (=player1), away (=player2), status, score,
market{home_yes_ask, away_yes_ask, open_home_yes, source, as_of},
model{home_win_prob, model_name, tested}, edge_home_c, edge_away_c,
context{injuries, recent_form, rest_days, head_to_head, notes[]}, props[].

Usage:
  .venv\Scripts\python.exe -m Dashboard.BuildBoard --date 2026-11-14 --out E:\AgentDocs\Tennibot\_dashboard\board_latest.json
  .\RunJob.ps1 -Name build_board -Module Dashboard.BuildBoard -ModuleArgs "--date 2026-11-14 --out E:\AgentDocs\Tennibot\_dashboard\board_latest.json"
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from zoneinfo import ZoneInfo

from Backtest.Odds import TIM_FEE, KalshiFee
from Backtest.PredictDay import BuildState, Resolve, FRESH_DAYS, MIN_MATCHES
from Backtest.PropMarkets import SERIES as PROP_SERIES_MAP, ParsePropMarket
from Backtest.LoadMatches import LoadMatches
from LiveOdds.KalshiPublic import KalshiPublic, MATCH_SERIES

ET = ZoneInfo("America/New_York")
DEFAULT_OUT = Path(r"E:\AgentDocs\Tennibot\_dashboard\board_latest.json")
MODEL_NAME = "surface_elo"
# Surface Elo has a leak-free holdout in this repo (Kalshi-reality / market-blend)
# but it did not pass: blend weight on the model was ~0 and no fee-adjusted edge survived.
# SCHEMA.md gates a green VALUE badge on tested=true AND edge >= 3c, so tested is False.
MODEL_TESTED = False
HOLDOUT_NOTE = (
    "surface Elo was holdout-tested and lost to Kalshi "
    "(market-blend model weight c~0; no validated fee-adjusted edge); "
    "tested=false so GutCall will not flag VALUE; numbers are decision support, not a betting edge"
)
FEE_NOTE = (
    "edge_home_c / edge_away_c use Tim's $0.02 fee estimate (tim_est_0.02, not a verified fee); "
    "edge_home_c_kalshi / edge_away_c_kalshi use Kalshi's verified taker fee 0.07*p*(1-p)"
)
OPEN_PRICE_NOTE = "open_home_yes is null unless a clearly named open-price field is on the public market list; last trade is not used as open"
SACKMANN_CUTOFF_NOTE = "Sackmann match data ends {last}; recent_form and rest_days are stale/null vs this slate; head_to_head is local history through that date only"
FORM_N = 5


def EdgeCents(prob, ask, fee=TIM_FEE):
    """Fee-adjusted edge in cents, or None when prob or ask is missing/invalid."""
    if prob is None or ask is None:
        return None
    try:
        prob = float(prob)
        ask = float(ask)
    except (TypeError, ValueError):
        return None
    if not (0.0 < ask < 1.0) or not (0.0 <= prob <= 1.0):
        return None
    return round((prob - ask - fee) * 100.0, 2)


def KalshiEdgeCents(prob, ask):
    if prob is None or ask is None:
        return None
    try:
        ask = float(ask)
    except (TypeError, ValueError):
        return None
    if not (0.0 < ask < 1.0):
        return None
    return EdgeCents(prob, ask, fee=KalshiFee(ask))


def ParseStart(text):
    if not text:
        return None
    try:
        return dt.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None


def EventSuffix(event_ticker):
    if not event_ticker or "-" not in event_ticker:
        return event_ticker or ""
    return event_ticker.split("-", 1)[1]


def InDateWindow(start, date_from, date_to):
    if start is None:
        return False
    if start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    day = start.astimezone(ET).date()
    return date_from <= day <= date_to


def OpenPrice(market):
    """Only a clearly named open-price field counts; last/previous are not open."""
    if not market:
        return None
    for key in ("yes_open_dollars", "open_price_dollars", "yes_open", "open_yes_dollars"):
        value = market.get(key)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def JsonSafe(value):
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if hasattr(value, "item") and not isinstance(value, (bytes, str, dict, list)):
        try:
            return JsonSafe(value.item())
        except (ValueError, AttributeError):
            pass
    if isinstance(value, dt.datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, dt.date):
        return value.isoformat()
    return value


def Clean(obj):
    if isinstance(obj, dict):
        return {k: Clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [Clean(v) for v in obj]
    return JsonSafe(obj)


def RecentForm(frame, pid, n=FORM_N):
    if frame is None or pid is None or len(frame) == 0:
        return None
    rows = frame[(frame["winner_id"] == pid) | (frame["loser_id"] == pid)].sort_values("date")
    if rows.empty:
        return None
    letters = []
    for rec in rows.tail(n).itertuples(index=False):
        won = rec.winner_id == pid
        tag = "W" if won else "L"
        score = str(getattr(rec, "score", "") or "")
        if "RET" in score.upper() or "DEF" in score.upper():
            tag += "*"
        letters.append(tag)
    last = rows["date"].max()
    last_date = last.date().isoformat() if hasattr(last, "date") else str(last)[:10]
    return {"results": "".join(letters), "through": last_date, "n": int(len(letters))}


def RestDays(frame, pid, slate_date):
    if frame is None or pid is None or len(frame) == 0:
        return None
    rows = frame[(frame["winner_id"] == pid) | (frame["loser_id"] == pid)]
    if rows.empty:
        return None
    last = rows["date"].max()
    last_date = last.date() if hasattr(last, "date") else last
    return int((slate_date - last_date).days)


def HeadToHead(frame, id_home, id_away):
    if frame is None or id_home is None or id_away is None or len(frame) == 0:
        return None
    mask = (
        ((frame["winner_id"] == id_home) & (frame["loser_id"] == id_away))
        | ((frame["winner_id"] == id_away) & (frame["loser_id"] == id_home))
    )
    rows = frame.loc[mask].sort_values("date")
    if rows.empty:
        return None
    home_wins = int((rows["winner_id"] == id_home).sum())
    away_wins = int((rows["winner_id"] == id_away).sum())
    last = rows["date"].max()
    last_date = last.date().isoformat() if hasattr(last, "date") else str(last)[:10]
    return (
        f"home {home_wins}-{away_wins} away in {int(len(rows))} meetings through {last_date} "
        f"(local Sackmann only; later meetings unknown)"
    )


def BuildContext(frame, id_home, id_away, slate_date, sackmann_last, extra_notes=None):
    notes = list(extra_notes or [])
    injuries = None
    recent_form = None
    rest_days = None
    head_to_head = None
    last = sackmann_last
    if last is not None and hasattr(last, "isoformat"):
        last_s = last.isoformat()
    elif last is not None:
        last_s = str(last)
    else:
        last_s = None
    if last_s:
        notes.append(SACKMANN_CUTOFF_NOTE.format(last=last_s))
    else:
        notes.append("no local Sackmann match table loaded; recent_form, rest_days, head_to_head left null")

    stale = last is None or (slate_date - last).days > FRESH_DAYS
    covered = id_home is not None and id_away is not None and frame is not None and len(frame) > 0
    if not covered:
        notes.append("player identity or local match history missing; form/rest/h2h left null")
    else:
        if stale:
            notes.append(
                f"recent_form and rest_days set null because local results (through {last_s}) "
                f"are stale relative to slate {slate_date.isoformat()} (fresh window {FRESH_DAYS} days)"
            )
        else:
            def FormText(block):
                if not block:
                    return None
                return f"{block['results']} last {block['n']} through {block['through']}"
            home_form = FormText(RecentForm(frame, id_home))
            away_form = FormText(RecentForm(frame, id_away))
            if home_form or away_form:
                recent_form = {"home": home_form, "away": away_form}
            rest_days = {"home": RestDays(frame, id_home, slate_date), "away": RestDays(frame, id_away, slate_date)}
        head_to_head = HeadToHead(frame, id_home, id_away)
        if head_to_head is None:
            notes.append(f"no head-to-head meetings in local data through {last_s}; later meetings unknown")
    notes.append("no injury feed; injuries left null")
    notes.append("fatigue unknown unless rest_days is filled from fresh local results")
    notes.append("retirement status after the local cutoff is unknown")
    notes.append(HOLDOUT_NOTE)
    notes.append(FEE_NOTE)
    # de-dupe while preserving order
    seen, uniq = set(), []
    for note in notes:
        if note not in seen:
            seen.add(note)
            uniq.append(note)
    return {
        "injuries": injuries,
        "recent_form": recent_form,
        "rest_days": rest_days,
        "head_to_head": head_to_head,
        "notes": uniq,
    }


def BuildGame(game_id, start_time, home, away, status, score, market, model_prob,
              model_name=MODEL_NAME, tested=MODEL_TESTED, context=None, props=None,
              fee=TIM_FEE):
    market = dict(market or {})
    market.setdefault("home_yes_ask", None)
    market.setdefault("away_yes_ask", None)
    market.setdefault("open_home_yes", None)
    market.setdefault("source", "Kalshi")
    market.setdefault("as_of", None)
    if market.get("open_home_yes") is None:
        market["open_home_yes"] = OpenPrice(market)
    p_home = model_prob
    p_away = None if p_home is None else (1.0 - p_home)
    game = {
        "game_id": game_id,
        "start_time": start_time,
        "home": home,
        "away": away,
        "status": status,
        "score": score,
        "market": {
            "home_yes_ask": JsonSafe(market.get("home_yes_ask")),
            "away_yes_ask": JsonSafe(market.get("away_yes_ask")),
            "open_home_yes": JsonSafe(market.get("open_home_yes")),
            "source": market.get("source") or "Kalshi",
            "as_of": market.get("as_of"),
        },
        "model": {
            "home_win_prob": JsonSafe(p_home if p_home is None else round(float(p_home), 6)),
            "model_name": model_name,
            "tested": bool(tested),
            "fee_basis": "tim_est_0.02",
            "holdout_note": HOLDOUT_NOTE if model_name == MODEL_NAME else ("this model has no holdout backtest in the repo" if not tested else None),
        },
        "edge_home_c": EdgeCents(p_home, market.get("home_yes_ask"), fee=fee),
        "edge_away_c": EdgeCents(p_away, market.get("away_yes_ask"), fee=fee),
        "edge_home_c_kalshi": KalshiEdgeCents(p_home, market.get("home_yes_ask")),
        "edge_away_c_kalshi": KalshiEdgeCents(p_away, market.get("away_yes_ask")),
        "context": context or {
            "injuries": None,
            "recent_form": None,
            "rest_days": None,
            "head_to_head": None,
            "notes": [],
        },
        "props": list(props or []),
    }
    return Clean(game)


def BuildBoard(games, generated_at, sport="TENNIS"):
    return Clean({"sport": sport, "generated_at": generated_at, "games": list(games or [])})


def PropRecord(market):
    parsed = ParsePropMarket({
        "ticker": market.get("ticker"),
        "event_ticker": market.get("event_ticker"),
        "series": market.get("series"),
        "title": market.get("title") or "",
        "yes_sub_title": market.get("yes_sub_title") or market.get("player") or "",
    })
    yes_ask = market.get("yes_ask")
    yes_bid = market.get("yes_bid")
    line = parsed.get("line")
    if line is None:
        line_s = parsed.get("side_label") or None
    elif isinstance(line, (int, float)) and float(line).is_integer():
        line_s = str(int(line))
    else:
        line_s = str(line)
    return Clean({
        # SCHEMA.md keys; model_prob/edge_c stay null (prices only, no prop model)
        "player": parsed.get("player") or market.get("yes_sub_title") or market.get("player"),
        "market": parsed.get("kind") or parsed.get("side_label") or market.get("series"),
        "line": line_s,
        "yes_ask": JsonSafe(yes_ask),
        "model_prob": None,
        "edge_c": None,
        "tested": False,
        "ticker": market.get("ticker"),
        "event_ticker": market.get("event_ticker"),
        "series": market.get("series"),
        "kind": parsed.get("kind"),
        "title": market.get("title"),
        "yes_sub_title": market.get("yes_sub_title") or market.get("player"),
        "side_label": parsed.get("side_label"),
        "yes_bid": JsonSafe(yes_bid),
        "as_of": market.get("fetched_at"),
        "source": "Kalshi",
    })


def AttachProps(games, prop_markets, related=None):
    """Join prop markets onto games by event ticker, milestone-related tickers, or suffix."""
    related = related or {}
    by_event = {}
    by_suffix = {}
    for market in prop_markets or []:
        rec = PropRecord(market)
        event = rec.get("event_ticker") or ""
        by_event.setdefault(event, []).append(rec)
        by_suffix.setdefault(EventSuffix(event), []).append(rec)
        # also index any milestone-related alias later
    for game in games:
        gid = game.get("game_id") or ""
        suffix = EventSuffix(gid)
        attached = []
        seen = set()
        candidates = list(by_event.get(gid, []))
        for alias in related.get(gid, []) or []:
            candidates.extend(by_event.get(alias, []))
        candidates.extend(by_suffix.get(suffix, []))
        for rec in candidates:
            ticker = rec.get("ticker")
            if ticker in seen:
                continue
            seen.add(ticker)
            attached.append(rec)
        game["props"] = attached
    return games


def SlamBestOf(competition, tour, level):
    text = (competition or "").lower()
    if tour == "atp" and level == "tour" and any(k in text for k in (
        "australian open", "french open", "roland", "wimbledon", "us open",
    )):
        return 5
    return 3


def GameStatus(start, market_status, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    if start is not None and start.tzinfo is None:
        start = start.replace(tzinfo=dt.timezone.utc)
    status = (market_status or "").lower() or None
    if status in ("finalized", "settled", "determined", "final"):
        return "final"
    if start is not None and start > now:
        return "scheduled"
    # SCHEMA.md allows only scheduled|live|final; map in-progress Kalshi states to live.
    if start is not None and start <= now:
        return "live"
    if status in ("active", "open", "live"):
        return "live"
    return None


def ScoreEvent(event_ticker, event, states, context_frames, date_from, date_to):
    tour = event.get("tour")
    state = states.get(tour)
    if state is None:
        return None
    markets = sorted(event.get("markets") or [], key=lambda m: m.get("ticker") or "")
    if len(markets) < 2:
        return None
    home_m, away_m = markets[0], markets[1]
    start = ParseStart(event.get("start"))
    if not InDateWindow(start, date_from, date_to):
        return None
    slate_date = start.astimezone(ET).date()
    competition = (event.get("details") or {}).get("tournament_name") or home_m.get("rules") or home_m.get("title") or ""
    surface, surface_source = state["surfaces"].Get(competition)
    best_of = SlamBestOf(competition, tour, event.get("level"))
    home_name = home_m.get("player") or home_m.get("yes_sub_title")
    away_name = away_m.get("player") or away_m.get("yes_sub_title")
    id_home = Resolve(state["index"], home_name, state["last_seen"])
    id_away = Resolve(state["index"], away_name, state["last_seen"])
    n_home = int(state["n_matches"].get(id_home, 0)) if id_home is not None else 0
    n_away = int(state["n_matches"].get(id_away, 0)) if id_away is not None else 0
    rated = (
        id_home is not None and id_away is not None
        and n_home >= 1 and n_away >= 1
        and id_home in state["elo"].count and id_away in state["elo"].count
    )
    p_home = None
    if rated:
        p_home = state["elo"].Predict(id_home, id_away, surface, best_of)

    extra_notes = [
        f"surface: {surface} ({surface_source})",
        f"tour/level: {tour}/{event.get('level')}; best_of={best_of}",
        "home is player1 (Kalshi market with the earlier ticker); away is player2",
        OPEN_PRICE_NOTE,
    ]
    if surface_source == "guess_default_hard":
        extra_notes.append("surface was not found in the local lookup; Hard is the explicit default, not a measured value")
    if not rated:
        extra_notes.append("home_win_prob left null because one or both players are unrated in surface Elo")
    elif min(n_home, n_away) < MIN_MATCHES:
        extra_notes.append(f"thin Elo history (< {MIN_MATCHES} rated matches for at least one player)")
    sackmann_last = state.get("sackmann_last_date")
    context = BuildContext(context_frames.get(tour), id_home, id_away, slate_date, sackmann_last, extra_notes)

    as_of = home_m.get("fetched_at") or away_m.get("fetched_at")
    market = {
        "home_yes_ask": home_m.get("yes_ask"),
        "away_yes_ask": away_m.get("yes_ask"),
        "open_home_yes": OpenPrice(home_m),
        "source": "Kalshi",
        "as_of": as_of,
    }
    status = GameStatus(start, home_m.get("status") or event.get("status"))
    start_iso = start.isoformat(timespec="seconds") if start is not None else None
    return BuildGame(
        game_id=event_ticker,
        start_time=start_iso,
        home=home_name,
        away=away_name,
        status=status,
        score=None,
        market=market,
        model_prob=p_home,
        context=context,
        props=[],
    )


def RelatedTickers(events):
    """Map match event ticker -> other tickers listed on the same milestone."""
    out = {}
    suffix_to_match = {}
    for ticker, event in (events or {}).items():
        suffix_to_match[EventSuffix(ticker)] = ticker
        details = event.get("details") or {}
        aliases = []
        for key in ("related_event_tickers", "primary_event_tickers", "event_tickers"):
            aliases.extend(event.get(key) or details.get(key) or [])
        if aliases:
            out.setdefault(ticker, [])
            out[ticker].extend(aliases)
    # also let callers join by shared suffix via AttachProps
    return out


def PropSeriesList():
    series = []
    for names in PROP_SERIES_MAP.values():
        series.extend(names)
    # drop empties / keep order unique
    seen, out = set(), []
    for name in series:
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def WriteAtomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def WriteBoard(payload, out_path, date_tag):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    WriteAtomic(out_path, text)
    dated = out_path.with_name(f"board_{date_tag}.json")
    if dated.resolve() != out_path.resolve():
        WriteAtomic(dated, text)
    return out_path, dated


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", default=None, help="first ET date YYYY-MM-DD (default: today ET); also includes the next day")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="path for board_latest.json; a dated copy is written alongside")
    parser.add_argument("--tours", nargs="+", default=["atp", "wta"])
    parser.add_argument("--levels", nargs="+", default=["tour", "challenger"])
    args = parser.parse_args(argv)

    generated_at = dt.datetime.now(ET).isoformat(timespec="seconds")
    date_from = dt.date.fromisoformat(args.date) if args.date else dt.datetime.now(ET).date()
    date_to = date_from + dt.timedelta(days=1)

    states = {tour: BuildState(tour, date_from) for tour in args.tours}
    context_frames = {tour: LoadMatches(tour) for tour in args.tours}

    series = [s for s, (t, lvl) in MATCH_SERIES.items() if t in args.tours and lvl in args.levels]
    since = (dt.datetime.combine(date_from, dt.time(0), ET) - dt.timedelta(days=1)).astimezone(dt.timezone.utc)
    client = KalshiPublic()
    events = client.Slate(series, since.strftime("%Y-%m-%dT%H:%M:%SZ"))
    prop_markets = []
    for prop_series in PropSeriesList():
        prop_markets.extend(client.OpenMarkets(prop_series))

    games = []
    for ticker, event in sorted((events or {}).items()):
        game = ScoreEvent(ticker, event, states, context_frames, date_from, date_to)
        if game:
            games.append(game)
    AttachProps(games, prop_markets, RelatedTickers(events))

    payload = BuildBoard(games, generated_at)
    latest, dated = WriteBoard(payload, args.out, date_from.isoformat())
    n_model = sum(1 for g in games if g["model"]["home_win_prob"] is not None)
    n_props = sum(1 for g in games if g["props"])
    print(f"wrote {latest} and {dated}")
    print(f"games={len(games)} with_model={n_model} with_props={n_props} window={date_from.isoformat()}..{date_to.isoformat()} generated_at={generated_at}")


if __name__ == "__main__":
    Main()
