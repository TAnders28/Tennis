r"""Point-by-point Monte Carlo of tennis matches: fair in-play price grid.

For each serve-probability cell (p_a, p_b) and best-of-3 / best-of-5, simulate
N matches (default 1,000,000) point by point. While a match runs, every-visit
counts at the start of each game accumulate P(A wins) from that state. Pregame
P(A wins) is the fraction of simulated matches A wins (coin-toss first server).

This table is a later live-model input. It is not a betting recommendation;
no edges, stakes, or PASS/BET calls are made from it.

Exact Markov values from Backtest.Markov are stored beside the MC estimates as
a sanity check, not as a substitute for the requested simulation.

Usage:
  .venv\Scripts\python.exe -m Backtest.InPlayMonteCarlo --out-dir E:\AgentDocs\Tennibot\2026-11-13_residual-search --workers 7
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from Backtest.Markov import LiveWinProb, MatchProb

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = Path(r"E:\AgentDocs\Tennibot\2026-11-13_residual-search")
DEFAULT_BASE = "2026-11-13_residual-search"
N_SIMS = 1_000_000
P_LO, P_HI, P_STEP = 0.50, 0.80, 0.01
BATCH = 200_000
MAX_POINTS = 800


def Pack(sets_a, sets_b, games_a, games_b, server):
    sets_a = np.asarray(sets_a, dtype=np.int32)
    sets_b = np.asarray(sets_b, dtype=np.int32)
    games_a = np.asarray(games_a, dtype=np.int32)
    games_b = np.asarray(games_b, dtype=np.int32)
    server = np.asarray(server, dtype=np.int32)
    return ((sets_a << 11) | (sets_b << 8) | (games_a << 4) | (games_b << 1) | server).astype(np.int32)


def Unpack(key):
    key = np.asarray(key, dtype=np.int32)
    return (key >> 11) & 7, (key >> 8) & 7, (key >> 4) & 15, (key >> 1) & 15, key & 1


def PGrid(step=P_STEP):
    return np.round(np.arange(P_LO, P_HI + step / 2, step), 2)


def Simulate(p_a, p_b, best_of, n, rng):
    """Vectorized point-by-point simulation of n matches.

    Returns a_won, done, visit_match_index, visit_key.
    """
    need = best_of // 2 + 1
    sets_a = np.zeros(n, dtype=np.int8)
    sets_b = np.zeros(n, dtype=np.int8)
    games_a = np.zeros(n, dtype=np.int8)
    games_b = np.zeros(n, dtype=np.int8)
    pts_a = np.zeros(n, dtype=np.int8)
    pts_b = np.zeros(n, dtype=np.int8)
    server = rng.integers(0, 2, size=n, dtype=np.int8)
    in_tb = np.zeros(n, dtype=np.bool_)
    tb_first = np.zeros(n, dtype=np.int8)
    tb_played = np.zeros(n, dtype=np.int8)
    done = np.zeros(n, dtype=np.bool_)
    a_won = np.zeros(n, dtype=np.int8)
    u = np.empty(n, dtype=np.float64)
    rec_i = []
    rec_k = []
    recorded = np.zeros(n, dtype=np.bool_)

    for _ in range(MAX_POINTS):
        active = ~done
        if not active.any():
            break
        tb = in_tb & active
        rg = (~in_tb) & active

        start_game = rg & (pts_a == 0) & (pts_b == 0) & (~recorded)
        if start_game.any():
            idx = np.flatnonzero(start_game)
            rec_i.append(idx)
            rec_k.append(Pack(sets_a[idx], sets_b[idx], games_a[idx], games_b[idx], server[idx]))
            recorded[start_game] = True

        rng.random(n, out=u)
        a_point = u < np.where(server == 0, p_a, 1.0 - p_b)

        if tb.any():
            pts_a[tb & a_point] += 1
            pts_b[tb & ~a_point] += 1
            tb_played[tb] += 1
            a_tb_win = tb & (pts_a >= 7) & ((pts_a - pts_b) >= 2)
            b_tb_win = tb & (pts_b >= 7) & ((pts_b - pts_a) >= 2)
            if a_tb_win.any() or b_tb_win.any():
                sets_a[a_tb_win] += 1
                sets_b[b_tb_win] += 1
                finished_tb = a_tb_win | b_tb_win
                server[finished_tb] = 1 - tb_first[finished_tb]
                games_a[finished_tb] = 0
                games_b[finished_tb] = 0
                pts_a[finished_tb] = 0
                pts_b[finished_tb] = 0
                in_tb[finished_tb] = False
                tb_played[finished_tb] = 0
                recorded[finished_tb] = False
                just_won = finished_tb & (sets_a >= need)
                just_lost = finished_tb & (sets_b >= need)
                done[just_won | just_lost] = True
                a_won[just_won] = 1
            still_tb = in_tb & active & (~done)
            if still_tb.any():
                nxt = (tb_played[still_tb] + 1) // 2
                server[still_tb] = np.where(nxt % 2 == 0, tb_first[still_tb], 1 - tb_first[still_tb])

        if rg.any():
            pts_a[rg & a_point] += 1
            pts_b[rg & ~a_point] += 1
            a_gw = rg & (pts_a >= 4) & ((pts_a - pts_b) >= 2)
            b_gw = rg & (pts_b >= 4) & ((pts_b - pts_a) >= 2)
            if a_gw.any() or b_gw.any():
                games_a[a_gw] += 1
                games_b[b_gw] += 1
                finished_g = a_gw | b_gw
                pts_a[finished_g] = 0
                pts_b[finished_g] = 0
                server[finished_g] = 1 - server[finished_g]
                recorded[finished_g] = False
                a_sw = finished_g & (games_a >= 6) & ((games_a - games_b) >= 2)
                b_sw = finished_g & (games_b >= 6) & ((games_b - games_a) >= 2)
                if a_sw.any() or b_sw.any():
                    sets_a[a_sw] += 1
                    sets_b[b_sw] += 1
                    finished_s = a_sw | b_sw
                    games_a[finished_s] = 0
                    games_b[finished_s] = 0
                    recorded[finished_s] = False
                    just_won = finished_s & (sets_a >= need)
                    just_lost = finished_s & (sets_b >= need)
                    done[just_won | just_lost] = True
                    a_won[just_won] = 1
                start_tb = finished_g & (~done) & (games_a == 6) & (games_b == 6)
                if start_tb.any():
                    in_tb[start_tb] = True
                    tb_first[start_tb] = server[start_tb]
                    tb_played[start_tb] = 0
                    pts_a[start_tb] = 0
                    pts_b[start_tb] = 0

    if rec_i:
        visit_i = np.concatenate(rec_i)
        visit_k = np.concatenate(rec_k)
    else:
        visit_i = np.empty(0, dtype=np.int64)
        visit_k = np.empty(0, dtype=np.int32)
    return a_won, done, visit_i, visit_k


def RunCell(p_a, p_b, best_of, n_sims, seed, with_exact_inplay=True):
    rng = np.random.default_rng(seed)
    visits = np.zeros(1 << 14, dtype=np.int64)
    wins = np.zeros(1 << 14, dtype=np.int64)
    won_sum = 0
    n_done = 0
    unfinished = 0
    left = n_sims
    while left > 0:
        batch = min(BATCH, left)
        a_won, done, vis_i, vis_k = Simulate(p_a, p_b, best_of, batch, rng)
        unfinished += int((~done).sum())
        n_done += int(done.sum())
        won_sum += int(a_won[done].sum())
        if vis_k.size:
            keep = done[vis_i]
            vis_k = vis_k[keep]
            vis_i = vis_i[keep]
            if vis_k.size:
                np.add.at(visits, vis_k, 1)
                np.add.at(wins, vis_k, a_won[vis_i])
        left -= batch
    p_mc = float(won_sum / n_done) if n_done else float("nan")
    exact = float(MatchProb(p_a, p_b, best_of))
    pregame = {
        "p_serve_a": float(p_a), "p_serve_b": float(p_b), "best_of": int(best_of),
        "sets_a": 0, "sets_b": 0, "games_a": 0, "games_b": 0, "server": "either",
        "n": int(n_done), "unfinished": int(unfinished),
        "mc_p_win_a": p_mc, "exact_p_win_a": exact,
        "mc_minus_exact": (p_mc - exact) if n_done else float("nan"),
    }
    inplay = []
    keys = np.flatnonzero(visits)
    if keys.size:
        sa, sb, ga, gb, sv = Unpack(keys)
        for i in range(keys.size):
            n_vis = int(visits[keys[i]])
            p_hat = float(wins[keys[i]] / n_vis)
            exact_s = float("nan")
            if with_exact_inplay:
                try:
                    exact_s = float(LiveWinProb(
                        p_a, p_b, int(sa[i]), int(sb[i]), int(ga[i]), int(gb[i]),
                        0, 0, "a" if sv[i] == 0 else "b", best_of=best_of))
                except Exception:
                    exact_s = float("nan")
            inplay.append({
                "p_serve_a": float(p_a), "p_serve_b": float(p_b), "best_of": int(best_of),
                "sets_a": int(sa[i]), "sets_b": int(sb[i]),
                "games_a": int(ga[i]), "games_b": int(gb[i]),
                "server": "a" if sv[i] == 0 else "b",
                "n": n_vis, "unfinished": 0,
                "mc_p_win_a": p_hat, "exact_p_win_a": exact_s,
                "mc_minus_exact": (p_hat - exact_s) if np.isfinite(exact_s) else float("nan"),
            })
    return pregame, inplay


def _Init():
    try:
        import psutil
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass


def _EvalTask(task):
    p_a, p_b, best_of, n_sims, seed = task
    return RunCell(p_a, p_b, best_of, n_sims, seed)


def CellSeed(p_a, p_b, best_of, base_seed):
    return int(base_seed + 1009 * int(round(p_a * 100)) + 13 * int(round(p_b * 100)) + 7 * best_of) & 0x7FFFFFFF


def BuildTasks(n_sims, base_seed, p_values=None, best_ofs=(3, 5), step=P_STEP):
    grid = p_values if p_values is not None else PGrid(step)
    tasks = []
    for best_of in best_ofs:
        for p_a in grid:
            for p_b in grid:
                tasks.append((float(p_a), float(p_b), int(best_of), int(n_sims),
                              CellSeed(p_a, p_b, best_of, base_seed)))
    return tasks


def WriteFrames(pregame_rows, inplay_rows, out_dir, base, n_sims, workers, elapsed_s, p_step=P_STEP):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_date = dt.date.today().isoformat()
    pregame = pd.DataFrame(pregame_rows)
    inplay = pd.DataFrame(inplay_rows)
    for frame, kind in ((pregame, "pregame"), (inplay, "inplay")):
        if len(frame) == 0:
            continue
        frame.insert(0, "date", run_date)
        frame.insert(1, "series", [f"bo{int(bo)}_{kind}" for bo in frame["best_of"].to_numpy()])
    pre_path = out_dir / f"{base}_inplay-pregame.csv"
    grid_path = out_dir / f"{base}_inplay-grid.csv"
    if len(pregame):
        pregame.sort_values(["best_of", "p_serve_a", "p_serve_b"]).to_csv(pre_path, index=False)
    if len(inplay):
        inplay.sort_values(["best_of", "p_serve_a", "p_serve_b", "sets_a", "sets_b", "games_a", "games_b", "server"]).to_csv(grid_path, index=False)
    abs_err = np.abs(pregame["mc_minus_exact"].to_numpy()) if len(pregame) else np.array([])
    summary = pd.DataFrame([{
        "date": run_date,
        "series": "inplay_mc_summary",
        "n_sims_per_cell": int(n_sims),
        "n_cells": int(len(pregame)),
        "n_inplay_rows": int(len(inplay)),
        "p_lo": P_LO, "p_hi": P_HI, "p_step": float(p_step),
        "workers": int(workers),
        "elapsed_s": round(float(elapsed_s), 1),
        "mean_abs_mc_minus_exact_pregame": float(np.nanmean(abs_err)) if abs_err.size else np.nan,
        "max_abs_mc_minus_exact_pregame": float(np.nanmax(abs_err)) if abs_err.size else np.nan,
        "unfinished_total": int(pregame["unfinished"].sum()) if len(pregame) else 0,
        "note": "Monte Carlo fair-price grid only; no betting claims.",
    }])
    sum_path = out_dir / f"{base}_inplay-summary.csv"
    summary.to_csv(sum_path, index=False)
    return pre_path, grid_path, sum_path, summary.iloc[0].to_dict()


def Main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--n-sims", type=int, default=N_SIMS)
    parser.add_argument("--seed", type=int, default=20261113)
    parser.add_argument("--p-step", type=float, default=P_STEP)
    parser.add_argument("--smoke", action="store_true", help="tiny grid and 5k sims, for tests")
    args = parser.parse_args(argv)

    global BATCH
    if args.smoke:
        args.n_sims = min(args.n_sims, 5000)
        p_values = np.array([0.55, 0.65, 0.75])
        best_ofs = (3,)
        BATCH = min(BATCH, args.n_sims)
        p_step = 0.10
    else:
        p_values = PGrid(args.p_step)
        best_ofs = (3, 5)
        p_step = args.p_step

    cap = max(1, (os.cpu_count() or 28) // 4)
    workers = max(1, min(int(args.workers), cap))
    tasks = BuildTasks(args.n_sims, args.seed, p_values=p_values, best_ofs=best_ofs, step=p_step)
    print(f"inplay MC: cells={len(tasks)} n_sims={args.n_sims:,} workers={workers} "
          f"p={p_values[0]:.2f}..{p_values[-1]:.2f} step={p_step:.2f}", flush=True)

    check_n = min(20_000, args.n_sims)
    pre, _ = RunCell(0.65, 0.62, 3, check_n, args.seed)
    print(f"self-check bo3 p=0.65/0.62 n={check_n:,}: mc={pre['mc_p_win_a']:.4f} "
          f"exact={pre['exact_p_win_a']:.4f} unfinished={pre['unfinished']}", flush=True)
    if not np.isfinite(pre["mc_p_win_a"]) or abs(pre["mc_minus_exact"]) > 0.05:
        raise SystemExit(f"self-check failed: {pre}")

    t0 = dt.datetime.now()
    pregame_rows, inplay_rows = [], []
    pool = None
    if workers == 1:
        iterator = map(_EvalTask, tasks)
    else:
        pool = Pool(workers, initializer=_Init)
        iterator = pool.imap_unordered(_EvalTask, tasks, chunksize=1)
    try:
        done = 0
        for pre, inp in iterator:
            pregame_rows.append(pre)
            inplay_rows.extend(inp)
            done += 1
            if done == 1 or done % 25 == 0 or done == len(tasks):
                elapsed = (dt.datetime.now() - t0).total_seconds()
                print(f"  {done}/{len(tasks)} cells  {elapsed:.0f}s", flush=True)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
    elapsed = (dt.datetime.now() - t0).total_seconds()
    pre_path, grid_path, sum_path, summary = WriteFrames(
        pregame_rows, inplay_rows, args.out_dir, args.base, args.n_sims, workers, elapsed, p_step=p_step)
    print("wrote", pre_path, flush=True)
    print("wrote", grid_path, flush=True)
    print("wrote", sum_path, flush=True)
    print(summary, flush=True)


if __name__ == "__main__":
    Main()
