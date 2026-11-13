# tennis

Tennis statistics and pre-match / in-play win-probability models with leak-free backtests, scored against
real prices (Kalshi public quotes; tennis-data.co.uk bookmaker odds once downloaded manually).
Same layout and conventions as `basketball` / `football` (see `AGENTS.md`).

## Setup

    powershell -ExecutionPolicy Bypass -File .\bootstrap.ps1

Python 3.14 (`E:\Installs\Python314`), binary wheels only (no compiler on this PC). Pins in `requirements.lock.txt`.

## Models

- **Surface Elo**: overall + Hard/Clay/Grass ratings blended with weight w on surface; K = a/(n+b)^c per player;
  best-of-5 stretches the rating gap; carpet is treated as Hard.
- **Serve/return**: per-player serve-points-won and return-points-won deviations (surface + all-surface, opponent
  adjusted, decayed, shrunk to the tour mean) -> point-level p_serve for each player -> exact Markov match probability.
- **Blend**: logit blend of the two (weight tuned on validation seasons).
- **In-play Markov**: exact P(win) from any score state (bo3/bo5; final-set rules by slam and year: advantage,
  7-pt tiebreak at 6-6 or 12-12, 10-pt tiebreak at 6-6), with optional in-match learning of p_serve.

## Data sources and licences

- **Jeff Sackmann tennis_atp / tennis_wta / tennis_slam_pointbypoint** - CC BY-NC-SA 4.0 (attribution,
  non-commercial, share-alike). Data by Jeff Sackmann / Tennis Abstract. The original GitHub repos returned 404
  from about 2026-10-14; this repo pulls the data-only archival mirror `Aneeshers/tennis-sackmann-archive`
  (June 2026 snapshot, so results end in early June 2026). Non-singles point-by-point (tennis_pointbypoint) is not
  available in that mirror.
- **Kalshi public market data** (read-only, no account): tennis match markets from June 2025. Candles have
  top-of-book bid/ask but no size; the live market list has sizes but no quote timestamp.
- **tennis-data.co.uk** - free for private individuals, not for bots/scrapers/AI. Scripted requests get HTTP 403,
  so files are downloaded by hand into `data/raw/tennis_data/<tour>/<year>.xlsx` (see `DataPull/PullTennisData.py`).

## Outputs

Backtest outputs go to `data/backtest/` (gitignored). Finished reports go to `E:\AgentDocs\Tennibot\<date>_<slug>\`.