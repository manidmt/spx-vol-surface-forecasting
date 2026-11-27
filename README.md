# S&P 500 Implied Volatility Surface Forecasting

Master's thesis, Master in Quantitative Finance (IEB, Madrid).

The goal is to forecast the **next-day implied volatility surface of the S&P 500 (SPX)** from
information available at today's close. The current stage is a methodological review of what
neural networks can do on the surface, which is forecast directly on its grid; static no-arbitrage
constraints are left for a later stage. Forecasts are evaluated against the surface actually traded
the next day, tenor by tenor, and always relative to non-trivial baselines.

## Approach

1. **Data.** SPX implied volatility by tenor and moneyness, the underlying, and cross-asset and
   derivatives-market series (rates, credit, FX smiles, oil, sector ETF vols, macro releases),
   downloaded from a Bloomberg terminal.
2. **Representation.** The Bloomberg surface on a fixed grid (7 tenors × 9 moneyness levels),
   forecast as a whole by a single model rather than smile by smile.
3. **Models.** Targets in daily changes, walk-forward evaluation. Econometric baselines first,
   then neural networks (convolutional, LSTM), with exogenous inputs added by blocks.
4. **Evaluation.** MAE and RMSE by tenor and moneyness, Diebold-Mariano tests against the
   baselines, and an economic backtest of straddles and strangles with transaction costs.

Baselines: persistence (sticky strike / sticky moneyness), random walk in total variance and
autoregressive models on the surface factors.

## Current status

The first step was the **30-day smile** (9 moneyness points, 80-120 %) forecast one day ahead;
the second, the **full surface** (7 tenors × 9 moneyness levels).

| Notebook | Content |
|---|---|
| `01_eda_series.ipynb` | Exploratory analysis of the 30-day smile: stylised facts, bad-print detection, stationarity, distribution of daily changes, autocorrelation and PCA of smile changes |
| `02_baselines.ipynb` | Linear baselines on daily changes, walk-forward with monthly refits: per-strike AR(1) (expanding and 252-day windows, OLS and Huber), multivariate ridge using every strike as input, and a VAR(1) on three PCA factors |
| `03_cnn.ipynb` | Small 2D convolutional network over (days × moneyness) with change and level channels, Huber loss, early stopping and a 5-seed ensemble, same walk-forward protocol; ablations with the same network without nonlinearities and with a ridge on the same flattened input window |
| `04_surface_eda.ipynb` | Exploratory analysis of the full surface: bad-print detection across all tenors, volatility of daily changes by tenor and moneyness, correlation across tenors and PCA of surface changes |
| `05_surface_cnn.ipynb` | 3D convolutional network over (days × tenor × moneyness) forecasting the whole surface at once, with a shared 1×1 or a dense output head, against persistence and a per-point AR(1) |

Out-of-sample results on the 30-day smile (2021-01-04 to 2026-10-07, 1,440 days common to all
models, errors in volatility points averaged over the 9 moneyness levels):

| Model | MAE | RMSE | MAE vs persistence | R² OOS vs persistence |
|---|---|---|---|---|
| Persistence | 0.906 | 1.453 | — | — |
| AR(1), expanding window | 0.921 | 1.453 | +1.7 % | −0.000 |
| AR(1), 252-day window, Huber | 0.898 | 1.435 | −0.9 % | +0.024 |
| Ridge, 1 lag, 252-day window | 0.905 | 1.444 | −0.1 % | +0.012 |
| VAR(1) on PCA factors, 252-day window | 0.916 | 1.448 | +1.1 % | +0.007 |
| Ridge on the CNN input window (5 days, changes and levels) | 0.916 | 1.444 | +1.1 % | +0.012 |
| CNN, 5-day window | 0.889 | 1.423 | −1.9 % | +0.041 |
| CNN, 20-day window | 0.892 | 1.436 | −1.5 % | +0.023 |
| CNN, 20-day window, linear | 0.892 | 1.430 | −1.5 % | +0.031 |

No linear baseline beats persistence significantly: the best one (AR(1) with a 252-day window and
Huber loss) improves MAE by about 1 %, with Diebold-Mariano p-values of 0.06 (squared loss) and
0.08 (absolute loss). The CNN with a 5-day window is the first model to beat persistence at the
5 % level under absolute loss (Diebold-Mariano p = 0.02; p = 0.24 under squared loss), but it is not
significantly better than the best AR(1), and the same network without nonlinearities performs just
as well: the gain does not come from nonlinear structure. A ridge regression on exactly the same input window loses to
persistence (+1.1 % MAE) and to the linear CNN (Diebold-Mariano p < 0.001 under absolute loss), so
the gain comes from how the network is fitted (Huber loss, weights shared across strikes, early
stopping) rather than from its inputs.

Out-of-sample results on the full surface (same period, 1,444 days, errors averaged over the 63
grid points):

| Model | MAE | RMSE | MAE vs persistence | R² OOS vs persistence |
|---|---|---|---|---|
| Persistence | 0.460 | 0.828 | — | — |
| AR(1) per point, 252-day window, Huber | 0.457 | 0.822 | −0.6 % | +0.014 |
| 3D CNN, 5-day window, shared head | 0.455 | 0.824 | −1.1 % | +0.009 |
| 3D CNN, 5-day window, dense head | 0.455 | 0.823 | −1.1 % | +0.011 |

The CNN improves on persistence at every tenor, from −1.7 % at 30 days to −0.4 % at 24 months
(Diebold-Mariano p = 0.06 under absolute loss). Its gains concentrate on the short-dated downside
wings, while the AR(1) does better at long-dated at-the-money points, where it captures noise in
the Bloomberg surface that reverts the next day.

**Model round 1** (surface as the only input; models and hyperparameters fixed before any result,
see `docs/superpowers/plans/2026-11-27-model-round-1.md`). Same 1,444 days; Diebold-Mariano
against persistence, with p-values also Holm-adjusted over the seven models compared:

| Model | MAE | RMSE | MAE vs persistence | R² OOS | DM p (abs.) | Holm p (abs.) | DM p (sq.) |
|---|---|---|---|---|---|---|---|
| Persistence | 0.4597 | 0.8277 | — | — | — | — | — |
| AR(1) per point, 252 days, Huber | 0.4571 | 0.8218 | −0.6 % | +0.014 | 0.215 | 0.430 | 0.224 |
| 3D CNN, shared head | 0.4546 | 0.8241 | −1.1 % | +0.009 | 0.058 | 0.233 | 0.617 |
| 3D CNN, dense head | 0.4546 | 0.8231 | −1.1 % | +0.011 | 0.104 | 0.312 | 0.594 |
| LSTM (1 layer, 64 units) | 0.4559 | 0.8239 | −0.8 % | +0.009 | 0.304 | 0.430 | 0.669 |
| ConvLSTM (16 channels, 3×3) | 0.4528 | 0.8168 | −1.5 % | +0.026 | 0.028 | 0.138 | 0.258 |
| Ensemble CNN + AR(1), equal weights | 0.4534 | 0.8180 | −1.4 % | +0.024 | 0.003 | 0.019 | 0.097 |
| Ensemble CNN + LSTM + ConvLSTM + AR(1) | 0.4520 | 0.8162 | −1.7 % | +0.028 | 0.003 | 0.019 | 0.118 |

The two ensembles are the first forecasts to beat persistence significantly after correcting for
multiple comparisons (absolute loss); under squared loss no model is significant. The ConvLSTM
is the best single network. The CNN + AR(1) ensemble was motivated by the earlier observation
that the two models win in different regions of the surface, an observation made on this same
test period; the four-model ensemble is free of that choice.

**Model round 2** (simple exogenous inputs already on disk, see
`docs/superpowers/plans/2026-11-27-model-round-2.md`): a *calendar* block (business days to the
next scheduled FOMC decision, CPI and payrolls release, and whether one falls before the next
session, from the schedule as it was known at each close, revisions included) and an *underlying*
block (SPX return, its negative part, 5- and 21-day realised volatility, daily range). Each block
enters the network through one zero-initialised linear layer added to its output. Same protocol,
1,445 days common to these runs; Holm adjustment over the six exogenous runs against persistence
and over the three variants of each network against that network:

| Model | MAE | MAE vs persistence | DM p vs persistence (abs.) | Holm p | DM p vs same network without inputs (abs.) |
|---|---|---|---|---|---|
| 3D CNN, shared head | 0.4544 | −1.1 % | 0.052 | — | — |
| + calendar | 0.4554 | −0.9 % | 0.119 | 0.227 | 0.504 (worse) |
| + underlying | 0.4543 | −1.2 % | 0.042 | 0.179 | 0.924 |
| + calendar + underlying | 0.4542 | −1.2 % | 0.036 | 0.179 | 0.877 |
| ConvLSTM | 0.4527 | −1.5 % | 0.025 | — | — |
| + calendar | 0.4548 | −1.1 % | 0.114 | 0.227 | 0.184 (worse) |
| + underlying | 0.4540 | −1.2 % | 0.048 | 0.179 | 0.387 (worse) |
| + calendar + underlying | 0.4537 | −1.3 % | 0.026 | 0.156 | 0.578 (worse) |

Neither block improves on the same network without it, and no model is significant under squared
loss. The underlying block shifts the gains from the short end to the long end of the surface
(−1.1 % at 24 months against −0.4 % for the CNN alone, but −1.3 % instead of −1.7 % at 30 days).
On the day of a scheduled payrolls release persistence errs more (MAE 0.57 against 0.44 on days
with no scheduled event) and every network does worse than persistence there (+0.9 % to +3.2 %
MAE, 69 days); the calendar block does not change this consistently.

**Model round 3a** (cross-asset inputs from the first Bloomberg download, see
`docs/superpowers/plans/2026-11-27-model-round-3a.md`), same linear branch, four blocks:
*rates* (2- and 10-year Treasury and 10-year Bund changes, curve slope, change in the priced
fed funds path, SOFR minus effective fed funds, both published the next morning and lagged a
day), *credit* (IG and HY OAS changes, HYG implied vol), *cross-asset vol* (MOVE, OVX, EURUSD and
USDJPY one-month ATM vol, USDJPY risk reversal) and *sectors* (mean and dispersion of the implied
vols of the 11 SPDR sector ETFs, dispersion of their returns). Holm over the ten runs against
persistence and over the five variants of each network against that network (absolute loss):

| Inputs | CNN: MAE vs persistence | DM p vs CNN alone (Holm) | ConvLSTM: MAE vs persistence | DM p vs ConvLSTM alone (Holm) |
|---|---|---|---|---|
| none (round 1) | −1.1 % | — | −1.5 % | — |
| rates | +0.3 % | 0.001 (0.004), worse | +0.1 % | < 0.001 (0.001), worse |
| credit | −0.7 % | 0.234 (0.602) | −0.3 % | 0.026 (0.053), worse |
| cross-asset vol | −0.7 % | 0.201 (0.602) | −0.4 % | 0.004 (0.012), worse |
| sectors | −0.9 % | 0.304 (0.602) | −1.2 % | 0.330 (0.330) |
| all four | +0.0 % | 0.017 (0.070), worse | +0.5 % | 0.001 (0.004), worse |

No block helps and several make the networks significantly worse; none of the ten runs beats
persistence significantly. The damage is largest early in the test period and fades as the
training window grows (the ConvLSTM with rates loses 1.9 % against persistence in 2022 and gains
3.0 % in 2026), the pattern of a linear map fitted on a short history whose relations do not hold
in the next regime (2019-2020 for the first refits, then the 2022 hiking cycle). On the 5 % of
days with the largest moves of the 30-day ATM vol (73 days, persistence MAE 1.96 against 0.38 on
the rest) the ConvLSTM without exogenous inputs gains 3.0 %, more than any variant with them.

## Repository layout

```
src/volsurf/      experiment framework (see below)
configs/          one TOML file per experiment
tests/            pytest suite
notebooks/        numbered analysis notebooks (01-06 are the historical record of the first stage)
docs/             design notes and implementation plans
data/             not included (see below)
pyproject.toml    dependencies (managed with uv)
uv.lock
```

## Experiment framework

Every model runs under one walk-forward protocol, implemented once in `src/volsurf/`:

```
data.py            load the clean surface; drop the hold-out period unless asked
protocol.py        monthly refit schedule and the walk-forward runner
models/base.py     Forecaster interface: fit(history) and predict_change(history)
models/baselines.py  persistence and per-point AR(1) with Huber loss
models/torch_base.py shared network trainer (Huber loss, AdamW, early stopping, seed ensemble)
models/cnn.py      3D convolutional network over (days × tenor × moneyness)
metrics.py         MAE/RMSE by tenor and point, R² OOS, Diebold-Mariano
run.py, evaluate.py  command-line entry points
```

**Timing convention.** The surface of day *t* is forecast with information up to the close of
*t−1*. Models predict the change over the last observed surface and the runner adds it back, so a
model that predicts zero is persistence. The runner refits each model on the first test day of every
month using only earlier dates, and asks for each day's forecast with history up to the previous
day, as a private copy of the earlier rows: a model cannot see future data even if it has a bug.
A test checks this for every model by replacing with noise all data after a date, and then all data
from that date on, and requiring identical forecasts up to that date (at a refit date and mid-month).

**Running experiments.**

```bash
uv run python -m volsurf.run configs/persistence.toml        # writes runs/persistence/
uv run python -m volsurf.run configs/cnn_conv_head.toml      # writes runs/cnn_conv_head/
uv run python -m volsurf.evaluate runs/persistence runs/cnn_conv_head
```

Each run saves its predictions (`predictions.parquet`), a copy of its config and metadata (git
commit, runtime, epochs per refit); a rerun replaces the previous run only once it is complete.
`evaluate` compares any set of runs on the days they all cover: MAE and RMSE overall, by tenor and by
grid point, R² OOS and Diebold-Mariano tests against persistence and, with `--reference NAME`,
against that run as well.
Dates from 2026-10-08 onwards are a clean hold-out: both commands drop them unless `--holdout` is
passed.

**Adding a model.** A network is one `nn.Module` plus a factory like `make_cnn` and a registry entry
in `models/__init__.py`; any other model subclasses `Forecaster`. Then add a TOML config.

**Tests.** `uv run pytest` runs the suite; tests that need the (private) data are marked `slow` and
skipped when the data are missing.

## Data

The data come from a Bloomberg terminal and **are not included** in this repository for
licensing reasons, so the notebooks will not run as-is. They expect this layout:

```
data/raw/bloomberg/
├── ovdv_impvol/SPX_IMPVOL_<tenor>.csv   # tenor: 30DAY, 60DAY, 3MTH, 6MTH, 12MTH, 18MTH, 24MTH
├── spx/SPX_OHLCV.csv
├── series/series_px_last_long.csv
├── fx/fx_smile_long.csv
├── ivol_etf/ivol30d_atm_long.csv
└── macro/eco_cpi_nfp.csv
```

Each `SPX_IMPVOL_<tenor>.csv` is a daily table (2019-01-02 onwards) with a `date` column and
one column per moneyness level named `<tenor>_IMPVOL_<m>%MNY_DF`, exactly as Bloomberg returns
the field (e.g. `30DAY_IMPVOL_80%MNY_DF`, `30DAY_IMPVOL_97.5%MNY_DF`), for
m ∈ {80, 90, 95, 97.5, 100, 102.5, 105, 110, 120}.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python -m ipykernel install --user --name vol-surfaces-tfm
uv run jupyter lab
```
