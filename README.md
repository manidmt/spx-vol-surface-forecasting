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
