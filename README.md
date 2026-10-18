# S&P 500 Implied Volatility Surface Forecasting

Master's thesis, Master in Quantitative Finance (IEB, Madrid).

The goal is to forecast the **next-day implied volatility surface of the S&P 500 (SPX)** from
information available at today's close, and to do it **without static arbitrage**. Forecasts are
evaluated against the surface actually traded the next day, tenor by tenor, and always relative to
non-trivial baselines.

## Approach

1. **Data.** SPX implied volatility by tenor and moneyness, the underlying, and cross-asset and
   derivatives-market series (rates, credit, FX smiles, oil, sector ETF vols, macro releases),
   downloaded from a Bloomberg terminal.
2. **Surface construction.** Rebuild the surface from option chains: implied forward from put-call
   parity, Black-76 inversion, OTM selection, log-moneyness and total variance, static
   no-arbitrage checks and SVI calibration.
3. **Representation.** Forecast a low-dimensional, arbitrage-aware description of the surface
   (SVI/SSVI parameters, with PCA factors as a benchmark) instead of each grid point on its own.
4. **Models.** Targets in daily changes, walk-forward evaluation. Econometric baselines first,
   then gradient boosting and neural networks (convolutional, LSTM).
5. **Evaluation.** MAE and RMSE by tenor and moneyness, Diebold-Mariano tests against the
   baselines, and an economic backtest of straddles and strangles with transaction costs.

Baselines: persistence (sticky strike / sticky moneyness), random walk in total variance and
autoregressive models on the surface factors.

## Current status

The first step is the **30-day smile** (9 moneyness points, 80-120 %) forecast one day ahead,
before extending to more tenors and to the full surface.

| Notebook | Content |
|---|---|
| `01_eda_series.ipynb` | Exploratory analysis of the 30-day smile: stylised facts, bad-print detection, stationarity, distribution of daily changes, autocorrelation and PCA of smile changes |
| `02_baselines.ipynb` | Linear baselines on daily changes, walk-forward with monthly refits: per-strike AR(1) (expanding and 252-day windows, OLS and Huber), multivariate ridge using every strike as input, and a VAR(1) on three PCA factors |

Out-of-sample results on the 30-day smile (2021-01-04 to 2026-10-07, 1,440 days common to all
models, errors in volatility points averaged over the 9 moneyness levels):

| Model | MAE | RMSE | MAE vs persistence | R² OOS vs persistence |
|---|---|---|---|---|
| Persistence | 0.906 | 1.453 | — | — |
| AR(1), expanding window | 0.921 | 1.453 | +1.7 % | −0.000 |
| AR(1), 252-day window, Huber | 0.898 | 1.435 | −0.9 % | +0.024 |
| Ridge, 1 lag, 252-day window | 0.905 | 1.444 | −0.1 % | +0.012 |
| VAR(1) on PCA factors, 252-day window | 0.916 | 1.448 | +1.1 % | +0.007 |

No linear baseline beats persistence significantly: the best one (AR(1) with a 252-day window and
Huber loss) improves MAE by about 1 %, with Diebold-Mariano p-values of 0.06 (squared loss) and
0.08 (absolute loss).

## Repository layout

```
notebooks/        numbered analysis notebooks
data/             not included (see below)
pyproject.toml    dependencies (managed with uv)
uv.lock
```

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
