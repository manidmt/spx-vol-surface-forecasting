# Model round 3a (cross-asset inputs already on disk) — design and plan

Date: 2026-11-27. Builds on round 2 (`docs/superpowers/plans/2026-11-27-model-round-2.md`).

## Intent

Round 2 found that the event calendar and the underlying, entering through a linear branch, do
not improve the networks. Round 3a asks the same question of the **cross-asset** data already
downloaded in the first Bloomberg visit: rates, credit, the implied volatility of other markets
and the dispersion of the S&P 500 sectors. They carry information the surface does not contain
by construction (the exclusion rule of the project: no input that is a deterministic function of
the target). The data of the second visit come in round 3b.

Same question as round 2, block by block: for the same network, does adding a block of inputs
beat the same network without it?

## Timing convention

Unchanged from round 2: the panel row of date *d* holds only what is known at the end of the
trading day *d*; the forecast for day *t* uses the row of *t−1*.

Each series has its own calendar (foreign holidays, bond-market closures). A series is aligned to
the surface dates **as of** each date: the last value dated on or before *d*, at most 5 business
days old (older → NaN). Changes are taken on the aligned series, between consecutive surface
dates, so a change never uses a value dated after *d*.

Closing times relative to the surface (SPX options close at 16:15 ET):

| Series | Snapshot | Publication lag |
|---|---|---|
| Sector ETFs (`PX_LAST`), sector and HYG 30-day ATM implied vol, OVX | 16:00-16:15 ET | 0 |
| Fed funds futures FF1-FF4 | CME settlement, afternoon ET | 0 |
| Bund yields `GDBR2`, `GDBR10` | European close, 11:30 ET | 0 (older information) |
| Treasury yields `USGG*`, MOVE, credit OAS indices | end of the U.S. day ⚠️ exact time unverified | 0 |
| FX smile (`BGN`) | 17:00 ET | 0 (45 min after the surface: declared asymmetry, still before the target close of *t+1*) |
| `SOFRRATE`, `FEDL01` (SOFR, effective fed funds) | **dated by the day the rate applies to, published the next business morning** (8:00 / 9:00 ET) | **1 surface day** |

The SOFR row was checked on the data: the repo spike of 2019-09-17 (5.25 %) is dated
2019-09-17, but it was published on 2019-09-18.

## Pre-registration

Fixed now, before any run; every run reported; Holm-adjusted DM p-values within each family.
Units: yields and spreads in basis points, vols in vol points, returns in percent.

**Block `rates`** (6 features)
- `us2y_chg`, `us10y_chg`: daily change of `USGG2YR`, `USGG10YR` (bp);
- `us_slope`: `USGG10YR − USGG2YR` (bp, level);
- `bund10_chg`: daily change of `GDBR10` (bp);
- `ff_path_chg`: daily change of the priced policy path over three months,
  `(100 − FF4) − (100 − FF1)` (bp);
- `funding`: `SOFRRATE − FEDL01` (bp, level, both lagged one day).

**Block `credit`** (4)
- `ig_oas_chg`, `hy_oas_chg`: daily change of `LUACOAS`, `LF98OAS` (bp);
- `hyg_iv`, `hyg_iv_chg`: HYG 30-day ATM implied vol, level and daily change.

**Block `cross_vol`** (6)
- `move_chg`, `ovx_chg`: 100 · daily log change of MOVE and OVX;
- `eur_atm_chg`, `jpy_atm_chg`: daily change of `EURUSDV1M`, `USDJPYV1M`;
- `jpy_rr`, `jpy_rr_chg`: `USDJPY25R1M` (25-delta 1-month risk reversal), level and daily change.

**Block `sectors`** (3), over the 11 SPDR sector ETFs
- `sec_iv_chg`: daily change of the mean 30-day ATM implied vol;
- `sec_iv_disp`: cross-sectional standard deviation of the implied vols (level);
- `sec_ret_disp`: cross-sectional standard deviation of the daily log returns (percent).

Left out on purpose: oil and FX returns (they act on the surface through the SPX return, already
tested in round 2; OVX carries the oil shock), the E-mini futures (they need roll alignment and
belong with the implied dividend of the option chains, round 3b), PCA of the curves (it would have
to be fitted inside each refit; the hand-made level / slope / change features are the simple
version).

**Network branch**: unchanged from round 2 (standardisation on the training rows, zero-initialised
`Linear(n_features, 63)` added to the output, training rows with non-finite inputs dropped).

| Run name | Network | Blocks |
|---|---|---|
| `cnn_conv_head_rates`, `convlstm_rates` | CNN shared head / ConvLSTM | rates |
| `cnn_conv_head_credit`, `convlstm_credit` | same | credit |
| `cnn_conv_head_xvol`, `convlstm_xvol` | same | cross_vol |
| `cnn_conv_head_sectors`, `convlstm_sectors` | same | sectors |
| `cnn_conv_head_xasset`, `convlstm_xasset` | same | rates + credit + cross_vol + sectors |

Comparisons: each run vs persistence (one Holm family of ten) and each run vs its network without
exogenous inputs (one family of five per network). Also reported: errors on the days of the 5 %
largest absolute moves of the 30-day ATM vol (identified from the realised surface, for reporting
only) vs other days.

## Tasks

### Task A — features (`src/volsurf/features.py`)
- `load_market_series(paths=...) -> pd.DataFrame`: wide frame (date × ticker) from
  `series/series_px_last_long.csv`, `fx/fx_smile_long.csv` and `ivol_etf/ivol30d_atm_long.csv`
  (implied-vol tickers prefixed so `HYG US Equity` price and implied vol stay distinct).
- `align_asof(series, index, max_age=5, lag=0)`: as-of alignment with staleness limit and
  publication lag in surface days.
- `rates_features`, `credit_features`, `cross_vol_features`, `sectors_features`, each
  `(market: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame` with the columns above.
- `build_exog`: new block names `"rates"`, `"credit"`, `"cross_vol"`, `"sectors"`; the market
  frame is loaded once.
- Tests (synthetic): as-of alignment across a foreign holiday; staleness → NaN; lag of one day
  for SOFR / EFFR; every feature formula on hand-made numbers; **no look-ahead**: changing any raw
  value dated *D* or later leaves every panel row before *D* identical (for each block). Slow
  real-data test: every block finite on ≥ 99 % of the test-period rows; the SOFR lag (row
  2019-09-17 does not contain the 5.25 % print, row 2019-09-18 does).

### Task B — configs
Ten configs from the table, `[params] exog = [...]`, otherwise identical to `cnn_conv_head.toml`
and `convlstm.toml`.

### Task C — controller
Review, merge, run the ten configs on the GPU one at a time, evaluate the two families and the
large-move breakdown, update the README.
