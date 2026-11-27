# Model round 2 (simple exogenous inputs) — design and plan

Date: 2026-11-27. Builds on the experiment framework and round 1
(`docs/superpowers/plans/2026-11-27-model-round-1.md`).

## Intent

Round 1: with the surface as the only input, the best single network is the ConvLSTM (−1.5 % MAE
vs persistence, DM p = 0.028, Holm 0.14) and no model is significant under squared loss: large
moves are not anticipated. Round 2 asks whether **inputs already on disk** help: the scheduled
event calendar (the event premium is in the surface before an event and collapses after it) and
the underlying (leverage effect, realised volatility). The new Bloomberg data come in round 3.

The question is an ablation: for the same network, does adding a block of inputs beat the same
network without it? Each block is measured separately and together.

## Timing convention for exogenous inputs

An exogenous panel is indexed by date *d* and holds only what is **known at the close of d**.
The forecast of the surface on day *t* uses the panel row of *t−1*, exactly like the surface
window. The runner slices the panel with the same rules as the surface (fit: rows before the
refit date; predict: rows up to *t−1*), so the leakage guarantee and its tests extend to it.

## Pre-registration

Fixed now, not tuned on the test period; every run is reported; Holm-adjusted DM p-values within
each family.

**Block `calendar`** (6 features). Events: scheduled FOMC decision days (statement day, from the
Federal Reserve's calendar pages, unscheduled meetings excluded because they are not known in
advance; committed as `src/volsurf/resources/fomc_meetings.csv` with its source), CPI and NFP
release dates (`release_date` of `data/raw/bloomberg/macro/eco_cpi_nfp.csv`, tickers
`CPI CHNG Index` and `NFP TCH Index`). For each event type *j* ∈ {fomc, cpi, nfp} and date *d*:
- `{j}_next`: 1 if an event falls in (*d*, next trading date], else 0. The next trading date is the
  next date of the surface index (the exchange calendar is known in advance); for the last row,
  *d* + 1 business day.
- `{j}_days_to_next`: business days from *d* to the first event strictly after *d*
  (`numpy.busday_count`), capped at 40; NaN if no later event is known.

**Block `underlying`** (5 features), from `data/raw/bloomberg/spx/SPX_OHLCV.csv` (`PX_LAST`,
`PX_HIGH`, `PX_LOW`), in percent:
- `ret`: 100 · log(C_d / C_{d−1}); `ret_neg`: min(`ret`, 0);
- `rv5`, `rv21`: √252 · standard deviation of `ret` over the last 5 / 21 days including *d*;
- `range`: 100 · log(H_d / L_d).

**Network branch.** Exogenous inputs are standardised with the training window's mean and
standard deviation (std 0 → 1) and enter through one zero-initialised `Linear(n_features, 63)`
whose output is added to the network's output (in the same scaled-change space). Linear on
purpose (Ockham); identical for every architecture, so the untrained model is still persistence.
Training rows with any non-finite exogenous value are dropped, like non-finite windows.

| Run name | Network | Blocks |
|---|---|---|
| `cnn_conv_head_cal` | 3D CNN, shared head (as `cnn_conv_head`) | calendar |
| `cnn_conv_head_und` | same | underlying |
| `cnn_conv_head_calund` | same | calendar + underlying |
| `convlstm_cal` | ConvLSTM (as `convlstm`) | calendar |
| `convlstm_und` | same | underlying |
| `convlstm_calund` | same | calendar + underlying |

Comparisons: each run vs persistence (one Holm family of six), and each run vs its network
without exogenous inputs (`cnn_conv_head`, `convlstm`; one Holm family of three per network).
Also reported: errors on event days (forecast day *t* with `{j}_next` = 1 at *t−1*) vs other days.

## Tasks (A and B in parallel worktrees from `experiment-framework`)

Common rules as in round 1 (English, type hints, docstrings, Keras-equivalent comments, ruff,
TDD, short commits without `Co-Authored-By`, no GPU or full-size runs; real-data tests skip in
worktrees, unit tests use synthetic data).

### Task A — features (`src/volsurf/features.py`, `src/volsurf/resources/fomc_meetings.csv`)
- `fomc_meetings.csv`: columns `decision_date`, `meeting_days`; scheduled meetings 2019–2026 as
  given by the controller; header comment line with the source URLs.
- `load_event_dates(path_eco: Path = ECO_PATH) -> dict[str, pd.DatetimeIndex]` with keys
  `"fomc"`, `"cpi"`, `"nfp"` (sorted, unique).
- `calendar_features(index: pd.DatetimeIndex, events: dict[str, pd.DatetimeIndex]) -> pd.DataFrame`
  — the 6 columns above, indexed by `index`.
- `underlying_features(spx: pd.DataFrame) -> pd.DataFrame` — the 5 columns above, indexed by
  the SPX dates; `load_spx(path: Path = SPX_PATH) -> pd.DataFrame`.
- `build_exog(blocks: list[str], index: pd.DatetimeIndex) -> pd.DataFrame` — concatenates the
  requested blocks (`"calendar"`, `"underlying"`), reindexed to `index`; unknown block →
  `ValueError` listing the valid ones.
- Tests on hand-made calendars and prices: `{j}_next` on the day before an event, across a
  weekend and a holiday gap in the index, and on the last row; `days_to_next` values and cap;
  returns, negative part, rolling vols (first 4 / 20 rows NaN), range; `build_exog` column order
  and reindexing. A slow real-data test: every surface date has finite underlying features after
  the first 21 rows, event counts per year are 8 FOMC and 12 CPI and 12 NFP (± 1 at the edges).

### Task B — exogenous plumbing
- `Forecaster.fit(history, exog=None)` and `Forecaster.predict_change(history, exog=None)`;
  baselines accept and ignore `exog`.
- `walk_forward(model, surface, test_start=TEST_START, exog=None)`: when given, `exog` must cover
  every surface date (`ValueError` otherwise); it is reindexed to the surface index and sliced
  with exactly the surface's rules, as an isolated copy.
- `TorchForecaster(..., n_exog: int = 0)`: when `n_exog > 0`, wraps the network in
  `ExogBranch(net, n_exog, n_points)` (`forward(x, z) = net(x) + linear(z)`, zero-initialised
  linear), standardises `exog` on the training rows, drops training rows with non-finite exog,
  and predicts from the last exog row. With `n_exog = 0` behaviour and results are unchanged.
- Configs: optional `[params] exog = ["calendar", ...]`; `run.main` calls
  `features.build_exog(blocks, surface.index)` (import inside the exog branch so Task B does not
  depend on Task A's module at import time) and passes `n_exog` and the panel; meta records the
  blocks and feature names.
- `tests/leakage.py`: `assert_no_leakage` also perturbs the exogenous panel (future noise and
  same-day noise) when one is given.
- Tests: exog slicing (spy sees exog rows ≤ *t−1* and < refit date), coverage error, untrained
  `ExogBranch` output zero, tiny CPU leakage test for the CNN with a synthetic exog panel, and a
  regression check that `n_exog = 0` reproduces the previous forecasts exactly on a tiny run.
- Six configs from the table above (exog blocks `["calendar"]`, `["underlying"]`,
  `["calendar", "underlying"]`).

### Task C — controller
Merge A and B; joint review; run the six configs on the GPU one at a time; evaluate the two
families and the event-day breakdown; update the README.
