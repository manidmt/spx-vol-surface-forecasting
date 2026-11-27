# Experiment framework for next-day surface forecasts — design

Date: 2026-11-27. Status: draft, pending approval.

## Intent

Until now every model lives in a notebook (`02_baselines`, `03_cnn`, `05_surface_cnn`), and each
notebook re-implements the same protocol: data loading (`load_tenor` is copied four times), the
monthly walk-forward loop, the evaluation mask, `r2_oos` and `diebold_mariano` (three copies each).
The next phase adds many models (LSTM, ConvLSTM, ensembles) and exogenous inputs. Differences
between models are of the order of 1 % of MAE, so the comparison is only credible if **every model
runs under one protocol that cannot leak future information**.

The goal is a small, readable Python package that fixes that protocol once, so that adding a model
means writing only the model.

## Goals

1. One package, `src/volsurf/`, readable, typed, with English docstrings and comments.
2. A walk-forward runner that gives each model **only past data**, so that a model cannot use
   future information even if it has a bug.
3. A common model interface shared by linear baselines and neural networks.
4. A shared PyTorch training loop, so a new network is only its `nn.Module`.
5. Experiments launched from scripts with one TOML config each; predictions saved to disk in a
   common format; evaluation done afterwards from saved predictions.
6. **Reproduce the current results** of `05_surface_cnn` before anything new is added.

## Non-goals (this iteration)

- Exogenous features, LSTM, ConvLSTM, ensembles: separate follow-up plans built on this skeleton.
  The design leaves the extension points (see below) but does not implement them.
- Hyperparameter search.
- No-arbitrage constraints and the straddle/strangle backtest.
- Migrating the 30-day smile notebooks (01–03). They stay as they are, as historical record.

## Anti-goals

- A heavy framework (Lightning, Hydra, Darts…). The training loop is ~25 lines; the package must be
  readable end to end by someone who knows Keras but not PyTorch.
- Results that change after the refactor without an explanation.
- Any path through which a model sees data after its forecast date.

## Constraints

- Python 3.12, uv (`uv add`, never `pip`), PyTorch with CUDA on a single RTX 4070.
- Public repo: all code, comments, configs and outputs in English. One commit per change, short
  English messages, no `Co-Authored-By`.
- Data are licensed and git-ignored (`data/`). Input: `data/processed/surface_clean.pkl`, a
  DataFrame of 1,952 days (2019-01-02 to 2026-10-07) × 63 columns, MultiIndex
  `(tenor, moneyness)`, 7 tenors × 9 moneyness, bad Bloomberg prints already set to NaN.
- `.gitignore` contains `models/`, which would also ignore `src/volsurf/models/`. It must become
  `/models/` (root only).

## Approaches considered

- **A. Common base class plus a runner that owns time (chosen).** The runner decides refit dates
  and slices history; models only implement fit/predict. Leakage protection is structural, and
  baselines and networks share one interface, so evaluation and ensembles treat them alike.
- **B. Precomputed tensors, models as bare `nn.Module`s.** Fewer classes, good for networks, but
  statsmodels baselines fit badly and leakage control is back to index arithmetic (the delicate
  `train[:-1] &= …` line in notebook 05). Its one good idea is kept inside A: torch models may
  cache their input windows, as long as the leakage test passes.
- **C. Existing framework (Darts, neuralforecast, Lightning + Hydra).** Rejected: series-oriented
  libraries do not fit a tenor × moneyness grid (3D CNN, ConvLSTM), their components are hard to
  defend in the thesis, and the dependency is not worth it.

**Deep learning library: PyTorch.** Already installed with CUDA and used in notebooks 03 and 05.
Keras 3 on the torch backend was considered (the author knows Keras); kept PyTorch because the
training loop is written once in `torch_base.py` and later work (custom losses, no-arbitrage
penalties) needs direct control of it. Architecture files read like a subclassed Keras `Model`.

## Design

### Layout

```
src/volsurf/
  data.py          # load the clean surface; later, exogenous panels
  protocol.py      # walk-forward schedule and history slicing
  models/
    base.py        # Forecaster interface
    baselines.py   # persistence, AR(1) Huber per point
    torch_base.py  # shared training loop for networks
    cnn.py         # SurfaceCNN (nn.Module), port of notebook 05
  metrics.py       # MAE/RMSE by tenor and point, R² OOS, Diebold-Mariano
  run.py           # python -m volsurf.run configs/<name>.toml
  evaluate.py      # python -m volsurf.evaluate runs/<a> runs/<b> ...
configs/           # one TOML per experiment
tests/             # pytest
```

The package is installed in the uv environment so notebooks can `import volsurf`.

### Timing convention

The surface of day *t* is forecast with information up to the close of *t−1*. Models predict the
**change** with respect to the last observed surface; the runner adds that surface back. So a model
that outputs zero is persistence, and networks start there (zero-initialised head, as in notebook 05).

### Protocol (`protocol.py`)

- Out-of-sample start 2021-01-01; refit on the first trading day of each month (as now).
- At a refit date *s*, the model is fitted on history strictly before *s*; training pairs
  (inputs at *t*, target change *t → t+1*) are used only if *t+1 < s*.
- The forecast for day *t* is computed from history up to *t−1* only.
- Window type (expanding or rolling *N* days) is a model setting (AR(1) uses rolling 252, the CNN
  expanding).
- Inputs may be forward-filled (past only); targets keep their NaN. Days whose target or any
  model's forecast is NaN are dropped from evaluation by a **common mask** across the runs compared.
- **Clean hold-out:** data after 2026-10-07 (the second Bloomberg download) are cut off by default.
  Evaluating on them requires an explicit flag, and is meant to be done once, at the end.

**Leakage test** (the key test): run a model, then overwrite all data after a date *d* with noise
and run again; forecasts for days ≤ *d* must be identical.

### Model interface (`models/base.py`)

A `Forecaster` with `fit(history)` and `predict(history) -> change for the next day`, plus its
window setting. The runner calls them; models never touch the calendar. Exact signatures (per-day
versus per-month batched prediction) are left to the builder, provided the leakage test passes.

### Networks (`models/torch_base.py`)

Shared trainer reproducing notebook 05: Huber loss (δ = 1), AdamW (lr 1e-3, weight decay 1e-4),
batch 64, up to 300 epochs, early stopping with patience 30 on the last 15 % of the training window,
best weights restored, average of 5 seeds; input scaling (change channel by its std, level channel
standardised) and per-point target scaling, all fitted on the training window only. Seeds fixed for
reproducibility. A new architecture supplies only its `nn.Module`.

### Configs and outputs

One TOML per run (model class, its parameters, window, seeds). A run writes to `runs/<name>/`
(already git-ignored): predictions as Parquet (days × 63 points), a copy of the config, and metadata
(git commit, date, epochs per refit, runtime).

### Evaluation (`evaluate.py`, `metrics.py`)

Loads any set of runs, applies the common mask and reports: MAE and RMSE overall, by tenor and by
grid point; MAE relative to persistence; R² OOS; Diebold-Mariano (absolute and squared loss, HAC
with 5 lags) against persistence and against a chosen reference. Same formulas as notebook 05.

### Extension points (not built now)

- **Exogenous inputs:** `data.py` will expose date-indexed feature panels, each value keyed on the
  day it is known at the close. Networks will take them through a shared dense branch added to the
  output, the same for every architecture, so architecture and features can be compared separately.
- **Ensembles:** combinations of saved runs, evaluated as one more run, without refitting.
- **New networks** (LSTM, ConvLSTM): one `nn.Module` and one config each.

## Acceptance

1. Reproduction against notebook 05 on the same 1,444 days: persistence MAE 0.460; AR(1) Huber
   about −0.6 % versus persistence; CNN with conv head about −1.1 % and with dense head about
   −1.1 % (tolerance ±0.2 percentage points for GPU non-determinism). Both heads are ported (the
   head is a constructor flag), so the common mask is built from the same four models. Persistence
   adds back the raw surface of *t−1*, not the forward-filled one. Any larger gap is explained
   before moving on.
2. Leakage test passes for persistence, AR(1) and the CNN. For the CNN it runs inside pytest on a
   reduced configuration (CPU, deterministic algorithms, one seed, a few epochs, a few refits) and
   requires **exact** equality; the full-size runs are not repeated for it.
3. `ruff check` and `ruff format` clean; pytest suite passes (protocol dates, slicing, leakage,
   metrics against a hand-computed case).
4. `evaluate` produces, from the new runs, the same tables as notebook 05 (overall, by tenor, by
   point, R² OOS, Diebold-Mariano). Notebooks 04–06 are left untouched as historical record; new
   analysis notebooks import `volsurf`.

## Left to the builder

Exact class signatures, Parquet column layout, CLI flags, test fixtures, how windows are cached,
and how packaging is declared in `pyproject.toml`.

## Afterwards (separate plans, in this order)

Exogenous branch with calendar and underlying features; LSTM and ConvLSTM (candidates for parallel
delegation to agents in separate branches, one GPU run at a time); ensembles; the hold-out
evaluation once the second Bloomberg download is in.
