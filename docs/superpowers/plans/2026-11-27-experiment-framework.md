# Experiment Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the notebook-only workflow with a small `volsurf` package that runs every surface
model under one leak-proof walk-forward protocol, and reproduce notebook 05's results with it.

**Architecture:** A runner (`protocol.walk_forward`) owns time: it fits each model on history strictly
before the monthly refit date and asks for each day's forecast with history up to the previous day
only. Models implement a two-method `Forecaster` interface and predict the change over the last
observed surface. Networks share one trainer (`TorchForecaster`); runs are launched from TOML configs
and saved to `runs/`, then compared with `evaluate`.

**Tech Stack:** Python 3.12, uv, pandas, numpy, statsmodels, PyTorch (CUDA), pytest, ruff, pyarrow.

**Spec:** `docs/superpowers/specs/2026-11-27-experiment-framework-design.md`

## Global Constraints

- Python 3.12; dependencies with `uv add` (dev: `uv add --dev`), never `pip`. Run everything with `uv run`.
- Public repo: code, comments, docstrings, configs, test names and outputs in **English**.
- One commit per task, short English message, **no** `Co-Authored-By` line, no session link.
- Work on branch `experiment-framework` in the main checkout (not a separate worktree: `data/` is
  git-ignored and would be missing there).
- Input: `data/processed/surface_clean.pkl`, 1,952 days (2019-01-02 to 2026-10-07) × 63 columns,
  MultiIndex `(tenor, moneyness)` with names `["tenor", "moneyness"]`, tenors in the order
  `30DAY, 60DAY, 3MTH, 6MTH, 12MTH, 18MTH, 24MTH`, moneyness floats `80.0 … 120.0` (9 values).
  Bad prints are NaN. **Column order is the grid order everywhere; never sort it.**
- Out-of-sample start `2021-01-01`; refit on the first test day of each calendar month.
- Hold-out: dates `>= 2026-10-08` are dropped unless explicitly requested.
- No heavy frameworks (no Lightning, Hydra, Darts). Docstrings and type hints on public functions.
- The forecast for day *t* = raw surface of *t−1* (not forward-filled) + predicted change.
- Reference numbers (notebook 05, 1,444 evaluated days, 2021-01-04 to 2026-10-07), from
  `data/processed/surface_preds.pkl` (dict with keys `persistence`, `ar1_rolling252_huber`,
  `cnn_conv_head`, `cnn_dense_head`, each a DataFrame indexed by test day, same columns as the surface):

  | model | MAE | RMSE | MAE vs persistence | R² OOS |
  |---|---|---|---|---|
  | persistence | 0.4597 | 0.8277 | 0 | 0 |
  | ar1_rolling252_huber | 0.4571 | 0.8218 | −0.0057 | 0.0143 |
  | cnn_conv_head | 0.4546 | 0.8241 | −0.0112 | 0.0089 |
  | cnn_dense_head | 0.4546 | 0.8231 | −0.0112 | 0.0111 |

  Diebold-Mariano vs persistence, absolute loss: cnn_conv_head t = 1.894, p = 0.058;
  ar1 t = 1.241, p = 0.215. Squared loss: cnn_conv_head t = 0.500, p = 0.617.

## Review Focus

1. A bad-print day (NaN row, like 2022-09-20) inside history: fitting must not crash, inputs are
   forward-filled, the NaN target is excluded, and the forecast for the following day is NaN
   (raw *t−1* is NaN) — pinned in Task 3 and Task 4.
2. Column order surviving disk round trips (Parquet cannot store the `(str, float)` MultiIndex
   directly; a sorted or string-typed column index silently scrambles the 7 × 9 grid) — Task 7.
3. Runs covering different date ranges compared together: evaluation must use the intersection
   of non-NaN days (common mask), never align by position — Task 5.
4. The first refit month, with less history than a rolling window (AR(1) asks for 252 pairs):
   use what exists, do not fail — Task 4.
5. Hold-out dates present in the data file: dropped by default, kept only with the flag — Task 2.

---

### Task 1: Package skeleton and tooling

**Files:**
- Modify: `pyproject.toml`, `.gitignore`
- Create: `src/volsurf/__init__.py`, `src/volsurf/models/__init__.py` (empty for now), `tests/conftest.py`, `tests/test_package.py`

**Interfaces:**
- Produces: importable package `volsurf`; pytest fixtures `synthetic_surface` (DataFrame, 600
  business days from 2019-01-01, the real 7 × 9 MultiIndex columns and names, a seeded random walk
  around 20 with per-tenor noise scale, one all-NaN row at position 400) and `real_surface`
  (loads `data/processed/surface_clean.pkl`, `pytest.skip` if missing); constant
  `TENORS`/`MONEYNESS` lists for tests built in conftest.

- [ ] **Step 1:** `git switch -c experiment-framework`.
- [ ] **Step 2: Write the failing test** `tests/test_package.py::test_package_imports`: `import volsurf; assert volsurf.__version__ == "0.1.0"`.
- [ ] **Step 3:** Run `uv run pytest tests/test_package.py -v`. Expected: FAIL (no module).
- [ ] **Step 4: Packaging.** In `pyproject.toml` add a hatchling `[build-system]` and
  `[tool.hatch.build.targets.wheel] packages = ["src/volsurf"]`; `uv add --dev pytest`; add
  `[tool.ruff] line-length = 110`, `[tool.ruff.lint] select = ["E", "F", "I", "UP", "B"]`,
  `[tool.pytest.ini_options] testpaths = ["tests"]` and a marker `slow` (real data, > 10 s).
  `__init__.py` defines `__version__ = "0.1.0"` and a one-paragraph module docstring.
  In `.gitignore` change `models/` to `/models/`. Write `tests/conftest.py` with the fixtures above.
- [ ] **Step 5:** `uv sync && uv run pytest -v` → PASS; `uv run ruff check src tests && uv run ruff format --check src tests` → clean.
- [ ] **Step 6:** Commit: `git add -A pyproject.toml uv.lock .gitignore src tests && git commit -m "Add volsurf package skeleton"`.

### Task 2: Data loading and refit schedule

**Files:**
- Create: `src/volsurf/data.py`, `src/volsurf/protocol.py`, `tests/test_data.py`, `tests/test_protocol.py`

**Interfaces:**
- Produces:
  - `data.SURFACE_PATH: Path` (repo-root `data/processed/surface_clean.pkl`, resolved from the package location), `data.HOLDOUT_START = pd.Timestamp("2026-10-08")`
  - `data.load_surface(path: Path = SURFACE_PATH, include_holdout: bool = False) -> pd.DataFrame`
  - `data.drop_holdout(df: pd.DataFrame, include_holdout: bool = False) -> pd.DataFrame`
  - `data.grid_shape(surface: pd.DataFrame) -> tuple[int, int]` (n_tenors, n_moneyness, from the column levels, unique in order)
  - `protocol.TEST_START = pd.Timestamp("2021-01-01")`
  - `protocol.refit_schedule(index: pd.DatetimeIndex, test_start: pd.Timestamp = TEST_START) -> list[tuple[pd.Timestamp, pd.DatetimeIndex]]` — one entry per calendar month with test days: (first test day of the month, all test days of that month)

- [ ] **Step 1: Failing tests.**
  - `test_drop_holdout_default`: a frame with dates 2026-10-07 and 2026-10-08 keeps only 2026-10-07; with `include_holdout=True` keeps both.
  - `test_grid_shape(synthetic_surface)`: `== (7, 9)`.
  - `test_load_surface_real(real_surface)` (marked slow): shape `(1952, 63)`, first date 2019-01-02, last 2026-10-07, column names `["tenor", "moneyness"]`, first tenor level values in the order of Global Constraints.
  - `test_refit_schedule(synthetic_surface)`: every refit date is the first element of its month's days; months are consecutive; the union of all days equals `index[index >= test_start]`; the first refit date is the first index date `>= test_start`.
- [ ] **Step 2:** `uv run pytest tests/test_data.py tests/test_protocol.py -v` → FAIL (import errors).
- [ ] **Step 3:** Implement the five functions/constants above. `load_surface` reads the pickle and applies `drop_holdout`.
- [ ] **Step 4:** Tests → PASS (`-m "slow or not slow"` to include the real-data one); ruff clean.
- [ ] **Step 5:** Commit `"Add surface loading and refit schedule"`.

### Task 3: Forecaster interface and walk-forward runner

**Files:**
- Create: `src/volsurf/models/base.py`, `tests/test_walk_forward.py`
- Modify: `src/volsurf/protocol.py`

**Interfaces:**
- Consumes: `refit_schedule`, `TEST_START` (Task 2).
- Produces:
  - `models.base.Forecaster` (ABC) with:
    - `name: str` attribute
    - `fit(self, history: pd.DataFrame) -> None` — `history` is the surface strictly before the refit date (raw, with NaN rows)
    - `predict_change(self, history: pd.DataFrame) -> np.ndarray` — `history` ends at *t−1*; returns shape `(n_points,)`, the forecast of `surface[t] − surface[t−1]` in column order
    - `fit_log: list[dict]` — per-refit diagnostics appended by `fit` (may be empty)
  - `protocol.walk_forward(model: Forecaster, surface: pd.DataFrame, test_start: pd.Timestamp = TEST_START) -> pd.DataFrame` — level forecasts indexed by test day, same columns as `surface`; for each month calls `model.fit(surface[surface.index < refit_date])`, then for each day *t* `surface.shift(1).loc[t] + model.predict_change(surface.loc[:t_prev])` where `t_prev` is the previous index date.

- [ ] **Step 1: Failing tests** with a `SpyForecaster` defined in the test that records `history.index[-1]` on every call and returns zeros:
  - `test_fit_sees_only_past(synthetic_surface)`: for every recorded fit, last history date `<` its refit date.
  - `test_predict_sees_up_to_previous_day(synthetic_surface)`: for every test day *t*, the recorded predict history ends at the index date just before *t*.
  - `test_zero_change_is_persistence(synthetic_surface)`: output equals `synthetic_surface.shift(1).loc[test days]` exactly (NaN in the same places).
  - `test_nan_row_propagates(synthetic_surface)`: the forecast for the day after the NaN row (position 401) is all NaN (Review Focus 1).
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `Forecaster` and `walk_forward` (module docstring explaining the timing convention of the spec). Per-day prediction loop; no vectorised shortcut here.
- [ ] **Step 4:** Run → PASS; ruff clean.
- [ ] **Step 5:** Commit `"Add forecaster interface and walk-forward runner"`.

### Task 4: Baselines (persistence and AR(1) Huber)

**Files:**
- Create: `src/volsurf/models/baselines.py`, `tests/test_baselines.py`, `tests/leakage.py` (shared helper)

**Interfaces:**
- Consumes: `Forecaster`, `walk_forward` (Task 3).
- Produces:
  - `baselines.Persistence(Forecaster)` — `name = "persistence"`, predicts zeros.
  - `baselines.AR1Huber(Forecaster)` — `__init__(self, window: int | None = 252)`, `name = "ar1_rolling252_huber"` when window is 252; per column: regress `d[t]` on `d[t−1]` (d = `history.diff()`), pairs with NaN dropped, last `window` pairs, `statsmodels` `RLM` with `HuberT`; `predict_change` returns `c + b * d_last` with `d_last = history.iloc[-1] − history.iloc[-2]` per column (NaN if either is NaN). Fewer than `window` pairs → use all available (Review Focus 4).
  - `tests/leakage.py::assert_no_leakage(make_model: Callable[[], Forecaster], surface: pd.DataFrame, cutoff: pd.Timestamp, test_start: pd.Timestamp) -> None` — runs `walk_forward` on `surface` and on a copy whose rows after `cutoff` are replaced by seeded Gaussian noise (scale 5), and asserts forecasts for days `<= cutoff` are identical with `np.testing.assert_array_equal` (NaN-aware).

- [ ] **Step 1: Failing tests.**
  - `test_persistence_matches_notebook(real_surface)` (slow): `walk_forward(Persistence(), real_surface)` equals `surface_preds.pkl["persistence"]` with `assert_frame_equal(check_freq=False)`.
  - `test_ar1_matches_notebook(real_surface)` (slow): `walk_forward(AR1Huber(252), real_surface)` vs `surface_preds.pkl["ar1_rolling252_huber"]`, `atol=1e-8`.
  - `test_persistence_no_leakage(synthetic_surface)` and `test_ar1_no_leakage(synthetic_surface)`: `assert_no_leakage` with test_start 2020-01-01 and cutoff in the middle of a month (e.g. the 450th index date).
  - `test_ar1_survives_nan_row(synthetic_surface)`: walk-forward runs, and forecasts are finite except where `shift(1)` is NaN.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement both classes.
- [ ] **Step 4:** Run including slow tests → PASS; ruff clean.
- [ ] **Step 5:** Commit `"Add persistence and AR(1) baselines"`.

### Task 5: Metrics

**Files:**
- Create: `src/volsurf/metrics.py`, `tests/test_metrics.py`

**Interfaces:**
- Produces:
  - `common_mask(actual: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> pd.Index` — dates where `actual` and every prediction exist and are all finite (intersection by date, Review Focus 3).
  - `errors(actual: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]` — `actual − pred` on the common mask.
  - `summary(errs: dict[str, pd.DataFrame], reference: str = "persistence") -> pd.DataFrame` — columns `mae, rmse, mae_vs_ref, r2_oos_vs_ref`, formulas of notebook 05.
  - `mae_by_tenor(errs) -> pd.DataFrame` (models × tenors, tenor order preserved), `mae_by_point(errs) -> pd.DataFrame` (points × models).
  - `diebold_mariano(e_ref: pd.DataFrame, e: pd.DataFrame, loss: Literal["squared", "absolute"], maxlags: int = 5) -> tuple[float, float]` — daily mean loss differential `loss(e_ref) − loss(e)`, OLS on a constant with HAC (`maxlags`), returns (t, p); positive t means `e` is better.
  - `dm_table(errs, references: list[str], models: list[str]) -> pd.DataFrame` — index (reference, model, loss), columns `t_stat, p_value`.

- [ ] **Step 1: Failing tests.**
  - `test_common_mask_intersects_dates`: two prediction frames over different date ranges with one NaN day → mask is the intersection minus the NaN day.
  - `test_summary_hand_case`: errors `[[1, -1], [2, 0]]` vs reference `[[2, 2], [2, 2]]` → mae 1.0, rmse `sqrt(1.5)`, mae_vs_ref −0.5, r2 `1 − 6/16`.
  - `test_reproduces_notebook_tables(real_surface)` (slow): from `surface_preds.pkl` and `real_surface`, the mask has 1,444 days, first 2021-01-04, last 2026-10-07; `summary` matches the Global Constraints table to 4 decimals; DM (absolute) persistence vs cnn_conv_head gives t = 1.894, p = 0.058 to 3 decimals; squared t = 0.500.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run → PASS; ruff clean.
- [ ] **Step 5:** Commit `"Add evaluation metrics and Diebold-Mariano test"`.

### Task 6: Shared network trainer and the surface CNN

**Files:**
- Create: `src/volsurf/models/torch_base.py`, `src/volsurf/models/cnn.py`, `tests/test_torch.py`

**Interfaces:**
- Consumes: `Forecaster` (Task 3), `grid_shape` (Task 2), `assert_no_leakage` (Task 4).
- Produces:
  - `torch_base.make_windows(levels: np.ndarray, lookback: int) -> np.ndarray` — `levels` shape `(T, n_tenors, n_moneyness)`; returns float32 `(T, 2, lookback, n_tenors, n_moneyness)`, channel 0 = daily change, channel 1 = level, row `i` holds days `i−lookback+1 … i`; rows `< lookback` are NaN (as notebook 05).
  - `torch_base.TorchForecaster(Forecaster)` — `__init__(self, name: str, net_factory: Callable[[int, int], nn.Module], lookback: int = 5, seeds: int = 5, epochs: int = 300, patience: int = 30, batch: int = 64, val_frac: float = 0.15, lr: float = 1e-3, weight_decay: float = 1e-4, huber_delta: float = 1.0, device: str | None = None, deterministic: bool = False)`; `net_factory(n_tenors, n_moneyness)` builds a fresh network whose output is `(batch, n_points)` scaled changes.
    - `fit`: windows from `history.ffill()`, targets `history.diff().shift(-1)` (raw, not filled); keep rows where window and target are finite; scaling fitted on those rows exactly as notebook 05 (`sd_change` = std of channel 0; level channel standardised; target divided by per-point std); for each seed: `torch.manual_seed(seed)`, build net, train with the notebook 05 loop (last `max(int(val_frac·n), 20)` rows as validation, `torch.randperm` minibatches, Huber, AdamW, early stopping with best weights restored). Appends `{"refit": history.index[-1], "epochs": [...]}` to `fit_log`.
    - `predict_change`: the single window ending at the last row, built from `history.ffill().iloc[-(lookback + 1):]` (never the whole history: it is called once per test day), scaled with stored stats, mean over seed networks, times per-point std.
    - `deterministic=True` → `torch.use_deterministic_algorithms(True)` during fit/predict; `device=None` → CUDA if available.
  - `cnn.SurfaceCNN(nn.Module)` — `__init__(self, n_tenors: int, n_moneyness: int, channels: int = 16, head: Literal["conv", "dense"] = "conv")`; port of notebook 05 (two `Conv3d(…, 3, padding=1)` + GELU, mean over time, 1×1 `Conv2d` head or `Linear` head, head zero-initialised). Docstring compares it to a Keras subclassed `Model`.
  - `cnn.make_cnn(name: str, head: str = "conv", channels: int = 16, **trainer_kwargs) -> TorchForecaster`.

- [ ] **Step 1: Failing tests.**
  - `test_make_windows_shapes_and_content`: on a small `(12, 2, 3)` array, shape `(12, 2, 5, 2, 3)`, rows 0–4 NaN, row 7 channel 1 equals `levels[3:8]`, channel 0 equals `levels[3:8] - levels[2:7]`.
  - `test_untrained_cnn_is_persistence`: `SurfaceCNN(7, 9)` on random input returns zeros (both heads).
  - `test_cnn_no_leakage(synthetic_surface)`: `assert_no_leakage(lambda: make_cnn("tiny", seeds=1, epochs=3, patience=2, channels=4, device="cpu", deterministic=True), …)`, test_start 2020-06-01 so only a few refits run.
  - `test_cnn_fit_log(synthetic_surface)`: after a tiny walk-forward, `fit_log` has one entry per refit month.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement `make_windows`, `TorchForecaster`, `SurfaceCNN`, `make_cnn`. Comments explain each PyTorch idiom with its Keras equivalent (training loop ≈ `fit`, early stopping ≈ `EarlyStopping(restore_best_weights=True)`, channels-first layout).
- [ ] **Step 4:** Run → PASS (CPU, should take under a minute); ruff clean.
- [ ] **Step 5:** Commit `"Add shared network trainer and surface CNN"`.

### Task 7: Configs, run and evaluate commands

**Files:**
- Create: `src/volsurf/io.py`, `src/volsurf/run.py`, `src/volsurf/evaluate.py`, `src/volsurf/models/__init__.py` (registry), `configs/persistence.toml`, `configs/ar1_rolling252_huber.toml`, `configs/cnn_conv_head.toml`, `configs/cnn_dense_head.toml`, `tests/test_io.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: everything above.
- Produces:
  - `models.build_model(kind: str, name: str, params: dict) -> Forecaster` — kinds `"persistence"`, `"ar1"`, `"cnn"`; unknown kind raises `ValueError` listing the valid ones.
  - Config format (TOML, read with `tomllib`):
    ```toml
    name = "cnn_conv_head"
    kind = "cnn"
    test_start = "2021-01-01"
    [params]
    head = "conv"
    lookback = 5
    seeds = 5
    ```
  - `io.save_run(run_dir: Path, preds: pd.DataFrame, config: dict, meta: dict) -> None` — writes `predictions.parquet` (columns flattened as `"{tenor}|{moneyness:g}"`), `config.toml` (copy of the source file text) and `meta.json`.
  - `io.load_run(run_dir: Path) -> tuple[pd.DataFrame, dict]` — restores the MultiIndex `(tenor: str, moneyness: float)` with names and **original order** (Review Focus 2); returns (predictions, meta).
  - `run.main(argv: list[str] | None = None) -> Path` — `python -m volsurf.run CONFIG [--runs-dir runs] [--holdout]`; loads the surface, builds the model, `walk_forward`, saves to `runs/<name>/`, meta = git commit (`git rev-parse HEAD`, or `"unknown"`), UTC timestamp, runtime seconds, `include_holdout`, `fit_log` (JSON-serialisable). Prints the run directory.
  - `evaluate.main(argv: list[str] | None = None) -> dict[str, pd.DataFrame]` — `python -m volsurf.evaluate RUN_DIR [RUN_DIR ...] [--reference persistence] [--holdout]`; loads runs and the actual surface, prints `summary`, `mae_by_tenor` (absolute and relative to the reference) and `dm_table` (reference vs every other run, both losses), returns them keyed `"summary"`, `"by_tenor"`, `"dm"`.

- [ ] **Step 1: Failing tests.**
  - `test_run_roundtrip_preserves_columns(tmp_path, synthetic_surface)`: `save_run` then `load_run` returns a frame equal to the input with `assert_frame_equal` (column order, level names, float moneyness).
  - `test_build_model_unknown_kind`: `ValueError` mentioning `"persistence"`.
  - `test_run_and_evaluate_persistence(tmp_path, real_surface)` (slow): `run.main([configs/persistence.toml, "--runs-dir", tmp])` creates the three files; `evaluate.main([run_dir])["summary"].loc["persistence", "mae"]` rounds to 0.4597.
- [ ] **Step 2:** Run → FAIL.
- [ ] **Step 3:** Implement, plus the four configs (CNN configs mirror notebook 05: lookback 5, seeds 5, heads conv/dense).
- [ ] **Step 4:** Run → PASS; ruff clean.
- [ ] **Step 5:** Commit `"Add run and evaluate commands with configs"`.

### Task 8: Full reproduction and documentation (controller, GPU)

**Files:**
- Modify: `README.md` (new section "Code layout and running experiments")

- [ ] **Step 1:** Run the four configs one at a time on the GPU: `uv run python -m volsurf.run configs/<name>.toml` (persistence, AR(1), both CNNs); record runtimes.
- [ ] **Step 2:** `uv run python -m volsurf.evaluate runs/persistence runs/ar1_rolling252_huber runs/cnn_conv_head runs/cnn_dense_head`. Expected: 1,444 days; persistence and AR(1) equal to the reference table to 4 decimals; both CNNs MAE vs persistence within −0.0112 ± 0.002. A larger gap is investigated (diff against `surface_preds.pkl` per month) before continuing.
- [ ] **Step 3:** Full suite `uv run pytest -m "slow or not slow"` and `uv run ruff check . && uv run ruff format --check src tests` → clean.
- [ ] **Step 4:** README section: package layout, how to add a model (one `nn.Module` or `Forecaster` + one config), run/evaluate commands, the timing convention and the hold-out flag. English.
- [ ] **Step 5:** Commit `"Document the experiment framework"`.
