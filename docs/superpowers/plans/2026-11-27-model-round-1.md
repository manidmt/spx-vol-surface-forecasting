# Model round 1 (no exogenous inputs) — design and plan

Date: 2026-11-27. Builds on `docs/superpowers/specs/2026-11-27-experiment-framework-design.md`.

## Intent

The experiment framework reproduces notebook 05: the 3D CNN improves MAE on persistence by 1.1 %
but no model is significant (best: CNN conv head, Diebold-Mariano p = 0.058 under absolute loss).
Round 1 tries the remaining model families agreed with the supervisor, **with the surface as the
only input**. Round 2 adds simple exogenous inputs already on disk (event calendar, SPX returns);
round 3 the new Bloomberg data.

## Pre-registration (anti data-snooping)

Many models tried and the best p-value reported would find a "significant" model by chance. So:

- The model list and every hyperparameter below are fixed **now**, before seeing any result,
  and are not tuned on the test period (2021-01-04 to 2026-10-07).
- Every model in the round is reported, not only the winner.
- Diebold-Mariano p-values are also reported **Holm-adjusted** within each family
  (same reference, same loss, all models of the table).
- The clean hold-out (from 2026-10-08) stays untouched until the end of the thesis.

All networks share the existing trainer and its settings (lookback 5 days, 5 seeds, Huber δ = 1,
AdamW lr 1e-3 wd 1e-4, batch 64, ≤ 300 epochs, patience 30, last 15 % as validation), the same
inputs (channel 0 daily change, channel 1 level, shape `(batch, 2, lookback, 7, 9)`) and the same
zero-initialised output head, so the untrained network is persistence.

| Run name | Model | Fixed hyperparameters |
|---|---|---|
| `lstm` | LSTM over days; each day is the flattened 2 × 63 input vector (126 features) | 1 layer, hidden size 64, last hidden state → `Linear(64, 63)` zero-initialised |
| `convlstm` | ConvLSTM over days on the 7 × 9 grid (convolutional gates), as in Adamski and Ślepaczuk (2026) | 1 ConvLSTM layer, 16 hidden channels, 3 × 3 kernel with padding 1, last hidden state → `Conv2d(16, 1, 1)` zero-initialised |
| `ens_cnn_ar1` | Equal-weight average of the saved `cnn_conv_head` and `ar1_rolling252_huber` forecasts | weights 0.5 / 0.5 |
| `ens_all` | Equal-weight average of `cnn_conv_head`, `lstm`, `convlstm` and `ar1_rolling252_huber` | weights 0.25 each |

Ensembles average saved forecasts (all share the same raw *t−1* base, so this equals averaging the
predicted changes). They need no refitting and inherit the members' leakage guarantees.

## Tasks (parallel; each in its own worktree branched from `experiment-framework`)

Common rules: English code/comments/tests; docstrings and type hints; comments explain PyTorch
idioms with their Keras equivalent; ruff clean; TDD; short English commits without
`Co-Authored-By`; no GPU / full-size runs (the controller runs them one at a time); `data/` is
absent in worktrees, so real-data tests skip — unit tests use the `synthetic_surface` fixture.
Mirror `tests/test_torch.py` for network tests.

### Task A — LSTM (`src/volsurf/models/lstm.py`)
- `SurfaceLSTM(nn.Module)`: `__init__(self, n_tenors: int, n_moneyness: int, hidden: int = 64)`;
  `forward(x)` maps `(batch, 2, lookback, n_tenors, n_moneyness)` → `(batch, n_tenors * n_moneyness)`
  (permute to `(batch, lookback, 2 * n_points)`, `nn.LSTM(batch_first=True)`, last hidden state,
  zero-initialised `Linear`). The flattening keeps channel-major grid order (document it).
- `make_lstm(name: str, hidden: int = 64, **trainer_kwargs) -> TorchForecaster`.
- Registry: kind `"lstm"` in `models/__init__.py` (`KINDS` and `build_model`).
- `configs/lstm.toml`: `name = "lstm"`, `kind = "lstm"`, `test_start = "2021-01-01"`,
  `[params] hidden = 64, lookback = 5, seeds = 5`.
- Tests (`tests/test_lstm.py`): untrained output is exactly zero; output shape `(4, 63)`;
  `assert_no_leakage` on a tiny CPU deterministic config (seeds 1, epochs 3, patience 2, hidden 8)
  with both leakage checks of `tests/leakage.py`; `build_model("lstm", "x", {...})` returns a
  forecaster named `"x"`.

### Task B — ConvLSTM (`src/volsurf/models/convlstm.py`)
- `ConvLSTMCell(nn.Module)`: `__init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3)`;
  one `Conv2d(in + hidden, 4 * hidden, kernel, padding=kernel // 2)` producing the input, forget,
  output and candidate gates; `forward(x, state) -> (h, c)`.
- `SurfaceConvLSTM(nn.Module)`: `__init__(self, n_tenors: int, n_moneyness: int, hidden_channels: int = 16, kernel_size: int = 3)`;
  loops the cell over the lookback days from zero state; last `h` → zero-initialised
  `Conv2d(hidden, 1, 1)` → flatten to `(batch, n_tenors * n_moneyness)`.
- `make_convlstm(name: str, hidden_channels: int = 16, kernel_size: int = 3, **trainer_kwargs) -> TorchForecaster`.
- Registry kind `"convlstm"`; `configs/convlstm.toml` (`hidden_channels = 16`, `kernel_size = 3`,
  `lookback = 5`, `seeds = 5`).
- Tests (`tests/test_convlstm.py`): cell output shapes; untrained model output exactly zero;
  shape `(4, 63)`; tiny CPU leakage test as in Task A; registry test.

### Task C — Ensembles and Holm adjustment
- `src/volsurf/ensemble.py`: `average_runs(runs_dir: Path, members: list[str], weights: list[float] | None = None) -> tuple[pd.DataFrame, dict]`
  loads members with `io.load_run`, requires identical index and columns (else `ValueError`
  naming the member), weights default to equal and must be non-negative and sum to 1, returns the
  weighted average and member metadata (name, git commit, timestamp).
- `run.main`: a config with `kind = "ensemble"` and `[params] members = [...]` (optional
  `weights`) builds the forecast with `average_runs` instead of `walk_forward`; meta records the
  members. `configs/ens_cnn_ar1.toml` and `configs/ens_all.toml` as in the table above.
- `metrics.dm_table`: add column `p_holm` — Holm-adjusted p-values within each
  (reference, loss) group, over all models in the table (statsmodels `multipletests(method="holm")`).
  `evaluate` prints it.
- Tests: averaging two synthetic runs saved with `save_run` gives their mean; mismatched index or
  columns raise; bad weights raise; an ensemble config run through `run.main` publishes a run;
  `p_holm` matches a hand-computed Holm case and is ≥ the raw p-value.

### Task D — controller
Merge A, B, C into `experiment-framework`; one review of the merged diff; run `lstm`, `convlstm`
(GPU, one at a time) then both ensembles; evaluate all runs together against persistence, AR(1)
and the CNN; report every model with raw and Holm-adjusted p-values; update the README results.
