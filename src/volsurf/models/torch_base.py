"""Shared trainer for neural forecasters (port of the training loop in notebook 05).

The surface enters the networks as a window of the last ``lookback`` days with two channels,
the daily change and the level. PyTorch uses a *channels-first* layout, so one sample is
``(channels, time, tenor, moneyness)`` and a batch is ``(batch, 2, lookback, n_tenors,
n_moneyness)``; Keras' default is channels-last, ``(batch, time, tenor, moneyness, channels)``.

Optional exogenous inputs (``n_exog > 0``) enter through ``ExogBranch``: a linear map of the
scaled exogenous row of the window's last day, added to the network's output.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from torch import nn

from volsurf.data import grid_shape
from volsurf.models.base import Forecaster


def _check_memory(memory) -> tuple[int, ...]:
    """``memory`` as a tuple of integers of at least 2, else ``ValueError``."""
    memory = tuple(memory)
    if any(not isinstance(m, int) or isinstance(m, bool) or m < 2 for m in memory):
        raise ValueError(f"memory must be a list of integers of at least 2 days, got {list(memory)}")
    return memory


def memory_rows(lookback: int, memory=()) -> int:
    """Days of history the last window needs: ``lookback`` plus the day before, or the memory span."""
    memory = _check_memory(memory)
    return lookback + max([1, *(m - 1 for m in memory)])


def make_windows(levels: np.ndarray, lookback: int, memory=()) -> np.ndarray:
    """Stack the last ``lookback`` days of every date as a multi-channel window.

    ``levels`` has shape ``(T, n_tenors, n_moneyness)``. Returns float32
    ``(T, 2 + len(memory), lookback, n_tenors, n_moneyness)``: row ``i`` holds days
    ``i-lookback+1 … i``, channel 0 is the daily change and channel 1 the level. Rows
    ``< lookback`` are NaN (the first change of the window needs the day before it).

    Each ``m`` in ``memory`` adds a channel: at every day of the window, the level minus the
    mean of the levels of the ``m`` days ending at that day (past only). In rows whose window
    start has fewer than ``m - 1`` earlier days that channel is partly NaN, so fitting drops them.
    """
    memory = _check_memory(memory)
    levels = np.asarray(levels, dtype=float)
    change = np.full_like(levels, np.nan)
    change[1:] = levels[1:] - levels[:-1]
    deviations = []
    for m in memory:
        deviation = np.full_like(levels, np.nan)
        csum = np.cumsum(levels, axis=0)
        trailing = csum[m - 1 :].copy()
        trailing[1:] -= csum[:-m]
        deviation[m - 1 :] = levels[m - 1 :] - trailing / m
        deviations.append(deviation)
    X = np.full((len(levels), 2 + len(memory), lookback, *levels.shape[1:]), np.nan, dtype=np.float32)
    for i in range(lookback, len(levels)):
        X[i, 0] = change[i - lookback + 1 : i + 1]
        X[i, 1] = levels[i - lookback + 1 : i + 1]
        for k, deviation in enumerate(deviations):
            X[i, 2 + k] = deviation[i - lookback + 1 : i + 1]
    return X


class ExogBranch(nn.Module):
    """Wrap a surface network: ``forward(x, z) = net(x) + linear(z)``.

    ``z`` is ``(batch, n_exog)`` scaled exogenous inputs; ``linear`` maps them to the
    ``n_points`` scaled changes and starts at zero, so the untrained wrapper forecasts exactly
    what ``net`` does (persistence for the zero-initialised heads). In Keras this would be a
    functional model with two inputs, ``Add()([net(x), Dense(n_points, kernel_initializer="zeros",
    bias_initializer="zeros")(z)])``; in PyTorch a module can simply take two arguments.
    """

    def __init__(self, net: nn.Module, n_exog: int, n_points: int) -> None:
        super().__init__()
        self.net = net  # assigning a Module as an attribute registers its weights (≈ a nested layer)
        self.linear = nn.Linear(n_exog, n_points)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """Map windows ``x`` and exogenous rows ``z`` to ``(batch, n_points)`` scaled changes."""
        return self.net(x) + self.linear(z)


@contextmanager
def _deterministic(enabled: bool) -> Iterator[None]:
    """Turn on PyTorch's deterministic algorithms inside the block and restore the old setting."""
    if not enabled:
        yield
        return
    previous = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=warn_only)


class TorchForecaster(Forecaster):
    """Forecaster that trains ``seeds`` copies of a network and averages their forecasts.

    ``net_factory(n_tenors, n_moneyness)`` must return a fresh ``nn.Module`` mapping a batch of
    scaled windows ``(batch, 2, lookback, n_tenors, n_moneyness)`` to scaled changes
    ``(batch, n_tenors * n_moneyness)``. Training follows notebook 05: Huber loss, AdamW,
    shuffled minibatches, early stopping on the last ``val_frac`` of the training rows (at
    least 20) with the best weights restored.

    ``device=None`` picks CUDA when available. ``deterministic=True`` switches on
    ``torch.use_deterministic_algorithms`` during ``fit`` and ``predict_change`` (on CUDA this
    also needs the environment variable ``CUBLAS_WORKSPACE_CONFIG=:4096:8``).

    ``n_exog > 0`` wraps every network in ``ExogBranch``: ``fit`` and ``predict_change`` then
    require an ``exog`` panel with ``n_exog`` columns and the same dates as ``history``. The
    exogenous row of day ``i`` goes with the window ending at ``i``. Rows with non-finite
    inputs are dropped from fitting and produce NaN forecasts. ``exog_scaling="standard"``
    uses the mean and standard deviation of the usable fitting rows. ``"robust"`` uses
    the median and IQR / 1.349, falling back to the standard deviation for a zero IQR.
    A zero standard deviation is replaced by 1. ``exog_clip`` optionally bounds scaled
    inputs symmetrically during fitting and prediction.

    ``exog_l2`` penalises the squared Frobenius norm of the branch weight matrix in the
    training loss. A list selects the penalty per refit by seed 0's plain Huber validation
    loss, favouring the larger penalty in a tie. The fit log records the selected penalty,
    seed 0's validation losses in candidate order (one entry for a scalar), and the final
    branch weight norm for each seed. Exogenous settings require ``n_exog > 0`` when they
    differ from their defaults.
    """

    def __init__(
        self,
        name: str,
        net_factory: Callable[[int, int], nn.Module],
        lookback: int = 5,
        seeds: int = 5,
        epochs: int = 300,
        patience: int = 30,
        batch: int = 64,
        val_frac: float = 0.15,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        huber_delta: float = 1.0,
        device: str | None = None,
        deterministic: bool = False,
        n_exog: int = 0,
        exog_scaling: str = "standard",
        exog_clip: float | None = None,
        exog_l2: list[float] | float = 0.0,
        memory: list[int] | tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        self.name = name
        self.net_factory = net_factory
        self.lookback = lookback
        self.seeds = seeds
        self.epochs = epochs
        self.patience = patience
        self.batch = batch
        self.val_frac = val_frac
        self.lr = lr
        self.weight_decay = weight_decay
        self.huber_delta = huber_delta
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.deterministic = deterministic
        self.n_exog = n_exog
        if exog_scaling not in ("standard", "robust"):
            raise ValueError("exog_scaling must be 'standard' or 'robust'")
        if exog_clip is not None and not exog_clip > 0:
            raise ValueError("exog_clip must be positive")
        penalties = exog_l2 if isinstance(exog_l2, list) else [exog_l2]
        if not penalties or any(not np.isfinite(value) or value < 0 for value in penalties):
            raise ValueError("exog_l2 must contain one or more finite, non-negative penalties")
        if not n_exog and (exog_scaling != "standard" or exog_clip is not None or exog_l2 != 0.0):
            raise ValueError("non-default exog_* settings require n_exog > 0")
        self.exog_scaling = exog_scaling
        self.exog_clip = exog_clip
        self.exog_l2 = exog_l2
        self.memory = _check_memory(memory)
        self.nets: list[nn.Module] = []
        # scaling statistics, fitted on the training rows at each refit
        self.sd_change: np.float32 | None = None
        self.mu_level: np.float32 | None = None
        self.sd_level: np.float32 | None = None
        self.sd_y: np.ndarray | None = None
        self.sd_memory: np.ndarray | None = None  # one scale per memory channel
        self.mu_z: np.ndarray | None = None  # exogenous centre and scale (n_exog > 0 only)
        self.sd_z: np.ndarray | None = None

    def fit(self, history: pd.DataFrame, exog: pd.DataFrame | None = None) -> None:
        """Train ``seeds`` networks on every complete (window, next-day change) pair in ``history``.

        Inputs are windows of the forward-filled surface (filling only uses the past); targets
        are the raw next-day changes, so rows whose target involves a NaN print are dropped.
        The last row has no next day inside ``history`` and drops out the same way. With
        exogenous inputs, rows with a non-finite exogenous value are dropped too.

        Raises ``ValueError`` (naming the refit by the last date of ``history``) before any
        network is built if the usable pairs cannot fill both a training and a validation
        split, or if a scaling statistic is zero, e.g. a grid point that never moves.
        """
        refit = f"refit on history ending {history.index[-1].date()}" if len(history) else "refit"
        self._check_exog(history, exog)
        with _deterministic(self.deterministic):
            n_tenors, n_moneyness = grid_shape(history)
            levels = history.ffill().to_numpy(dtype=float).reshape(len(history), n_tenors, n_moneyness)
            X = make_windows(levels, self.lookback, self.memory)
            Y = history.diff().shift(-1).to_numpy(dtype=np.float32)
            ok = np.isfinite(X).all(axis=(1, 2, 3, 4)) & np.isfinite(Y).all(axis=1)
            if self.n_exog:
                Z = exog.to_numpy(dtype=np.float32)
                ok &= np.isfinite(Z).all(axis=1)
            Xtr, Ytr = X[ok], Y[ok]
            n_val = self._n_val(len(Xtr))
            if len(Xtr) - n_val < 1:
                raise ValueError(
                    f"{self.name}: {refit}: {len(Xtr)} usable training pairs (lookback {self.lookback}) "
                    f"cannot fill a training split and a validation split of {n_val}"
                )

            # scaling fitted on the training rows only; each target point by its own std
            # (short tenors move about four times more than long ones)
            self.sd_change = Xtr[:, 0].std()
            self.mu_level, self.sd_level = Xtr[:, 1].mean(), Xtr[:, 1].std()
            self.sd_y = Ytr.std(axis=0)
            flat = ~(self.sd_y > 0)
            if flat.any():
                raise ValueError(
                    f"{self.name}: {refit}: the target change has zero standard deviation "
                    f"at {list(history.columns[flat])}"
                )
            self.sd_memory = Xtr[:, 2:].std(axis=(0, 2, 3, 4))
            if not (self.sd_memory > 0).all():
                raise ValueError(f"{self.name}: {refit}: a memory channel has zero standard deviation")
            if not (self.sd_change > 0 and self.sd_level > 0):
                raise ValueError(
                    f"{self.name}: {refit}: zero standard deviation of the input changes "
                    f"({self.sd_change}) or levels ({self.sd_level})"
                )

            inputs = [self._scale(Xtr)]
            if self.n_exog:
                Ztr = Z[ok]
                self.sd_z = Ztr.std(axis=0)
                if self.exog_scaling == "robust":
                    self.mu_z = np.median(Ztr, axis=0)
                    q25, q75 = np.percentile(Ztr, [25, 75], axis=0)
                    robust_scale = ((q75 - q25) / 1.349).astype(Ztr.dtype)
                    self.sd_z = np.where(robust_scale > 0, robust_scale, self.sd_z)
                else:
                    self.mu_z = Ztr.mean(axis=0)
                self.sd_z[self.sd_z == 0] = 1.0  # a constant column, e.g. no event in the window
                inputs.append(self._scale_exog(Ztr))
            Y_scaled = torch.tensor(Ytr / self.sd_y, device=self.device)
            self.nets, epochs = [], []
            penalties = self.exog_l2 if isinstance(self.exog_l2, list) else [self.exog_l2]
            chosen = penalties[0]
            val_losses = []
            for seed in range(self.seeds):
                if seed == 0:
                    best_key = (np.inf, 0.0)
                    for penalty in penalties:
                        candidate, candidate_epoch, val = self._train(
                            inputs, Y_scaled, seed, n_tenors, n_moneyness, penalty
                        )
                        val_losses.append(val)
                        key = (val, -penalty)
                        if key < best_key:
                            best_key = key
                            net, last_epoch, chosen = candidate, candidate_epoch, penalty
                        del candidate
                else:
                    net, last_epoch, _ = self._train(inputs, Y_scaled, seed, n_tenors, n_moneyness, chosen)
                self.nets.append(net)
                epochs.append(last_epoch)
            log = {"refit": history.index[-1], "epochs": epochs}
            if self.n_exog:
                log.update(
                    exog_l2=chosen,
                    exog_val_loss=val_losses,
                    exog_weight_norm=[net.linear.weight.detach().norm().item() for net in self.nets],
                )
        self.fit_log.append(log)

    def predict_change(self, history: pd.DataFrame, exog: pd.DataFrame | None = None) -> np.ndarray:
        """Mean forecast of the seed networks from the window (and exogenous row) of the last day."""
        if not self.nets:
            raise RuntimeError("fit must be called before predict_change")
        self._check_exog(history, exog)
        with _deterministic(self.deterministic):
            n_tenors, n_moneyness = grid_shape(history)
            # lookback changes need one extra day; memory channels need their own span
            recent = history.ffill().iloc[-memory_rows(self.lookback, self.memory) :]
            levels = recent.to_numpy(dtype=float).reshape(len(recent), n_tenors, n_moneyness)
            inputs = [self._scale(make_windows(levels, self.lookback, self.memory)[-1:])]
            if self.n_exog:
                Z = exog.iloc[-1:].to_numpy(dtype=np.float32)
                if self.exog_clip is not None and not np.isfinite(Z).all():
                    return np.full(n_tenors * n_moneyness, np.nan, dtype=np.float32)
                inputs.append(self._scale_exog(Z))
            # torch.no_grad() ≈ model.predict: no gradients are recorded
            with torch.no_grad():
                out = [net(*inputs).cpu().numpy() * self.sd_y for net in self.nets]
        return np.mean(out, axis=0)[0]

    def _check_exog(self, history: pd.DataFrame, exog: pd.DataFrame | None) -> None:
        """``ValueError`` unless ``exog`` matches ``n_exog`` (columns) and ``history`` (dates)."""
        if exog is None:
            if self.n_exog:
                raise ValueError(f"{self.name}: n_exog={self.n_exog} but no exog was given")
            return
        if exog.shape[1] != self.n_exog:
            raise ValueError(f"{self.name}: n_exog={self.n_exog} but exog has {exog.shape[1]} columns")
        if not exog.index.equals(history.index):
            raise ValueError(f"{self.name}: exog must have the same dates as history")

    def _scale_exog(self, Z: np.ndarray) -> torch.Tensor:
        """Scale and optionally clip exogenous rows, then return them on the device."""
        scaled = (Z - self.mu_z) / self.sd_z
        if self.exog_clip is not None:
            scaled = np.clip(scaled, -self.exog_clip, self.exog_clip)
        return torch.tensor(scaled, device=self.device)

    def _n_val(self, n_pairs: int) -> int:
        """Size of the early-stopping split: the last ``val_frac`` of the pairs, at least 20."""
        return max(int(self.val_frac * n_pairs), 20)

    def _scale(self, X: np.ndarray) -> torch.Tensor:
        """Change and memory channels divided by their std, level standardised; on the device."""
        X = X.copy()
        X[:, 0] /= self.sd_change
        X[:, 1] = (X[:, 1] - self.mu_level) / self.sd_level
        for k, sd in enumerate(self.sd_memory):
            X[:, 2 + k] /= sd
        return torch.tensor(X, device=self.device)

    def _train(
        self,
        inputs: list[torch.Tensor],
        Y: torch.Tensor,
        seed: int,
        n_tenors: int,
        n_moneyness: int,
        exog_l2: float,
    ) -> tuple[nn.Module, int, float]:
        """Train one seed and return its best network, last epoch and best plain Huber loss.

        ``inputs`` holds the scaled windows and, with exogenous inputs, the scaled exogenous rows,
        row-aligned with ``Y``; the network receives them as positional arguments.
        """
        # PyTorch has no model.fit(): the loop below is what Keras' fit() does internally
        torch.manual_seed(seed)  # ≈ keras.utils.set_random_seed: fixes init and shuffling
        n_val = self._n_val(len(Y))
        train_inputs, val_inputs = [t[:-n_val] for t in inputs], [t[-n_val:] for t in inputs]
        Y_train, Y_val = Y[:-n_val], Y[-n_val:]
        net = self.net_factory(n_tenors, n_moneyness)
        if self.n_exog:
            net = ExogBranch(net, self.n_exog, n_tenors * n_moneyness)
        net = net.to(self.device)  # .to(device) ≈ placing on GPU
        # ≈ model.compile(optimizer=AdamW(...), loss=Huber(delta))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.HuberLoss(delta=self.huber_delta)
        # early stopping ≈ EarlyStopping(monitor="val_loss", patience, restore_best_weights=True)
        best, best_state, wait = np.inf, None, 0
        for epoch in range(self.epochs):  # noqa: B007 (the last epoch is returned)
            net.train()  # ≈ training=True: layers such as dropout behave in training mode
            for b in torch.randperm(len(Y_train)).split(self.batch):  # shuffled minibatches
                opt.zero_grad()  # gradients accumulate in PyTorch, so reset them every step
                # net(*tensors) unpacks the list into positional arguments (windows[, exog])
                loss = loss_fn(net(*(t[b] for t in train_inputs)), Y_train[b])
                if exog_l2:
                    loss = loss + exog_l2 * net.linear.weight.square().sum()
                loss.backward()
                opt.step()  # apply the update ≈ optimizer.apply_gradients
            net.eval()  # ≈ training=False
            with torch.no_grad():
                val = loss_fn(net(*val_inputs), Y_val).item()
            if val < best:
                # state_dict() ≈ get_weights(); clone so later updates do not overwrite the copy
                best, best_state, wait = val, {k: v.clone() for k, v in net.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is None:
            raise RuntimeError(f"{self.name}: validation loss was never finite (seed {seed})")
        net.load_state_dict(best_state)  # ≈ set_weights(best)
        net.eval()
        return net, epoch, best
