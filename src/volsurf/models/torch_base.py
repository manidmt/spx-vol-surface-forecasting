"""Shared trainer for neural forecasters (port of the training loop in notebook 05).

The surface enters the networks as a window of the last ``lookback`` days with two channels,
the daily change and the level. PyTorch uses a *channels-first* layout, so one sample is
``(channels, time, tenor, moneyness)`` and a batch is ``(batch, 2, lookback, n_tenors,
n_moneyness)``; Keras' default is channels-last, ``(batch, time, tenor, moneyness, channels)``.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch
from torch import nn

from volsurf.data import grid_shape
from volsurf.models.base import Forecaster


def make_windows(levels: np.ndarray, lookback: int) -> np.ndarray:
    """Stack the last ``lookback`` days of every date as a two-channel window.

    ``levels`` has shape ``(T, n_tenors, n_moneyness)``. Returns float32
    ``(T, 2, lookback, n_tenors, n_moneyness)``: row ``i`` holds days ``i-lookback+1 … i``,
    channel 0 is the daily change and channel 1 the level. Rows ``< lookback`` are NaN
    (the first change of the window needs the day before it).
    """
    levels = np.asarray(levels, dtype=float)
    change = np.full_like(levels, np.nan)
    change[1:] = levels[1:] - levels[:-1]
    X = np.full((len(levels), 2, lookback, *levels.shape[1:]), np.nan, dtype=np.float32)
    for i in range(lookback, len(levels)):
        X[i, 0] = change[i - lookback + 1 : i + 1]
        X[i, 1] = levels[i - lookback + 1 : i + 1]
    return X


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
        self.nets: list[nn.Module] = []
        # scaling statistics, fitted on the training rows at each refit
        self.sd_change: np.float32 | None = None
        self.mu_level: np.float32 | None = None
        self.sd_level: np.float32 | None = None
        self.sd_y: np.ndarray | None = None

    def fit(self, history: pd.DataFrame) -> None:
        """Train ``seeds`` networks on every complete (window, next-day change) pair in ``history``.

        Inputs are windows of the forward-filled surface (filling only uses the past); targets
        are the raw next-day changes, so rows whose target involves a NaN print are dropped.
        The last row has no next day inside ``history`` and drops out the same way.
        """
        with _deterministic(self.deterministic):
            n_tenors, n_moneyness = grid_shape(history)
            levels = history.ffill().to_numpy(dtype=float).reshape(len(history), n_tenors, n_moneyness)
            X = make_windows(levels, self.lookback)
            Y = history.diff().shift(-1).to_numpy(dtype=np.float32)
            ok = np.isfinite(X).all(axis=(1, 2, 3, 4)) & np.isfinite(Y).all(axis=1)
            Xtr, Ytr = X[ok], Y[ok]

            # scaling fitted on the training rows only; each target point by its own std
            # (short tenors move about four times more than long ones)
            self.sd_change = Xtr[:, 0].std()
            self.mu_level, self.sd_level = Xtr[:, 1].mean(), Xtr[:, 1].std()
            self.sd_y = Ytr.std(axis=0)

            X_scaled = self._scale(Xtr)
            Y_scaled = torch.tensor(Ytr / self.sd_y, device=self.device)
            self.nets, epochs = [], []
            for seed in range(self.seeds):
                net, last_epoch = self._train(X_scaled, Y_scaled, seed, n_tenors, n_moneyness)
                self.nets.append(net)
                epochs.append(last_epoch)
        self.fit_log.append({"refit": history.index[-1], "epochs": epochs})

    def predict_change(self, history: pd.DataFrame) -> np.ndarray:
        """Mean forecast of the seed networks from the window ending at the last row of ``history``."""
        if not self.nets:
            raise RuntimeError("fit must be called before predict_change")
        with _deterministic(self.deterministic):
            n_tenors, n_moneyness = grid_shape(history)
            recent = history.ffill().iloc[-(self.lookback + 1) :]  # lookback changes need one extra day
            levels = recent.to_numpy(dtype=float).reshape(len(recent), n_tenors, n_moneyness)
            x = self._scale(make_windows(levels, self.lookback)[-1:])
            # torch.no_grad() ≈ model.predict: no gradients are recorded
            with torch.no_grad():
                out = [net(x).cpu().numpy() * self.sd_y for net in self.nets]
        return np.mean(out, axis=0)[0]

    def _scale(self, X: np.ndarray) -> torch.Tensor:
        """Change channel divided by its std, level channel standardised; returned on the device."""
        X = X.copy()
        X[:, 0] /= self.sd_change
        X[:, 1] = (X[:, 1] - self.mu_level) / self.sd_level
        return torch.tensor(X, device=self.device)

    def _train(
        self, X: torch.Tensor, Y: torch.Tensor, seed: int, n_tenors: int, n_moneyness: int
    ) -> tuple[nn.Module, int]:
        """One seed of the notebook 05 loop; returns the best-validation network and the last epoch."""
        # PyTorch has no model.fit(): the loop below is what Keras' fit() does internally
        torch.manual_seed(seed)  # ≈ keras.utils.set_random_seed: fixes init and shuffling
        n_val = max(int(self.val_frac * len(X)), 20)
        X_train, Y_train, X_val, Y_val = X[:-n_val], Y[:-n_val], X[-n_val:], Y[-n_val:]
        net = self.net_factory(n_tenors, n_moneyness).to(self.device)  # .to(device) ≈ placing on GPU
        # ≈ model.compile(optimizer=AdamW(...), loss=Huber(delta))
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.HuberLoss(delta=self.huber_delta)
        # early stopping ≈ EarlyStopping(monitor="val_loss", patience, restore_best_weights=True)
        best, best_state, wait = np.inf, None, 0
        for epoch in range(self.epochs):  # noqa: B007 (the last epoch is returned)
            net.train()  # ≈ training=True: layers such as dropout behave in training mode
            for b in torch.randperm(len(X_train)).split(self.batch):  # shuffled minibatches
                opt.zero_grad()  # gradients accumulate in PyTorch, so reset them every step
                loss_fn(net(X_train[b]), Y_train[b]).backward()  # backpropagation
                opt.step()  # apply the update ≈ optimizer.apply_gradients
            net.eval()  # ≈ training=False
            with torch.no_grad():
                val = loss_fn(net(X_val), Y_val).item()
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
        return net, epoch
