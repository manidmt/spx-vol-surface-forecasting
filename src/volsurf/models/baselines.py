"""Baseline forecasters: persistence and rolling AR(1) with a Huber loss."""

import numpy as np
import pandas as pd
import statsmodels.api as sm

from volsurf.models.base import Forecaster


class Persistence(Forecaster):
    """Tomorrow's surface equals today's: the predicted change is zero."""

    name = "persistence"

    def fit(self, history: pd.DataFrame) -> None:
        """Nothing to fit."""

    def predict_change(self, history: pd.DataFrame) -> np.ndarray:
        """Zero change at every grid point."""
        return np.zeros(history.shape[1])


class AR1Huber(Forecaster):
    """Per-column AR(1) on daily changes, fitted by Huber robust regression.

    ``window`` is the number of most recent (d[t-1], d[t]) pairs used at each refit
    (all of them if fewer are available); ``None`` means an expanding window.
    """

    def __init__(self, window: int | None = 252) -> None:
        super().__init__()
        self.window = window
        self.name = "ar1_expanding_huber" if window is None else f"ar1_rolling{window}_huber"
        self.params: np.ndarray | None = None  # (n_points, 2): intercept, slope

    def fit(self, history: pd.DataFrame) -> None:
        """Estimate intercept and slope for every column from the changes in ``history``."""
        d = history.diff()
        params = np.full((history.shape[1], 2), np.nan)
        for i in range(history.shape[1]):
            s = d.iloc[:, i]
            pairs = pd.concat([s, s.shift(1)], axis=1, keys=["y", "x"]).dropna()
            if self.window is not None:
                pairs = pairs.iloc[-self.window :]
            if len(pairs) < 3:
                continue
            fit = sm.RLM(pairs["y"], sm.add_constant(pairs["x"]), M=sm.robust.norms.HuberT()).fit()
            params[i] = fit.params.to_numpy()
        self.params = params
        self.fit_log.append({"n_history": len(history)})

    def predict_change(self, history: pd.DataFrame) -> np.ndarray:
        """``c + b * d_last`` with ``d_last = history[-1] - history[-2]`` (NaN if undefined)."""
        if self.params is None:
            raise RuntimeError("fit must be called before predict_change")
        d_last = history.iloc[-1].to_numpy(dtype=float) - history.iloc[-2].to_numpy(dtype=float)
        return self.params[:, 0] + self.params[:, 1] * d_last
