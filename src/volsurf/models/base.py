"""Forecaster interface.

Timing convention: the runner owns time. For a forecast day ``t`` it hands the model
``history`` ending at ``t-1`` (the previous index date); ``fit`` receives data strictly
before the refit date. The model returns the forecast of ``surface[t] - surface[t-1]``
in column order, and never sees anything dated ``t`` or later.
"""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class Forecaster(ABC):
    """One-day-ahead forecaster of surface changes."""

    name: str

    def __init__(self) -> None:
        self.fit_log: list[dict] = []

    @abstractmethod
    def fit(self, history: pd.DataFrame) -> None:
        """Fit on the raw surface strictly before the refit date (NaN rows included)."""

    @abstractmethod
    def predict_change(self, history: pd.DataFrame) -> np.ndarray:
        """Forecast ``surface[t] - surface[t-1]``, shape ``(n_points,)``; ``history`` ends at t-1."""
