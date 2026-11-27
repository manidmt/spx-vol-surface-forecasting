"""Shared leakage checks for forecasters.

Two perturbations of the surface, each compared with an unperturbed run:

- future noise: rows strictly after a date ``d`` are replaced by noise; forecasts up to and
  including ``d`` must not change.
- same-day noise: rows from ``d`` inclusive are replaced by noise; forecasts up to and including
  ``d`` must still not change, since the forecast for ``d`` may only use data before ``d``. This
  catches a model that reads its own target.
"""

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

from volsurf.models.base import Forecaster
from volsurf.protocol import refit_schedule, walk_forward


def _with_noise(surface: pd.DataFrame, rows: np.ndarray) -> pd.DataFrame:
    noisy = surface.copy()
    rng = np.random.default_rng(123)
    noisy.loc[rows, :] = rng.normal(scale=5.0, size=(int(rows.sum()), noisy.shape[1]))
    return noisy


def _assert_same_forecasts_through(
    make_model: Callable[[], Forecaster],
    surface: pd.DataFrame,
    noisy: pd.DataFrame,
    test_start: pd.Timestamp,
    date: pd.Timestamp,
) -> None:
    base = walk_forward(make_model(), surface, test_start)
    assert date in base.index, f"{date} is not a forecast day"
    perturbed = walk_forward(make_model(), noisy, test_start)
    days = base.index[base.index <= date]
    np.testing.assert_array_equal(
        base.loc[days].to_numpy(), perturbed.loc[days].to_numpy(), err_msg=f"forecasts through {date}"
    )


def assert_future_noise_invariant(
    make_model: Callable[[], Forecaster], surface: pd.DataFrame, test_start: pd.Timestamp, date: pd.Timestamp
) -> None:
    """Forecasts through ``date`` must not change when the rows after ``date`` become noise."""
    noisy = _with_noise(surface, surface.index > date)
    _assert_same_forecasts_through(make_model, surface, noisy, test_start, date)


def assert_same_day_invariant(
    make_model: Callable[[], Forecaster], surface: pd.DataFrame, test_start: pd.Timestamp, date: pd.Timestamp
) -> None:
    """Forecasts through ``date`` must not change when the rows from ``date`` inclusive become noise."""
    noisy = _with_noise(surface, surface.index >= date)
    _assert_same_forecasts_through(make_model, surface, noisy, test_start, date)


def assert_no_leakage(
    make_model: Callable[[], Forecaster],
    surface: pd.DataFrame,
    test_start: pd.Timestamp,
    dates: Iterable[pd.Timestamp],
) -> None:
    """Run both the future-noise and the same-day checks at every forecast date in ``dates``."""
    for date in dates:
        assert_future_noise_invariant(make_model, surface, test_start, date)
        assert_same_day_invariant(make_model, surface, test_start, date)


def leakage_dates(surface: pd.DataFrame, test_start: pd.Timestamp, month: int = 1) -> list[pd.Timestamp]:
    """A monthly refit boundary (first test day of a month) and an ordinary mid-month day of it.

    ``month`` is the position of the month in the refit schedule (the second month by default,
    so that the boundary is a refit with earlier forecasts before it).
    """
    _, days = refit_schedule(surface.index, test_start)[month]
    return [days[0], days[len(days) // 2]]
