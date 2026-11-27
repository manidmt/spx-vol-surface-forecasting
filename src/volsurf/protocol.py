"""Walk-forward evaluation protocol."""

import numpy as np
import pandas as pd

from volsurf.models.base import Forecaster

TEST_START = pd.Timestamp("2021-01-01")


def refit_schedule(
    index: pd.DatetimeIndex, test_start: pd.Timestamp = TEST_START
) -> list[tuple[pd.Timestamp, pd.DatetimeIndex]]:
    """Group test days by calendar month: (first test day of the month, all its test days)."""
    test_days = index[index >= test_start]
    schedule = []
    for _, days in test_days.to_series().groupby(test_days.to_period("M")):
        month_days = pd.DatetimeIndex(days.index)
        schedule.append((month_days[0], month_days))
    return schedule


def walk_forward(
    model: Forecaster, surface: pd.DataFrame, test_start: pd.Timestamp = TEST_START
) -> pd.DataFrame:
    """Level forecasts for every test day: ``surface[t-1] + model.predict_change(surface[:t-1])``.

    The model is refit on the first test day of each month using only dates strictly before
    it. The previous-day level is the raw surface (not forward-filled), so a NaN row makes the
    next day's forecast NaN. Output has the same columns as ``surface``, indexed by test day.
    """
    index = surface.index
    if not (index.is_monotonic_increasing and index.is_unique):
        raise ValueError("surface.index must be sorted ascending with unique dates")
    test_days = index[index >= test_start]
    out = pd.DataFrame(np.nan, index=test_days, columns=surface.columns, dtype=float)
    for refit_date, days in refit_schedule(index, test_start):
        model.fit(surface[index < refit_date])
        for t in days:
            pos = index.get_loc(t)
            assert pos > 0, "test_start must be after the first date of the data"
            history = surface.iloc[:pos]
            previous = history.iloc[-1].to_numpy(dtype=float)
            out.loc[t] = previous + np.asarray(model.predict_change(history), dtype=float)
    return out
