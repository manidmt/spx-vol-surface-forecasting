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


def _materialise(surface: pd.DataFrame, stop: int) -> pd.DataFrame:
    """Rows ``[0, stop)`` of ``surface`` as a fresh frame sharing no storage with it.

    A positional slice of a DataFrame is a view: its values and dates keep a reference to the
    full arrays (reachable through ``ndarray.base``), so a model could read rows after its
    forecast date. Values and dates are copied here, and only the visible rows are copied.
    """
    values = surface.iloc[:stop].to_numpy(dtype=float, copy=True)
    dates = pd.DatetimeIndex(surface.index[:stop].to_numpy(copy=True), name=surface.index.name)
    return pd.DataFrame(values, index=dates, columns=surface.columns.copy(), copy=False)


def walk_forward(
    model: Forecaster, surface: pd.DataFrame, test_start: pd.Timestamp = TEST_START
) -> pd.DataFrame:
    """Level forecasts for every test day: ``surface[t-1] + model.predict_change(surface[:t-1])``.

    The model is refit on the first test day of each month using only dates strictly before
    it. Every history handed to the model is a private copy of the rows before the date in
    question, so the model cannot reach later rows even through the backing arrays. The
    previous-day level is the raw surface (not forward-filled), read from the runner's own
    copy, so a NaN row makes the next day's forecast NaN. Output has the same columns as
    ``surface``, indexed by test day.

    Raises ``ValueError`` if the index is not sorted and unique, if ``test_start`` is not
    after the first date, or if ``predict_change`` does not return shape ``(n_points,)``.
    """
    index = surface.index
    if not (index.is_monotonic_increasing and index.is_unique):
        raise ValueError("surface.index must be sorted ascending with unique dates")
    test_days = index[index >= test_start]
    if len(test_days) > 0 and test_days[0] == index[0]:
        raise ValueError(f"test_start ({test_start}) must be after the first date of the data ({index[0]})")
    levels = surface.to_numpy(dtype=float, copy=True)  # the runner's copy, never handed to the model
    n_points = surface.shape[1]
    out = np.full((len(test_days), n_points), np.nan)
    row = 0
    for refit_date, days in refit_schedule(index, test_start):
        model.fit(_materialise(surface, index.get_loc(refit_date)))
        for t in days:
            pos = index.get_loc(t)
            change = np.asarray(model.predict_change(_materialise(surface, pos)), dtype=float)
            if change.shape != (n_points,):
                raise ValueError(
                    f"{model.name}: predict_change for {t.date()} returned shape {change.shape}, "
                    f"expected ({n_points},)"
                )
            out[row] = levels[pos - 1] + change
            row += 1
    return pd.DataFrame(out, index=test_days, columns=surface.columns)
