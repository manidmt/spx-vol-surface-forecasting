"""Walk-forward evaluation protocol."""

from collections.abc import Callable

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
    """Rows ``[0, stop)`` of ``surface`` (or of the exogenous panel) as a fresh frame sharing no storage.

    A positional slice of a DataFrame is a view: its values and dates keep a reference to the
    full arrays (reachable through ``ndarray.base``), so a model could read rows after its
    forecast date. Values and dates are copied here, and only the visible rows are copied.
    """
    values = surface.iloc[:stop].to_numpy(dtype=float, copy=True)
    dates = pd.DatetimeIndex(surface.index[:stop].to_numpy(copy=True), name=surface.index.name)
    return pd.DataFrame(values, index=dates, columns=surface.columns.copy(), copy=False)


def _align_exog(exog: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """``exog`` reindexed to the surface dates; ``ValueError`` if it misses any of them."""
    if not exog.index.is_unique:
        raise ValueError("exog.index must have unique dates")
    missing = index.difference(exog.index)
    if len(missing) > 0:
        raise ValueError(
            f"exog must cover every surface date; it misses {len(missing)} surface dates, "
            f"first {missing[0].date()}"
        )
    return exog.reindex(index)


def walk_forward(
    model: Forecaster,
    surface: pd.DataFrame,
    test_start: pd.Timestamp = TEST_START,
    exog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Level forecasts for every test day: ``surface[t-1] + model.predict_change(surface[:t-1])``.

    The model is refit on the first test day of each month using only dates strictly before
    it. Every history handed to the model is a private copy of the rows before the date in
    question, so the model cannot reach later rows even through the backing arrays. The
    previous-day level is the raw surface (not forward-filled), read from the runner's own
    copy, so a NaN row makes the next day's forecast NaN. Output has the same columns as
    ``surface``, indexed by test day.

    ``exog`` (optional) is a panel of exogenous inputs whose row ``d`` holds only what is known
    at the close of ``d``. It is reindexed to the surface dates and sliced with exactly the
    surface's rules: ``fit`` gets the rows strictly before the refit date and the forecast for
    ``t`` the rows up to ``t-1``, as private copies, passed as the ``exog`` keyword. Without
    ``exog`` the model is called with the history alone.

    Raises ``ValueError`` if the index is not sorted and unique, if ``test_start`` is not
    after the first date, if ``exog`` has duplicate dates or misses a surface date, or if
    ``predict_change`` does not return shape ``(n_points,)``.
    """
    index = surface.index
    if not (index.is_monotonic_increasing and index.is_unique):
        raise ValueError("surface.index must be sorted ascending with unique dates")
    test_days = index[index >= test_start]
    if len(test_days) > 0 and test_days[0] == index[0]:
        raise ValueError(f"test_start ({test_start}) must be after the first date of the data ({index[0]})")
    aligned = None if exog is None else _align_exog(exog, index)

    def call(method: Callable, stop: int):
        """Call ``model.fit`` or ``model.predict_change`` on fresh copies of rows ``[0, stop)``."""
        history = _materialise(surface, stop)
        if aligned is None:
            return method(history)
        return method(history, exog=_materialise(aligned, stop))

    levels = surface.to_numpy(dtype=float, copy=True)  # the runner's copy, never handed to the model
    n_points = surface.shape[1]
    out = np.full((len(test_days), n_points), np.nan)
    row = 0
    for refit_date, days in refit_schedule(index, test_start):
        call(model.fit, index.get_loc(refit_date))
        for t in days:
            pos = index.get_loc(t)
            change = np.asarray(call(model.predict_change, pos), dtype=float)
            if change.shape != (n_points,):
                raise ValueError(
                    f"{model.name}: predict_change for {t.date()} returned shape {change.shape}, "
                    f"expected ({n_points},)"
                )
            out[row] = levels[pos - 1] + change
            row += 1
    return pd.DataFrame(out, index=test_days, columns=surface.columns)
