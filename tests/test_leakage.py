"""The leakage helpers must catch a forecaster that reads its own target."""

import numpy as np
import pandas as pd
import pytest
from leakage import assert_future_noise_invariant, assert_no_leakage, assert_same_day_invariant, leakage_dates

from volsurf import protocol
from volsurf.models.base import Forecaster
from volsurf.models.baselines import Persistence

TEST_START = pd.Timestamp("2020-01-01")


class BackingArrayCheat(Forecaster):
    """Returns the forecast day's own change when it can reach it through the backing array."""

    name = "backing_array_cheat"

    def fit(self, history: pd.DataFrame) -> None:
        pass

    def predict_change(self, history: pd.DataFrame) -> np.ndarray:
        n_rows, n_points = history.shape
        array = history.to_numpy()
        while isinstance(array, np.ndarray):
            if array.ndim == 2 and array.shape[0] == n_points and array.shape[1] > n_rows:
                array = array.T  # pandas may keep the block as (points, days)
            if array.ndim == 2 and array.shape[1] == n_points and array.shape[0] > n_rows:
                return array[n_rows] - array[n_rows - 1]  # the target day minus the last visible day
            array = array.base
        return np.zeros(n_points)


def _view_history(surface: pd.DataFrame, stop: int) -> pd.DataFrame:
    """The runner's old behaviour: a positional view that keeps the full arrays reachable."""
    return surface.iloc[:stop]


@pytest.fixture
def short_surface(synthetic_surface):
    return synthetic_surface.loc[:"2020-04-30"]


def test_leakage_dates_are_refit_boundary_and_mid_month(short_surface):
    boundary, mid = leakage_dates(short_surface, TEST_START)
    schedule = protocol.refit_schedule(short_surface.index, TEST_START)
    assert boundary in [refit for refit, _ in schedule]
    assert mid not in [refit for refit, _ in schedule]
    assert boundary.to_period("M") == mid.to_period("M")


def test_same_day_check_catches_cheat_on_leaky_runner(monkeypatch, short_surface):
    monkeypatch.setattr(protocol, "_materialise", _view_history)
    out = protocol.walk_forward(BackingArrayCheat(), short_surface, TEST_START)
    np.testing.assert_allclose(out.to_numpy(), short_surface.loc[out.index].to_numpy())  # it does cheat
    dates = leakage_dates(short_surface, TEST_START)
    for date in dates:
        # the future-noise check alone (the old helper) misses it
        assert_future_noise_invariant(BackingArrayCheat, short_surface, TEST_START, date)
        with pytest.raises(AssertionError):
            assert_same_day_invariant(BackingArrayCheat, short_surface, TEST_START, date)
    with pytest.raises(AssertionError):
        assert_no_leakage(BackingArrayCheat, short_surface, TEST_START, dates)


def test_cheat_finds_nothing_on_isolated_runner(short_surface):
    out = protocol.walk_forward(BackingArrayCheat(), short_surface, TEST_START)
    pd.testing.assert_frame_equal(out, protocol.walk_forward(Persistence(), short_surface, TEST_START))
    assert_no_leakage(BackingArrayCheat, short_surface, TEST_START, leakage_dates(short_surface, TEST_START))
