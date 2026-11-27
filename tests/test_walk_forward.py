import numpy as np
import pandas as pd

from volsurf.models.base import Forecaster
from volsurf.protocol import refit_schedule, walk_forward

TEST_START = pd.Timestamp("2020-01-01")


class SpyForecaster(Forecaster):
    """Records the last history date seen by every fit and predict call; forecasts no change."""

    name = "spy"

    def __init__(self) -> None:
        super().__init__()
        self.fit_ends: list[pd.Timestamp] = []
        self.predict_ends: list[pd.Timestamp] = []

    def fit(self, history: pd.DataFrame) -> None:
        self.fit_ends.append(history.index[-1])

    def predict_change(self, history: pd.DataFrame) -> np.ndarray:
        self.predict_ends.append(history.index[-1])
        return np.zeros(history.shape[1])


def test_fit_sees_only_past(synthetic_surface):
    spy = SpyForecaster()
    walk_forward(spy, synthetic_surface, test_start=TEST_START)
    schedule = refit_schedule(synthetic_surface.index, TEST_START)
    assert len(spy.fit_ends) == len(schedule)
    for last_seen, (refit_date, _) in zip(spy.fit_ends, schedule, strict=True):
        assert last_seen < refit_date


def test_predict_sees_up_to_previous_day(synthetic_surface):
    spy = SpyForecaster()
    out = walk_forward(spy, synthetic_surface, test_start=TEST_START)
    index = synthetic_surface.index
    expected = [index[index.get_loc(t) - 1] for t in out.index]
    assert spy.predict_ends == expected


def test_zero_change_is_persistence(synthetic_surface):
    out = walk_forward(SpyForecaster(), synthetic_surface, test_start=TEST_START)
    expected = synthetic_surface.shift(1).loc[out.index]
    pd.testing.assert_frame_equal(out, expected)
    assert out.index[0] >= TEST_START


def test_nan_row_propagates(synthetic_surface):
    out = walk_forward(SpyForecaster(), synthetic_surface, test_start=TEST_START)
    nan_day = synthetic_surface.index[400]
    after = synthetic_surface.index[401]
    assert nan_day in out.index
    assert out.loc[after].isna().all()
    assert out.loc[synthetic_surface.index[402]].notna().all()
