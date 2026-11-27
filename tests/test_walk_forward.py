import numpy as np
import pandas as pd
import pytest

from volsurf.models.base import Forecaster
from volsurf.protocol import refit_schedule, walk_forward

TEST_START = pd.Timestamp("2020-01-01")


class SpyForecaster(Forecaster):
    """Records the last history date seen by every fit and predict call; forecasts no change."""

    name = "spy"

    def __init__(self) -> None:
        super().__init__()
        self.fit_indexes: list[pd.DatetimeIndex] = []
        self.predict_ends: list[pd.Timestamp] = []

    def fit(self, history: pd.DataFrame) -> None:
        self.fit_indexes.append(history.index.copy())

    def predict_change(self, history: pd.DataFrame) -> np.ndarray:
        self.predict_ends.append(history.index[-1])
        return np.zeros(history.shape[1])


def test_fit_sees_only_past(synthetic_surface):
    spy = SpyForecaster()
    walk_forward(spy, synthetic_surface, test_start=TEST_START)
    schedule = refit_schedule(synthetic_surface.index, TEST_START)
    index = synthetic_surface.index
    assert len(spy.fit_indexes) == len(schedule)
    for seen, (refit_date, _) in zip(spy.fit_indexes, schedule, strict=True):
        # the full prefix, ending at the observation immediately before the refit date
        pd.testing.assert_index_equal(seen, index[index < refit_date], check_exact=True)
        assert seen[-1] == index[index.get_loc(refit_date) - 1]


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


def test_unsorted_index_raises(synthetic_surface):
    shuffled = synthetic_surface.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError, match="sorted"):
        walk_forward(SpyForecaster(), shuffled, test_start=TEST_START)


def test_duplicate_date_raises(synthetic_surface):
    duplicated = pd.concat([synthetic_surface, synthetic_surface.iloc[[500]]]).sort_index()
    with pytest.raises(ValueError, match="unique"):
        walk_forward(SpyForecaster(), duplicated, test_start=TEST_START)


def _base_chain(array: np.ndarray) -> list[np.ndarray]:
    """``array`` and every array reachable through ``.base``."""
    chain = []
    while isinstance(array, np.ndarray):
        chain.append(array)
        array = array.base
    return chain


class IsolationSpy(Forecaster):
    """Checks that every history it receives is a private copy bounded at its last row."""

    name = "isolation_spy"

    def __init__(self, surface: pd.DataFrame) -> None:
        super().__init__()
        self.full_values = surface.to_numpy()
        self.full_dates = surface.index.asi8
        self.calls = 0

    def _check(self, history: pd.DataFrame) -> None:
        values = history.to_numpy()
        assert not np.shares_memory(values, self.full_values)
        for array in (*_base_chain(values), *_base_chain(history.index.asi8)):
            assert array.size <= history.size  # nothing beyond the last history row is reachable
            assert not np.shares_memory(array, self.full_values)
            assert not np.shares_memory(array, self.full_dates)
        self.calls += 1

    def fit(self, history: pd.DataFrame) -> None:
        self._check(history)

    def predict_change(self, history: pd.DataFrame) -> np.ndarray:
        self._check(history)
        return np.zeros(history.shape[1])


def test_history_shares_no_storage_with_surface(synthetic_surface):
    spy = IsolationSpy(synthetic_surface)
    out = walk_forward(spy, synthetic_surface, test_start=TEST_START)
    assert spy.calls == len(out) + len(refit_schedule(synthetic_surface.index, TEST_START))


class Vandal(Forecaster):
    """Overwrites every history it is given in place; forecasts no change."""

    name = "vandal"

    def fit(self, history: pd.DataFrame) -> None:
        history.iloc[:, :] = 1e6

    def predict_change(self, history: pd.DataFrame) -> np.ndarray:
        history.iloc[:, :] = 1e6
        return np.zeros(history.shape[1])


def test_previous_level_independent_of_model_history(synthetic_surface):
    original = synthetic_surface.copy()
    out = walk_forward(Vandal(), synthetic_surface, test_start=TEST_START)
    pd.testing.assert_frame_equal(synthetic_surface, original)
    pd.testing.assert_frame_equal(out, original.shift(1).loc[out.index])


def test_test_start_at_first_date_raises_before_fit(synthetic_surface):
    spy = SpyForecaster()
    with pytest.raises(ValueError, match="first date"):
        walk_forward(spy, synthetic_surface, test_start=synthetic_surface.index[0])
    assert spy.fit_indexes == []


class FixedOutput(Forecaster):
    """Returns the same (possibly malformed) change for every day."""

    name = "fixed_output"

    def __init__(self, change) -> None:
        super().__init__()
        self.change = change

    def fit(self, history: pd.DataFrame) -> None:
        pass

    def predict_change(self, history: pd.DataFrame) -> np.ndarray:
        return self.change


@pytest.mark.parametrize("change", [0.0, np.zeros(1), np.zeros((1, 63)), np.zeros(62)])
def test_wrong_output_shape_raises(synthetic_surface, change):
    first_day = synthetic_surface.index[synthetic_surface.index >= TEST_START][0]
    with pytest.raises(ValueError, match=rf"fixed_output.*{first_day.date()}.*\(63,\)"):
        walk_forward(FixedOutput(change), synthetic_surface, test_start=TEST_START)


@pytest.fixture
def synthetic_exog(synthetic_surface) -> pd.DataFrame:
    """Three exogenous columns on the surface dates, with a NaN in one of them."""
    rng = np.random.default_rng(1)
    exog = pd.DataFrame(
        rng.normal(size=(len(synthetic_surface), 3)), index=synthetic_surface.index, columns=["a", "b", "c"]
    )
    exog.iloc[50, 1] = np.nan
    return exog


class ExogSpy(Forecaster):
    """Records the exogenous rows seen by every fit and predict call; forecasts no change."""

    name = "exog_spy"

    def __init__(self, full_exog: pd.DataFrame) -> None:
        super().__init__()
        self.full_values = full_exog.to_numpy()
        self.full_dates = full_exog.index.asi8
        self.fit_seen: list[pd.DataFrame] = []
        self.predict_seen: list[pd.DataFrame] = []

    def _check(self, history: pd.DataFrame, exog: pd.DataFrame | None) -> pd.DataFrame:
        assert exog is not None
        pd.testing.assert_index_equal(exog.index, history.index, check_exact=True)
        values = exog.to_numpy()
        for array in (*_base_chain(values), *_base_chain(exog.index.asi8)):
            assert array.size <= exog.size  # nothing beyond the last visible row is reachable
            assert not np.shares_memory(array, self.full_values)
            assert not np.shares_memory(array, self.full_dates)
        return exog.copy()

    def fit(self, history: pd.DataFrame, exog: pd.DataFrame | None = None) -> None:
        self.fit_seen.append(self._check(history, exog))

    def predict_change(self, history: pd.DataFrame, exog: pd.DataFrame | None = None) -> np.ndarray:
        self.predict_seen.append(self._check(history, exog))
        return np.zeros(history.shape[1])


def test_exog_sliced_like_surface(synthetic_surface, synthetic_exog):
    spy = ExogSpy(synthetic_exog)
    out = walk_forward(spy, synthetic_surface, test_start=TEST_START, exog=synthetic_exog)
    index = synthetic_surface.index
    schedule = refit_schedule(index, TEST_START)
    assert len(spy.fit_seen) == len(schedule)
    for seen, (refit_date, _) in zip(spy.fit_seen, schedule, strict=True):
        pd.testing.assert_frame_equal(seen, synthetic_exog.loc[index < refit_date], check_freq=False)
    assert len(spy.predict_seen) == len(out)
    for seen, t in zip(spy.predict_seen, out.index, strict=True):
        pd.testing.assert_frame_equal(seen, synthetic_exog.loc[index < t], check_freq=False)
        assert seen.index[-1] == index[index.get_loc(t) - 1]
    pd.testing.assert_frame_equal(out, synthetic_surface.shift(1).loc[out.index])


def test_exog_reindexed_to_surface(synthetic_surface, synthetic_exog):
    # extra dates (before, inside and after the surface) and a shuffled order are dropped by reindexing
    extra = pd.DataFrame(
        99.0,
        index=pd.DatetimeIndex(["2018-12-31", "2019-06-01", "2030-01-01"]),
        columns=synthetic_exog.columns,
    )
    messy = pd.concat([synthetic_exog, extra]).sample(frac=1.0, random_state=0)
    spy = ExogSpy(messy)
    walk_forward(spy, synthetic_surface, test_start=TEST_START, exog=messy)
    last = spy.predict_seen[-1]
    pd.testing.assert_frame_equal(last, synthetic_exog.iloc[: len(last)], check_freq=False)


def test_exog_must_cover_every_surface_date(synthetic_surface, synthetic_exog):
    spy = ExogSpy(synthetic_exog)
    missing = synthetic_exog.drop(synthetic_exog.index[[10, 300]])
    with pytest.raises(ValueError, match=r"exog.*2 surface dates.*2019-01-15"):
        walk_forward(spy, synthetic_surface, test_start=TEST_START, exog=missing)
    assert spy.fit_seen == []


def test_exog_duplicate_date_raises(synthetic_surface, synthetic_exog):
    duplicated = pd.concat([synthetic_exog, synthetic_exog.iloc[[7]]])
    with pytest.raises(ValueError, match="exog.*unique"):
        walk_forward(ExogSpy(synthetic_exog), synthetic_surface, test_start=TEST_START, exog=duplicated)


def test_exog_vandal_cannot_change_caller_panel(synthetic_surface, synthetic_exog):
    class ExogVandal(Forecaster):
        name = "exog_vandal"

        def fit(self, history, exog=None):
            exog.iloc[:, :] = 1e6

        def predict_change(self, history, exog=None):
            exog.iloc[:, :] = 1e6
            return np.zeros(history.shape[1])

    original = synthetic_exog.copy()
    walk_forward(ExogVandal(), synthetic_surface, test_start=TEST_START, exog=synthetic_exog)
    pd.testing.assert_frame_equal(synthetic_exog, original)
