import pandas as pd
import pytest

from volsurf.data import drop_holdout, grid_shape, load_surface


def _frame() -> pd.DataFrame:
    index = pd.to_datetime(["2026-10-07", "2026-10-08"])
    return pd.DataFrame({"a": [1.0, 2.0]}, index=index)


def test_drop_holdout_default():
    out = drop_holdout(_frame())
    assert list(out.index) == [pd.Timestamp("2026-10-07")]
    assert len(drop_holdout(_frame(), include_holdout=True)) == 2


def test_grid_shape(synthetic_surface):
    assert grid_shape(synthetic_surface) == (7, 9)


@pytest.mark.slow
def test_load_surface_real(real_surface):
    df = load_surface()
    assert df.shape == (1952, 63)
    assert df.index[0] == pd.Timestamp("2019-01-02")
    assert df.index[-1] == pd.Timestamp("2026-10-07")
    assert list(df.columns.names) == ["tenor", "moneyness"]
    expected = ["30DAY", "60DAY", "3MTH", "6MTH", "12MTH", "18MTH", "24MTH"]
    assert list(df.columns.get_level_values("tenor").unique()) == expected
