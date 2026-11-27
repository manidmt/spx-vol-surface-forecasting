import pandas as pd
import pytest
from leakage import assert_no_leakage, leakage_dates

from volsurf.models.baselines import AR1Huber, Persistence
from volsurf.protocol import walk_forward

PREDS_PATH = "data/processed/surface_preds.pkl"
TEST_START = pd.Timestamp("2020-01-01")


@pytest.mark.slow
def test_persistence_matches_notebook(real_surface):
    ref = pd.read_pickle(PREDS_PATH)["persistence"]
    out = walk_forward(Persistence(), real_surface)
    pd.testing.assert_frame_equal(out, ref, check_freq=False)


@pytest.mark.slow
def test_ar1_matches_notebook(real_surface):
    ref = pd.read_pickle(PREDS_PATH)["ar1_rolling252_huber"]
    out = walk_forward(AR1Huber(252), real_surface)
    pd.testing.assert_frame_equal(out, ref, check_freq=False, atol=1e-8, rtol=0)


def test_persistence_no_leakage(synthetic_surface):
    dates = [*leakage_dates(synthetic_surface, TEST_START), synthetic_surface.index[449]]
    assert_no_leakage(Persistence, synthetic_surface, TEST_START, dates)


def test_ar1_no_leakage(synthetic_surface):
    surface = synthetic_surface.loc[:"2020-04-30"]  # three refits are enough and keep the test fast
    assert_no_leakage(lambda: AR1Huber(252), surface, TEST_START, leakage_dates(surface, TEST_START))


def test_ar1_survives_nan_row(synthetic_surface):
    out = walk_forward(AR1Huber(252), synthetic_surface, TEST_START)
    # a NaN row at t-1 or t-2 makes the forecast undefined (level or last change is NaN)
    undefined = out.index.isin(synthetic_surface.index[[401, 402]])
    assert out[~undefined].notna().all().all()
    assert out[undefined].isna().all().all()


def test_ar1_names():
    assert AR1Huber(252).name == "ar1_rolling252_huber"
    assert AR1Huber(100).name == "ar1_rolling100_huber"
    assert AR1Huber(None).name == "ar1_expanding_huber"
