import numpy as np
import pandas as pd
import pytest
import torch
from leakage import assert_no_leakage, leakage_dates

from volsurf.models.cnn import SurfaceCNN, make_cnn
from volsurf.models.torch_base import make_windows
from volsurf.protocol import refit_schedule, walk_forward

TEST_START = pd.Timestamp("2020-06-01")


def tiny_cnn():
    return make_cnn("tiny", seeds=1, epochs=3, patience=2, channels=4, device="cpu", deterministic=True)


def test_make_windows_shapes_and_content():
    levels = np.arange(12 * 2 * 3, dtype=float).reshape(12, 2, 3) ** 1.5
    X = make_windows(levels, 5)
    assert X.shape == (12, 2, 5, 2, 3)
    assert X.dtype == np.float32
    assert np.isnan(X[:5]).all()
    assert np.isfinite(X[5:]).all()
    np.testing.assert_allclose(X[7, 1], levels[3:8], rtol=1e-6)
    np.testing.assert_allclose(X[7, 0], levels[3:8] - levels[2:7], rtol=1e-6)


@pytest.mark.parametrize("head", ["conv", "dense"])
def test_untrained_cnn_is_persistence(head):
    net = SurfaceCNN(7, 9, head=head)
    out = net(torch.randn(4, 2, 5, 7, 9))
    assert out.shape == (4, 63)
    assert torch.all(out == 0)


def test_cnn_no_leakage(synthetic_surface):
    surface = synthetic_surface.loc[:"2020-08-31"]  # three refits
    assert_no_leakage(tiny_cnn, surface, TEST_START, leakage_dates(surface, TEST_START))


def test_cnn_fit_log(synthetic_surface):
    model = tiny_cnn()
    out = walk_forward(model, synthetic_surface, TEST_START)
    assert not torch.are_deterministic_algorithms_enabled()  # global setting restored
    schedule = refit_schedule(synthetic_surface.index, TEST_START)
    assert len(model.fit_log) == len(schedule)
    index = synthetic_surface.index
    for entry, (refit_date, _) in zip(model.fit_log, schedule, strict=True):
        assert entry["refit"] == index[index < refit_date][-1]
        assert len(entry["epochs"]) == 1
    # the NaN row at position 400 only breaks the forecast for the next day (raw t-1 level)
    undefined = out.index == index[401]
    assert out[~undefined].notna().all().all()
    assert out[undefined].isna().all().all()
