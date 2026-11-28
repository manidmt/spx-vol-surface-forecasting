"""Surface-memory channels: level minus its trailing m-day mean, per window step."""

import numpy as np
import pandas as pd
import pytest
import torch
from leakage import assert_no_leakage, leakage_dates

from volsurf.models.cnn import SurfaceCNN, make_cnn
from volsurf.models.convlstm import SurfaceConvLSTM, make_convlstm
from volsurf.models.lstm import SurfaceLSTM, make_lstm
from volsurf.models.torch_base import make_windows
from volsurf.protocol import walk_forward

TEST_START = pd.Timestamp("2020-06-01")


def test_make_windows_memory_channels_hand_case():
    levels = np.arange(20 * 2 * 3, dtype=float).reshape(20, 2, 3) ** 1.5
    X = make_windows(levels, 4, memory=[3, 5])
    assert X.shape == (20, 4, 4, 2, 3)
    # row i is valid when the window start (i - 3) has m - 1 earlier days: i >= 4 + 5 - 2 = 7
    assert np.isnan(X[:7]).any(axis=(1, 2, 3, 4)).all()  # dropped from training
    assert np.isfinite(X[7:]).all()
    i, step = 10, 1  # window of row 10 covers days 7..10; step 1 is day 8
    day = i - 4 + 1 + step
    np.testing.assert_allclose(X[i, 2, step], levels[day] - levels[day - 2 : day + 1].mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(X[i, 3, step], levels[day] - levels[day - 4 : day + 1].mean(axis=0), rtol=1e-6)
    # the first two channels are those of the plain windows
    np.testing.assert_array_equal(X[:, :2][7:], make_windows(levels, 4)[7:])


def test_make_windows_rejects_short_memory():
    with pytest.raises(ValueError, match="memory"):
        make_windows(np.zeros((10, 2, 3)), 3, memory=[1])


@pytest.mark.parametrize(
    "net",
    [
        SurfaceCNN(7, 9, in_channels=4),
        SurfaceConvLSTM(7, 9, in_channels=4),
        SurfaceLSTM(7, 9, in_channels=4),
    ],
)
def test_untrained_networks_with_memory_are_persistence(net):
    out = net(torch.randn(3, 4, 5, 7, 9))
    assert out.shape == (3, 63)
    assert torch.all(out == 0)


def _tiny(factory):
    def make():
        return factory("tiny", seeds=1, epochs=3, patience=2, device="cpu", deterministic=True, memory=[3, 6])

    return make


@pytest.mark.parametrize("factory", [make_cnn, make_convlstm, make_lstm])
def test_memory_models_no_leakage(synthetic_surface, factory):
    surface = synthetic_surface.loc[:"2020-08-31"]
    assert_no_leakage(_tiny(factory), surface, TEST_START, leakage_dates(surface, TEST_START))


def test_memory_forecasts_are_finite_and_differ_from_plain(synthetic_surface):
    surface = synthetic_surface.loc[:"2020-07-31"]
    with_memory = walk_forward(_tiny(make_cnn)(), surface, TEST_START)
    plain = walk_forward(
        make_cnn("tiny", seeds=1, epochs=3, patience=2, device="cpu", deterministic=True), surface, TEST_START
    )
    after_nan = with_memory.index == surface.index[401]  # the NaN print at row 400 breaks one forecast
    assert np.isfinite(with_memory[~after_nan].to_numpy()).all()
    assert not np.allclose(with_memory.to_numpy(), plain.to_numpy())


def test_memory_must_be_positive_integers_of_at_least_two():
    with pytest.raises(ValueError, match="memory"):
        make_cnn("bad", memory=[0])
