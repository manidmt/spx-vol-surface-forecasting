"""Shared leakage check for forecasters."""

from collections.abc import Callable

import numpy as np
import pandas as pd

from volsurf.models.base import Forecaster
from volsurf.protocol import walk_forward


def assert_no_leakage(
    make_model: Callable[[], Forecaster],
    surface: pd.DataFrame,
    cutoff: pd.Timestamp,
    test_start: pd.Timestamp,
) -> None:
    """Forecasts for days <= cutoff must not change when the data after cutoff is replaced by noise."""
    noisy = surface.copy()
    rng = np.random.default_rng(123)
    after = noisy.index > cutoff
    noisy.loc[after, :] = rng.normal(scale=5.0, size=(int(after.sum()), noisy.shape[1]))

    base = walk_forward(make_model(), surface, test_start)
    perturbed = walk_forward(make_model(), noisy, test_start)
    days = base.index[base.index <= cutoff]
    assert len(days) > 0
    np.testing.assert_array_equal(base.loc[days].to_numpy(), perturbed.loc[days].to_numpy())
