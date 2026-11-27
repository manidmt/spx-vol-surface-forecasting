"""Adding the exogenous branch must leave surface-only forecasts bit-for-bit unchanged.

``golden/n_exog0_forecasts.npz`` holds the forecasts of tiny CPU runs recorded with the code
before the exogenous branch existed (commit 7ed624f), with one intra-op thread: the order of
floating-point reductions depends on the thread count, so the comparison pins it to one.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from volsurf.models.cnn import make_cnn
from volsurf.models.convlstm import make_convlstm
from volsurf.models.lstm import make_lstm
from volsurf.protocol import walk_forward

GOLDEN = Path(__file__).resolve().parent / "golden" / "n_exog0_forecasts.npz"
TEST_START = pd.Timestamp("2020-06-01")
COMMON = dict(seeds=1, epochs=3, patience=2, device="cpu", deterministic=True)
MODELS = {
    "cnn": lambda: make_cnn("tiny", channels=4, **COMMON),
    "convlstm": lambda: make_convlstm("tiny", hidden_channels=4, **COMMON),
    "lstm": lambda: make_lstm("tiny", hidden=8, **COMMON),
}


@pytest.fixture
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("kind", sorted(MODELS))
def test_surface_only_forecasts_unchanged(synthetic_surface, one_thread, kind):
    surface = synthetic_surface.loc[:"2020-07-31"]
    out = walk_forward(MODELS[kind](), surface, TEST_START)
    np.testing.assert_array_equal(out.to_numpy(), np.load(GOLDEN)[kind])
