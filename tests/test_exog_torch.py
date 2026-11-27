"""Exogenous branch of the neural forecasters."""

import numpy as np
import pandas as pd
import pytest
import torch
from leakage import assert_no_leakage, leakage_dates

from volsurf.models.cnn import SurfaceCNN, make_cnn
from volsurf.models.convlstm import make_convlstm
from volsurf.models.torch_base import ExogBranch
from volsurf.protocol import walk_forward

TEST_START = pd.Timestamp("2020-06-01")


def tiny_cnn(n_exog: int = 0):
    return make_cnn(
        "tiny", seeds=1, epochs=3, patience=2, channels=4, device="cpu", deterministic=True, n_exog=n_exog
    )


@pytest.fixture
def exog(synthetic_surface) -> pd.DataFrame:
    """Three exogenous columns on the surface dates: noise, a surface change and a dummy; a few NaN."""
    rng = np.random.default_rng(3)
    panel = pd.DataFrame(
        {
            "noise": rng.normal(size=len(synthetic_surface)),
            "change": synthetic_surface.iloc[:, 0].ffill().diff().to_numpy(),
            "dummy": (np.arange(len(synthetic_surface)) % 21 == 0).astype(float),
        },
        index=synthetic_surface.index,
    )
    panel.iloc[100:103, 0] = np.nan
    return panel


class _Ones(torch.nn.Module):
    """A stand-in network whose output is one everywhere."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[0], 63)


def test_untrained_exog_branch_adds_zero():
    branch = ExogBranch(_Ones(), n_exog=4, n_points=63)
    out = branch(torch.randn(5, 2, 5, 7, 9), torch.randn(5, 4))
    assert out.shape == (5, 63)
    assert torch.all(out == 1)
    cnn = ExogBranch(SurfaceCNN(7, 9), n_exog=4, n_points=63)
    assert torch.all(cnn(torch.randn(5, 2, 5, 7, 9), torch.randn(5, 4)) == 0)  # still persistence


def test_exog_branch_is_linear_in_exog():
    branch = ExogBranch(_Ones(), n_exog=2, n_points=63)
    torch.nn.init.ones_(branch.linear.weight)
    z = torch.tensor([[1.0, 2.0]])
    assert torch.allclose(branch(torch.zeros(1, 2, 5, 7, 9), z), torch.full((1, 63), 4.0))


def test_cnn_with_exog_no_leakage(synthetic_surface, exog):
    surface = synthetic_surface.loc[:"2020-08-31"]  # three refits
    dates = leakage_dates(surface, TEST_START)
    assert_no_leakage(lambda: tiny_cnn(n_exog=3), surface, TEST_START, dates, exog=exog.loc[surface.index])


def test_exog_changes_forecasts_and_fits_on_finite_rows(synthetic_surface, exog):
    surface = synthetic_surface.loc[:"2020-07-31"]
    with_exog = walk_forward(tiny_cnn(n_exog=3), surface, TEST_START, exog=exog)
    without = walk_forward(tiny_cnn(), surface, TEST_START)
    assert with_exog.notna().sum().sum() == without.notna().sum().sum()  # NaN exog rows only drop pairs
    assert not np.allclose(with_exog.to_numpy(), without.to_numpy(), equal_nan=True)


def test_predict_uses_last_exog_row_only(synthetic_surface, exog):
    history, panel = synthetic_surface.iloc[:200], exog.iloc[:200]
    model = tiny_cnn(n_exog=3)
    model.fit(history, exog=panel)
    torch.nn.init.normal_(model.nets[0].linear.weight)  # make the branch visibly non-zero
    base = model.predict_change(history, exog=panel)
    earlier = panel.copy()
    earlier.iloc[:-1] = 1e3
    np.testing.assert_array_equal(model.predict_change(history, exog=earlier), base)
    last = panel.copy()
    last.iloc[-1] += 1.0
    assert not np.allclose(model.predict_change(history, exog=last), base)


def test_exog_standardised_on_finite_training_rows(synthetic_surface, exog):
    history, panel = synthetic_surface.iloc[:200].copy(), exog.iloc[:200].copy()
    panel.iloc[150:155, 1] = 1e6  # rows dropped because another column is NaN there
    panel.iloc[150:155, 0] = np.nan
    panel.iloc[-1] = 1e6  # the last row has no next-day target, so it is not a training row
    panel["dummy"] = 0.0  # constant column: standard deviation 0 is replaced by 1
    model = tiny_cnn(n_exog=3)
    model.fit(history, exog=panel)
    rows = panel.iloc[5:-1].dropna()  # lookback 5: windows start at row 5
    np.testing.assert_allclose(model.mu_z, rows.mean().to_numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(model.sd_z, [*rows.std(ddof=0).to_numpy()[:2], 1.0], rtol=1e-5)


def test_nan_exog_on_forecast_day_gives_nan_forecast(synthetic_surface, exog):
    history, panel = synthetic_surface.iloc[:200], exog.iloc[:200].copy()
    model = tiny_cnn(n_exog=3)
    model.fit(history, exog=panel)
    panel.iloc[-1, 2] = np.nan
    assert np.isnan(model.predict_change(history, exog=panel)).all()


def test_exog_mismatches_raise(synthetic_surface, exog):
    history, panel = synthetic_surface.iloc[:200], exog.iloc[:200]
    with pytest.raises(ValueError, match="n_exog=3.*no exog"):
        tiny_cnn(n_exog=3).fit(history)
    with pytest.raises(ValueError, match="n_exog=2.*3 columns"):
        tiny_cnn(n_exog=2).fit(history, exog=panel)
    with pytest.raises(ValueError, match="n_exog=0.*3 columns"):
        tiny_cnn().fit(history, exog=panel)
    with pytest.raises(ValueError, match="same dates"):
        tiny_cnn(n_exog=3).fit(history, exog=exog.iloc[1:201])
    model = tiny_cnn(n_exog=3)
    model.fit(history, exog=panel)
    with pytest.raises(ValueError, match="n_exog=3.*no exog"):
        model.predict_change(history)


def test_convlstm_accepts_exog(synthetic_surface, exog):
    model = make_convlstm("tiny", seeds=1, epochs=2, patience=1, hidden_channels=4, device="cpu", n_exog=3)
    model.fit(synthetic_surface.iloc[:200], exog=exog.iloc[:200])
    assert isinstance(model.nets[0], ExogBranch)
    assert model.predict_change(synthetic_surface.iloc[:200], exog=exog.iloc[:200]).shape == (63,)
