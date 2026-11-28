"""Exogenous branch of the neural forecasters."""

from pathlib import Path

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


def tiny_cnn(n_exog: int = 0, **kwargs):
    params = dict(seeds=1, epochs=3, patience=2, channels=4, device="cpu", deterministic=True)
    return make_cnn("tiny", n_exog=n_exog, **(params | kwargs))


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


@pytest.mark.parametrize("settings", [{}, dict(exog_scaling="robust", exog_clip=3.0, exog_l2=[0.0, 0.1])])
def test_cnn_with_exog_no_leakage(synthetic_surface, exog, settings):
    surface = synthetic_surface.loc[:"2020-08-31"]  # three refits
    dates = leakage_dates(surface, TEST_START)
    assert_no_leakage(
        lambda: tiny_cnn(n_exog=3, **settings), surface, TEST_START, dates, exog=exog.loc[surface.index]
    )


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


@pytest.mark.parametrize("n_exog", [0, 3])
def test_exog_defaults_match_original_trainer(synthetic_surface, exog, n_exog):
    history = synthetic_surface.iloc[:80]
    panel = exog.loc[history.index] if n_exog else None
    implicit = tiny_cnn(n_exog)
    explicit = tiny_cnn(n_exog, exog_scaling="standard", exog_clip=None, exog_l2=0.0)
    for model in (implicit, explicit):
        model.fit(history, panel)
    np.testing.assert_array_equal(
        implicit.predict_change(history, panel), explicit.predict_change(history, panel)
    )
    assert implicit.fit_log == explicit.fit_log
    assert ("exog_l2" in implicit.fit_log[0]) == bool(n_exog)
    if n_exog:
        # Recorded with the pre-regularisation trainer and one Torch thread.
        golden = Path(__file__).with_name("golden") / "default_exog_forecast.npy"
        previous = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            implicit.fit(history, panel)
            np.testing.assert_array_equal(implicit.predict_change(history, panel), np.load(golden))
        finally:
            torch.set_num_threads(previous)


def test_robust_scaling_and_fallbacks(synthetic_surface):
    history = synthetic_surface.iloc[:34]
    panel = pd.DataFrame(7.0, index=history.index, columns=["spread", "dummy", "constant"])
    panel.iloc[5:32, 0] = np.repeat([0.0, 2.0, 4.0], 9)
    panel.iloc[5:32, 1] = np.r_[1.0, np.zeros(26)]
    panel.iloc[-2:] = [1e6, np.nan, 1e6]
    panel.iloc[-1] = 1e6
    model = tiny_cnn(3, exog_scaling="robust")
    model.fit(history, panel)
    np.testing.assert_array_equal(model.mu_z, [2.0, 0.0, 7.0])
    np.testing.assert_allclose(model.sd_z, [4.0 / 1.349, np.sqrt(26) / 27, 1.0])


def test_exog_clipped_during_fit_and_prediction(synthetic_surface, exog, monkeypatch):
    history = synthetic_surface.iloc[:80]
    panel = exog.loc[history.index].copy()
    panel.iloc[15:17, 0] = [-1e6, 1e6]
    model = tiny_cnn(3, exog_scaling="robust", exog_clip=3.0)
    train = model._train

    def capture(inputs, *args):
        rows = panel.iloc[5:-1].to_numpy(dtype=np.float32)
        scaled = (rows - model.mu_z) / model.sd_z
        assert scaled[:, 0].min() < -3 < 3 < scaled[:, 0].max()
        np.testing.assert_array_equal(inputs[1].numpy(), np.clip(scaled, -3, 3))
        return train(inputs, *args)

    monkeypatch.setattr(model, "_train", capture)
    model.fit(history, panel)
    baseline = model.predict_change(history, panel)
    panel.iloc[-1] = 1e6
    clipped = model.predict_change(history, panel)
    panel.iloc[-1] = model.mu_z + 4 * model.sd_z
    np.testing.assert_array_equal(clipped, model.predict_change(history, panel))
    assert not np.array_equal(baseline, clipped)
    panel.iloc[-1, 0] = np.inf
    assert np.isnan(model.predict_change(history, panel)).all()


@pytest.mark.parametrize("constant", [False, True])
def test_ridge_grid_selects_and_reuses_seed_zero(synthetic_surface, exog, monkeypatch, constant):
    history = synthetic_surface.iloc[:80]
    panel = exog.loc[history.index].copy()
    if constant:
        panel.iloc[:] = 1.0  # all scaled inputs are zero, so every penalty ties
    grid = [0.1, 1.0, 0.0]
    model = tiny_cnn(3, exog_l2=grid, seeds=2)
    train = model._train
    calls = []

    def capture(inputs, targets, seed, n_tenors, n_moneyness, penalty):
        net, epoch, loss = train(inputs, targets, seed, n_tenors, n_moneyness, penalty)
        n_val = model._n_val(len(targets))
        with torch.no_grad():
            plain_loss = torch.nn.functional.huber_loss(
                net(*(t[-n_val:] for t in inputs)), targets[-n_val:], delta=model.huber_delta
            ).item()
        assert loss == plain_loss
        calls.append((seed, penalty, net, loss))
        return net, epoch, loss

    monkeypatch.setattr(model, "_train", capture)
    model.fit(history, panel)
    log = model.fit_log[0]
    losses = log["exog_val_loss"]
    assert losses == [call[3] for call in calls[: len(grid)]]
    if constant:
        assert losses == [losses[0]] * len(grid)
        assert log["exog_l2"] == max(grid)
    best = min(range(len(grid)), key=lambda i: (losses[i], -grid[i]))
    assert log["exog_l2"] == grid[best]
    assert [(seed, penalty) for seed, penalty, _, _ in calls] == [
        *((0, penalty) for penalty in grid),
        (1, grid[best]),
    ]
    assert model.nets[0] is calls[best][2]
    assert log["exog_weight_norm"] == [net.linear.weight.norm().item() for net in model.nets]


def test_large_ridge_shrinks_branch_weights(synthetic_surface, exog):
    history = synthetic_surface.iloc[:100]
    panel = exog.loc[history.index]
    plain, ridge = [tiny_cnn(3, epochs=25, patience=25, lr=1e-4, exog_l2=value) for value in (0.0, 1e6)]
    for model in (plain, ridge):
        model.fit(history, panel)
    plain_norm = plain.fit_log[0]["exog_weight_norm"][0]
    ridge_norm = ridge.fit_log[0]["exog_weight_norm"][0]
    assert ridge_norm < 0.01
    assert ridge_norm < plain_norm / 5
    torch.testing.assert_close(plain.nets[0].linear.bias, ridge.nets[0].linear.bias, atol=0.01, rtol=0)


@pytest.mark.parametrize(
    ("n_exog", "settings", "match"),
    [
        (3, {"exog_scaling": "unknown"}, "exog_scaling"),
        (3, {"exog_clip": 0.0}, "exog_clip"),
        (3, {"exog_clip": -1.0}, "exog_clip"),
        (3, {"exog_clip": np.nan}, "exog_clip"),
        (3, {"exog_l2": -1.0}, "exog_l2"),
        (3, {"exog_l2": np.inf}, "exog_l2"),
        (3, {"exog_l2": [0.0, np.nan]}, "exog_l2"),
        (3, {"exog_l2": []}, "exog_l2"),
        (0, {"exog_scaling": "robust"}, "n_exog"),
        (0, {"exog_clip": 3.0}, "n_exog"),
        (0, {"exog_l2": 0.1}, "n_exog"),
        (0, {"exog_l2": [0.0]}, "n_exog"),
    ],
)
def test_invalid_exog_settings(n_exog, settings, match):
    with pytest.raises(ValueError, match=match):
        tiny_cnn(n_exog, **settings)
