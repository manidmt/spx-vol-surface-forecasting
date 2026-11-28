"""Factor model (PCA factors of the changes, HAR terms, ridge)."""

import numpy as np
import pandas as pd
import pytest
from leakage import assert_no_leakage, leakage_dates

from volsurf.models import build_model
from volsurf.models.factor import FactorHAR
from volsurf.protocol import walk_forward

TEST_START = pd.Timestamp("2020-06-01")


def _one_factor_surface(n=600, phi=0.5, seed=0):
    """Changes driven by one AR(1) factor with fixed loadings, plus small idiosyncratic noise."""
    rng = np.random.default_rng(seed)
    cols = pd.MultiIndex.from_product(
        [["30DAY", "60DAY", "3MTH"], [90.0, 100.0, 110.0]], names=["tenor", "moneyness"]
    )
    loadings = np.linspace(1.0, 0.3, len(cols))
    f = np.zeros(n)
    for t in range(1, n):
        f[t] = phi * f[t - 1] + rng.normal()
    changes = f[:, None] * loadings + 0.1 * rng.normal(size=(n, len(cols)))
    levels = 20 + np.cumsum(changes, axis=0)
    return pd.DataFrame(levels, index=pd.bdate_range("2019-01-01", periods=n), columns=cols)


def test_recovers_an_ar1_factor_and_beats_persistence():
    surface = _one_factor_surface()
    preds = walk_forward(FactorHAR(n_factors=2), surface, TEST_START)
    actual = surface.loc[preds.index]
    previous = surface.shift(1).loc[preds.index]
    mae_model = (actual - preds).abs().to_numpy().mean()
    mae_persistence = (actual - previous).abs().to_numpy().mean()
    assert mae_model < 0.95 * mae_persistence


def test_shapes_and_fit_log(synthetic_surface):
    model = FactorHAR()
    out = walk_forward(model, synthetic_surface.loc[:"2020-08-31"], TEST_START)
    assert out.shape[1] == 63
    assert {"refit", "alpha", "val_mae", "n_rows"} <= model.fit_log[0].keys()
    assert model.fit_log[0]["alpha"] in model.alphas
    assert model.loadings.shape == (63, 5)


def test_factor_no_leakage(synthetic_surface):
    surface = synthetic_surface.loc[:"2020-08-31"]
    assert_no_leakage(FactorHAR, surface, TEST_START, leakage_dates(surface, TEST_START))


def test_factor_with_exog_no_leakage(synthetic_surface):
    surface = synthetic_surface.loc[:"2020-08-31"]
    rng = np.random.default_rng(1)
    exog = pd.DataFrame(rng.normal(size=(len(surface), 3)), index=surface.index, columns=["a", "b", "c"])
    assert_no_leakage(
        lambda: FactorHAR(n_exog=3), surface, TEST_START, leakage_dates(surface, TEST_START), exog=exog
    )


def test_exog_must_match():
    surface = _one_factor_surface(n=200)
    with pytest.raises(ValueError, match="exog"):
        FactorHAR(n_exog=2).fit(surface)


def test_registry_builds_factor_har():
    model = build_model("factor_har", "my_factor", {"n_factors": 3, "n_exog": 4})
    assert isinstance(model, FactorHAR)
    assert model.name == "my_factor"
    assert model.n_factors == 3
