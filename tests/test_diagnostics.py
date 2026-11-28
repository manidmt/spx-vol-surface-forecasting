import numpy as np
import pandas as pd
import pytest
from scipy import stats

from volsurf.diagnostics import (
    by_state,
    by_year,
    clark_west,
    days_for_power,
    giacomini_white,
    newey_west_variance,
    state_instrument,
)
from volsurf.metrics import diebold_mariano


def _errors(n=600, seed=0, gain=0.0):
    """Reference and model errors on a 2x2 grid; the model's errors shrink by ``gain``."""
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-04", periods=n)
    cols = pd.MultiIndex.from_product([["30DAY", "60DAY"], [90.0, 100.0]], names=["tenor", "moneyness"])
    e_ref = pd.DataFrame(rng.normal(size=(n, 4)), index=index, columns=cols)
    noise = pd.DataFrame(rng.normal(scale=0.3, size=(n, 4)), index=index, columns=cols)
    e = e_ref * (1 - gain) + noise
    return e_ref, e


def test_newey_west_variance_matches_hand_case():
    x = np.array([1.0, -1.0, 2.0, 0.0, -2.0])
    u = x - x.mean()
    g0 = (u @ u) / 5
    g1 = (u[1:] @ u[:-1]) / 5
    assert newey_west_variance(x, maxlags=1) == pytest.approx(g0 + 2 * 0.5 * g1)
    assert newey_west_variance(x, maxlags=0) == pytest.approx(g0)


def test_clark_west_adjusted_differential_and_one_sided_p():
    e_ref, e = _errors(gain=0.2)
    t, p = clark_west(e_ref, e, maxlags=5)
    f = (e_ref**2 - e**2 + (e - e_ref) ** 2).mean(axis=1).to_numpy()
    se = np.sqrt(newey_west_variance(f, 5) / len(f))
    assert t == pytest.approx(f.mean() / se)
    assert p == pytest.approx(stats.norm.sf(t))
    assert t > 2  # a model with smaller errors is detected


def test_clark_west_adjustment_is_never_below_the_mse_differential():
    e_ref, e = _errors(gain=-0.1, seed=3)  # the larger model is worse
    f = (e_ref**2 - e**2 + (e - e_ref) ** 2).mean(axis=1)
    plain = (e_ref**2 - e**2).mean(axis=1)
    assert (f >= plain - 1e-12).all()


def test_giacomini_white_with_a_constant_equals_dm_squared():
    e_ref, e = _errors(gain=0.05, seed=1)
    instruments = pd.DataFrame({"const": 1.0}, index=e.index)
    w, p, n = giacomini_white(e_ref, e, instruments, loss="absolute", maxlags=5)
    t, _ = diebold_mariano(e_ref, e, "absolute", maxlags=5)
    assert w == pytest.approx(t**2, rel=1e-2)  # statsmodels adds a small-sample factor n / (n - 1)
    assert p == pytest.approx(stats.chi2.sf(w, 1))
    assert n == len(e)


def test_giacomini_white_drops_days_without_instruments():
    e_ref, e = _errors(seed=2)
    instruments = pd.DataFrame({"const": 1.0, "lag": np.arange(len(e), dtype=float)}, index=e.index)
    instruments.iloc[:10, 1] = np.nan
    _, _, n = giacomini_white(e_ref, e, instruments, loss="squared")
    assert n == len(e) - 10


def test_days_for_power_formula():
    e_ref, e = _errors(gain=0.1, seed=4)
    d = (e_ref.abs() - e.abs()).mean(axis=1).to_numpy()
    sigma = np.sqrt(newey_west_variance(d, 5))
    mean_ref = e_ref.abs().to_numpy().mean()
    z = stats.norm.ppf(0.975) + stats.norm.ppf(0.8)
    assert days_for_power(e_ref, e, effect=0.01) == pytest.approx((z * sigma / (0.01 * mean_ref)) ** 2)


def test_state_instrument_uses_only_past_changes():
    index = pd.bdate_range("2021-01-04", periods=40)
    cols = pd.MultiIndex.from_product([["30DAY"], [100.0]], names=["tenor", "moneyness"])
    actual = pd.DataFrame(np.arange(40.0) ** 1.5, index=index, columns=cols)
    state = state_instrument(actual, window=5)
    changes = actual[("30DAY", 100.0)].diff().abs()
    # the value for day t is the mean absolute change of the 5 days ending at t-1
    assert state.iloc[10] == pytest.approx(changes.iloc[5:10].mean())
    altered = actual.copy()
    altered.iloc[10:] = 0.0
    pd.testing.assert_series_equal(state.iloc[:11], state_instrument(altered, window=5).iloc[:11])


def test_by_year_and_by_state_tables():
    e_ref, e = _errors(n=520, gain=0.1, seed=5)
    years = by_year(e_ref, e)
    assert list(years.index) == sorted(set(e.index.year))
    assert {"n_days", "mae_vs_ref", "dm_t_abs", "dm_p_abs", "cw_t", "cw_p"} <= set(years.columns)
    assert (years["mae_vs_ref"] < 0).all()
    state = pd.Series(np.arange(len(e), dtype=float), index=e.index)
    terciles = by_state(e_ref, e, state)
    assert list(terciles.index) == ["low", "mid", "high"]
    assert terciles["n_days"].sum() == len(e)


def test_diagnose_cli_on_synthetic_runs(tmp_path, monkeypatch, synthetic_surface):
    from volsurf import diagnose
    from volsurf.io import save_run

    monkeypatch.setattr(diagnose, "load_surface", lambda include_holdout=False: synthetic_surface)
    actual = synthetic_surface.loc["2020-01-01":]
    persistence = synthetic_surface.shift(1).loc[actual.index]
    better = 0.8 * persistence + 0.2 * actual
    dirs = []
    for name, frame in {"persistence": persistence, "better": better}.items():
        save_run(tmp_path / name, frame, {"source": f'name = "{name}"\n'}, {})
        dirs.append(str(tmp_path / name))
    tables = diagnose.main([*dirs, "--reference", "persistence", "--out", str(tmp_path / "out")])
    assert tables["overall"].loc["better", "mae_vs_ref"] < 0
    assert tables["overall"].loc["better", "cw_p"] < 0.05
    assert (tmp_path / "out" / "by_year.csv").exists()
