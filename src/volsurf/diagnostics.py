"""Forecast diagnostics beyond Diebold-Mariano: nested models, state dependence and power.

Every function works on error frames (``actual - forecast``, dates x grid points) already
restricted to a common mask, as returned by ``volsurf.metrics.errors``. Daily losses are the mean
over the grid points, as in ``volsurf.metrics.diebold_mariano``.

- ``clark_west``: Clark and West (2007) test for nested models under squared loss. The larger
  model estimates parameters that are zero under the null, which inflates its squared error; the
  test adds back the squared forecast difference. One-sided: H1 says the larger model is better.
- ``giacomini_white``: Giacomini and White (2006) conditional predictive ability for one-step
  forecasts: does the loss differential depend on information known before the forecast?
- ``days_for_power``: the number of test days needed to detect a given relative MAE gain.
"""

from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

Loss = Literal["squared", "absolute"]
ATM_30D = ("30DAY", 100.0)


def newey_west_variance(x: np.ndarray, maxlags: int = 5) -> float:
    """Long-run variance of ``x`` (Bartlett kernel, ``maxlags`` lags, no small-sample factor)."""
    u = np.asarray(x, dtype=float) - np.mean(x)
    n = len(u)
    s = u @ u / n
    for lag in range(1, maxlags + 1):
        s += 2 * (1 - lag / (maxlags + 1)) * (u[lag:] @ u[:-lag]) / n
    return float(s)


def _newey_west_matrix(z: np.ndarray, maxlags: int) -> np.ndarray:
    """Long-run covariance matrix of the rows of ``z`` (n x q), Bartlett kernel, centred."""
    u = z - z.mean(axis=0)
    n = len(u)
    s = u.T @ u / n
    for lag in range(1, maxlags + 1):
        g = u[lag:].T @ u[:-lag] / n
        s += (1 - lag / (maxlags + 1)) * (g + g.T)
    return s


def _daily(e: pd.DataFrame, loss: Loss) -> pd.Series:
    if loss == "squared":
        return (e**2).mean(axis=1)
    if loss == "absolute":
        return e.abs().mean(axis=1)
    raise ValueError(f"unknown loss: {loss!r}")


def loss_differential(e_ref: pd.DataFrame, e: pd.DataFrame, loss: Loss) -> pd.Series:
    """Daily ``loss(e_ref) - loss(e)``: positive when the model beats the reference that day."""
    return _daily(e_ref, loss) - _daily(e, loss)


def clark_west(e_ref: pd.DataFrame, e: pd.DataFrame, maxlags: int = 5) -> tuple[float, float]:
    """Clark-West test, ``e_ref`` from the nested (smaller) model. Returns (t, one-sided p).

    Adjusted differential per day: mean over points of ``e_ref² - e² + (e - e_ref)²``, where
    ``e - e_ref`` is the difference between the two forecasts. A positive t favours ``e``.
    """
    f = (e_ref**2 - e**2 + (e - e_ref) ** 2).mean(axis=1).to_numpy()
    se = np.sqrt(newey_west_variance(f, maxlags) / len(f))
    if not se > 0:
        return float("nan"), float("nan")
    t = float(f.mean() / se)
    return t, float(stats.norm.sf(t))


def giacomini_white(
    e_ref: pd.DataFrame,
    e: pd.DataFrame,
    instruments: pd.DataFrame,
    loss: Loss = "absolute",
    maxlags: int = 5,
) -> tuple[float, float, int]:
    """Giacomini-White test of conditional predictive ability. Returns (Wald, p, days used).

    ``instruments`` is indexed by forecast day: row ``t`` must hold only information known
    before the forecast of ``t`` (the caller lags it). With ``d_t`` the loss differential,
    ``Z_t = h_t d_t`` and the statistic is ``n Z̄ᵀ S⁻¹ Z̄`` with a Newey-West ``S``, χ²(q) under the
    null of equal conditional predictive ability. Days with a missing instrument are dropped.
    A constant alone reproduces the squared Diebold-Mariano statistic.
    """
    d = loss_differential(e_ref, e, loss)
    h = instruments.reindex(d.index)
    ok = h.notna().all(axis=1) & d.notna()
    z = h[ok].to_numpy(dtype=float) * d[ok].to_numpy()[:, None]
    n, q = z.shape
    zbar = z.mean(axis=0)
    s = _newey_west_matrix(z, maxlags)
    w = float(n * zbar @ np.linalg.solve(s, zbar))
    return w, float(stats.chi2.sf(w, q)), n


def days_for_power(
    e_ref: pd.DataFrame,
    e: pd.DataFrame,
    effect: float,
    loss: Loss = "absolute",
    power: float = 0.8,
    alpha: float = 0.05,
    maxlags: int = 5,
) -> float:
    """Days needed for a two-sided DM test at ``alpha`` to detect a relative gain of ``effect``.

    Uses the long-run standard deviation of the observed differential and the mean loss of the
    reference: ``n = ((z_{1-α/2} + z_power) σ / (effect · mean loss_ref))²``.
    """
    d = loss_differential(e_ref, e, loss).to_numpy()
    sigma = np.sqrt(newey_west_variance(d, maxlags))
    target = effect * _daily(e_ref, loss).mean()
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float((z * sigma / target) ** 2)


def state_instrument(actual: pd.DataFrame, window: int = 21, point: tuple = ATM_30D) -> pd.Series:
    """Trailing mean absolute daily change of ``point`` over ``window`` days, known before each day.

    The value for day ``t`` uses the changes up to ``t-1``, so it can condition a forecast of
    ``t`` (an ex-ante state, not the realised move of ``t``).
    """
    change = actual[point].diff().abs()
    return change.rolling(window, min_periods=window).mean().shift(1)


def default_instruments(
    e_ref: pd.DataFrame, e: pd.DataFrame, actual: pd.DataFrame, loss: Loss
) -> pd.DataFrame:
    """Constant, the previous day's loss differential and the ex-ante vol state."""
    d = loss_differential(e_ref, e, loss)
    state = state_instrument(actual).reindex(d.index)
    return pd.DataFrame({"const": 1.0, "lagged_differential": d.shift(1), "state": state}, index=d.index)


def relative_mae(e_ref: pd.DataFrame, e: pd.DataFrame) -> float:
    """MAE of ``e`` relative to that of ``e_ref``, minus one."""
    return float(e.abs().to_numpy().mean() / e_ref.abs().to_numpy().mean() - 1)


def dm_absolute(e_ref: pd.DataFrame, e: pd.DataFrame, maxlags: int = 5) -> tuple[float, float]:
    """Diebold-Mariano on absolute loss with the Newey-West variance above (two-sided p)."""
    d = loss_differential(e_ref, e, "absolute").to_numpy()
    se = np.sqrt(newey_west_variance(d, maxlags) / len(d))
    if not se > 0:
        return float("nan"), float("nan")
    t = float(d.mean() / se)
    return t, float(2 * stats.norm.sf(abs(t)))


def _subset_row(e_ref: pd.DataFrame, e: pd.DataFrame, maxlags: int) -> dict:
    dm_t, dm_p = dm_absolute(e_ref, e, maxlags)
    cw_t, cw_p = clark_west(e_ref, e, maxlags)
    return {
        "n_days": len(e),
        "mae_vs_ref": relative_mae(e_ref, e),
        "dm_t_abs": dm_t,
        "dm_p_abs": dm_p,
        "cw_t": cw_t,
        "cw_p": cw_p,
    }


def by_year(e_ref: pd.DataFrame, e: pd.DataFrame, maxlags: int = 5) -> pd.DataFrame:
    """Relative MAE, DM (absolute loss, two-sided) and Clark-West for each calendar year."""
    rows = {
        year: _subset_row(e_ref.loc[days], e.loc[days], maxlags)
        for year, days in e.index.to_series().groupby(e.index.year).groups.items()
    }
    return pd.DataFrame.from_dict(rows, orient="index")


def by_state(e_ref: pd.DataFrame, e: pd.DataFrame, state: pd.Series, maxlags: int = 5) -> pd.DataFrame:
    """The same table by tercile of an ex-ante ``state`` (days without a state are left out)."""
    state = state.reindex(e.index).dropna()
    groups = pd.qcut(state, 3, labels=["low", "mid", "high"])
    rows = {}
    for label in ["low", "mid", "high"]:
        days = groups.index[groups == label]
        rows[label] = _subset_row(e_ref.loc[days], e.loc[days], maxlags)
    return pd.DataFrame.from_dict(rows, orient="index")
