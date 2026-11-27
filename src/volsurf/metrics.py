"""Forecast evaluation: error frames, summary tables and the Diebold-Mariano test."""

from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm

Loss = Literal["squared", "absolute"]


def common_mask(actual: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> pd.Index:
    """Dates present in `actual` and every prediction frame, with all values finite in each."""
    dates: pd.Index | None = None
    for frame in [actual, *preds.values()]:
        finite = frame.index[np.isfinite(frame.to_numpy(dtype=float)).all(axis=1)]
        dates = finite if dates is None else dates.intersection(finite)
    assert dates is not None
    return dates.sort_values()


def errors(actual: pd.DataFrame, preds: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """`actual - prediction` for each model, restricted to the common mask."""
    mask = common_mask(actual, preds)
    return {name: (actual.loc[mask] - p.loc[mask]) for name, p in preds.items()}


def _r2_oos(e: pd.DataFrame, e_ref: pd.DataFrame) -> float:
    return 1 - float((e**2).to_numpy().sum() / (e_ref**2).to_numpy().sum())


def summary(errs: dict[str, pd.DataFrame], reference: str = "persistence") -> pd.DataFrame:
    """MAE, RMSE, MAE relative to the reference and out-of-sample R2 against it."""
    e_ref = errs[reference]
    mae_ref = e_ref.abs().to_numpy().mean()
    rows = {}
    for name, e in errs.items():
        mae = e.abs().to_numpy().mean()
        rows[name] = {
            "mae": mae,
            "rmse": float(np.sqrt((e**2).to_numpy().mean())),
            "mae_vs_ref": mae / mae_ref - 1,
            "r2_oos_vs_ref": _r2_oos(e, e_ref),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def mae_by_tenor(errs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """MAE per tenor (models x tenors), keeping the tenor order of the columns."""
    rows = {name: e.abs().T.groupby(level="tenor", sort=False).mean().T.mean() for name, e in errs.items()}
    return pd.DataFrame.from_dict(rows, orient="index")


def mae_by_point(errs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """MAE per surface point (points x models), in grid order."""
    return pd.DataFrame({name: e.abs().mean() for name, e in errs.items()})


def diebold_mariano(
    e_ref: pd.DataFrame, e: pd.DataFrame, loss: Loss, maxlags: int = 5
) -> tuple[float, float]:
    """Diebold-Mariano test on the daily mean loss differential `loss(e_ref) - loss(e)`.

    Regresses the differential on a constant with HAC standard errors. Returns (t, p);
    a positive t means `e` has the lower loss.
    """
    if loss == "squared":
        fn = np.square
    elif loss == "absolute":
        fn = np.abs
    else:
        raise ValueError(f"unknown loss: {loss!r}")
    dl = pd.DataFrame(fn(e_ref)).mean(axis=1) - pd.DataFrame(fn(e)).mean(axis=1)
    res = sm.OLS(dl.to_numpy(), np.ones(len(dl))).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    return float(res.tvalues[0]), float(res.pvalues[0])


def dm_table(errs: dict[str, pd.DataFrame], references: list[str], models: list[str]) -> pd.DataFrame:
    """Diebold-Mariano results for each (reference, model, loss) combination."""
    rows = {}
    for ref in references:
        for model in models:
            for loss in ("absolute", "squared"):
                t, p = diebold_mariano(errs[ref], errs[model], loss)
                rows[(ref, model, loss)] = {"t_stat": t, "p_value": p}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index = pd.MultiIndex.from_tuples(out.index, names=["reference", "model", "loss"])
    return out
