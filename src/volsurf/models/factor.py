"""Factor model: forecast a few PCA factors of the daily changes with HAR terms and ridge.

Instead of 63 correlated grid points, the model forecasts ``n_factors`` principal components of
the scaled daily changes (Cont and da Fonseca, 2002, on the dynamics of implied-volatility
surfaces) and maps them back to the grid. Grid directions outside the factors are forecast as
no change (persistence).

At each refit, on the training rows:

- every grid point's change is scaled by its standard deviation, and the loadings are the first
  ``n_factors`` right singular vectors of the scaled changes (no centring: daily changes have a
  mean close to zero and the intercept of the regression absorbs any drift);
- the regressors of day ``t`` are the factor scores of the change of ``t``, their means over the
  last 5 and 22 days (HAR terms), and the scores of the level minus its 22-day mean; with
  exogenous inputs, also the exogenous row of ``t`` (robust-scaled and clipped);
- the targets are the factor scores of the change of ``t+1``; one ridge regression with an
  intercept for all factors, its penalty chosen on the last ``val_frac`` of the rows by the MAE of
  the reconstructed grid change, then refitted on all rows.

Inputs use the forward-filled surface (past only); targets use the raw changes, so a NaN print
drops the pairs it touches, as in the networks.
"""

import numpy as np
import pandas as pd

from volsurf.models.base import Forecaster

HAR_WINDOWS = (5, 22)
LEVEL_WINDOW = 22


def _trailing_mean(x: np.ndarray, window: int) -> np.ndarray:
    """Mean of the last ``window`` rows ending at each row; NaN if any of them is NaN."""
    return pd.DataFrame(x).rolling(window, min_periods=window).mean().to_numpy()


class FactorHAR(Forecaster):
    """Ridge forecasts of PCA factors of the daily changes with HAR terms; see the module docstring."""

    def __init__(
        self,
        n_factors: int = 5,
        alphas: tuple[float, ...] | list[float] = (0.1, 1.0, 10.0, 100.0, 1000.0),
        val_frac: float = 0.15,
        n_exog: int = 0,
        exog_clip: float = 3.0,
    ) -> None:
        super().__init__()
        if n_factors < 1:
            raise ValueError("n_factors must be at least 1")
        if not alphas or any(not a > 0 for a in alphas):
            raise ValueError("alphas must be positive")
        self.name = "factor_har"
        self.n_factors = n_factors
        self.alphas = list(alphas)
        self.val_frac = val_frac
        self.n_exog = n_exog
        self.exog_clip = exog_clip
        self.sd: np.ndarray | None = None
        self.loadings: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.intercept: np.ndarray | None = None
        self.mu_x: np.ndarray | None = None
        self.sd_x: np.ndarray | None = None
        self.mu_z: np.ndarray | None = None
        self.sd_z: np.ndarray | None = None

    # ----- features -----------------------------------------------------------------------------
    def _regressors(self, levels: np.ndarray, Z: np.ndarray | None) -> np.ndarray:
        """Regressor rows for every day of ``levels`` (forward-filled), NaN where undefined."""
        change = np.full_like(levels, np.nan)
        change[1:] = levels[1:] - levels[:-1]
        scores = (change / self.sd) @ self.loadings
        parts = [scores, *(_trailing_mean(scores, w) for w in HAR_WINDOWS)]
        deviation = levels - _trailing_mean(levels, LEVEL_WINDOW)
        parts.append((deviation / self.sd) @ self.loadings)
        if self.n_exog:
            parts.append(np.clip((Z - self.mu_z) / self.sd_z, -self.exog_clip, self.exog_clip))
        return np.concatenate(parts, axis=1)

    def _check_exog(self, history: pd.DataFrame, exog: pd.DataFrame | None) -> np.ndarray | None:
        if not self.n_exog:
            return None
        if exog is None or exog.shape[1] != self.n_exog or not exog.index.equals(history.index):
            raise ValueError(f"{self.name}: exog must have {self.n_exog} columns and the history's dates")
        return exog.to_numpy(dtype=float)

    # ----- fitting ------------------------------------------------------------------------------
    @staticmethod
    def _ridge(X: np.ndarray, Y: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """Ridge with an unpenalised intercept on standardised ``X``: (coefficients, intercept)."""
        x_mean, y_mean = X.mean(axis=0), Y.mean(axis=0)
        Xc, Yc = X - x_mean, Y - y_mean
        coef = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ Yc)
        return coef, y_mean - x_mean @ coef

    def _reconstruct(self, scores: np.ndarray) -> np.ndarray:
        return (scores @ self.loadings.T) * self.sd

    def fit(self, history: pd.DataFrame, exog: pd.DataFrame | None = None) -> None:
        """Estimate loadings, scaling and ridge coefficients from ``history`` (see module docstring)."""
        Z = self._check_exog(history, exog)
        raw_change = history.diff().to_numpy(dtype=float)
        finite = np.isfinite(raw_change).all(axis=1)
        self.sd = raw_change[finite].std(axis=0)
        if not (self.sd > 0).all():
            raise ValueError(f"{self.name}: a grid point never moves in the training data")
        _, _, vt = np.linalg.svd(raw_change[finite] / self.sd, full_matrices=False)
        self.loadings = vt[: self.n_factors].T  # (n_points, n_factors)
        if self.n_exog:
            ok_z = np.isfinite(Z).all(axis=1)
            self.mu_z = np.median(Z[ok_z], axis=0)
            q25, q75 = np.percentile(Z[ok_z], [25, 75], axis=0)
            scale = (q75 - q25) / 1.349
            fallback = Z[ok_z].std(axis=0)
            self.sd_z = np.where(scale > 0, scale, np.where(fallback > 0, fallback, 1.0))

        levels = history.ffill().to_numpy(dtype=float)
        X = self._regressors(levels, Z)[:-1]
        target_change = raw_change[1:]
        Y = (target_change / self.sd) @ self.loadings
        ok = np.isfinite(X).all(axis=1) & np.isfinite(target_change).all(axis=1)
        X, Y, target_change = X[ok], Y[ok], target_change[ok]
        n_val = max(int(self.val_frac * len(X)), 20)
        if len(X) - n_val < X.shape[1] + 1:
            raise ValueError(f"{self.name}: too few usable rows ({len(X)}) for {X.shape[1]} regressors")

        self.mu_x, self.sd_x = X[:-n_val].mean(axis=0), X[:-n_val].std(axis=0)
        self.sd_x[self.sd_x == 0] = 1.0
        Xs = (X - self.mu_x) / self.sd_x
        best, chosen = (np.inf, 0.0), self.alphas[-1]
        val_mae = []
        for alpha in self.alphas:
            coef, intercept = self._ridge(Xs[:-n_val], Y[:-n_val], alpha)
            pred = self._reconstruct(Xs[-n_val:] @ coef + intercept)
            mae = float(np.abs(target_change[-n_val:] - pred).mean())
            val_mae.append(mae)
            if (mae, -alpha) < best:
                best, chosen = (mae, -alpha), alpha
        self.mu_x, self.sd_x = X.mean(axis=0), X.std(axis=0)
        self.sd_x[self.sd_x == 0] = 1.0
        self.coef, self.intercept = self._ridge((X - self.mu_x) / self.sd_x, Y, chosen)
        self.fit_log.append(
            {"refit": history.index[-1], "alpha": chosen, "val_mae": val_mae, "n_rows": len(X)}
        )

    def predict_change(self, history: pd.DataFrame, exog: pd.DataFrame | None = None) -> np.ndarray:
        """Reconstructed grid change from the forecast factor scores of the next day."""
        if self.coef is None:
            raise RuntimeError("fit must be called before predict_change")
        Z = self._check_exog(history, exog)
        rows = max(*HAR_WINDOWS, LEVEL_WINDOW) + 1
        levels = history.ffill().to_numpy(dtype=float)[-rows:]
        x = self._regressors(levels, None if Z is None else Z[-rows:])[-1]
        scores = ((x - self.mu_x) / self.sd_x) @ self.coef + self.intercept
        return self._reconstruct(scores[None, :])[0]
