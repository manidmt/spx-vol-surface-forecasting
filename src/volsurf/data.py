"""Loading of the cleaned volatility surface."""

from pathlib import Path

import pandas as pd

SURFACE_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "surface_clean.pkl"
HOLDOUT_START = pd.Timestamp("2026-10-08")


def drop_holdout(df: pd.DataFrame, include_holdout: bool = False) -> pd.DataFrame:
    """Drop rows dated on or after ``HOLDOUT_START`` unless ``include_holdout`` is True."""
    if include_holdout:
        return df
    return df.loc[df.index < HOLDOUT_START]


def load_surface(path: Path = SURFACE_PATH, include_holdout: bool = False) -> pd.DataFrame:
    """Load the cleaned surface (bad prints are NaN); the hold-out is dropped by default."""
    return drop_holdout(pd.read_pickle(path), include_holdout)


def grid_shape(surface: pd.DataFrame) -> tuple[int, int]:
    """Return (n_tenors, n_moneyness) from the unique column levels."""
    return (
        surface.columns.get_level_values("tenor").nunique(),
        surface.columns.get_level_values("moneyness").nunique(),
    )
