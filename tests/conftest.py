from pathlib import Path

import numpy as np
import pandas as pd
import pytest

TENORS = ["30DAY", "60DAY", "3MTH", "6MTH", "12MTH", "18MTH", "24MTH"]
MONEYNESS = [80.0, 90.0, 95.0, 97.5, 100.0, 102.5, 105.0, 110.0, 120.0]

REAL_SURFACE_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "surface_clean.pkl"


def _columns() -> pd.MultiIndex:
    return pd.MultiIndex.from_product([TENORS, MONEYNESS], names=["tenor", "moneyness"])


@pytest.fixture
def synthetic_surface() -> pd.DataFrame:
    """600 business days of a seeded random walk around 20, with one all-NaN row at position 400."""
    rng = np.random.default_rng(0)
    index = pd.bdate_range("2019-01-01", periods=600)
    columns = _columns()
    scale = np.repeat(np.linspace(0.5, 0.2, len(TENORS)), len(MONEYNESS))
    steps = rng.normal(size=(len(index), len(columns))) * scale
    values = 20.0 + np.cumsum(steps, axis=0)
    values[400] = np.nan  # mimics a bad Bloomberg print
    return pd.DataFrame(values, index=index, columns=columns)


@pytest.fixture
def real_surface() -> pd.DataFrame:
    if not REAL_SURFACE_PATH.exists():
        pytest.skip(f"{REAL_SURFACE_PATH} not available")
    return pd.read_pickle(REAL_SURFACE_PATH)
