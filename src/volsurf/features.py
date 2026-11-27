"""Calendar and underlying features available at the close of each surface date."""

from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

_BLOOMBERG_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "bloomberg"
ECO_PATH = _BLOOMBERG_PATH / "macro" / "eco_cpi_nfp.csv"
SPX_PATH = _BLOOMBERG_PATH / "spx" / "SPX_OHLCV.csv"

_EVENT_TYPES = ("fomc", "cpi", "nfp")
_BLOCKS = ("calendar", "underlying")


def load_event_dates(path_eco: Path = ECO_PATH) -> dict[str, pd.DatetimeIndex]:
    """Load sorted, unique FOMC decision dates and CPI and NFP release dates."""
    resource = files("volsurf").joinpath("resources", "fomc_meetings.csv")
    with resource.open("rb") as source:
        fomc = pd.read_csv(source, comment="#", parse_dates=["decision_date"])
    eco = pd.read_csv(path_eco, parse_dates=["release_date"])
    dates = {
        "fomc": fomc["decision_date"],
        "cpi": eco.loc[eco["ticker"] == "CPI CHNG Index", "release_date"],
        "nfp": eco.loc[eco["ticker"] == "NFP TCH Index", "release_date"],
    }
    return {name: pd.DatetimeIndex(values.dropna().unique()).sort_values() for name, values in dates.items()}


def calendar_features(index: pd.DatetimeIndex, events: dict[str, pd.DatetimeIndex]) -> pd.DataFrame:
    """Flag events through the next trading date using ordered surface and event dates."""
    days = index.to_numpy(dtype="datetime64[D]")
    last_next = (index[-1:] + pd.offsets.BDay()).to_numpy(dtype="datetime64[D]")
    next_trading = np.concatenate((days[1:], last_next))
    columns = {}
    for name in _EVENT_TYPES:
        event_days = events[name].to_numpy(dtype="datetime64[D]")
        positions = np.searchsorted(event_days, days, side="right")
        has_later_event = positions < len(event_days)
        later_events = event_days[positions[has_later_event]]

        flags = np.zeros(len(index), dtype=int)
        flags[has_later_event] = later_events <= next_trading[has_later_event]
        distances = np.full(len(index), np.nan)
        distances[has_later_event] = np.minimum(
            np.busday_count(days[has_later_event], later_events),
            40,
        )
        columns[f"{name}_next"] = flags
        columns[f"{name}_days_to_next"] = distances
    return pd.DataFrame(columns, index=index)


def load_spx(path: Path = SPX_PATH) -> pd.DataFrame:
    """Load daily SPX prices sorted by date."""
    return pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()


def underlying_features(spx: pd.DataFrame) -> pd.DataFrame:
    """Compute percentage returns, annualized sample volatility and range from ordered SPX prices."""
    close = spx["PX_LAST"]
    ret = 100 * np.log(close / close.shift())
    annualization = np.sqrt(252)
    return pd.DataFrame(
        {
            "ret": ret,
            "ret_neg": ret.clip(upper=0),
            "rv5": annualization * ret.rolling(5).std(),
            "rv21": annualization * ret.rolling(21).std(),
            "range": 100 * np.log(spx["PX_HIGH"] / spx["PX_LOW"]),
        },
        index=spx.index,
    )


def build_exog(blocks: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the requested blocks in order and align their rows to the surface dates."""
    for block in blocks:
        if block not in _BLOCKS:
            raise ValueError(f"Unknown exogenous block {block!r}. Valid blocks are {', '.join(_BLOCKS)}.")

    panels = []
    for block in blocks:
        if block == "calendar":
            panels.append(calendar_features(index, load_event_dates()))
        else:
            panels.append(underlying_features(load_spx()).reindex(index))
    return pd.concat(panels, axis=1) if panels else pd.DataFrame(index=index)
