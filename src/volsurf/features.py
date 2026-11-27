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


_SCHEDULE_COLUMNS = ["event", "date", "known_from", "withdrawn_from"]
_ECO_TICKERS = {"cpi": "CPI CHNG Index", "nfp": "NFP TCH Index"}


def _read_resource(name: str, **kwargs) -> pd.DataFrame:
    with files("volsurf").joinpath("resources", name).open("rb") as source:
        return pd.read_csv(source, comment="#", **kwargs)


def load_event_schedule(path_eco: Path = ECO_PATH) -> pd.DataFrame:
    """Load the point-in-time schedule of FOMC decisions and CPI and NFP releases.

    One row per (event, date), with the closes between which that date was part of the published
    schedule: the entry is known at the close of d iff ``known_from <= d < withdrawn_from``. Final
    dates (the FOMC resource and the Bloomberg ``release_date``) are always known and never withdrawn
    (``Timestamp.min`` / ``Timestamp.max``) unless ``event_schedule_revisions.csv`` lists them; that
    file also adds withdrawn original dates and releases missing from the export.
    """
    fomc = _read_resource("fomc_meetings.csv", parse_dates=["decision_date"])
    eco = pd.read_csv(path_eco, parse_dates=["release_date"])
    finals = [pd.DataFrame({"event": "fomc", "date": fomc["decision_date"]})]
    for name, ticker in _ECO_TICKERS.items():
        finals.append(pd.DataFrame({"event": name, "date": eco.loc[eco["ticker"] == ticker, "release_date"]}))
    final = pd.concat(finals, ignore_index=True).dropna().drop_duplicates()
    final["known_from"] = pd.Timestamp.min
    final["withdrawn_from"] = pd.Timestamp.max

    revisions = _read_resource(
        "event_schedule_revisions.csv", parse_dates=["date", "known_from", "withdrawn_from"]
    )[_SCHEDULE_COLUMNS]
    revisions = revisions.fillna({"known_from": pd.Timestamp.min, "withdrawn_from": pd.Timestamp.max})
    # A final date listed in the revisions takes the revision's bounds.
    keys = pd.MultiIndex.from_frame(final[["event", "date"]])
    revised = keys.isin(pd.MultiIndex.from_frame(revisions[["event", "date"]]))
    schedule = pd.concat([final[~revised], revisions], ignore_index=True)
    schedule = schedule.astype({column: "datetime64[ns]" for column in _SCHEDULE_COLUMNS[1:]})
    return schedule.sort_values(["event", "date"], ignore_index=True)


def calendar_features(index: pd.DatetimeIndex, schedule: pd.DataFrame) -> pd.DataFrame:
    """Flag events through the next trading date from the schedule known at each close.

    At the close of each date d only the entries with ``known_from <= d < withdrawn_from`` are used,
    so a later change of the schedule cannot reach earlier rows.
    """
    days = index.to_numpy(dtype="datetime64[D]")
    last_next = (index[-1:] + pd.offsets.BDay()).to_numpy(dtype="datetime64[D]")
    next_trading = np.concatenate((days[1:], last_next))
    # Compare in nanoseconds: the unbounded ends (Timestamp.min / max) do not fit a coarser unit.
    day = index.to_numpy(dtype="datetime64[ns]")[:, None]
    never = np.datetime64(pd.Timestamp.max.value, "ns")
    columns = {}
    for name in _EVENT_TYPES:
        entries = schedule[schedule["event"] == name]
        event_days, known_from, withdrawn_from = (
            entries[column].to_numpy(dtype="datetime64[ns]") for column in _SCHEDULE_COLUMNS[1:]
        )
        # Rows x entries: the entry is public at the close of the row and falls strictly after it.
        active = (known_from <= day) & (day < withdrawn_from) & (event_days > day)
        has_later_event = active.any(axis=1)
        later_events = np.where(active, event_days, never).min(axis=1, initial=never)[has_later_event]
        later_events = later_events.astype("datetime64[D]")

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
            panels.append(calendar_features(index, load_event_schedule()))
        else:
            panels.append(underlying_features(load_spx()).reindex(index))
    return pd.concat(panels, axis=1) if panels else pd.DataFrame(index=index)
