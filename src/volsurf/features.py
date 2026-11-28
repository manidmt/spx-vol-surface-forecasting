"""Exogenous features aligned to surface dates for next-day forecasts."""

from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd

_BLOOMBERG_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "bloomberg"
ECO_PATH = _BLOOMBERG_PATH / "macro" / "eco_cpi_nfp.csv"
SPX_PATH = _BLOOMBERG_PATH / "spx" / "SPX_OHLCV.csv"

_EVENT_TYPES = ("fomc", "cpi", "nfp")
MARKET_PATHS = (
    _BLOOMBERG_PATH / "series" / "series_px_last_long.csv",
    _BLOOMBERG_PATH / "fx" / "fx_smile_long.csv",
    _BLOOMBERG_PATH / "ivol_etf" / "ivol30d_atm_long.csv",
)
_IVOL_PREFIX = "ivol:"
_SECTOR_TICKERS = tuple(
    f"{symbol} US Equity" for symbol in "XLB XLC XLE XLF XLI XLK XLP XLRE XLU XLV XLY".split()
)


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


def session_features(spx: pd.DataFrame) -> pd.DataFrame:
    """Split the SPX close-to-close return into its overnight and intraday parts, in percent."""
    close, open_ = spx["PX_LAST"], spx["PX_OPEN"]
    return pd.DataFrame(
        {
            "ret_overnight": 100 * (open_ / close.shift() - 1),
            "ret_intraday": 100 * (close / open_ - 1),
        },
        index=spx.index,
    )


def _trading_calendar(index: pd.DatetimeIndex, horizon_days: int = 120) -> pd.DatetimeIndex:
    """The index dates, extended past its end with business days (``BDay``)."""
    tail = pd.bdate_range(index[-1] + pd.offsets.BDay(), periods=horizon_days)
    return pd.DatetimeIndex(index.append(tail)).normalize()


def _on_or_before(day: pd.Timestamp, calendar: pd.DatetimeIndex) -> pd.Timestamp:
    """``day`` if it is a trading day of ``calendar``, else the trading day before it.

    Days after the end of ``calendar`` are taken as business days (``BDay``).
    """
    if day > calendar[-1]:
        return day if day.weekday() < 5 else day - pd.offsets.BDay()
    return calendar[calendar.searchsorted(day, side="right") - 1]


def _third_friday(year: int, month: int) -> pd.Timestamp:
    first = pd.Timestamp(year=year, month=month, day=1)
    return first + pd.Timedelta(days=(4 - first.weekday()) % 7 + 14)


def option_expiries(calendar: pd.DatetimeIndex) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Monthly SPX expiries and VIX settlement days within the span of ``calendar``.

    Monthly expiry: the third Friday, or the trading day before it if that Friday is not a
    trading day. VIX settlement: 30 calendar days before the next month's third Friday (before
    the trading day preceding that Friday if it is a holiday), moved to the preceding trading
    day if it is not one itself.
    """
    months = pd.period_range(calendar[0], calendar[-1], freq="M")
    monthly, vix = [], []
    for month in months:
        friday = _third_friday(month.year, month.month)
        monthly.append(_on_or_before(friday, calendar))
        nxt = month + 1
        next_friday = _third_friday(nxt.year, nxt.month)
        if next_friday in calendar or next_friday > calendar[-1]:
            base = next_friday
        else:  # an exchange holiday: 30 days before the trading day preceding that Friday
            base = _on_or_before(next_friday - pd.Timedelta(days=1), calendar)
        vix.append(_on_or_before(base - pd.Timedelta(days=30), calendar))
    return pd.DatetimeIndex(monthly), pd.DatetimeIndex(vix)


def options_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Deterministic option-market calendar for the next session after each close.

    Row ``d`` describes the next trading date ``d+`` (the next index date; business days past
    the end of the index), which is known at the close of ``d`` because the exchange calendar is
    published in advance: the calendar-day gap to ``d+`` (weekends and holidays), whether ``d+``
    is a monthly or quarterly SPX expiry or a VIX settlement day, the trading days from ``d`` to
    the next monthly expiry (divided by 21), and whether ``d+`` is the last or first trading day
    of its month.
    """
    calendar = _trading_calendar(index)
    days = index.normalize()
    positions = calendar.get_indexer(days)
    nxt = calendar[positions + 1]
    after_next = calendar[positions + 2]
    monthly, vix = option_expiries(calendar)
    quarterly = monthly[monthly.month.isin([3, 6, 9, 12])]
    expiry_pos = calendar.get_indexer(monthly)
    upcoming = expiry_pos[np.searchsorted(expiry_pos, positions, side="right")]
    return pd.DataFrame(
        {
            "gap_days": (nxt - days).days.astype(float),
            "next_monthly_expiry": nxt.isin(monthly).astype(float),
            "next_quarterly_expiry": nxt.isin(quarterly).astype(float),
            "days_to_monthly_expiry": (upcoming - positions) / 21,
            "next_vix_settlement": nxt.isin(vix).astype(float),
            "next_month_end": (after_next.month != nxt.month).astype(float),
            "next_month_start": (nxt.month != days.month).astype(float),
        },
        index=index,
    )


def load_market_series(paths: tuple[Path, Path, Path] = MARKET_PATHS) -> pd.DataFrame:
    """Load Bloomberg series, FX smiles and ETF implied vols in their original units."""
    frames = []
    for path, value in zip(paths, ("px_last", "px_last", "ivol_30d_atm"), strict=True):
        observations = pd.read_csv(path, parse_dates=["date"], usecols=["date", "ticker", value])
        wide = observations.pivot(index="date", columns="ticker", values=value)
        frames.append(wide.add_prefix(_IVOL_PREFIX) if value == "ivol_30d_atm" else wide)
    return pd.concat(frames, axis=1, sort=True)


def align_asof(series: pd.Series, index: pd.DatetimeIndex, max_age: int = 5, lag: int = 0) -> pd.Series:
    """Carry the latest eligible observation at most max_age weekdays to each surface date.

    With a publication lag, the observation date is capped by the surface date lag rows back.
    Measure age from the raw observation to the destination surface date.
    """
    if max_age < 0 or lag < 0:
        raise ValueError("max_age and lag must be nonnegative.")
    observed = series.dropna().sort_index()
    eligible = index.to_series().shift(lag)
    positions = observed.index.get_indexer(eligible, method="pad")
    matched = (positions >= 0) & eligible.notna().to_numpy()
    values = np.full(len(index), np.nan)
    source = observed.index.take(positions[matched])
    age = np.busday_count(
        source.to_numpy(dtype="datetime64[D]"), index[matched].to_numpy(dtype="datetime64[D]")
    )
    values[matched] = np.where(age <= max_age, observed.to_numpy()[positions[matched]], np.nan)
    return pd.Series(values, index=index, name=series.name)


def rates_features(market: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute yield changes, curve slope, policy path change and funding spread in bp."""
    us2y = 100 * align_asof(market["USGG2YR Index"], index)
    us10y = 100 * align_asof(market["USGG10YR Index"], index)
    bund10 = 100 * align_asof(market["GDBR10 Index"], index)
    ff1 = align_asof(market["FF1 Comdty"], index)
    ff4 = align_asof(market["FF4 Comdty"], index)
    sofr = align_asof(market["SOFRRATE Index"], index, lag=1)
    effr = align_asof(market["FEDL01 Index"], index, lag=1)
    return pd.DataFrame(
        {
            "us2y_chg": us2y.diff(),
            "us10y_chg": us10y.diff(),
            "us_slope": us10y - us2y,
            "bund10_chg": bund10.diff(),
            "ff_path_chg": 100 * (ff1 - ff4).diff(),  # (100 - FF4) - (100 - FF1)
            "funding": 100 * (sofr - effr),
        }
    )


def credit_features(market: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute OAS changes in bp and HYG implied volatility in vol points."""
    ig_oas = align_asof(market["LUACOAS Index"], index)
    hy_oas = align_asof(market["LF98OAS Index"], index)
    hyg_iv = align_asof(market[f"{_IVOL_PREFIX}HYG US Equity"], index)
    return pd.DataFrame(
        {
            "ig_oas_chg": 100 * ig_oas.diff(),
            "hy_oas_chg": 100 * hy_oas.diff(),
            "hyg_iv": hyg_iv,
            "hyg_iv_chg": hyg_iv.diff(),
        }
    )


def cross_vol_features(market: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute volatility index log changes in percent and FX smile features in vol points."""
    move = align_asof(market["MOVE Index"], index)
    ovx = align_asof(market["OVX Index"], index)
    eur_atm = align_asof(market["EURUSDV1M BGN Curncy"], index)
    jpy_atm = align_asof(market["USDJPYV1M BGN Curncy"], index)
    jpy_rr = align_asof(market["USDJPY25R1M BGN Curncy"], index)
    return pd.DataFrame(
        {
            "move_chg": 100 * np.log(move).diff(),
            "ovx_chg": 100 * np.log(ovx).diff(),
            "eur_atm_chg": eur_atm.diff(),
            "jpy_atm_chg": jpy_atm.diff(),
            "jpy_rr": jpy_rr,
            "jpy_rr_chg": jpy_rr.diff(),
        }
    )


def sectors_features(market: pd.DataFrame, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Compute mean IV change and population dispersion across all 11 sector ETFs."""
    iv = pd.concat(
        [align_asof(market[f"{_IVOL_PREFIX}{ticker}"], index) for ticker in _SECTOR_TICKERS], axis=1
    )
    prices = pd.concat([align_asof(market[ticker], index) for ticker in _SECTOR_TICKERS], axis=1)
    returns = 100 * np.log(prices).diff()
    return pd.DataFrame(
        {
            "sec_iv_chg": iv.mean(axis=1, skipna=False).diff(),
            "sec_iv_disp": iv.std(axis=1, ddof=0, skipna=False),
            "sec_ret_disp": returns.std(axis=1, ddof=0, skipna=False),
        }
    )


_MARKET_BLOCKS = {
    "rates": rates_features,
    "credit": credit_features,
    "cross_vol": cross_vol_features,
    "sectors": sectors_features,
}
_BLOCKS = ("calendar", "underlying", "options_calendar", "session", *_MARKET_BLOCKS)


def build_exog(blocks: list[str], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Build the requested blocks in order and align their rows to the surface dates."""
    for block in blocks:
        if block not in _BLOCKS:
            raise ValueError(f"Unknown exogenous block {block!r}. Valid blocks are {', '.join(_BLOCKS)}.")

    market = load_market_series() if any(block in _MARKET_BLOCKS for block in blocks) else pd.DataFrame()
    panels = []
    for block in blocks:
        if block == "calendar":
            panels.append(calendar_features(index, load_event_schedule()))
        elif block == "underlying":
            panels.append(underlying_features(load_spx()).reindex(index))
        elif block == "options_calendar":
            panels.append(options_calendar_features(index))
        elif block == "session":
            panels.append(session_features(load_spx()).reindex(index))
        else:
            panels.append(_MARKET_BLOCKS[block](market, index))
    return pd.concat(panels, axis=1) if panels else pd.DataFrame(index=index)
