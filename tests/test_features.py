from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from volsurf import features

CALENDAR_COLUMNS = [
    "fomc_next",
    "fomc_days_to_next",
    "cpi_next",
    "cpi_days_to_next",
    "nfp_next",
    "nfp_days_to_next",
]
UNDERLYING_COLUMNS = ["ret", "ret_neg", "rv5", "rv21", "range"]


# Bloomberg rows around the 2025 lapse in appropriations, as in the local export.
ECO_2025 = (
    "ticker,period_end,actual,survey_median,release_date\n"
    "CPI CHNG Index,2025-08-31,0.3,0.3,2025-09-11\n"
    "CPI CHNG Index,2025-09-30,0.3,0.4,2025-10-24\n"
    "CPI CHNG Index,2025-12-31,0.3,0.3,2026-01-13\n"
    "NFP TCH Index,2025-08-31,-70,75,2025-09-05\n"
    "NFP TCH Index,2025-09-30,76,52,2025-11-20\n"
    "NFP TCH Index,2025-10-31,-140,-25,2025-12-16\n"
    "NFP TCH Index,2025-11-30,41,50,2025-12-16\n"
    "NFP TCH Index,2025-12-31,-17,70,2026-01-09\n"
    "CPI CHNG Index,2026-01-31,0.2,0.3,2026-02-13\n"
    "NFP TCH Index,2026-01-31,130,70,2026-02-11\n"
)
SCHEDULE_COLUMNS = ["event", "date", "known_from", "withdrawn_from"]


def _schedule(events: dict[str, list]) -> pd.DataFrame:
    """A schedule whose entries were always known and never withdrawn."""
    rows = [(name, pd.Timestamp(date)) for name, dates in events.items() for date in dates]
    out = pd.DataFrame(rows, columns=["event", "date"]).astype({"date": "datetime64[ns]"})
    out["known_from"] = pd.Timestamp.min
    out["withdrawn_from"] = pd.Timestamp.max
    return out


def _entry(schedule: pd.DataFrame, event: str, date: str) -> pd.Series:
    rows = schedule[(schedule["event"] == event) & (schedule["date"] == pd.Timestamp(date))]
    assert len(rows) == 1
    return rows.iloc[0]


@pytest.fixture
def eco_2025(tmp_path: Path) -> Path:
    path = tmp_path / "eco.csv"
    path.write_text(ECO_2025, encoding="utf-8")
    return path


def test_load_event_schedule(tmp_path: Path) -> None:
    path = tmp_path / "eco.csv"
    path.write_text(
        "ticker,period_end,actual,survey_median,release_date\n"
        "CPI CHNG Index,2024-02-29,2,2,2024-03-12\n"
        "NFP TCH Index,2024-01-31,3,3,2024-02-02\n"
        "CPI CHNG Index,2024-01-31,1,1,2024-02-13\n"
        "CPI CHNG Index,2024-01-31,1,1,2024-02-13\n"
        "NFP TCH Index,2024-02-29,,,2024-03-08\n"
        "OTHER Index,2024-01-31,1,1,2024-02-01\n",
        encoding="utf-8",
    )
    schedule = features.load_event_schedule(path)
    assert list(schedule.columns) == SCHEDULE_COLUMNS
    assert set(schedule["event"]) == {"fomc", "cpi", "nfp"}
    assert not schedule.duplicated(["event", "date"]).any()
    always = schedule[
        (schedule["known_from"] == pd.Timestamp.min) & (schedule["withdrawn_from"] == pd.Timestamp.max)
    ]
    by_event = {name: pd.DatetimeIndex(group["date"]) for name, group in always.groupby("event")}
    assert by_event["cpi"].tolist() == list(pd.to_datetime(["2024-02-13", "2024-03-12"]))
    assert by_event["nfp"].tolist() == list(pd.to_datetime(["2024-02-02", "2024-03-08"]))
    assert by_event["fomc"].is_monotonic_increasing
    assert by_event["fomc"][0] == pd.Timestamp("2019-01-30")
    assert by_event["fomc"][-1] == pd.Timestamp("2026-12-09")
    # The March 2020 meeting stays scheduled until the unscheduled meeting of 15 March replaced it.
    cancelled = _entry(schedule, "fomc", "2020-03-18")
    assert cancelled["known_from"] == pd.Timestamp.min
    assert cancelled["withdrawn_from"] == pd.Timestamp("2020-03-15")


def test_load_event_schedule_revisions(eco_2025: Path) -> None:
    schedule = features.load_event_schedule(eco_2025)
    lapse = pd.Timestamp("2025-10-01")
    for event, date in [
        ("cpi", "2025-10-15"),
        ("cpi", "2025-11-13"),
        ("cpi", "2025-12-10"),
        ("nfp", "2025-10-03"),
        ("nfp", "2025-11-07"),
        ("nfp", "2025-12-05"),
    ]:
        original = _entry(schedule, event, date)
        assert (original["known_from"], original["withdrawn_from"]) == (pd.Timestamp.min, lapse)
    for event, date in [("cpi", "2026-02-11"), ("nfp", "2026-02-06")]:
        original = _entry(schedule, event, date)
        assert (original["known_from"], original["withdrawn_from"]) == (
            pd.Timestamp.min,
            pd.Timestamp("2026-01-31"),
        )
    for event, date, known_from in [
        ("cpi", "2025-10-24", "2025-10-10"),
        ("cpi", "2025-12-18", "2025-12-11"),
        ("nfp", "2025-11-20", "2025-11-19"),
        ("nfp", "2025-12-16", "2025-12-11"),
        ("cpi", "2026-02-13", "2026-02-04"),
        ("nfp", "2026-02-11", "2026-02-04"),
    ]:
        revised = _entry(schedule, event, date)
        assert (revised["known_from"], revised["withdrawn_from"]) == (
            pd.Timestamp(known_from),
            pd.Timestamp.max,
        )
    for event, date in [("cpi", "2026-01-13"), ("nfp", "2026-01-09"), ("nfp", "2025-09-05")]:
        unchanged = _entry(schedule, event, date)
        assert (unchanged["known_from"], unchanged["withdrawn_from"]) == (pd.Timestamp.min, pd.Timestamp.max)


def test_calendar_features() -> None:
    index = pd.DatetimeIndex(
        ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-12", "2024-01-16", "2024-01-19"], name="date"
    )
    events = {
        "fomc": ["2024-01-05", "2024-01-09", "2024-10-01"],
        "cpi": ["2024-01-08", "2024-01-15", "2024-01-22"],
        "nfp": ["2024-01-23"],
    }
    out = features.calendar_features(index, _schedule(events))
    pd.testing.assert_index_equal(out.index, index)
    assert list(out.columns) == CALENDAR_COLUMNS
    # The first FOMC event is on the current day. CPI spans a weekend and a holiday gap.
    np.testing.assert_array_equal(out["fomc_next"], [0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(out["cpi_next"], [1, 0, 0, 1, 0, 1])
    np.testing.assert_array_equal(out["nfp_next"], np.zeros(len(index)))
    np.testing.assert_array_equal(out["fomc_days_to_next"], [2, 1, 40, 40, 40, 40])
    np.testing.assert_array_equal(out["cpi_days_to_next"], [1, 5, 4, 1, 4, 1])
    np.testing.assert_array_equal(out["nfp_days_to_next"], [12, 11, 10, 7, 5, 2])
    exhausted = features.calendar_features(index, _schedule({**events, "cpi": events["cpi"][:1]})).loc[
        index[1] :
    ]
    assert (exhausted["cpi_next"] == 0).all()
    assert exhausted["cpi_days_to_next"].isna().all()
    no_nfp = features.calendar_features(index, _schedule({**events, "nfp": []}))
    assert (no_nfp["nfp_next"] == 0).all()
    assert no_nfp["nfp_days_to_next"].isna().all()
    assert features.calendar_features(index[:0], _schedule(events)).empty


def test_calendar_features_vintage() -> None:
    index = pd.bdate_range("2024-01-02", "2024-01-31", name="date")
    revisions = pd.DataFrame(
        {
            "event": ["cpi", "cpi"],
            "date": pd.to_datetime(["2024-01-10", "2024-01-17"]),
            "known_from": [pd.Timestamp.min, pd.Timestamp("2024-01-08")],
            "withdrawn_from": [pd.Timestamp("2024-01-08"), pd.Timestamp.max],
        }
    )
    schedule = pd.concat([_schedule({"cpi": ["2024-01-30"]}), revisions], ignore_index=True)
    out = features.calendar_features(index, schedule)
    # The 10 January release is seen until it is moved to 17 January at the close of 8 January.
    assert out.loc["2024-01-05", "cpi_days_to_next"] == np.busday_count("2024-01-05", "2024-01-10")
    assert out.loc["2024-01-08", "cpi_days_to_next"] == np.busday_count("2024-01-08", "2024-01-17")
    assert out.loc["2024-01-09", "cpi_next"] == 0
    assert out.loc["2024-01-16", "cpi_next"] == 1
    # No FOMC or NFP entry at all.
    assert out["fomc_days_to_next"].isna().all()
    assert (out["nfp_next"] == 0).all()


@pytest.mark.parametrize("change", ["move", "add"])
def test_calendar_features_no_look_ahead(eco_2025: Path, change: str) -> None:
    """A schedule change public from the close of D leaves every row up to D - 1 unchanged."""
    schedule = features.load_event_schedule(eco_2025)
    index = pd.bdate_range("2025-08-01", "2026-02-27", name="date")
    published = pd.Timestamp("2025-11-03")
    changed = schedule.copy()
    if change == "move":
        # Move a release whose revision is made public at D, together with its announcement date.
        row = (changed["event"] == "nfp") & (changed["date"] == pd.Timestamp("2025-11-20"))
        changed.loc[row, "date"] = pd.Timestamp("2025-11-05")
        changed.loc[row, "known_from"] = published
    else:
        extra = pd.DataFrame(
            {
                "event": ["cpi"],
                "date": [pd.Timestamp("2025-11-04")],
                "known_from": [published],
                "withdrawn_from": [pd.Timestamp.max],
            }
        )
        changed = pd.concat([changed, extra], ignore_index=True)
    base = features.calendar_features(index, schedule)
    out = features.calendar_features(index, changed)
    before = published - pd.Timedelta(days=1)
    pd.testing.assert_frame_equal(out.loc[:before], base.loc[:before])
    assert not out.loc[published].equals(base.loc[published])


def test_calendar_features_2025_lapse(eco_2025: Path) -> None:
    index = pd.bdate_range("2025-09-01", "2026-01-30", name="date")
    out = features.calendar_features(index, features.load_event_schedule(eco_2025))
    # Before the lapse the original releases are still scheduled.
    assert out.loc["2025-09-30", "cpi_days_to_next"] == np.busday_count("2025-09-30", "2025-10-15")
    assert out.loc["2025-09-30", "nfp_days_to_next"] == np.busday_count("2025-09-30", "2025-10-03")
    # From the lapse the originals are withdrawn; the rescheduled CPI is public from 10 October.
    assert out.loc["2025-10-02", "cpi_days_to_next"] == 40
    assert out.loc["2025-10-09", "cpi_days_to_next"] == 40
    assert out.loc["2025-10-10", "cpi_days_to_next"] == np.busday_count("2025-10-10", "2025-10-24")
    assert out.loc["2025-10-23", "cpi_next"] == 1
    # Revised releases without an announcement date are known from the business day before.
    assert out.loc["2025-11-18", "nfp_next"] == 0
    assert out.loc["2025-11-19", "nfp_next"] == 1
    # The December revised releases are known from the schedule update of 11 December.
    assert out.loc["2025-12-10", "nfp_days_to_next"] == np.busday_count("2025-12-10", "2026-01-09")
    assert out.loc["2025-12-11", "nfp_days_to_next"] == np.busday_count("2025-12-11", "2025-12-16")
    assert out.loc["2025-12-11", "cpi_days_to_next"] == np.busday_count("2025-12-11", "2025-12-18")
    assert out.loc["2025-12-15", "nfp_next"] == 1
    assert out.loc["2025-12-17", "cpi_next"] == 1


def test_calendar_features_2026_lapse(eco_2025: Path) -> None:
    index = pd.bdate_range("2026-01-26", "2026-02-27", name="date")
    out = features.calendar_features(index, features.load_event_schedule(eco_2025))
    # Before the 2026 lapse the original February releases are still scheduled.
    assert out.loc["2026-01-30", "nfp_days_to_next"] == np.busday_count("2026-01-30", "2026-02-06")
    assert out.loc["2026-01-30", "cpi_days_to_next"] == np.busday_count("2026-01-30", "2026-02-11")
    # Withdrawn from the lapse; the revised dates are public from 4 February.
    assert out.loc["2026-02-03", "nfp_next"] == 0
    assert np.isnan(out.loc["2026-02-03", "nfp_days_to_next"])
    assert out.loc["2026-02-04", "nfp_days_to_next"] == np.busday_count("2026-02-04", "2026-02-11")
    assert out.loc["2026-02-04", "cpi_days_to_next"] == np.busday_count("2026-02-04", "2026-02-13")
    assert out.loc["2026-02-05", "nfp_next"] == 0
    assert out.loc["2026-02-10", "nfp_next"] == 1
    assert out.loc["2026-02-12", "cpi_next"] == 1


def test_calendar_features_march_2020_fomc(eco_2025: Path) -> None:
    index = pd.bdate_range("2020-03-02", "2020-03-31", name="date")
    out = features.calendar_features(index, features.load_event_schedule(eco_2025))
    assert out.loc["2020-03-13", "fomc_days_to_next"] == 3
    assert out.loc["2020-03-17", "fomc_next"] == 0
    assert out.loc["2020-03-16", "fomc_days_to_next"] == np.busday_count("2020-03-16", "2020-04-29")


def test_load_spx(tmp_path: Path) -> None:
    path = tmp_path / "spx.csv"
    path.write_text(
        "date,PX_OPEN,PX_HIGH,PX_LOW,PX_LAST,PX_VOLUME\n"
        "2024-01-03,11,13,10,12,90\n"
        "2024-01-02,10,12,9,11,100\n",
        encoding="utf-8",
    )
    out = features.load_spx(path)
    assert out.index.tolist() == list(pd.to_datetime(["2024-01-02", "2024-01-03"]))
    assert out["PX_LAST"].tolist() == [11, 12]


def _prices() -> tuple[pd.DataFrame, np.ndarray]:
    returns = np.arange(-12, 13, dtype=float) ** 3 / 144
    close = 100 * np.exp(np.r_[0.0, returns.cumsum()] / 100)
    index = pd.bdate_range("2024-01-02", periods=len(close), name="date")
    spx = pd.DataFrame({"PX_LAST": close, "PX_HIGH": close * 1.02, "PX_LOW": close * 0.99}, index=index)
    return spx, returns


def test_underlying_features() -> None:
    spx, returns = _prices()
    out = features.underlying_features(spx)
    pd.testing.assert_index_equal(out.index, spx.index)
    assert list(out.columns) == UNDERLYING_COLUMNS
    expected_ret = np.r_[np.nan, returns]
    np.testing.assert_allclose(out["ret"], expected_ret, atol=1e-12)
    np.testing.assert_allclose(out["ret_neg"], np.minimum(expected_ret, 0), atol=1e-12)
    np.testing.assert_allclose(out["range"], 100 * np.log(1.02 / 0.99))
    for window in (5, 21):
        # There is no prior close for the first row, so it cannot contribute a return.
        expected_vol = [
            np.sqrt(252) * returns[end - window : end].std(ddof=1) for end in range(window, len(returns) + 1)
        ]
        assert out[f"rv{window}"].iloc[:window].isna().all()
        np.testing.assert_allclose(out[f"rv{window}"].iloc[window:], expected_vol)


@pytest.mark.parametrize("blocks", [["calendar", "underlying"], ["underlying", "calendar"]])
def test_build_exog(blocks: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    spx, _ = _prices()
    index = spx.index[[2, 4]].append(pd.DatetimeIndex(["2024-02-29"], name="date"))
    schedule = _schedule({name: list(spx.index[3:4]) for name in ("fomc", "cpi", "nfp")})
    monkeypatch.setattr(features, "load_event_schedule", lambda: schedule)
    monkeypatch.setattr(features, "load_spx", lambda: spx)
    out = features.build_exog(blocks, index)
    columns = {"calendar": CALENDAR_COLUMNS, "underlying": UNDERLYING_COLUMNS}
    assert list(out.columns) == columns[blocks[0]] + columns[blocks[1]]
    pd.testing.assert_index_equal(out.index, index)
    assert out["fomc_next"].tolist() == [1, 0, 0]
    pd.testing.assert_frame_equal(out[UNDERLYING_COLUMNS], features.underlying_features(spx).reindex(index))


def test_build_exog_empty_and_unknown() -> None:
    index = pd.bdate_range("2024-01-02", periods=3)
    pd.testing.assert_frame_equal(features.build_exog([], index), pd.DataFrame(index=index))
    with pytest.raises(ValueError, match="typo.*calendar.*underlying"):
        features.build_exog(["calendar", "typo"], index)


@pytest.mark.slow
def test_features_real(real_surface: pd.DataFrame) -> None:
    for path in (features.ECO_PATH, features.SPX_PATH):
        if not path.exists():
            pytest.skip(f"{path} not available")
    index = real_surface.index
    panel = features.build_exog(["calendar", "underlying"], index)
    pd.testing.assert_index_equal(panel.index, index)
    assert np.isfinite(panel[UNDERLYING_COLUMNS].iloc[21:]).all().all()

    expected = {
        # The scheduled March 2020 FOMC meeting was cancelled.
        "fomc": [8, 7, 8, 8, 8, 8, 8, 8],
        # The local macro export has truncated edge years; October 2025 CPI was cancelled.
        "cpi": [11, 12, 12, 12, 12, 12, 11, 10],
        # Two 2025 payroll periods share a single December release date.
        "nfp": [11, 12, 12, 12, 12, 12, 11, 11],
    }
    # Count the final dates: entries that were never withdrawn.
    schedule = features.load_event_schedule()
    final = schedule[schedule["withdrawn_from"] == pd.Timestamp.max]
    assert set(final["event"]) == set(expected)
    for name, group in final.groupby("event"):
        counts = group["date"].dt.year.value_counts()
        assert counts.reindex(range(2019, 2027)).tolist() == expected[name]
