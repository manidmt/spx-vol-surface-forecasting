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


def test_load_event_dates(tmp_path: Path) -> None:
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
    events = features.load_event_dates(path)
    assert set(events) == {"fomc", "cpi", "nfp"}
    assert events["cpi"].tolist() == list(pd.to_datetime(["2024-02-13", "2024-03-12"]))
    assert events["nfp"].tolist() == list(pd.to_datetime(["2024-02-02", "2024-03-08"]))
    assert events["fomc"].is_monotonic_increasing and events["fomc"].is_unique
    assert events["fomc"][0] == pd.Timestamp("2019-01-30")
    assert events["fomc"][-1] == pd.Timestamp("2026-12-09")


def test_calendar_features() -> None:
    index = pd.DatetimeIndex(
        ["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-12", "2024-01-16", "2024-01-19"], name="date"
    )
    events = {
        "fomc": pd.DatetimeIndex(["2024-01-05", "2024-01-09", "2024-10-01"]),
        "cpi": pd.DatetimeIndex(["2024-01-08", "2024-01-15", "2024-01-22"]),
        "nfp": pd.DatetimeIndex(["2024-01-23"]),
    }
    out = features.calendar_features(index, events)
    pd.testing.assert_index_equal(out.index, index)
    assert list(out.columns) == CALENDAR_COLUMNS
    # The first FOMC event is on the current day. CPI spans a weekend and a holiday gap.
    np.testing.assert_array_equal(out["fomc_next"], [0, 1, 0, 0, 0, 0])
    np.testing.assert_array_equal(out["cpi_next"], [1, 0, 0, 1, 0, 1])
    np.testing.assert_array_equal(out["nfp_next"], np.zeros(len(index)))
    np.testing.assert_array_equal(out["fomc_days_to_next"], [2, 1, 40, 40, 40, 40])
    np.testing.assert_array_equal(out["cpi_days_to_next"], [1, 5, 4, 1, 4, 1])
    np.testing.assert_array_equal(out["nfp_days_to_next"], [12, 11, 10, 7, 5, 2])
    exhausted = features.calendar_features(index, {**events, "cpi": events["cpi"][:1]}).loc[index[1] :]
    assert (exhausted["cpi_next"] == 0).all()
    assert exhausted["cpi_days_to_next"].isna().all()
    assert features.calendar_features(index[:0], events).empty


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
    events = {name: spx.index[3:4] for name in ("fomc", "cpi", "nfp")}
    monkeypatch.setattr(features, "load_event_dates", lambda: events)
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
        # The local macro export has truncated edge years and missing 2025 CPI dates.
        "cpi": [11, 12, 12, 12, 12, 12, 10, 10],
        # Two 2025 payroll periods share a single December release date.
        "nfp": [11, 12, 12, 12, 12, 12, 11, 11],
    }
    for name, dates in features.load_event_dates().items():
        counts = dates.year.value_counts()
        assert counts.reindex(range(2019, 2027)).tolist() == expected[name]
