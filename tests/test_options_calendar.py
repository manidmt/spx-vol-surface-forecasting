"""Option-market calendar and session-split features."""

import numpy as np
import pandas as pd
import pytest

from volsurf import features
from volsurf.features import option_expiries, options_calendar_features, session_features

# NYSE holidays in early 2022 (business days that were not trading days)
HOLIDAYS = pd.to_datetime(["2022-01-17", "2022-02-21", "2022-04-15", "2022-05-30"])


def _calendar(start="2022-01-03", end="2022-06-30"):
    days = pd.bdate_range(start, end)
    return days[~days.isin(HOLIDAYS)]


def test_monthly_expiry_moves_to_thursday_on_good_friday():
    monthly, _ = option_expiries(_calendar())
    assert pd.Timestamp("2022-04-14") in monthly  # third Friday 2022-04-15 was Good Friday
    assert pd.Timestamp("2022-02-18") in monthly
    assert pd.Timestamp("2022-04-15") not in monthly


def test_vix_settlement_rule():
    _, vix = option_expiries(_calendar())
    assert pd.Timestamp("2022-01-19") in vix  # 30 days before 2022-02-18
    assert pd.Timestamp("2022-03-15") in vix  # 30 days before Thursday 2022-04-14 (Good Friday)
    assert pd.Timestamp("2022-02-16") in vix  # 30 days before 2022-03-18
    assert pd.Timestamp("2022-06-15") in vix  # next month's Friday (2022-07-15) is past the calendar


def test_options_calendar_features_hand_cases():
    index = _calendar()
    out = options_calendar_features(index)
    assert out.loc["2022-01-07", "gap_days"] == 3  # Friday to Monday
    assert out.loc["2022-01-14", "gap_days"] == 4  # long weekend (Martin Luther King Day)
    assert out.loc["2022-04-14", "gap_days"] == 4  # Thursday before Good Friday
    assert out.loc["2022-04-13", "next_monthly_expiry"] == 1
    assert out.loc["2022-04-14", "next_monthly_expiry"] == 0
    assert out.loc["2022-03-17", "next_quarterly_expiry"] == 1
    assert out.loc["2022-02-17", "next_quarterly_expiry"] == 0
    assert out.loc["2022-03-14", "next_vix_settlement"] == 1
    assert out.loc["2022-02-17", "days_to_monthly_expiry"] == pytest.approx(1 / 21)
    assert out.loc["2022-02-18", "days_to_monthly_expiry"] == pytest.approx(19 / 21)  # to 2022-03-18
    assert out.loc["2022-03-30", "next_month_end"] == 1  # next session 2022-03-31
    assert out.loc["2022-03-31", "next_month_start"] == 1
    assert out.loc["2022-03-29", "next_month_end"] == 0
    assert out.notna().all().all()


def test_options_calendar_features_last_row_uses_business_days():
    index = _calendar(end="2022-06-30")
    out = options_calendar_features(index)
    assert out.iloc[-1]["gap_days"] == 1  # Thursday 2022-06-30 to Friday
    assert out.iloc[-1]["next_month_start"] == 1


def test_session_features_split_the_return():
    index = pd.bdate_range("2022-01-03", periods=3)
    spx = pd.DataFrame({"PX_OPEN": [100.0, 102.0, 99.0], "PX_LAST": [101.0, 100.0, 98.0]}, index=index)
    out = session_features(spx)
    assert np.isnan(out.iloc[0, 0])
    assert out.iloc[1]["ret_overnight"] == pytest.approx(100 * (102 / 101 - 1))
    assert out.iloc[1]["ret_intraday"] == pytest.approx(100 * (100 / 102 - 1))
    total = (1 + out.iloc[1]["ret_overnight"] / 100) * (1 + out.iloc[1]["ret_intraday"] / 100)
    assert total == pytest.approx(100 / 101)


def test_build_exog_new_blocks(monkeypatch):
    index = _calendar()
    spx = pd.DataFrame({"PX_OPEN": 100.0, "PX_LAST": 101.0}, index=index)
    monkeypatch.setattr(features, "load_spx", lambda: spx)
    out = features.build_exog(["options_calendar", "session"], index)
    assert list(out.columns[-2:]) == ["ret_overnight", "ret_intraday"]
    assert out.index.equals(index)


def test_options_calendar_real(real_surface):
    out = options_calendar_features(real_surface.index)
    assert out.notna().all().all()
    # one monthly expiry per month, and a few dozen VIX settlements over the sample
    assert out["next_monthly_expiry"].groupby(real_surface.index.to_period("M")).sum().max() == 1
    assert out.loc["2025-04-17", "next_monthly_expiry"] == 0  # 2025-04-18 was Good Friday
    assert out.loc["2025-04-16", "next_monthly_expiry"] == 1
