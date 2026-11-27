"""Walk-forward evaluation protocol."""

import pandas as pd

TEST_START = pd.Timestamp("2021-01-01")


def refit_schedule(
    index: pd.DatetimeIndex, test_start: pd.Timestamp = TEST_START
) -> list[tuple[pd.Timestamp, pd.DatetimeIndex]]:
    """Group test days by calendar month: (first test day of the month, all its test days)."""
    test_days = index[index >= test_start]
    schedule = []
    for _, days in test_days.to_series().groupby(test_days.to_period("M")):
        month_days = pd.DatetimeIndex(days.index)
        schedule.append((month_days[0], month_days))
    return schedule
