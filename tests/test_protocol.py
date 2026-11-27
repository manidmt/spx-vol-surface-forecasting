import pandas as pd

from volsurf.protocol import TEST_START, refit_schedule


def test_refit_schedule(synthetic_surface):
    index = synthetic_surface.index
    schedule = refit_schedule(index)
    test_days = index[index >= TEST_START]
    assert schedule[0][0] == test_days[0]
    months = []
    for refit_date, days in schedule:
        assert refit_date == days[0]
        assert len(set(days.to_period("M"))) == 1
        months.append(days[0].to_period("M"))
    for a, b in zip(months[:-1], months[1:], strict=True):
        assert b == a + 1
    union = schedule[0][1].append([d for _, d in schedule[1:]])
    pd.testing.assert_index_equal(union, test_days)
