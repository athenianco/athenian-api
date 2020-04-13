from datetime import date, timedelta

import pytest

from athenian.api.models.web import Granularity


@pytest.mark.parametrize("value, result", [
    ("day", [date(2020, 1, 1) + i * timedelta(days=1) for i in range(63)]),
    ("1 day", [date(2020, 1, 1) + i * timedelta(days=1) for i in range(63)]),
    ("2 day", [date(2020, 1, 1) + 2 * i * timedelta(days=1) for i in range(31)] +
     [date(2020, 3, 3)]),
    ("week", [date(2020, 1, 1) + i * timedelta(days=7) for i in range(9)] + [date(2020, 3, 3)]),
    ("3 week", [date(2020, 1, 1) + 3 * i * timedelta(days=7) for i in range(3)] +
     [date(2020, 3, 3)]),
    ("month", [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1), date(2020, 3, 3)]),
    ("4 month", [date(2020, 1, 1), date(2020, 3, 3)]),
    ("year", [date(2020, 1, 1), date(2020, 3, 3)]),
    ("all", [date(2020, 1, 1), date(2020, 3, 3)]),
])
def test_split_correctness(value, result):
    assert result == Granularity.split(value, date(2020, 1, 1), date(2020, 3, 2))


def test_split_correctness_exact_month():
    assert [date(2020, 1, 1), date(2020, 2, 1), date(2020, 3, 1), date(2020, 3, 2)] == \
        Granularity.split("month", date(2020, 1, 1), date(2020, 3, 1))


def test_split_correctness_exact_leap_year():
    assert [date(2020, 1, 1), date(2021, 1, 1)] == \
        Granularity.split("year", date(2020, 1, 1), date(2020, 12, 31))


def test_split_correctness_exact_leap_year_plus_one():
    assert [date(2020, 1, 1), date(2021, 1, 1), date(2021, 1, 2)] == \
        Granularity.split("year", date(2020, 1, 1), date(2021, 1, 1))


@pytest.mark.parametrize("value", ["bullshit", "days", " day", "0 day", "01 day", "2day", "2 all",
                                   " all", "xx3 day", "2 day xxx", "dayll"])
def test_split_errors(value):
    with pytest.raises(ValueError):
        Granularity.split(value, date(2020, 1, 1), date(2020, 3, 2))
