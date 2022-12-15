from datetime import date

from athenian.api.internal.datetime_utils import (
    closed_dates_interval_to_datetimes,
    datetimes_to_closed_dates_interval,
)
from tests.testutils.time import dt


class TestClosedDatesIntervalToDatetimes:
    def test_base(self) -> None:
        valid_from = date(2001, 1, 1)
        expires_at = date(2001, 12, 31)

        assert closed_dates_interval_to_datetimes(valid_from, expires_at) == (
            dt(2001, 1, 1),
            dt(2002, 1, 1),
        )

    def test_conversion_round_trip(self) -> None:
        valid_from = date(2012, 7, 1)
        expires_at = date(2001, 9, 30)

        assert datetimes_to_closed_dates_interval(
            *closed_dates_interval_to_datetimes(valid_from, expires_at),
        ) == (valid_from, expires_at)


class TestGoalDatetimesToDates:
    def test_base(self) -> None:
        valid_from = dt(2012, 4, 1)
        expires_at = dt(2012, 7, 1)

        assert datetimes_to_closed_dates_interval(valid_from, expires_at) == (
            date(2012, 4, 1),
            date(2012, 6, 30),
        )

    def test_conversion_round_trip(self) -> None:
        valid_from = date(2020, 1, 1)
        expires_at = date(2020, 12, 31)

        assert datetimes_to_closed_dates_interval(
            *closed_dates_interval_to_datetimes(valid_from, expires_at),
        ) == (valid_from, expires_at)
