from datetime import date, datetime, timezone
from typing import Any

from athenian.api.align.goals.dates import (
    goal_dates_to_datetimes,
    goal_datetimes_to_dates,
    goal_initial_query_interval,
)


class TestGoalDatesToDatetimes:
    def test_base(self) -> None:
        valid_from = date(2001, 1, 1)
        expires_at = date(2001, 12, 31)

        assert goal_dates_to_datetimes(valid_from, expires_at) == (
            _dt(2001, 1, 1),
            _dt(2002, 1, 1),
        )

    def test_conversion_round_trip(self) -> None:
        valid_from = date(2012, 7, 1)
        expires_at = date(2001, 9, 30)

        assert goal_datetimes_to_dates(*goal_dates_to_datetimes(valid_from, expires_at)) == (
            valid_from,
            expires_at,
        )


class TestGoalDatetimesToDates:
    def test_base(self) -> None:
        valid_from = _dt(2012, 4, 1)
        expires_at = _dt(2012, 7, 1)

        assert goal_datetimes_to_dates(valid_from, expires_at) == (
            date(2012, 4, 1),
            date(2012, 6, 30),
        )

    def test_conversion_round_trip(self) -> None:
        valid_from = date(2020, 1, 1)
        expires_at = date(2020, 12, 31)

        assert goal_datetimes_to_dates(*goal_dates_to_datetimes(valid_from, expires_at)) == (
            valid_from,
            expires_at,
        )


class TestGoalInitialQueryInterval:
    def test_year_span(self) -> None:
        valid_from = _dt(2022, 1, 1)
        expires_at = _dt(2023, 1, 1)

        interval = goal_initial_query_interval(valid_from, expires_at)
        assert interval == (_dt(2021, 1, 1), _dt(2022, 1, 1))

    def test_quarter_span(self) -> None:
        valid_from = _dt(2020, 4, 1)
        expires_at = _dt(2020, 7, 1)

        interval = goal_initial_query_interval(valid_from, expires_at)
        assert interval == (_dt(2020, 1, 1), _dt(2020, 4, 1))

    def test_semester_span(self) -> None:
        valid_from = _dt(2019, 7, 1)
        expires_at = _dt(2020, 1, 1)

        interval = goal_initial_query_interval(valid_from, expires_at)
        assert interval == (_dt(2019, 1, 1), _dt(2019, 7, 1))

    def test_monthly_span(self) -> None:
        valid_from = _dt(2019, 5, 1)
        expires_at = _dt(2019, 6, 1)

        interval = goal_initial_query_interval(valid_from, expires_at)
        assert interval == (_dt(2019, 4, 1), _dt(2019, 5, 1))

    def test_custom_span(self) -> None:
        valid_from = _dt(2019, 3, 1)
        expires_at = _dt(2019, 10, 1)

        interval = goal_initial_query_interval(valid_from, expires_at)
        assert interval == (_dt(2018, 8, 1), _dt(2019, 3, 1))


def _dt(*args: Any, **kwargs: Any) -> datetime:
    kwargs.setdefault("tzinfo", timezone.utc)
    return datetime(*args, **kwargs)
