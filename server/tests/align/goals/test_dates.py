from datetime import date

from athenian.api.align.goals.dates import (
    GoalTimeseriesSpec,
    goal_dates_to_datetimes,
    goal_datetimes_to_dates,
    goal_initial_query_interval,
)
from athenian.api.models.web.goal import GoalSeriesGranularity
from tests.testutils.time import dt


class TestGoalDatesToDatetimes:
    def test_base(self) -> None:
        valid_from = date(2001, 1, 1)
        expires_at = date(2001, 12, 31)

        assert goal_dates_to_datetimes(valid_from, expires_at) == (dt(2001, 1, 1), dt(2002, 1, 1))

    def test_conversion_round_trip(self) -> None:
        valid_from = date(2012, 7, 1)
        expires_at = date(2001, 9, 30)

        assert goal_datetimes_to_dates(*goal_dates_to_datetimes(valid_from, expires_at)) == (
            valid_from,
            expires_at,
        )


class TestGoalDatetimesToDates:
    def test_base(self) -> None:
        valid_from = dt(2012, 4, 1)
        expires_at = dt(2012, 7, 1)

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
        valid_from = dt(2022, 1, 1)
        expires_at = dt(2023, 1, 1)

        interval = goal_initial_query_interval(valid_from, expires_at)
        assert interval == (dt(2021, 1, 1), dt(2022, 1, 1))

    def test_quarter_span(self) -> None:
        valid_from = dt(2020, 4, 1)
        expires_at = dt(2020, 7, 1)

        interval = goal_initial_query_interval(valid_from, expires_at)
        assert interval == (dt(2020, 1, 1), dt(2020, 4, 1))

    def test_semester_span(self) -> None:
        valid_from = dt(2019, 7, 1)
        expires_at = dt(2020, 1, 1)

        interval = goal_initial_query_interval(valid_from, expires_at)
        assert interval == (dt(2019, 1, 1), dt(2019, 7, 1))

    def test_monthly_span(self) -> None:
        valid_from = dt(2019, 5, 1)
        expires_at = dt(2019, 6, 1)

        interval = goal_initial_query_interval(valid_from, expires_at)
        assert interval == (dt(2019, 4, 1), dt(2019, 5, 1))

    def test_custom_span(self) -> None:
        valid_from = dt(2019, 3, 1)
        expires_at = dt(2019, 10, 1)

        interval = goal_initial_query_interval(valid_from, expires_at)
        assert interval == (dt(2018, 8, 1), dt(2019, 3, 1))


class TestGoalTimeseriesSpec:
    def test_from_timespan_year(self) -> None:
        spec = GoalTimeseriesSpec.from_timespan(dt(2021, 1, 1), dt(2022, 1, 1))

        assert spec.granularity == GoalSeriesGranularity.MONTH.value
        assert spec.intervals == (*(dt(2021, i, 1) for i in range(1, 13)), dt(2022, 1, 1))

    def test_from_timespan_quarter(self) -> None:
        spec = GoalTimeseriesSpec.from_timespan(dt(2021, 4, 1), dt(2021, 7, 1))

        assert spec.granularity == GoalSeriesGranularity.WEEK.value
        assert len(spec.intervals) == 14
        assert spec.intervals[:3] == (dt(2021, 4, 1), dt(2021, 4, 8), dt(2021, 4, 15))
        assert spec.intervals[-3:] == (dt(2021, 6, 17), dt(2021, 6, 24), dt(2021, 7, 1))

    def test_from_timespan_not_regular(self) -> None:
        spec = GoalTimeseriesSpec.from_timespan(dt(2022, 6, 23), dt(2022, 8, 12))

        assert spec.granularity == GoalSeriesGranularity.WEEK.value
        assert len(spec.intervals) == 9
        assert spec.intervals[:3] == (dt(2022, 6, 23), dt(2022, 6, 30), dt(2022, 7, 7))
        assert spec.intervals[-3:] == (dt(2022, 8, 4), dt(2022, 8, 11), dt(2022, 8, 12))
