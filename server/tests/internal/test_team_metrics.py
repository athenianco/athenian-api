from athenian.api.internal.team_metrics import (
    CalcTeamMetricsRequest,
    RequestedTeamDetails,
    _simplify_requests,
)
from athenian.api.models.web import PullRequestMetricID
from tests.testutils.time import dt


class TestSimplifyRequests:
    """Tests for the private function _simplify_requests."""

    def test_single_request(self) -> None:
        requests = [
            CalcTeamMetricsRequest(
                (PullRequestMetricID.PR_CLOSED,),
                ((dt(2001, 1, 1), dt(2001, 2, 1)),),
                {RequestedTeamDetails(1, 0, [10]), RequestedTeamDetails(2, 0, [10, 20])},
            ),
        ]
        simplified = _simplify_requests(requests)
        assert simplified == requests

    def test_metrics_merged(self) -> None:
        INTERVALS = ((dt(2001, 1, 1), dt(2001, 2, 1)),)
        requests = [
            CalcTeamMetricsRequest(
                (PullRequestMetricID.PR_CLOSED,),
                INTERVALS,
                [
                    RequestedTeamDetails(1, 0, [1]),
                    RequestedTeamDetails(2, 0, [10, 2]),
                ],
            ),
            CalcTeamMetricsRequest(
                (PullRequestMetricID.PR_RELEASE_TIME,),
                INTERVALS,
                [
                    RequestedTeamDetails(1, 0, [1]),
                    RequestedTeamDetails(2, 0, [10, 2]),
                ],
            ),
        ]
        simplified = _simplify_requests(requests)
        assert len(simplified) == 1
        expected = CalcTeamMetricsRequest(
            (PullRequestMetricID.PR_CLOSED, PullRequestMetricID.PR_RELEASE_TIME),
            INTERVALS,
            {
                RequestedTeamDetails(1, 0, [1]),
                RequestedTeamDetails(2, 0, [10, 2]),
            },
        )

        self._assert_team_requests_equal(simplified[0], expected)

    def test_teams_merged(self) -> None:
        INTERVALS = ((dt(2001, 1, 1), dt(2001, 2, 1)),)
        requests = [
            CalcTeamMetricsRequest(
                (PullRequestMetricID.PR_CLOSED,),
                INTERVALS,
                {RequestedTeamDetails(1, 0, [10])},
            ),
            CalcTeamMetricsRequest(
                (PullRequestMetricID.PR_RELEASE_TIME,),
                INTERVALS,
                {
                    RequestedTeamDetails(1, 0, [10]),
                    RequestedTeamDetails(2, 0, [10, 20]),
                },
            ),
            CalcTeamMetricsRequest(
                (PullRequestMetricID.PR_REVIEW_COUNT,),
                INTERVALS,
                {RequestedTeamDetails(2, 0, [10, 20])},
            ),
        ]

        simplified = sorted(
            _simplify_requests(requests), key=lambda r: {t.team_id for t in r.teams} == {2},
        )
        expected = [
            CalcTeamMetricsRequest(
                (PullRequestMetricID.PR_RELEASE_TIME, PullRequestMetricID.PR_CLOSED),
                INTERVALS,
                {RequestedTeamDetails(1, 0, [10])},
            ),
            CalcTeamMetricsRequest(
                (PullRequestMetricID.PR_RELEASE_TIME, PullRequestMetricID.PR_REVIEW_COUNT),
                INTERVALS,
                {RequestedTeamDetails(2, 0, [10, 20])},
            ),
        ]

        assert len(simplified) == 2
        self._assert_team_requests_equal(simplified[0], expected[0])
        self._assert_team_requests_equal(simplified[1], expected[1])

    def test_different_intervals_are_not_merged(self) -> None:
        INTERVALS_0 = ((dt(2001, 1, 1), dt(2001, 2, 1)),)
        INTERVALS_1 = ((dt(2011, 1, 1), dt(2021, 2, 1)),)
        requests = [
            CalcTeamMetricsRequest(
                (PullRequestMetricID.PR_CLOSED,),
                INTERVALS_0,
                {RequestedTeamDetails(1, 0, [1])},
            ),
            CalcTeamMetricsRequest(
                (PullRequestMetricID.PR_OPENED,),
                INTERVALS_1,
                {RequestedTeamDetails(1, 0, [1])},
            ),
            CalcTeamMetricsRequest(
                (PullRequestMetricID.PR_CLOSED,),
                INTERVALS_0,
                {RequestedTeamDetails(2, 0, [2])},
            ),
        ]
        simplified = sorted(
            _simplify_requests(requests), key=lambda r: {t.team_id for t in r.teams} == {2},
        )
        assert len(simplified) == 2

        expected = [
            CalcTeamMetricsRequest(
                (PullRequestMetricID.PR_CLOSED,),
                INTERVALS_0,
                {
                    RequestedTeamDetails(1, 0, [1]),
                    RequestedTeamDetails(2, 0, [2]),
                },
            ),
            CalcTeamMetricsRequest(
                (PullRequestMetricID.PR_OPENED,),
                INTERVALS_1,
                {RequestedTeamDetails(1, 0, [1])},
            ),
        ]
        self._assert_team_requests_equal(simplified[0], expected[0])
        self._assert_team_requests_equal(simplified[1], expected[1])

    @classmethod
    def _assert_team_requests_equal(
        cls,
        tr0: CalcTeamMetricsRequest,
        tr1: CalcTeamMetricsRequest,
    ) -> None:
        assert sorted(tr0.metrics) == sorted(tr1.metrics)
        assert sorted(tr0.time_intervals) == sorted(tr0.time_intervals)
        assert tr0.teams == tr1.teams
