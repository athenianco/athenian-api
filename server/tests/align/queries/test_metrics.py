from datetime import date
from functools import partial
from typing import Any, Optional

from freezegun import freeze_time
import pytest
import sqlalchemy as sa

from athenian.api.align.models import MetricParamsFields
from athenian.api.align.queries.metrics import (
    RequestedTeamDetails,
    TeamMetricsRequest,
    _simplify_requests,
)
from athenian.api.db import Database
from athenian.api.models.state.models import AccountJiraInstallation
from athenian.api.models.web import JIRAMetricID, PullRequestMetricID, ReleaseMetricID
from tests.align.utils import (
    align_graphql_request,
    assert_extension_error,
    build_recursive_fields_structure,
)
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.state import MappedJIRAIdentityFactory, TeamFactory
from tests.testutils.requester import Requester
from tests.testutils.time import dt


class BaseMetricsTest(Requester):
    def _query(self, depth=4):
        fragment = """
            fragment teamMetricValueFields on TeamMetricValue {
              team {
                id
              }
              value {
                str
                int
                float
              }
            }
        """

        recursive_fields = build_recursive_fields_structure(["...teamMetricValueFields"], depth)

        return (
            fragment
            + f"""
            query ($accountId: Int!,
                   $teamId: Int!,
                   $metrics: [String!],
                   $metricsWithParams: [MetricWithParamsInput!],
                   $validFrom: Date!,
                   $expiresAt: Date!,
                   $repositories: [String!],
                   $jiraPriorities: [String!],
                   $jiraProjects: [String!],
                   $jiraIssueTypes: [String!],
            ) {{
              metricsCurrentValues(accountId: $accountId, params: {{
                teamId: $teamId,
                metrics: $metrics,
                metricsWithParams: $metricsWithParams,
                validFrom: $validFrom,
                expiresAt: $expiresAt,
                repositories: $repositories,
                jiraPriorities: $jiraPriorities
                jiraProjects: $jiraProjects,
                jiraIssueTypes: $jiraIssueTypes,
              }}) {{
                metric
                value {{
                  {recursive_fields}
                }}
              }}
            }}
        """
        )

    async def _request(
        self,
        account_id: int,
        team_id: int,
        metrics: Optional[list[str]],
        validFrom: date,
        expiresAt: date,
        **extra: Any,  # other MetricParamsFields
    ) -> dict:
        body = {
            "query": self._query(),
            "variables": {
                "accountId": account_id,
                MetricParamsFields.teamId: team_id,
                MetricParamsFields.metrics: metrics,
                MetricParamsFields.validFrom: str(validFrom),
                MetricParamsFields.expiresAt: str(expiresAt),
                **extra,
            },
        }
        return await align_graphql_request(self.client, headers=self.headers, json=body)


@pytest.fixture(scope="function")
async def sample_teams(sdb: Database) -> None:
    await models_insert(
        sdb,
        TeamFactory(id=1, members=[40020, 39789, 40070]),
        TeamFactory(id=2, parent_id=1, members=[40191, 39926, 40418]),
        MappedJIRAIdentityFactory(github_user_id=40020, jira_user_id="5de5049e2c5dd20d0f9040c1"),
        MappedJIRAIdentityFactory(github_user_id=39789, jira_user_id="5dd58cb9c7ac480ee5674902"),
        MappedJIRAIdentityFactory(github_user_id=40191, jira_user_id="5ddec0b9be6c1f0d071ff82d"),
    )


class TestMetrics(BaseMetricsTest):
    async def test_fetch_all_kinds(self, sample_teams, mdb, precomputed_dead_prs) -> None:
        metrics = [
            PullRequestMetricID.PR_LEAD_TIME,
            ReleaseMetricID.RELEASE_PRS,
            JIRAMetricID.JIRA_RESOLVED,
        ]
        res = await self._request(1, 1, metrics, date(2016, 1, 1), date(2019, 1, 1))
        assert res == {
            "data": {
                "metricsCurrentValues": [
                    {
                        "metric": "pr-lead-time",
                        "value": {
                            "team": {"id": 1},
                            "value": {"str": "2939453s", "int": None, "float": None},
                            "children": [
                                {
                                    "team": {"id": 2},
                                    "value": {"str": "4337530s", "int": None, "float": None},
                                    "children": [],
                                },
                            ],
                        },
                    },
                    {
                        "metric": "release-prs",
                        "value": {
                            "team": {"id": 1},
                            "value": {"str": None, "int": 474, "float": None},
                            "children": [
                                {
                                    "team": {"id": 2},
                                    "value": {"str": None, "int": 382, "float": None},
                                    "children": [],
                                },
                            ],
                        },
                    },
                    {
                        "metric": "jira-resolved",
                        "value": {
                            "team": {
                                "id": 1,
                            },
                            "value": {"str": None, "int": 0, "float": None},
                            "children": [
                                {
                                    "team": {"id": 2},
                                    "value": {"str": None, "int": 0, "float": None},
                                    "children": [],
                                },
                            ],
                        },
                    },
                ],
            },
        }

    @pytest.mark.parametrize(
        "metric, value",
        [
            ("pr-lead-time", {"str": "2939453s", "children": [{"str": "4337530s"}]}),
            ("release-prs", {"int": 454, "children": [{"int": 370}]}),
            ("jira-resolved", {"int": 0, "children": [{"int": 0}]}),
        ],
    )
    async def test_fetch_one_kind(
        self,
        sample_teams,
        metric: str,
        value: dict[str, Any],
    ) -> None:
        res = await self._request(
            1,
            1,
            [metric],
            date(2016, 1, 1),
            date(2019, 1, 1),
            repositories=["github.com/src-d/go-git"],
        )
        assert len(mv := res["data"]["metricsCurrentValues"]) == 1
        assert mv[0]["metric"] == metric

        def validate_recursively(yours: dict[str, Any], mine: dict[str, Any]) -> None:
            for key, val in mine.items():
                if isinstance(val, list):
                    for yours_sub, mine_sub in zip(yours[key], val):
                        validate_recursively(yours_sub, mine_sub)
                else:
                    assert yours["value"][key] == val

        validate_recursively(mv[0]["value"], value)

    async def test_fetch_two(self, sample_teams):
        metrics = [JIRAMetricID.JIRA_RESOLVED, JIRAMetricID.JIRA_RESOLUTION_RATE]
        res = await self._request(1, 1, metrics, date(2019, 1, 1), date(2022, 1, 1))
        assert res == {
            "data": {
                "metricsCurrentValues": [
                    {
                        "metric": "jira-resolved",
                        "value": {
                            "team": {"id": 1},
                            "value": {"str": None, "int": 738, "float": None},
                            "children": [
                                {
                                    "team": {"id": 2},
                                    "value": {"str": None, "int": 163, "float": None},
                                    "children": [],
                                },
                            ],
                        },
                    },
                    {
                        "metric": "jira-resolution-rate",
                        "value": {
                            "team": {"id": 1},
                            "value": {"str": None, "int": None, "float": 0.9473684430122375},
                            "children": [
                                {
                                    "team": {"id": 2},
                                    "value": {
                                        "str": None,
                                        "int": None,
                                        "float": 0.8624338507652283,
                                    },
                                    "children": [],
                                },
                            ],
                        },
                    },
                ],
            },
        }

    async def test_missing_values_for_subteam(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[39789, 40070]),
            TeamFactory(id=2, parent_id=1, members=[39926]),
        )
        res = await self._request(
            1, 1, [PullRequestMetricID.PR_REVIEW_TIME], date(2005, 1, 1), date(2005, 3, 31),
        )
        pr_review_time_data = res["data"]["metricsCurrentValues"][0]
        assert pr_review_time_data["metric"] == PullRequestMetricID.PR_REVIEW_TIME
        team_1_data = pr_review_time_data["value"]
        assert team_1_data["team"]["id"] == 1
        assert team_1_data["value"] == {"float": None, "int": None, "str": None}

        assert len(team_1_data["children"]) == 1
        assert (team_2_data := team_1_data["children"][0])["team"]["id"] == 2
        assert team_2_data["value"] == {"float": None, "int": None, "str": None}

    async def test_repositories_param(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        metrics = [PullRequestMetricID.PR_ALL_COUNT]

        dates = date(2016, 1, 1), date(2019, 1, 1)

        res = await self._request(1, 1, metrics, *dates)
        value = res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"]
        assert value == 93

        res = await self._request(1, 1, metrics, *dates, repositories=["github.com/src-d/go-git"])
        value = res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"]
        # all data in big fixture is about go-git
        assert value == 93

        res = await self._request(1, 1, metrics, *dates, repositories=["github.com/src-d/gitbase"])
        value = res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"]
        assert value == 0

    async def test_jira_not_installed(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        await sdb.execute(sa.delete(AccountJiraInstallation))
        metrics = [PullRequestMetricID.PR_ALL_COUNT]

        dates = date(2016, 1, 1), date(2019, 1, 1)

        res = await self._request(1, 1, metrics, *dates)
        value = res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"]
        assert value == 93


class TestJIRAFiltering(BaseMetricsTest):
    async def test_pr_metric(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        metrics = [PullRequestMetricID.PR_ALL_COUNT]
        dates = (date(2016, 1, 1), date(2019, 1, 1))

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.JIRAProjectFactory(id="0", key="P0"),
                md_factory.JIRAProjectFactory(id="1", key="P1"),
                md_factory.JIRAPriorityFactory(id="100", name="extreme"),
                md_factory.JIRAPriorityFactory(id="101", name="medium"),
                md_factory.JIRAIssueTypeFactory(id="100", name="t0"),
                md_factory.JIRAIssueTypeFactory(id="101", name="t1"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162901, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162901, jira_id="21"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162907, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162908, jira_id="20"),
                md_factory.JIRAIssueFactory(
                    id="20",
                    project_id="0",
                    type="t0",
                    type_id="100",
                    priority_id="100",
                    priority_name="extreme",
                ),
                md_factory.JIRAIssueFactory(
                    id="21",
                    project_id="1",
                    type="t1",
                    type_id="101",
                    priority_id="101",
                    priority_name="medium",
                ),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            request = partial(self._request, 1, 1, metrics, *dates)

            res = await request(jiraPriorities=["extreme", "verylow"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 3

            res = await request(jiraPriorities=["verylow"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 0

            res = await request(jiraPriorities=["extreme"], jiraProjects=["P0"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 3

            res = await request(jiraPriorities=["extreme"], jiraProjects=["P1"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 0

            res = await request(jiraPriorities=["medium"], jiraProjects=["P0"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 0

            res = await request(jiraPriorities=["medium"], jiraProjects=["P1"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 1

            res = await request(jiraProjects=["P1"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 1

            res = await request(jiraIssueTypes=["T0"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 3

            res = await request(jiraIssueTypes=["T1"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 1

            res = await request(jiraIssueTypes=["t0"], jiraProjects=["P0"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 3

            res = await request(jiraIssueTypes=["T0"], jiraProjects=["P1"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 0

            res = await request(jiraProjects=["P2"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 0

    async def test_release_metric(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        metrics = [ReleaseMetricID.RELEASE_PRS]
        dates = (date(2016, 1, 1), date(2019, 1, 1))

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162901, jira_id="20"),
                md_factory.JIRAIssueFactory(
                    id="20", project_id="0", type_id="0", type="t0", priority_name="extreme",
                ),
                md_factory.JIRAProjectFactory(id="0", key="P0"),
                md_factory.JIRAIssueTypeFactory(id="0", name="t0"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            request = partial(self._request, 1, 1, metrics, *dates)

            res = await request()
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 450

            res = await request(jiraIssueTypes=["T99"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 0

            res = await request(jiraIssueTypes=["T0"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 71

            res = await request(jiraProjects=["P0"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 71

    async def test_jira_metric(self, sdb: Database, mdb_rw: Database) -> None:
        metrics = [JIRAMetricID.JIRA_OPEN]
        dates = (date(2019, 1, 1), date(2022, 1, 1))

        await models_insert(
            sdb,
            TeamFactory(id=1, members=[40020, 39789]),
            MappedJIRAIdentityFactory(
                github_user_id=40020, jira_user_id="5de5049e2c5dd20d0f9040c1",
            ),
            MappedJIRAIdentityFactory(
                github_user_id=39789, jira_user_id="5dd58cb9c7ac480ee5674902",
            ),
        )
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162901, jira_id="20"),
                md_factory.JIRAIssueFactory(id="20", priority_name="extreme"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)
            request = partial(self._request, 1, 1, metrics, *dates)

            res = await request()
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 15

            res = await request(jiraPriorities=["foo"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 0

            res = await request(jiraPriorities=["extreme"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 0

    async def test_jira_fields_normalization(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        metrics = [PullRequestMetricID.PR_ALL_COUNT]
        dates = (date(2016, 1, 1), date(2019, 1, 1))

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162901, jira_id="20"),
                md_factory.JIRAIssueTypeFactory(id="100", name="t0"),
                md_factory.JIRAPriorityFactory(id="100", name="VeryHigh"),
                md_factory.JIRAIssueFactory(
                    id="20",
                    type_id="100",
                    type="t0",
                    priority_id="100",
                    priority_name="VeryHigh",
                    project_id="0",
                ),
                md_factory.JIRAProjectFactory(id="0"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            request = partial(self._request, 1, 1, metrics, *dates)

            res = await request(jiraPriorities=["verYhigh", "verylow"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 1

            res = await request(jiraIssueTypes=["T0"])
            assert res["data"]["metricsCurrentValues"][0]["value"]["value"]["int"] == 1


class TestMetricsNasty(BaseMetricsTest):
    async def test_fetch_bad_team(self) -> None:
        res = await self._request(
            1,
            1,
            [JIRAMetricID.JIRA_RESOLVED],
            date(2019, 1, 1),
            date(2022, 1, 1),
        )
        assert res == {
            "errors": [
                {
                    "message": "Team not found",
                    "locations": [{"line": 24, "column": 15}],
                    "path": ["metricsCurrentValues"],
                    "extensions": {
                        "status": 404,
                        "type": "/errors/teams/TeamNotFound",
                        "detail": "Team 1 not found or access denied",
                    },
                },
            ],
        }

    async def test_fetch_bad_account(self, sample_teams) -> None:
        res = await self._request(
            2,
            1,
            [JIRAMetricID.JIRA_RESOLVED],
            date(2019, 1, 1),
            date(2022, 1, 1),
        )
        assert res == {
            "errors": [
                {
                    "message": "Team not found",
                    "locations": [{"line": 24, "column": 15}],
                    "path": ["metricsCurrentValues"],
                    "extensions": {
                        "status": 404,
                        "type": "/errors/teams/TeamNotFound",
                        "detail": "Team 1 not found or access denied",
                    },
                },
            ],
        }

    async def test_fetch_bad_metric(self, sample_teams) -> None:
        res = await self._request(
            1,
            1,
            ["whatever"],
            date(2019, 1, 1),
            date(2022, 1, 1),
        )
        assert res == {
            "errors": [
                {
                    "message": "Bad Request",
                    "locations": [{"line": 24, "column": 15}],
                    "path": ["metricsCurrentValues"],
                    "extensions": {
                        "status": 400,
                        "type": "/errors/InvalidRequestError",
                        "detail": "The following metrics are not supported: whatever",
                    },
                },
            ],
        }

    async def test_fetch_bad_dates_order(self, sample_teams) -> None:
        res = await self._request(
            1,
            1,
            [JIRAMetricID.JIRA_RESOLVED],
            date(2022, 1, 1),
            date(2019, 1, 1),
        )
        assert res == {
            "errors": [
                {
                    "message": "Bad Request",
                    "locations": [{"line": 24, "column": 15}],
                    "path": ["metricsCurrentValues"],
                    "extensions": {
                        "status": 400,
                        "type": "/errors/InvalidRequestError",
                        "detail": "validFrom must be less than or equal to expiresAt",
                    },
                },
            ],
        }

    async def test_dates_far_past(self, sample_teams) -> None:
        res = await self._request(
            1,
            1,
            [JIRAMetricID.JIRA_RESOLVED],
            date(1984, 1, 1),
            date(2002, 1, 1),
        )
        assert res["errors"]
        assert res.get("data") is None

    @freeze_time("2022-03-31")
    async def test_valid_from_in_the_future(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[39789]))

        res = await self._request(
            1,
            1,
            [PullRequestMetricID.PR_RELEASE_COUNT],
            date(2022, 4, 1),
            date(2022, 4, 10),
        )
        assert_extension_error(res, "validFrom cannot be in the future")
        assert res.get("data") is None

    async def test_fetch_invalid_metric(self, sample_teams) -> None:
        res = await self._request(
            1,
            1,
            ["whatever"],
            date(2019, 1, 1),
            date(2022, 1, 1),
            repositories=["github.com/src-d/go-git"],
        )
        assert res["errors"][0]["message"] == "Bad Request"
        assert_extension_error(res, "The following metrics are not supported: whatever")

    async def test_fetch_bad_repository(self, sample_teams) -> None:
        res = await self._request(
            1,
            1,
            [PullRequestMetricID.PR_ALL_COUNT],
            date(2019, 1, 1),
            date(2022, 1, 1),
            repositories=["github.com/src-d/not-existing"],
        )
        assert res["errors"][0]["message"] == "Forbidden"
        assert_extension_error(
            res,
            'The following repositories are access denied for account 1 (missing "github.com/"'
            " prefix?): {'src-d/not-existing'}",
        )

    async def test_neither_metrics_or_metric_with_params(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        res = await self._request(1, 1, None, date(2016, 1, 1), date(2017, 1, 1))
        assert res["errors"][0]["message"] == "Bad Request"
        assert_extension_error(res, "Use metrics or metricsWithParams")

    async def test_both_metrics_and_metric_with_params(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        with_params = [{"name": PullRequestMetricID.PR_LEAD_COUNT}]
        dates = (date(2016, 1, 1), date(2017, 1, 1))
        res = await self._request(
            1, 1, [PullRequestMetricID.PR_OPEN_COUNT], *dates, metricsWithParams=with_params,
        )
        assert res["errors"][0]["message"] == "Bad Request"
        assert_extension_error(res, "Use metrics or metricsWithParams")


class TestMetricsWithParams(BaseMetricsTest):
    async def test_base(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        metric = PullRequestMetricID.PR_REVIEW_COMMENTS_PER_ABOVE_THRESHOLD_RATIO
        request = partial(self._request, 1, 1, None, date(2016, 1, 1), date(2017, 1, 1))

        metrics_w_params = [{"name": metric, "metricParams": {"threshold": {"int": 5}}}]
        res = await request(metricsWithParams=metrics_w_params)
        val_threshold_5 = res["data"]["metricsCurrentValues"][0]["value"]["value"]["float"]

        metrics_w_params[0]["metricParams"]["threshold"]["int"] = 8
        res = await request(metricsWithParams=metrics_w_params)
        val_threshold_8 = res["data"]["metricsCurrentValues"][0]["value"]["value"]["float"]

        assert 0 < val_threshold_8 < val_threshold_5 < 1

    async def test_teams_overrides(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[40020, 39789]),
            TeamFactory(id=2, parent_id=1, members=[40020, 39789]),
        )
        dates = (date(2016, 1, 1), date(2017, 1, 1))
        metric = PullRequestMetricID.PR_REVIEW_TIME_BELOW_THRESHOLD_RATIO

        metrics_w_params = [
            {
                "name": metric,
                "metricParams": {"threshold": {"str": "172800s"}},
                "teamsMetricParams": [
                    {"teamId": 2, "metricParams": {"threshold": {"str": "123000s"}}},
                ],
            },
        ]
        res = await self._request(1, 1, None, *dates, metricsWithParams=metrics_w_params)
        metric_values = res["data"]["metricsCurrentValues"][0]["value"]

        # teams are the same, but team 2 is computed with a lower threshold due to
        # teamsMetricParams so will have a lower metric value
        assert metric_values["team"]["id"] == 1
        assert metric_values["value"]["float"] == pytest.approx(0.6923077)

        assert metric_values["children"][0]["team"]["id"] == 2
        assert metric_values["children"][0]["value"]["float"] == pytest.approx(0.653846)


class TestSimplifyRequests:
    """Tests for the private function _simplify_requests."""

    def test_single_request(self) -> None:
        requests = [
            TeamMetricsRequest(
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
            TeamMetricsRequest(
                (PullRequestMetricID.PR_CLOSED,),
                INTERVALS,
                [
                    RequestedTeamDetails(1, 0, [1]),
                    RequestedTeamDetails(2, 0, [10, 2]),
                ],
            ),
            TeamMetricsRequest(
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
        expected = TeamMetricsRequest(
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
            TeamMetricsRequest(
                (PullRequestMetricID.PR_CLOSED,),
                INTERVALS,
                {RequestedTeamDetails(1, 0, [10])},
            ),
            TeamMetricsRequest(
                (PullRequestMetricID.PR_RELEASE_TIME,),
                INTERVALS,
                {
                    RequestedTeamDetails(1, 0, [10]),
                    RequestedTeamDetails(2, 0, [10, 20]),
                },
            ),
            TeamMetricsRequest(
                (PullRequestMetricID.PR_REVIEW_COUNT,),
                INTERVALS,
                {RequestedTeamDetails(2, 0, [10, 20])},
            ),
        ]

        simplified = sorted(
            _simplify_requests(requests), key=lambda r: {t.team_id for t in r.teams} == {2},
        )
        expected = [
            TeamMetricsRequest(
                (PullRequestMetricID.PR_RELEASE_TIME, PullRequestMetricID.PR_CLOSED),
                INTERVALS,
                {RequestedTeamDetails(1, 0, [10])},
            ),
            TeamMetricsRequest(
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
            TeamMetricsRequest(
                (PullRequestMetricID.PR_CLOSED,),
                INTERVALS_0,
                {RequestedTeamDetails(1, 0, [1])},
            ),
            TeamMetricsRequest(
                (PullRequestMetricID.PR_OPENED,),
                INTERVALS_1,
                {RequestedTeamDetails(1, 0, [1])},
            ),
            TeamMetricsRequest(
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
            TeamMetricsRequest(
                (PullRequestMetricID.PR_CLOSED,),
                INTERVALS_0,
                {
                    RequestedTeamDetails(1, 0, [1]),
                    RequestedTeamDetails(2, 0, [2]),
                },
            ),
            TeamMetricsRequest(
                (PullRequestMetricID.PR_OPENED,),
                INTERVALS_1,
                {RequestedTeamDetails(1, 0, [1])},
            ),
        ]
        self._assert_team_requests_equal(simplified[0], expected[0])
        self._assert_team_requests_equal(simplified[1], expected[1])

    @classmethod
    def _assert_team_requests_equal(cls, tr0: TeamMetricsRequest, tr1: TeamMetricsRequest) -> None:
        assert sorted(tr0.metrics) == sorted(tr1.metrics)
        assert sorted(tr0.time_intervals) == sorted(tr0.time_intervals)
        assert tr0.teams == tr1.teams
