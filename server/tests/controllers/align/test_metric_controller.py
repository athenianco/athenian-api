from datetime import date
from functools import partial
from typing import Any, Optional, Sequence

import pytest
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.models.state.models import AccountJiraInstallation
from athenian.api.models.web import (
    JIRAMetricID,
    PullRequestMetricID,
    ReleaseMetricID,
    TeamMetricsRequest,
    TeamMetricWithParams,
)
from athenian.api.models.web.team_metrics import _TeamMetricParams
from tests.testutils.auth import force_request_auth
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.state import (
    MappedJIRAIdentityFactory,
    TeamFactory,
    UserAccountFactory,
)
from tests.testutils.requester import Requester
from tests.testutils.time import freeze_time


class BaseTeamMetricsTest(Requester):
    async def _request(
        self,
        assert_status: int = 200,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        path = "/private/align/team_metrics"
        with force_request_auth(user_id, self.headers) as headers:
            response = await self.client.request(
                method="POST", path=path, headers=headers, **kwargs,
            )
        assert response.status == assert_status
        return await response.json()

    @classmethod
    def _body(
        cls,
        team: int,
        metrics_with_params: Sequence[tuple | dict],
        valid_from: date,
        expires_at: date,
        **kwargs: Any,
    ) -> dict:
        request = TeamMetricsRequest(
            team=team,
            metrics_with_params=cls._conv_metrics_w_params(metrics_with_params),
            valid_from=valid_from,
            expires_at=expires_at,
            **kwargs,
        )
        body = request.to_dict()
        body["valid_from"] = body["valid_from"].isoformat()
        body["expires_at"] = body["expires_at"].isoformat()
        return body

    @classmethod
    def _conv_metrics_w_params(
        cls,
        raw_metrics_with_params: Sequence[tuple | dict],
    ) -> Sequence[TeamMetricWithParams]:
        converted = []
        for raw in raw_metrics_with_params:
            if isinstance(raw, dict):
                converted.append(raw)
                continue
            name, *rest = raw
            metric_params = None
            teams_metric_params = None
            if rest:
                metric_params, *rest = rest
                if rest:
                    teams_metric_params = [
                        _TeamMetricParams(team=t[0], metric_params=t[1]) for t in rest[0]
                    ]
            converted.append(
                TeamMetricWithParams(
                    name=name,
                    metric_params=metric_params,
                    teams_metric_params=teams_metric_params,
                ),
            )
        return converted


class TestTeamMetrics(BaseTeamMetricsTest):
    async def test_fetch_all_kinds(self, sample_teams, mdb, precomputed_dead_prs) -> None:
        metrics_with_params = [
            (PullRequestMetricID.PR_LEAD_TIME,),
            (ReleaseMetricID.RELEASE_PRS,),
            (JIRAMetricID.JIRA_RESOLVED,),
        ]
        body = self._body(
            1, metrics_with_params, valid_from=date(2016, 1, 1), expires_at=date(2019, 1, 1),
        )
        res = await self._request(json=body)
        assert res == [
            {
                "metric": "pr-lead-time",
                "value": {
                    "team": {"id": 1, "name": "T1"},
                    "value": "2939453s",
                    "children": [
                        {"team": {"id": 2, "name": "T2"}, "value": "4337530s", "children": []},
                    ],
                },
            },
            {
                "metric": "release-prs",
                "value": {
                    "team": {"id": 1, "name": "T1"},
                    "value": 474,
                    "children": [{"team": {"id": 2, "name": "T2"}, "value": 382, "children": []}],
                },
            },
            {
                "metric": "jira-resolved",
                "value": {
                    "team": {"id": 1, "name": "T1"},
                    "value": 0,
                    "children": [{"team": {"id": 2, "name": "T2"}, "value": 0, "children": []}],
                },
            },
        ]

    @pytest.mark.parametrize(
        "metric, value",
        [
            ("pr-lead-time", {"value": "2939453s", "children": [{"value": "4337530s"}]}),
            ("release-prs", {"value": 454, "children": [{"value": 370}]}),
            ("jira-resolved", {"value": 0, "children": [{"value": 0}]}),
        ],
    )
    async def test_fetch_one_kind(
        self,
        sample_teams,
        metric: str,
        value: dict[str, Any],
    ) -> None:
        body = self._body(
            1,
            [(metric,)],
            valid_from=date(2016, 1, 1),
            expires_at=date(2019, 1, 1),
            repositories=["github.com/src-d/go-git"],
        )
        res = await self._request(json=body)
        assert len(res) == 1
        assert res[0]["metric"] == metric

        def validate_recursively(yours: dict[str, Any], mine: dict[str, Any]) -> None:
            for key, val in mine.items():
                if isinstance(val, list):
                    for yours_sub, mine_sub in zip(yours[key], val):
                        validate_recursively(yours_sub, mine_sub)
                else:
                    assert yours["value"] == val

        validate_recursively(res[0]["value"], value)

    async def test_fetch_two(self, sample_teams):
        metrics = [(JIRAMetricID.JIRA_RESOLVED,), (JIRAMetricID.JIRA_RESOLUTION_RATE,)]
        body = self._body(1, metrics, date(2019, 1, 1), date(2022, 1, 1))
        res = await self._request(json=body)
        assert res == [
            {
                "metric": "jira-resolved",
                "value": {
                    "team": {"id": 1, "name": "T1"},
                    "value": 738,
                    "children": [{"team": {"id": 2, "name": "T2"}, "value": 163, "children": []}],
                },
            },
            {
                "metric": "jira-resolution-rate",
                "value": {
                    "team": {"id": 1, "name": "T1"},
                    "value": 0.9473684430122375,
                    "children": [
                        {
                            "team": {"id": 2, "name": "T2"},
                            "value": 0.8624338507652283,
                            "children": [],
                        },
                    ],
                },
            },
        ]

    async def test_missing_values_for_subteam(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[39789, 40070]),
            TeamFactory(id=2, name="T2", parent_id=1, members=[39926]),
        )
        body = self._body(
            1, [(PullRequestMetricID.PR_REVIEW_TIME,)], date(2005, 1, 1), date(2005, 3, 31),
        )
        res = await self._request(json=body)
        pr_review_time_data = res[0]  # ["data"]["metricsCurrentValues"][0]
        assert pr_review_time_data["metric"] == PullRequestMetricID.PR_REVIEW_TIME
        team_1_data = pr_review_time_data["value"]
        assert team_1_data["team"]["id"] == 1
        assert team_1_data["value"] is None

        assert len(team_1_data["children"]) == 1
        assert (team_2_data := team_1_data["children"][0])["team"] == {"id": 2, "name": "T2"}
        assert team_2_data["value"] is None

    async def test_repositories_param(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        metrics = [(PullRequestMetricID.PR_ALL_COUNT,)]
        dates = date(2016, 1, 1), date(2019, 1, 1)
        body = self._body(1, metrics, *dates)
        res = await self._request(json=body)
        assert res[0]["value"]["value"] == 93

        body = self._body(1, metrics, *dates, repositories=["github.com/src-d/go-git"])
        res = await self._request(json=body)
        # all data in big fixture is about go-git
        assert res[0]["value"]["value"] == 93

        body = self._body(1, metrics, *dates, repositories=["github.com/src-d/gitbase"])
        res = await self._request(json=body)
        assert res[0]["value"]["value"] == 0

    async def test_jira_not_installed(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        await sdb.execute(sa.delete(AccountJiraInstallation))
        metrics = [(PullRequestMetricID.PR_ALL_COUNT,)]
        body = self._body(1, metrics, date(2016, 1, 1), date(2019, 1, 1))
        res = await self._request(json=body)
        assert res[0]["value"]["value"] == 93


class TestJIRAFiltering(BaseTeamMetricsTest):
    async def test_pr_metric(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        metrics = [(PullRequestMetricID.PR_ALL_COUNT,)]
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

            body = partial(self._body, 1, metrics, *dates)

            res = await self._request(json=body(jira_priorities=["extreme", "verylow"]))
            assert res[0]["value"]["value"] == 3

            res = await self._request(json=body(jira_priorities=["verylow"]))
            assert res[0]["value"]["value"] == 0

            res = await self._request(json=body(jira_priorities=["extreme"], jira_projects=["P0"]))
            assert res[0]["value"]["value"] == 3

            res = await self._request(json=body(jira_priorities=["extreme"], jira_projects=["P1"]))
            assert res[0]["value"]["value"] == 0

            res = await self._request(json=body(jira_priorities=["medium"], jira_projects=["P0"]))
            assert res[0]["value"]["value"] == 0

            res = await self._request(json=body(jira_priorities=["medium"], jira_projects=["P1"]))
            assert res[0]["value"]["value"] == 1

            res = await self._request(json=body(jira_projects=["P1"]))
            assert res[0]["value"]["value"] == 1

            res = await self._request(json=body(jira_issue_types=["T0"]))
            assert res[0]["value"]["value"] == 3

            res = await self._request(json=body(jira_issue_types=["T1"]))
            assert res[0]["value"]["value"] == 1

            res = await self._request(json=body(jira_issue_types=["t0"], jira_projects=["P0"]))
            assert res[0]["value"]["value"] == 3

            res = await self._request(json=body(jira_issue_types=["T0"], jira_projects=["P1"]))
            assert res[0]["value"]["value"] == 0

            res = await self._request(json=body(jira_projects=["P2"]))
            assert res[0]["value"]["value"] == 0

    async def test_release_metric(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        metrics = [(ReleaseMetricID.RELEASE_PRS,)]
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

            body = partial(self._body, 1, metrics, *dates)

            res = await self._request(json=body())
            assert res[0]["value"]["value"] == 450

            res = await self._request(json=body(jira_issue_types=["T99"]))
            assert res[0]["value"]["value"] == 0

            res = await self._request(json=body(jira_issue_types=["T0"]))
            assert res[0]["value"]["value"] == 71

            res = await self._request(json=body(jira_projects=["P0"]))
            assert res[0]["value"]["value"] == 71

    async def test_jira_metric(self, sdb: Database, mdb_rw: Database) -> None:
        metrics = [(JIRAMetricID.JIRA_OPEN,)]
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
            body = partial(self._body, 1, metrics, *dates)

            res = await self._request(json=body())
            assert res[0]["value"]["value"] == 15

            res = await self._request(json=body(jira_priorities=["foo"]))
            assert res[0]["value"]["value"] == 0

            res = await self._request(json=body(jira_priorities=["extreme"]))
            assert res[0]["value"]["value"] == 0

    async def test_jira_fields_normalization(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        metrics = [(PullRequestMetricID.PR_ALL_COUNT,)]
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

            body = partial(self._body, 1, metrics, *dates)
            res = await self._request(json=body(jira_priorities=["verYhigh", "verylow"]))
            assert res[0]["value"]["value"] == 1

            res = await self._request(json=body(jira_issue_types=["T0"]))
            assert res[0]["value"]["value"] == 1


class TestMetricsNasty(BaseTeamMetricsTest):
    async def test_fetch_bad_team(self) -> None:
        body = self._body(1, [(JIRAMetricID.JIRA_RESOLVED,)], date(2019, 1, 1), date(2022, 1, 1))
        res = await self._request(404, json=body)
        assert res["type"] == "/errors/teams/TeamNotFound"
        assert res["detail"] == "Team 1 not found or access denied"
        assert res["title"] == "Team not found"

    async def test_fetch_bad_account(self, sdb, sample_teams) -> None:
        await models_insert(sdb, UserAccountFactory(account_id=2, user_id="gh|00"))
        body = self._body(1, [(JIRAMetricID.JIRA_RESOLVED,)], date(2019, 1, 1), date(2022, 1, 1))
        res = await self._request(404, user_id="gh|00", json=body)
        assert res["type"] == "/errors/teams/TeamNotFound"
        assert res["detail"] == "Team 1 not found or access denied"
        assert res["title"] == "Team not found"

    async def test_fetch_bad_metric(self, sample_teams) -> None:
        body = self._body(1, [("whatever",)], date(2019, 1, 1), date(2022, 1, 1))
        res = await self._request(400, json=body)
        assert res["type"] == "/errors/BadRequest"
        assert res["title"] == "Bad Request"
        assert "whatever" in res["detail"]

    async def test_fetch_bad_dates_order(self, sample_teams) -> None:
        body = self._body(1, [(JIRAMetricID.JIRA_RESOLVED,)], date(2022, 1, 1), date(2019, 1, 1))
        res = await self._request(400, json=body)
        assert res["type"] == "/errors/InvalidRequestError"
        assert res["title"] == "Bad Request"
        assert res["detail"] == "valid_from must be less than or equal to expires_at"

    async def test_dates_far_past(self, sample_teams) -> None:
        body = self._body(1, [(JIRAMetricID.JIRA_RESOLVED,)], date(1984, 1, 1), date(2002, 1, 1))
        res = await self._request(400, json=body)
        assert res["type"] == "/errors/InvalidRequestError"
        assert res["title"] == "Bad Request"
        assert "1984-01-01" in res["detail"]

    @freeze_time("2022-03-31")
    async def test_valid_from_in_the_future(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[39789]))
        body = self._body(1, [(PullRequestMetricID.PR_SIZE,)], date(2022, 4, 1), date(2022, 4, 10))
        res = await self._request(400, json=body)
        assert res["type"] == "/errors/InvalidRequestError"
        assert res["title"] == "Bad Request"
        assert res["detail"] == "valid_from cannot be in the future"

    async def test_fetch_bad_repository(self, sample_teams) -> None:
        body = self._body(
            1,
            [(PullRequestMetricID.PR_ALL_COUNT,)],
            date(2019, 1, 1),
            date(2022, 1, 1),
            repositories=["github.com/src-d/not-existing"],
        )
        res = await self._request(403, json=body)
        assert res["type"] == "/errors/ForbiddenError"
        assert res["title"] == "Forbidden"
        assert "src-d/not-existing" in res["detail"]

    async def test_no_metric_with_params(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        body = self._body(1, (), date(2016, 1, 1), date(2017, 1, 1))
        res = await self._request(400, json=body)
        assert res["type"] == "/errors/BadRequest"
        assert res["title"] == "Bad Request"
        assert "metrics_with_params" in res["detail"]


class TestMetricsWithParams(BaseTeamMetricsTest):
    async def test_base(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, members=[40020, 39789]))
        metric = PullRequestMetricID.PR_REVIEW_COMMENTS_PER_ABOVE_THRESHOLD_RATIO
        body = partial(self._body, 1, valid_from=date(2016, 1, 1), expires_at=date(2017, 1, 1))

        metrics_w_params = [{"name": metric, "metric_params": {"threshold": 5}}]
        res = await self._request(json=body(metrics_w_params))
        val_threshold_5 = res[0]["value"]["value"]

        metrics_w_params[0]["metric_params"]["threshold"] = 8
        res = await self._request(json=body(metrics_w_params))
        val_threshold_8 = res[0]["value"]["value"]

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
                "metric_params": {"threshold": "172800s"},
                "teams_metric_params": [{"team": 2, "metric_params": {"threshold": "123000s"}}],
            },
        ]
        body = self._body(1, metrics_w_params, *dates)
        res = await self._request(json=body)
        metric_values = res[0]["value"]

        # teams are the same, but team 2 is computed with a lower threshold due to
        # teamsMetricParams so will have a lower metric value
        assert metric_values["team"]["id"] == 1
        assert metric_values["value"] == pytest.approx(0.6923077)

        assert metric_values["children"][0]["team"]["id"] == 2
        assert metric_values["children"][0]["value"] == pytest.approx(0.653846)


@pytest.fixture(scope="function")
async def sample_teams(sdb: Database) -> None:
    await models_insert(
        sdb,
        TeamFactory(id=1, name="T1", members=[40020, 39789, 40070]),
        TeamFactory(id=2, name="T2", parent_id=1, members=[40191, 39926, 40418]),
        MappedJIRAIdentityFactory(github_user_id=40020, jira_user_id="5de5049e2c5dd20d0f9040c1"),
        MappedJIRAIdentityFactory(github_user_id=39789, jira_user_id="5dd58cb9c7ac480ee5674902"),
        MappedJIRAIdentityFactory(github_user_id=40191, jira_user_id="5ddec0b9be6c1f0d071ff82d"),
    )
