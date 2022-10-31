"""Tests for the measure_goals controller."""

from datetime import date, datetime, timedelta, timezone
from functools import partial
from typing import Any, Optional

import pytest
import sqlalchemy as sa
from sqlalchemy import delete

from athenian.api.db import Database
from athenian.api.internal.settings import ReleaseMatch
from athenian.api.models.state.models import AccountGitHubAccount, RepositorySet, TeamGoal
from athenian.api.models.web import (
    AlignGoalsRequest,
    JIRAMetricID,
    PullRequestMetricID,
    ReleaseMetricID,
)
from tests.testutils.auth import force_request_auth
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_MD_ACCOUNT_ID
from tests.testutils.factory.state import (
    AccountFactory,
    AccountGitHubAccountFactory,
    GoalFactory,
    LogicalRepositoryFactory,
    MappedJIRAIdentityFactory,
    ReleaseSettingFactory,
    RepositorySetFactory,
    TeamFactory,
    TeamGoalFactory,
    UserAccountFactory,
)
from tests.testutils.requester import Requester
from tests.testutils.time import dt


class BaseMeasureGoalsTest(Requester):
    async def _request(
        self,
        assert_status: int = 200,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> dict:
        with force_request_auth(user_id, self.headers) as headers:
            response = await self.client.request(
                method="POST", path="/private/align/goals", headers=headers, **kwargs,
            )
        assert response.status == assert_status
        return await response.json()

    def _body(
        self,
        team: int,
        account: int = 1,
        only_with_targets: Optional[bool] = None,
        include_series: Optional[bool] = None,
    ) -> dict:
        req = AlignGoalsRequest(
            account=account,
            team=team,
            only_with_targets=only_with_targets or False,
            include_series=include_series or False,
        )
        body = req.to_dict()
        if only_with_targets is None:
            body.pop("only_with_targets")
        if include_series is None:
            body.pop("include_series")
        return body

    @classmethod
    def _assert_team_goal_values(
        cls,
        team_goal: dict,
        initial: Any,
        current: Any,
        target: Any,
    ) -> None:
        assert team_goal["value"]["target"] == target
        assert team_goal["value"]["current"] == current
        assert team_goal["value"]["initial"] == initial


class TestMeasureGoalsErrors(BaseMeasureGoalsTest):
    async def test_not_existing_team(self, sdb: Database) -> None:
        res = await self._request(assert_status=404, json=self._body(1))
        assert res["detail"] == "Team 1 not found or access denied"

    async def test_implicit_root_team_not_team_existing(
        self,
        sdb: Database,
    ) -> None:
        res = await self._request(404, json=self._body(0))
        assert res["detail"] == "Root team not found or access denied"

    async def test_account_mismatch(self, sdb: Database) -> None:
        await sdb.execute(sa.delete(AccountGitHubAccount))
        await models_insert(
            sdb,
            AccountFactory(id=33),
            AccountGitHubAccountFactory(id=DEFAULT_MD_ACCOUNT_ID, account_id=33),
            UserAccountFactory(account_id=33, user_id="gh|XXX"),
            TeamFactory(id=10, members=[40191]),
        )
        await self._request(404, json=self._body(0), user_id="gh|XXX")

    async def test_jira_not_installed_filtering(self, sdb: Database, mdb_rw: Database) -> None:
        metric = PullRequestMetricID.PR_OPENED
        dates = {"valid_from": dt(2019, 1, 1), "expires_at": dt(2022, 1, 1)}
        await sdb.execute(sa.delete(AccountGitHubAccount))
        await models_insert(
            sdb,
            AccountFactory(id=33),
            AccountGitHubAccountFactory(id=DEFAULT_MD_ACCOUNT_ID, account_id=33),
            UserAccountFactory(account_id=33, user_id="gh|XXX"),
            TeamFactory(id=10, members=[39789, 40020, 40191], owner_id=33),
            GoalFactory(id=20, metric=metric, account_id=33, **dates),
            TeamGoalFactory(goal_id=20, team_id=10, jira_projects=["p1"]),
        )
        res = await self._request(assert_status=422, json=self._body(10, 33), user_id="gh|XXX")
        assert res["detail"] == "JIRA has not been installed to the metadata yet."


class TestMeasureGoals(BaseMeasureGoalsTest):
    async def test_no_goals_for_team(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1),
            TeamFactory(id=2, parent_id=1),
            GoalFactory(id=1),
            TeamGoalFactory(goal_id=1, team_id=1),
        )

        res = await self._request(json=self._body(2))
        assert res == []

    async def test_single_team_goal(self, sdb: Database) -> None:
        dates = {"valid_from": dt(2019, 1, 1), "expires_at": dt(2022, 1, 1)}
        await models_insert(
            sdb,
            TeamFactory(id=10, name="A-Team", members=[39789, 40020, 40191]),
            GoalFactory(id=20, metric=PullRequestMetricID.PR_REVIEW_COMMENTS_PER, **dates),
            TeamGoalFactory(goal_id=20, team_id=10, target=1.23),
        )
        res = await self._request(json=self._body(10))
        assert len(res) == 1
        goal = res[0]

        assert goal["id"] == 20
        assert goal["valid_from"] == "2019-01-01"
        assert goal["expires_at"] == "2021-12-31"
        for field in ("repositories", "jira_projects", "jira_priorities", "jira_issue_types"):
            assert field not in goal

        tg = goal["team_goal"]
        assert tg["team"]["id"] == 10
        assert tg["team"]["name"] == "A-Team"
        assert tg["children"] == []
        self._assert_team_goal_values(tg, pytest.approx(11.5, 0.1), pytest.approx(3.7, 0.1), 1.23)
        assert "series" not in tg["value"]
        assert "series_granularity" not in tg["value"]

    async def test_two_teams_jira_metric(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789]),
            TeamFactory(id=11, members=[40020], parent_id=10),
            GoalFactory(
                id=20,
                metric=PullRequestMetricID.PR_MEDIAN_SIZE,
                valid_from=dt(2019, 1, 1),
                expires_at=dt(2022, 1, 1),
            ),
            GoalFactory(
                id=21,
                metric=JIRAMetricID.JIRA_RESOLVED,
                valid_from=dt(2020, 1, 1),
                expires_at=dt(2022, 1, 1),
            ),
            TeamGoalFactory(goal_id=20, team_id=10, target=1),
            TeamGoalFactory(goal_id=20, team_id=11, target=4),
            TeamGoalFactory(goal_id=21, team_id=10, target=20),
            TeamGoalFactory(goal_id=21, team_id=11, target=40),
            MappedJIRAIdentityFactory(
                github_user_id=39789, jira_user_id="5dd58cb9c7ac480ee5674902",
            ),
            MappedJIRAIdentityFactory(
                github_user_id=40020, jira_user_id="5de5049e2c5dd20d0f9040c1",
            ),
        )
        res = await self._request(json=self._body(10))
        assert len(res) == 2

        goal_20 = res[0]
        assert goal_20["id"] == 20

        assert (team_goal_10 := goal_20["team_goal"])["team"]["id"] == 10
        assert team_goal_10["team"]["members_count"] == 1
        self._assert_team_goal_values(team_goal_10, 171, 64, 1)

        assert (team_goal_11 := team_goal_10["children"][0])["team"]["id"] == 11
        assert team_goal_11["team"]["members_count"] == 1
        assert team_goal_11["team"]["total_members_count"] == 1
        self._assert_team_goal_values(team_goal_11, 507, 47, 4)

        goal_21 = res[1]
        assert goal_21["id"] == 21

        assert (team_goal_10 := goal_21["team_goal"])["team"]["id"] == 10
        self._assert_team_goal_values(team_goal_10, 3, 572, 20)

        assert (team_goal_11 := team_goal_10["children"][0])["team"]["id"] == 11
        self._assert_team_goal_values(team_goal_11, 3, 528, 40)

    async def test_child_team_with_no_goal(self, sdb: Database) -> None:
        metric = PullRequestMetricID.PR_REVIEW_COMMENTS_PER
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[39789]),
            TeamFactory(id=10, parent_id=1, members=[39789]),
            GoalFactory(
                id=20, metric=metric, valid_from=dt(2019, 1, 1), expires_at=dt(2022, 1, 1),
            ),
            TeamGoalFactory(goal_id=20, team_id=1, target=1),
        )
        res = await self._request(json=self._body(0))
        assert len(res) == 1
        goal = res[0]

        assert (team_1_goal := goal["team_goal"])["team"]["id"] == 1
        assert (team_10_goal := team_1_goal["children"][0])["team"]["id"] == 10

        assert team_1_goal["value"]["initial"] is not None
        assert team_1_goal["value"]["current"] is not None
        # also metrics for child team with no goal are returned
        assert team_10_goal["value"]["initial"] is not None
        assert team_10_goal["value"]["current"] is not None

    async def test_multiple_teams(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[39789]),
            TeamFactory(id=10, parent_id=1, members=[39789]),
            TeamFactory(id=11, parent_id=10, members=[39789]),
            TeamFactory(id=12, parent_id=10, members=[39789]),
            TeamFactory(id=13, parent_id=11, members=[39789]),
            GoalFactory(id=20),
            TeamGoalFactory(goal_id=20, team_id=1, target=1),
            TeamGoalFactory(goal_id=20, team_id=10, target=10.1),
            TeamGoalFactory(goal_id=20, team_id=11, target="111s"),
            TeamGoalFactory(goal_id=20, team_id=13, target="222s"),
        )
        res = await self._request(json=self._body(10))
        assert len(res) == 1
        goal = res[0]

        assert goal["id"] == 20

        assert (team_goal_10 := goal["team_goal"])["team"]["id"] == 10
        assert len(team_goal_10["children"]) == 2
        assert team_goal_10["value"]["target"] == 10.1

        assert (team_goal_11 := team_goal_10["children"][0])["team"]["id"] == 11
        assert len(team_goal_11["children"]) == 1
        assert team_goal_11["value"]["target"] == "111s"

        assert (team_goal_12 := team_goal_10["children"][1])["team"]["id"] == 12
        assert not team_goal_12["children"]
        assert "target" not in team_goal_12["value"]

        assert (team_goal_13 := team_goal_11["children"][0])["team"]["id"] == 13
        assert not team_goal_13["children"]
        assert team_goal_13["value"]["target"] == "222s"

    async def test_multiple_goals(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789]),
            TeamFactory(id=11, parent_id=10, members=[39789]),
            TeamFactory(id=12, parent_id=10, members=[39789]),
            TeamFactory(id=13, parent_id=12, members=[39789]),
            GoalFactory(
                id=20,
                metric=PullRequestMetricID.PR_REVIEW_TIME,
                valid_from=dt(2000, 1, 1),
                expires_at=dt(2001, 1, 1),
            ),
            GoalFactory(
                id=21,
                metric=PullRequestMetricID.PR_REVIEW_COMMENTS_PER,
                valid_from=dt(2004, 7, 1),
                expires_at=dt(2004, 10, 1),
            ),
            TeamGoalFactory(goal_id=20, team_id=10, target=1),
            TeamGoalFactory(goal_id=20, team_id=12, target=2),
            TeamGoalFactory(goal_id=21, team_id=11, target=3),
            TeamGoalFactory(goal_id=21, team_id=12, target=4),
        )
        res = await self._request(json=self._body(10))
        assert len(res) == 2

        assert (goal_20 := res[0])["id"] == 20
        assert goal_20["valid_from"] == "2000-01-01"
        assert goal_20["expires_at"] == "2000-12-31"

        assert (team_goal_10 := goal_20["team_goal"])["team"]["id"] == 10
        assert team_goal_10["value"]["target"] == 1
        assert len(team_goal_10["children"]) == 2

        assert (team_goal_11 := team_goal_10["children"][0])["team"]["id"] == 11
        assert "target" not in team_goal_11["value"]
        assert not team_goal_11["children"]

        assert (team_goal_12 := team_goal_10["children"][1])["team"]["id"] == 12
        assert team_goal_12["value"]["target"] == 2
        assert len(team_goal_12["children"]) == 1

        assert (team_goal_13 := team_goal_12["children"][0])["team"]["id"] == 13
        assert "target" not in team_goal_13["value"]
        assert not team_goal_13["children"]

        # same tree is built for goal 21, with different values
        assert (goal_21 := res[1])["id"] == 21
        assert goal_21["valid_from"] == "2004-07-01"
        assert goal_21["expires_at"] == "2004-09-30"

        assert (team_goal_10 := goal_21["team_goal"])["team"]["id"] == 10
        assert "target" not in team_goal_10["value"]
        assert len(team_goal_10["children"]) == 2

        assert (team_goal_11 := team_goal_10["children"][0])["team"]["id"] == 11
        assert team_goal_11["value"]["target"] == 3
        assert not team_goal_11["children"]

        assert (team_goal_12 := team_goal_10["children"][1])["team"]["id"] == 12
        assert team_goal_12["value"]["target"] == 4
        assert len(team_goal_12["children"]) == 1

        assert (team_goal_13 := team_goal_12["children"][0])["team"]["id"] == 13
        assert "target" not in team_goal_13["value"]
        assert not team_goal_13["children"]

    async def test_timedelta_metric(self, sdb: Database) -> None:
        metric = PullRequestMetricID.PR_LEAD_TIME
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789]),
            GoalFactory(
                id=20, metric=metric, valid_from=dt(2019, 1, 1), expires_at=dt(2019, 7, 1),
            ),
            TeamGoalFactory(goal_id=20, team_id=10, target="20001s"),
        )

        res = await self._request(json=self._body(10))
        assert len(res) == 1

        goal = res[0]
        assert goal["team_goal"]["team"]["id"] == 10

        self._assert_team_goal_values(goal["team_goal"], "689712s", "4731059s", "20001s")

    async def test_only_with_targets_param(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, parent_id=None, members=[39789]),
            TeamFactory(id=11, parent_id=10, members=[39789]),
            TeamFactory(id=12, parent_id=11, members=[39789]),
            GoalFactory(id=20),
            TeamGoalFactory(goal_id=20, team_id=11, target=1.23),
        )
        res = await self._request(json=self._body(10, only_with_targets=True))

        assert len(res) == 1
        goal = res[0]
        assert (team_goal_10 := goal["team_goal"])["team"]["id"] == 10
        assert len(team_goal_10["children"]) == 1
        assert (team_goal_11 := team_goal_10["children"][0])["team"]["id"] == 11
        # team goal 12 is hidden
        assert len(team_goal_11["children"]) == 0

    async def test_archived_goals_are_excluded(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789]),
            GoalFactory(id=20),
            GoalFactory(id=21, archived=True),
            TeamGoalFactory(goal_id=20, team_id=10),
            TeamGoalFactory(goal_id=21, team_id=10),
        )
        res = await self._request(json=self._body(10))

        assert [goal["id"] for goal in res] == [20]

    async def test_repositories(
        self,
        sdb: Database,
        mdb_rw: Database,
        release_match_setting_tag_logical_db,
    ) -> None:
        await sdb.execute(delete(RepositorySet))
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789]),
            GoalFactory(id=20, repositories=[[1, None]]),
            GoalFactory(id=21, repositories=[[1, "a"], [1, "b"]]),
            TeamGoalFactory(goal_id=20, team_id=10, target=1),
            TeamGoalFactory(goal_id=21, team_id=10, target=20),
            *(
                ReleaseSettingFactory(logical_name=name, repo_id=1, match=ReleaseMatch.tag)
                for name in "ab"
            ),
            *(
                LogicalRepositoryFactory(name=name, repository_id=1, prs={"title": "x"})
                for name in "ab"
            ),
            RepositorySetFactory(
                items=[
                    ["github.com/athenianco/repo/a", 1],
                    ["github.com/athenianco/repo/b", 1],
                ],
            ),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [md_factory.RepositoryFactory(node_id=1, full_name="athenianco/repo")]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)
            res = await self._request(json=self._body(10))

        assert len(res) == 2
        assert res[0]["id"] == 20
        assert res[0]["repositories"] == ["github.com/athenianco/repo"]
        assert res[1]["id"] == 21
        assert res[1]["repositories"] == [
            "github.com/athenianco/repo/a",
            "github.com/athenianco/repo/b",
        ]

    async def test_repositories_metrics_filtering(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[39789]),
            TeamFactory(id=10, parent_id=1, members=[39789]),
            GoalFactory(
                id=20,
                metric=PullRequestMetricID.PR_ALL_COUNT,
                valid_from=dt(2019, 1, 1),
                expires_at=dt(2022, 1, 1),
                repositories=[[39652769, None]],
            ),
            TeamGoalFactory(goal_id=20, team_id=10, target=1, repositories=[[40550, None]]),
        )
        res = await self._request(json=self._body(0))
        goal = res[0]
        assert goal["id"] == 20
        assert goal["repositories"] == ["github.com/src-d/gitbase"]

        assert (team_1_goal := goal["team_goal"])["team"]["id"] == 1
        assert (team_10_goal := team_1_goal["children"][0])["team"]["id"] == 10

        # metrics for team 1 have been computed considering default goal repositories filter
        assert team_1_goal["value"]["initial"] == 0
        # metrics for team 10 have been computed considering repositories [[40550, None]] filter
        assert team_10_goal["value"]["initial"] == 88

    async def test_same_team_different_repos(self, sdb: Database) -> None:
        dates = {"valid_from": dt(2019, 1, 1), "expires_at": dt(2022, 1, 1)}
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[39789]),
            GoalFactory(id=20, metric=PullRequestMetricID.PR_ALL_COUNT, **dates),
            GoalFactory(id=21, metric=PullRequestMetricID.PR_ALL_COUNT, **dates),
            TeamGoalFactory(goal_id=20, team_id=1, repositories=[[40550, None]]),
            TeamGoalFactory(goal_id=21, team_id=1, repositories=[[39652769, None]]),
        )
        res = await self._request(json=self._body(0))
        goal_20 = res[0]
        assert goal_20["id"] == 20
        assert (tg := goal_20["team_goal"])["team"]["id"] == 1
        assert tg["value"]["initial"] == 88

        goal_21 = res[1]
        assert goal_21["id"] == 21
        assert (tg := goal_21["team_goal"])["team"]["id"] == 1
        assert tg["value"]["initial"] == 0

    @pytest.mark.xfail
    async def test_team_unassigned_multiple_goals(self, sdb: Database) -> None:
        # TODO: goal_id 0 passed to calculate_team_metrics conflicts for
        # the two goals with different filters so test fails
        dates = {"valid_from": dt(2019, 1, 1), "expires_at": dt(2022, 1, 1)}
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[39789]),
            TeamFactory(id=10, parent_id=1, members=[40020]),
            GoalFactory(id=20, metric=PullRequestMetricID.PR_ALL_COUNT, **dates),
            GoalFactory(
                id=21, metric=PullRequestMetricID.PR_ALL_COUNT, jira_projects=["INVALID"], **dates,
            ),
            TeamGoalFactory(goal_id=20, team_id=10),
            TeamGoalFactory(goal_id=21, team_id=10),
        )
        res = await self._request(json=self._body(0, only_with_targets=True))
        goal_20 = res[0]
        assert (tg := goal_20["team_goal"])["team"]["id"] == 1
        assert tg["value"]["initial"] == 93

        goal_21 = res[1]
        assert (tg := goal_21["team_goal"])["team"]["id"] == 1
        # value from goal 20 overwites the value for goal 21
        assert tg["value"]["initial"] == 0

    async def test_jira_fields(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789]),
            GoalFactory(id=20, jira_projects=["PR0"], jira_issue_types=["task", "bug"]),
            GoalFactory(id=21, jira_priorities=["low"]),
            TeamGoalFactory(goal_id=20, team_id=10),
            TeamGoalFactory(goal_id=21, team_id=10),
        )

        res = await self._request(json=self._body(10))

        assert len(res) == 2
        assert res[0]["id"] == 20
        assert res[0]["jira_projects"] == ["PR0"]
        assert "jira_priorities" not in res[0]
        assert res[0]["jira_issue_types"] == ["task", "bug"]
        assert res[1]["id"] == 21
        assert "jira_projects" not in res[1]
        assert res[1]["jira_priorities"] == ["low"]
        assert "jira_issue_types" not in res[1]

    async def test_jira_not_installed(self, sdb: Database, mdb_rw: Database) -> None:
        # jira installation is not needed when jira metrics/filters are not used
        metric = PullRequestMetricID.PR_OPENED
        dates = {"valid_from": dt(2019, 1, 1), "expires_at": dt(2022, 1, 1)}
        await sdb.execute(sa.delete(AccountGitHubAccount))

        await models_insert(
            sdb,
            AccountFactory(id=33),
            AccountGitHubAccountFactory(id=DEFAULT_MD_ACCOUNT_ID, account_id=33),
            UserAccountFactory(account_id=33, user_id="gh|XXX"),
            TeamFactory(id=10, members=[39789, 40020, 40191], owner_id=33),
            GoalFactory(id=20, metric=metric, account_id=33, **dates),
            TeamGoalFactory(goal_id=20, team_id=10),
            RepositorySetFactory(id=100, owner_id=33, items=[["github.com/a/b", 1]]),
        )
        res = await self._request(json=self._body(10, 33), user_id="gh|XXX")
        assert res[0]["team_goal"]["value"]["initial"] == 0

    async def test_metric_params(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789]),
            TeamFactory(id=11, parent_id=10, members=[39789]),
            GoalFactory(id=20, metric_params={"f": 1}),
            TeamGoalFactory(goal_id=20, team_id=10, metric_params=None),
            TeamGoalFactory(goal_id=20, team_id=11, metric_params={"f": 2}),
        )
        res = await self._request(json=self._body(10))
        assert len(res) == 1
        goal = res[0]

        assert goal["id"] == 20
        assert goal["metric_params"] == {"f": 1}

        assert (tg_10 := goal["team_goal"])["team"]["id"] == 10
        assert "metric_params" not in tg_10

        assert (tg_11 := tg_10["children"][0])["team"]["id"] == 11
        assert tg_11["metric_params"] == {"f": 2}


class TestMeasureGoalsJIRAFiltering(BaseMeasureGoalsTest):
    async def test_pr_metric_priority(self, sdb: Database, mdb_rw: Database) -> None:
        metric = PullRequestMetricID.PR_OPENED
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789, 40020, 40191]),
            GoalFactory(
                id=20, metric=metric, valid_from=dt(2019, 1, 1), expires_at=dt(2022, 1, 1),
            ),
            TeamGoalFactory(goal_id=20, team_id=10, jira_priorities=None),
        )
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162901, jira_id="20"),
                md_factory.JIRAIssueFactory(
                    id="20", priority_name="prio0", priority_id="100", project_id="1",
                ),
                md_factory.JIRAPriorityFactory(id="100", name="prio0"),
                md_factory.JIRAPriorityFactory(id="101", name="prio1"),
                md_factory.JIRAProjectFactory(id="1"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)
            res = await self._request(json=self._body(10))
            assert res[0]["team_goal"]["value"]["initial"] == 93

            # no issue with this priority, nothing is computed
            await self._update_team_goal(sdb, 10, 20, {TeamGoal.jira_priorities: ["prio1"]})
            res = await self._request(json=self._body(10))
            assert res[0]["team_goal"]["value"]["initial"] == 0

            # unknown priority, nothing is computed
            await self._update_team_goal(sdb, 10, 20, {TeamGoal.jira_priorities: ["prio2"]})
            res = await self._request(json=self._body(10))
            assert res[0]["team_goal"]["value"]["initial"] == 0

            await self._update_team_goal(sdb, 10, 20, {TeamGoal.jira_priorities: ["prio0"]})
            res = await self._request(json=self._body(10))
            # only pr 162901 is counted
            assert res[0]["team_goal"]["value"]["initial"] == 1

    async def test_pr_metric_unexisting_project(self, sdb: Database, mdb_rw: Database) -> None:
        dates = {"valid_from": dt(2019, 1, 1), "expires_at": dt(2022, 1, 1)}
        metric = PullRequestMetricID.PR_OPENED
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789, 40020, 40191]),
            GoalFactory(id=20, metric=metric, **dates),
            GoalFactory(id=21, metric=metric, **dates),
            TeamGoalFactory(goal_id=20, team_id=10, jira_projects=["P1"]),
            TeamGoalFactory(goal_id=21, team_id=10, jira_projects=None),
        )
        res = await self._request(json=self._body(10))
        assert len(res) == 2
        assert res[0]["team_goal"]["value"]["initial"] == 0
        assert res[1]["team_goal"]["value"]["initial"] == 93

    async def test_pr_metric_more_goals(self, sdb: Database, mdb_rw: Database) -> None:
        dates = {"valid_from": dt(2019, 1, 1), "expires_at": dt(2022, 1, 1)}
        metric = PullRequestMetricID.PR_OPENED
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789, 40020, 40191]),
            GoalFactory(id=20, metric=metric, **dates),
            GoalFactory(id=21, metric=metric, **dates),
            GoalFactory(id=22, metric=metric, **dates),
            TeamGoalFactory(goal_id=20, team_id=10, jira_projects=None),
            TeamGoalFactory(goal_id=21, team_id=10, jira_projects=["P0"]),
            TeamGoalFactory(goal_id=22, team_id=10, jira_projects=["P1"]),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162901, jira_id="20"),
                md_factory.JIRAIssueFactory(id="20", project_id="0"),
                md_factory.JIRAProjectFactory(id="0", key="P0"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)
            res = await self._request(json=self._body(10))
            assert len(res) == 3
            assert res[0]["id"] == 20
            assert res[0]["team_goal"]["value"]["initial"] == 93

            assert res[1]["id"] == 21
            assert res[1]["team_goal"]["value"]["initial"] == 1

            assert res[2]["id"] == 22
            assert res[2]["team_goal"]["value"]["initial"] == 0

    async def test_pr_metric_complex_filters(self, sdb: Database, mdb_rw: Database) -> None:
        dates = {"valid_from": dt(2019, 1, 1), "expires_at": dt(2022, 1, 1)}
        metric = PullRequestMetricID.PR_OPENED
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789, 40020, 40191]),
            TeamFactory(id=11, members=[39789, 40020, 40191], parent_id=10),
            GoalFactory(id=20, metric=metric, **dates),
            GoalFactory(id=21, metric=metric, **dates),
            TeamGoalFactory(goal_id=20, team_id=10, jira_priorities=["p0", "p1"]),
            TeamGoalFactory(
                goal_id=20, team_id=11, jira_projects=["PJ1"], jira_issue_types=["t0", "t1"],
            ),
            TeamGoalFactory(goal_id=21, team_id=10, jira_projects=["PJ0"], jira_priorities=["p0"]),
            TeamGoalFactory(goal_id=21, team_id=11, jira_projects=["PJ1"], jira_priorities=["t0"]),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.JIRAProjectFactory(id="0", key="PJ0"),
                md_factory.JIRAProjectFactory(id="1", key="PJ1"),
                md_factory.JIRAPriorityFactory(id="100", name="P0"),
                md_factory.JIRAPriorityFactory(id="101", name="P1"),
                md_factory.JIRAIssueTypeFactory(id="100", name="T0"),
                md_factory.JIRAIssueTypeFactory(id="101", name="T1"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162901, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162901, jira_id="21"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162908, jira_id="22"),
                *[
                    md_factory.JIRAIssueFactory(
                        id=id_,
                        project_id=project_id,
                        priority_id=priority_id,
                        priority_name=priority_name,
                        type_id=type_id,
                        type=type_,
                    )
                    for (id_, project_id, priority_id, priority_name, type_id, type_) in (
                        ("20", "0", "100", "P0", "100", "T0"),
                        ("21", "0", "101", "P1", "100", "T0"),
                        ("22", "1", "100", "P0", "101", "T1"),
                    )
                ],
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)
            res = await self._request(json=self._body(10))
            assert (goal_20 := res[0])["id"] == 20
            assert goal_20["team_goal"]["team"]["id"] == 10
            # both PRs are selected, priorities is P0, P1
            assert goal_20["team_goal"]["value"]["initial"] == 2

            assert goal_20["team_goal"]["children"][0]["team"]["id"] == 11
            # only 162908 is selected
            assert goal_20["team_goal"]["children"][0]["value"]["initial"] == 1

            # only 162901 is selected
            assert (goal_21 := res[1])["id"] == 21
            assert goal_21["team_goal"]["team"]["id"] == 10

            # no PR selected
            assert goal_21["team_goal"]["children"][0]["team"]["id"] == 11
            assert goal_21["team_goal"]["children"][0]["value"]["initial"] == 0

    async def test_jira_metric(self, sdb: Database, mdb_rw: Database) -> None:
        metric = JIRAMetricID.JIRA_OPEN
        t = dt(2005, 1, 2)

        GoalF = partial(
            GoalFactory, metric=metric, valid_from=t, expires_at=t + timedelta(days=30),
        )

        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789, 40020]),
            TeamFactory(id=11, members=[40020], parent_id=10),
            GoalF(id=20, jira_priorities=["p0"], jira_projects=["PJ0", "PJXX"]),
            GoalF(id=21),
            GoalF(id=22),
            TeamGoalFactory(goal_id=20, team_id=10, jira_projects=["PJ0"]),
            TeamGoalFactory(
                goal_id=21, team_id=10, jira_projects=["PJ1"], jira_priorities=["p0", "p1"],
            ),
            TeamGoalFactory(
                goal_id=22, team_id=10, jira_projects=["PJ0"], jira_issue_types=["t0"],
            ),
            TeamGoalFactory(
                goal_id=22, team_id=11, jira_projects=["PJ0"], jira_issue_types=["t0"],
            ),
            MappedJIRAIdentityFactory(
                github_user_id=39789, jira_user_id="5dd58cb9c7ac480ee5674902",
            ),
            MappedJIRAIdentityFactory(
                github_user_id=40020, jira_user_id="5de5049e2c5dd20d0f9040c1",
            ),
        )
        IF = partial(md_factory.JIRAIssueFactory, created=t)
        priority_0 = {"priority_id": "10", "priority_name": "P0"}
        priority_1 = {"priority_id": "11", "priority_name": "P1"}
        type_0 = {"type_id": "100", "type": "T0"}
        user0 = {"assignee_display_name": "waren long"}
        user1 = {"assignee_display_name": "vadim markovtsev"}
        models = [
            md_factory.JIRAProjectFactory(id="0", key="PJ0"),
            md_factory.JIRAProjectFactory(id="1", key="PJ1"),
            md_factory.JIRAPriorityFactory(id="10", name="P0"),
            md_factory.JIRAPriorityFactory(id="11", name="P1"),
            md_factory.JIRAIssueTypeFactory(id="100", name="T0"),
            IF(id="10", project_id="0", **user1, **priority_0),
            IF(id="11", project_id="0", **user1, **priority_0, **type_0),
            IF(id="12", project_id="1", **user1, **priority_0),
            IF(id="13", project_id="0", **user0, **priority_0, **type_0),
            IF(id="14", project_id="1", **user0, **priority_1),
            *[
                md_factory.JIRAAthenianIssueFactory(id=id_, updated=t)
                for id_ in ("10", "11", "12", "13", "14")
            ],
        ]

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            res = await self._request(json=self._body(10))
            assert len(res) == 3
            assert (goal_20 := res[0])["id"] == 20
            assert goal_20["team_goal"]["team"]["id"] == 10
            # issues 10, 11 and 13 counted
            assert goal_20["team_goal"]["team"]["id"] == 10
            assert goal_20["team_goal"]["value"]["current"] == 3

            # team goal is missing, filters taken from goal; issues 10 and 11 counted
            assert goal_20["team_goal"]["children"][0]["team"]["id"] == 11
            assert goal_20["team_goal"]["children"][0]["value"]["current"] == 2

            assert (goal_21 := res[1])["id"] == 21
            assert goal_21["team_goal"]["team"]["id"] == 10
            # issues 12 and 14 counted
            assert goal_21["team_goal"]["value"]["current"] == 2

            assert (goal_22 := res[2])["id"] == 22
            # issues 11 and 13 counted
            assert goal_22["team_goal"]["team"]["id"] == 10
            assert goal_22["team_goal"]["value"]["current"] == 2

            assert goal_22["team_goal"]["children"][0]["team"]["id"] == 11
            # issues 11 counted
            assert goal_22["team_goal"]["children"][0]["value"]["current"] == 1

    async def test_release_metric(self, sdb: Database, mdb_rw: Database) -> None:
        metric = ReleaseMetricID.RELEASE_COUNT
        dates = {"valid_from": dt(2019, 1, 1), "expires_at": dt(2022, 1, 1)}

        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789, 40020, 40191]),
            GoalFactory(id=20, metric=metric, **dates),
            GoalFactory(id=21, metric=metric, **dates),
            TeamGoalFactory(goal_id=20, team_id=10, jira_issue_types=["bug"]),
            TeamGoalFactory(goal_id=21, team_id=10, jira_issue_types=["task"]),
        )
        res = await self._request(json=self._body(10))
        assert len(res) == 2
        assert res[0]["team_goal"]["value"]["initial"] == 6
        assert res[1]["team_goal"]["value"]["initial"] == 11

    @classmethod
    async def _update_team_goal(
        cls,
        sdb: Database,
        team_id: int,
        goal_id: int,
        values: dict,
    ) -> None:
        values = {**values, TeamGoal.updated_at: datetime.now(timezone.utc)}
        await sdb.execute(
            sa.update(TeamGoal)
            .where(TeamGoal.team_id == team_id, TeamGoal.goal_id == goal_id)
            .values(values),
        )


class TestMeasureGoalsTimeseries(BaseMeasureGoalsTest):
    async def test_year_goal(self, sdb: Database) -> None:
        metric = PullRequestMetricID.PR_CLOSED
        dates = {"valid_from": dt(2019, 1, 1), "expires_at": dt(2020, 1, 1)}
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789, 40020, 40191]),
            GoalFactory(id=20, metric=metric, **dates),
            TeamGoalFactory(goal_id=20, team_id=10, target=200),
        )
        res = await self._request(json=self._body(10, include_series=True))
        assert len(res) == 1
        goal = res[0]

        assert goal["id"] == 20
        assert goal["valid_from"] == "2019-01-01"
        assert goal["expires_at"] == "2019-12-31"

        team_goal = goal["team_goal"]
        assert team_goal["team"]["id"] == 10
        self._assert_team_goal_values(team_goal, 9, 7, 200)

        # the series include a point for every month
        assert team_goal["value"]["series_granularity"] == "month"
        assert len(team_goal["value"]["series"]) == 12
        expected_dates = [date(2019, i, 1).isoformat() for i in range(1, 13)]
        expected_values = [0, 2, 1, 2, 0, 0, 2, 0, 0, 0, 0, 0]
        assert team_goal["value"]["series"] == [
            {"date": date, "value": value} for date, value in zip(expected_dates, expected_values)
        ]

    async def test_quarter_goal(self, sdb: Database) -> None:
        metric = ReleaseMetricID.RELEASE_AGE
        dates = {"valid_from": dt(2018, 4, 1), "expires_at": dt(2018, 7, 1)}
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789, 40020, 40191]),
            TeamFactory(id=11, members=[39789, 40020, 40191], parent_id=10),
            GoalFactory(id=20, metric=metric, **dates),
            TeamGoalFactory(goal_id=20, team_id=10, target="10s"),
            TeamGoalFactory(goal_id=20, team_id=11, target="12s", jira_issue_types=["bug"]),
        )
        res = await self._request(json=self._body(10, include_series=True))
        assert len(res) == 1
        goal = res[0]
        assert goal["valid_from"] == "2018-04-01"
        assert goal["expires_at"] == "2018-06-30"

        assert (tg_10 := goal["team_goal"])["team"]["id"] == 10
        self._assert_team_goal_values(tg_10, "4075735s", "1518598s", "10s")

        assert tg_10["value"]["series_granularity"] == "week"
        series = tg_10["value"]["series"]
        assert len(series) == 13
        expected_values = [
            "1892185s",
            "692688s",
            "536640s",
            *([None] * 3),
            "2494701s",
            *([None] * 2),
            "1976780s",
            *([None] * 3),
        ]
        assert [point["value"] for point in series] == expected_values

        assert [point["date"] for point in series[:2]] == [
            date(2018, 4, 1).isoformat(),
            date(2018, 4, 8).isoformat(),
        ]
        assert [point["date"] for point in series[-2:]] == [
            date(2018, 6, 17).isoformat(),
            date(2018, 6, 24).isoformat(),
        ]

        assert (tg_11 := goal["team_goal"]["children"][0])["team"]["id"] == 11
        self._assert_team_goal_values(tg_11, "2370238s", "1468535s", "12s")

        assert [p["date"] for p in tg_10["value"]["series"]] == [
            p["date"] for p in tg_11["value"]["series"]
        ]

        expected_values = ["1892185s", None, "536640s", *([None] * 6), "1976780s", *([None] * 3)]
        assert [point["value"] for point in tg_11["value"]["series"]] == expected_values
