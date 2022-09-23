from datetime import datetime, timedelta, timezone
from functools import partial
from typing import Optional, Sequence

import pytest
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.models.state.models import AccountGitHubAccount, TeamGoal
from athenian.api.models.web import JIRAMetricID, PullRequestMetricID, ReleaseMetricID
from tests.align.utils import (
    align_graphql_request,
    assert_extension_error,
    build_fragment,
    build_recursive_fields_structure,
)
from tests.testutils.auth import force_request_auth
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_MD_ACCOUNT_ID
from tests.testutils.factory.state import (
    AccountFactory,
    AccountGitHubAccountFactory,
    GoalFactory,
    MappedJIRAIdentityFactory,
    RepositorySetFactory,
    TeamFactory,
    TeamGoalFactory,
    UserAccountFactory,
)
from tests.testutils.requester import Requester
from tests.testutils.time import dt


class BaseGoalsTest(Requester):
    _ALL_GOAL_FIELDS = (
        "id",
        "name",
        "metric",
        "validFrom",
        "expiresAt",
        "repositories",
        "jiraProjects",
        "jiraPriorities",
        "jiraIssueTypes",
    )
    _ALL_VALUE_FIELDS = ("current", "initial", "target")
    _ALL_TEAM_FIELDS = ("id", "name", "totalTeamsCount", "totalMembersCount", "membersCount")

    def _query(
        self,
        goal_fields: Sequence[str],
        value_fields: Sequence[str],
        team_fields: Sequence[str],
        depth: int,
    ) -> str:
        goal_fragment = build_fragment("goalFields", "Goal", goal_fields)

        if team_fields:
            team_fragment = build_fragment("teamFields", "Team", team_fields)
        else:
            team_fragment = None

        if value_fields:
            fragment_fields = [
                f"    {field} {{\n        str\n        int\n        float\n    }}"
                for field in value_fields
            ]
            value_fragment = build_fragment("valueFields", "GoalValue", fragment_fields)
        else:
            value_fragment = None

        fields = """
            ...goalFields
        """

        actual_team_goal_fields = []
        if team_fragment:
            actual_team_goal_fields.append(
                """
            team {
            ...teamFields
            }
            """,
            )
        if value_fields:
            actual_team_goal_fields.append(
                """
            value {
            ...valueFields
            }
            """,
            )

        if actual_team_goal_fields:
            recursive_fields = build_recursive_fields_structure(actual_team_goal_fields, depth)
            fields = f"""
            {fields}
                teamGoal {{
                   {recursive_fields}
                }}
            """

        return f"""
        {goal_fragment}
        {team_fragment or ''}
        {value_fragment or ''}
        query ($accountId: Int!, $teamId: Int!, $onlyWithTargets: Boolean ) {{
          goals(accountId: $accountId, teamId: $teamId, onlyWithTargets: $onlyWithTargets) {{
             {fields}
          }}
        }}
        """

    async def _request(
        self,
        account_id: int,
        team_id: int,
        goal_fields=_ALL_GOAL_FIELDS,
        value_fields=_ALL_VALUE_FIELDS,
        team_fields=_ALL_TEAM_FIELDS,
        repositories: Optional[list[str]] = None,
        depth: int = 6,
        only_with_targets: bool = False,
        user_id: Optional[str] = None,
    ) -> dict:
        if repositories is None:
            repositories_kwargs = {}
        else:
            repositories_kwargs = {"repositories": repositories}
        body: dict[str, str | dict] = {
            "query": self._query(goal_fields, value_fields, team_fields, depth=depth),
            "variables": {
                "accountId": account_id,
                "teamId": team_id,
                "onlyWithTargets": only_with_targets,
                **repositories_kwargs,
            },
        }
        with force_request_auth(user_id, self.headers) as headers:
            return await align_graphql_request(self.client, headers=headers, json=body)


class TestGoalsErrors(BaseGoalsTest):
    async def test_not_existing_team(self, sdb: Database) -> None:
        res = await self._request(1, 1)
        assert_extension_error(res, "Team 1 not found or access denied")

    async def test_implicit_root_team_no_team_existing(
        self,
        sdb: Database,
    ) -> None:
        res = await self._request(1, 0)
        assert_extension_error(res, "Root team not found or access denied")

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
        res = await self._request(33, 10, user_id="gh|XXX")
        assert_extension_error(res, "JIRA has not been installed to the metadata yet.")


class TestGoals(BaseGoalsTest):
    async def test_no_goals_for_team(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1),
            TeamFactory(id=2, parent_id=1),
            GoalFactory(id=1),
            TeamGoalFactory(goal_id=1, team_id=1),
        )

        res = await self._request(1, 2)
        assert res["data"]["goals"] == []

    async def test_single_team_goal(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, name="A-Team", members=[39789, 40020, 40191]),
            GoalFactory(
                id=20,
                metric=PullRequestMetricID.PR_REVIEW_COMMENTS_PER,
                valid_from=datetime(2019, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2022, 1, 1, tzinfo=timezone.utc),
            ),
            TeamGoalFactory(goal_id=20, team_id=10, target=1.23),
        )
        res = await self._request(1, 10)

        assert len(res["data"]["goals"]) == 1
        goal = res["data"]["goals"][0]

        assert goal["id"] == 20
        assert goal["validFrom"] == "2019-01-01"
        assert goal["expiresAt"] == "2021-12-31"
        assert goal["repositories"] is None

        team_goal = goal["teamGoal"]
        assert team_goal["team"]["id"] == 10
        assert team_goal["team"]["name"] == "A-Team"
        assert team_goal["children"] == []
        assert team_goal["value"]["target"]["float"] == 1.23
        assert team_goal["value"]["current"]["float"] == pytest.approx(3.7, 0.1)
        assert team_goal["value"]["initial"]["float"] == pytest.approx(11.5, 0.1)

    async def test_two_teams_jira_metric(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789]),
            TeamFactory(id=11, members=[40020], parent_id=10),
            GoalFactory(
                id=20,
                metric=PullRequestMetricID.PR_MEDIAN_SIZE,
                valid_from=datetime(2019, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2022, 1, 1, tzinfo=timezone.utc),
            ),
            GoalFactory(
                id=21,
                metric=JIRAMetricID.JIRA_RESOLVED,
                valid_from=datetime(2020, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2022, 1, 1, tzinfo=timezone.utc),
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
        res = await self._request(1, 10)

        goal_20 = res["data"]["goals"][0]
        assert goal_20["id"] == 20

        assert (team_goal_10 := goal_20["teamGoal"])["team"]["id"] == 10
        assert team_goal_10["team"]["membersCount"] == 1
        assert team_goal_10["value"]["target"]["int"] == 1
        assert team_goal_10["value"]["initial"]["int"] == 171
        assert team_goal_10["value"]["current"]["int"] == 64

        assert (team_goal_11 := team_goal_10["children"][0])["team"]["id"] == 11
        assert team_goal_11["team"]["membersCount"] == 1
        assert team_goal_11["team"]["totalMembersCount"] == 1
        assert team_goal_11["value"]["target"]["int"] == 4
        assert team_goal_11["value"]["initial"]["int"] == 507
        assert team_goal_11["value"]["current"]["int"] == 47

        goal_21 = res["data"]["goals"][1]
        assert goal_21["id"] == 21

        assert (team_goal_10 := goal_21["teamGoal"])["team"]["id"] == 10
        assert team_goal_10["value"]["target"]["int"] == 20
        assert team_goal_10["value"]["initial"]["int"] == 3
        assert team_goal_10["value"]["current"]["int"] == 572

        assert (team_goal_11 := team_goal_10["children"][0])["team"]["id"] == 11
        assert team_goal_11["value"]["target"]["int"] == 40
        assert team_goal_11["value"]["initial"]["int"] == 3
        assert team_goal_11["value"]["current"]["int"] == 528

    async def test_child_team_with_no_goal(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[39789]),
            TeamFactory(id=10, parent_id=1, members=[39789]),
            GoalFactory(
                id=20,
                metric=PullRequestMetricID.PR_REVIEW_COMMENTS_PER,
                valid_from=datetime(2019, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2022, 1, 1, tzinfo=timezone.utc),
            ),
            TeamGoalFactory(goal_id=20, team_id=1, target=1),
        )
        res = await self._request(1, 0)
        goal = res["data"]["goals"][0]

        assert (team_1_goal := goal["teamGoal"])["team"]["id"] == 1
        assert (team_10_goal := team_1_goal["children"][0])["team"]["id"] == 10

        assert team_1_goal["value"]["initial"]["float"] is not None
        assert team_1_goal["value"]["current"]["float"] is not None
        # also metrics for child team with no goal are returned
        assert team_10_goal["value"]["initial"]["float"] is not None
        assert team_10_goal["value"]["current"]["float"] is not None

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
            TeamGoalFactory(goal_id=20, team_id=11, target="foo"),
            TeamGoalFactory(goal_id=20, team_id=13, target="bar"),
        )
        res = await self._request(1, 10)
        assert len(res["data"]["goals"]) == 1
        goal = res["data"]["goals"][0]

        assert goal["id"] == 20

        assert (team_goal_10 := goal["teamGoal"])["team"]["id"] == 10
        assert len(team_goal_10["children"]) == 2
        assert team_goal_10["value"]["target"]["float"] == 10.1

        assert (team_goal_11 := team_goal_10["children"][0])["team"]["id"] == 11
        assert len(team_goal_11["children"]) == 1
        assert team_goal_11["value"]["target"]["str"] == "foo"

        assert (team_goal_12 := team_goal_10["children"][1])["team"]["id"] == 12
        assert not team_goal_12["children"]
        assert team_goal_12["value"]["target"] is None

        assert (team_goal_13 := team_goal_11["children"][0])["team"]["id"] == 13
        assert not team_goal_13["children"]
        assert team_goal_13["value"]["target"]["str"] == "bar"

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
                valid_from=datetime(2000, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2001, 1, 1, tzinfo=timezone.utc),
            ),
            GoalFactory(
                id=21,
                metric=PullRequestMetricID.PR_REVIEW_COMMENTS_PER,
                valid_from=datetime(2004, 7, 1, tzinfo=timezone.utc),
                expires_at=datetime(2004, 10, 1, tzinfo=timezone.utc),
            ),
            TeamGoalFactory(goal_id=20, team_id=10, target=1),
            TeamGoalFactory(goal_id=20, team_id=12, target=2),
            TeamGoalFactory(goal_id=21, team_id=11, target=3),
            TeamGoalFactory(goal_id=21, team_id=12, target=4),
        )
        res = await self._request(1, 10)
        assert len(res["data"]["goals"]) == 2

        assert (goal_20 := res["data"]["goals"][0])["id"] == 20
        assert goal_20["validFrom"] == "2000-01-01"
        assert goal_20["expiresAt"] == "2000-12-31"

        assert (team_goal_10 := goal_20["teamGoal"])["team"]["id"] == 10
        assert team_goal_10["value"]["target"]["int"] == 1
        assert len(team_goal_10["children"]) == 2

        assert (team_goal_11 := team_goal_10["children"][0])["team"]["id"] == 11
        assert team_goal_11["value"]["target"] is None
        assert not team_goal_11["children"]

        assert (team_goal_12 := team_goal_10["children"][1])["team"]["id"] == 12
        assert team_goal_12["value"]["target"]["int"] == 2
        assert len(team_goal_12["children"]) == 1

        assert (team_goal_13 := team_goal_12["children"][0])["team"]["id"] == 13
        assert team_goal_13["value"]["target"] is None
        assert not team_goal_13["children"]

        # same tree is built for goal 21, with different values
        assert (goal_21 := res["data"]["goals"][1])["id"] == 21
        assert goal_21["validFrom"] == "2004-07-01"
        assert goal_21["expiresAt"] == "2004-09-30"

        assert (team_goal_10 := goal_21["teamGoal"])["team"]["id"] == 10
        assert team_goal_10["value"]["target"] is None
        assert len(team_goal_10["children"]) == 2

        assert (team_goal_11 := team_goal_10["children"][0])["team"]["id"] == 11
        assert team_goal_11["value"]["target"]["int"] == 3
        assert not team_goal_11["children"]

        assert (team_goal_12 := team_goal_10["children"][1])["team"]["id"] == 12
        assert team_goal_12["value"]["target"]["int"] == 4
        assert len(team_goal_12["children"]) == 1

        assert (team_goal_13 := team_goal_12["children"][0])["team"]["id"] == 13
        assert team_goal_13["value"]["target"] is None
        assert not team_goal_13["children"]

    async def test_timedelta_metric(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789]),
            GoalFactory(
                id=20,
                metric=PullRequestMetricID.PR_LEAD_TIME,
                valid_from=datetime(2019, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2019, 7, 1, tzinfo=timezone.utc),
            ),
            TeamGoalFactory(goal_id=20, team_id=10, target="20001s"),
        )

        res = await self._request(1, 10)
        assert len(res["data"]["goals"]) == 1

        goal = res["data"]["goals"][0]
        assert goal["teamGoal"]["team"]["id"] == 10

        assert goal["teamGoal"]["value"]["target"]["str"] == "20001s"
        assert goal["teamGoal"]["value"]["current"]["str"] == "4731059s"
        assert goal["teamGoal"]["value"]["initial"]["str"] == "689712s"

    async def test_only_with_targets_param(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, parent_id=None, members=[39789]),
            TeamFactory(id=11, parent_id=10, members=[39789]),
            TeamFactory(id=12, parent_id=11, members=[39789]),
            GoalFactory(id=20),
            TeamGoalFactory(goal_id=20, team_id=11, target=1.23),
        )
        res = await self._request(1, 10, only_with_targets=True)

        assert len(res["data"]["goals"]) == 1

        goal = res["data"]["goals"][0]

        assert (team_goal_10 := goal["teamGoal"])["team"]["id"] == 10
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
        res = await self._request(1, 10)

        goals = res["data"]["goals"]
        assert [goal["id"] for goal in goals] == [20]

    async def test_repositories(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789]),
            GoalFactory(id=20, repositories=[[1, None]]),
            GoalFactory(id=21, repositories=[[1, "a"], [1, "b"]]),
            TeamGoalFactory(goal_id=20, team_id=10, target=1),
            TeamGoalFactory(goal_id=21, team_id=10, target=20),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [md_factory.RepositoryFactory(node_id=1, full_name="athenianco/repo")]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)
            res = await self._request(1, 10)

        assert len(res["data"]["goals"]) == 2
        assert res["data"]["goals"][0]["id"] == 20
        assert res["data"]["goals"][0]["repositories"] == ["github.com/athenianco/repo"]
        assert res["data"]["goals"][1]["id"] == 21
        assert res["data"]["goals"][1]["repositories"] == [
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
                valid_from=datetime(2019, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2022, 1, 1, tzinfo=timezone.utc),
                repositories=[[39652769, None]],
            ),
            TeamGoalFactory(goal_id=20, team_id=10, target=1, repositories=[[40550, None]]),
        )
        res = await self._request(1, 0)
        goal = res["data"]["goals"][0]
        assert goal["id"] == 20
        assert goal["repositories"] == ["github.com/src-d/gitbase"]

        assert (team_1_goal := goal["teamGoal"])["team"]["id"] == 1
        assert (team_10_goal := team_1_goal["children"][0])["team"]["id"] == 10

        # metrics for team 1 have been computed considering default goal repositories filter
        assert team_1_goal["value"]["initial"]["int"] == 0
        # metrics for team 10 have been computed considering repositories [[40550, None]] filter
        assert team_10_goal["value"]["initial"]["int"] == 88

    async def test_same_team_different_repos(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[39789]),
            GoalFactory(
                id=20,
                metric=PullRequestMetricID.PR_ALL_COUNT,
                valid_from=datetime(2019, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2022, 1, 1, tzinfo=timezone.utc),
            ),
            GoalFactory(
                id=21,
                metric=PullRequestMetricID.PR_ALL_COUNT,
                valid_from=datetime(2019, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2022, 1, 1, tzinfo=timezone.utc),
            ),
            TeamGoalFactory(goal_id=20, team_id=1, repositories=[[40550, None]]),
            TeamGoalFactory(goal_id=21, team_id=1, repositories=[[39652769, None]]),
        )
        res = await self._request(1, 0)
        goal_20 = res["data"]["goals"][0]
        assert goal_20["id"] == 20
        assert (tg := goal_20["teamGoal"])["team"]["id"] == 1
        assert tg["value"]["initial"]["int"] == 88

        goal_21 = res["data"]["goals"][1]
        assert goal_21["id"] == 21
        assert (tg := goal_21["teamGoal"])["team"]["id"] == 1
        assert tg["value"]["initial"]["int"] == 0

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
        res = await self._request(1, 0, only_with_targets=True)
        goal_20 = res["data"]["goals"][0]
        assert (tg := goal_20["teamGoal"])["team"]["id"] == 1
        assert tg["value"]["initial"]["int"] == 93

        goal_21 = res["data"]["goals"][1]
        assert (tg := goal_21["teamGoal"])["team"]["id"] == 1
        # value from goal 20 overwites the value for goal 21
        assert tg["value"]["initial"]["int"] == 0

    async def test_jira_fields(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789]),
            GoalFactory(id=20, jira_projects=["PR0"], jira_issue_types=["task", "bug"]),
            GoalFactory(id=21, jira_priorities=["low"]),
            TeamGoalFactory(goal_id=20, team_id=10),
            TeamGoalFactory(goal_id=21, team_id=10),
        )

        res = await self._request(1, 10)

        assert len(res["data"]["goals"]) == 2
        assert res["data"]["goals"][0]["id"] == 20
        assert res["data"]["goals"][0]["jiraProjects"] == ["PR0"]
        assert res["data"]["goals"][0]["jiraPriorities"] is None
        assert res["data"]["goals"][0]["jiraIssueTypes"] == ["task", "bug"]
        assert res["data"]["goals"][1]["id"] == 21
        assert res["data"]["goals"][1]["jiraProjects"] is None
        assert res["data"]["goals"][1]["jiraPriorities"] == ["low"]
        assert res["data"]["goals"][1]["jiraIssueTypes"] is None

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
        res = await self._request(33, 10, user_id="gh|XXX")
        assert res["data"]["goals"][0]["teamGoal"]["value"]["initial"]["int"] == 0


class TestJIRAFiltering(BaseGoalsTest):
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
            res = await self._request(1, 10)
            assert res["data"]["goals"][0]["teamGoal"]["value"]["initial"]["int"] == 93

            # no issue with this priority, nothing is computed
            await self._update_team_goal(sdb, 10, 20, {TeamGoal.jira_priorities: ["prio1"]})
            res = await self._request(1, 10)
            assert res["data"]["goals"][0]["teamGoal"]["value"]["initial"]["int"] == 0

            # unknown priority, nothing is computed
            await self._update_team_goal(sdb, 10, 20, {TeamGoal.jira_priorities: ["prio2"]})
            res = await self._request(1, 10)
            assert res["data"]["goals"][0]["teamGoal"]["value"]["initial"]["int"] == 0

            await self._update_team_goal(sdb, 10, 20, {TeamGoal.jira_priorities: ["prio0"]})
            res = await self._request(1, 10)
            # only pr 162901 is counted
            assert res["data"]["goals"][0]["teamGoal"]["value"]["initial"]["int"] == 1

    async def test_pr_metric_unexisting_project(self, sdb: Database, mdb_rw: Database) -> None:
        metric = PullRequestMetricID.PR_OPENED
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789, 40020, 40191]),
            GoalFactory(
                id=20, metric=metric, valid_from=dt(2019, 1, 1), expires_at=dt(2022, 1, 1),
            ),
            GoalFactory(
                id=21, metric=metric, valid_from=dt(2019, 1, 1), expires_at=dt(2022, 1, 1),
            ),
            TeamGoalFactory(goal_id=20, team_id=10, jira_projects=["P1"]),
            TeamGoalFactory(goal_id=21, team_id=10, jira_projects=None),
        )
        res = await self._request(1, 10)
        assert res["data"]["goals"][0]["teamGoal"]["value"]["initial"]["int"] == 0
        assert res["data"]["goals"][1]["teamGoal"]["value"]["initial"]["int"] == 93

    async def test_pr_metric_more_goals(self, sdb: Database, mdb_rw: Database) -> None:
        metric = PullRequestMetricID.PR_OPENED
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789, 40020, 40191]),
            GoalFactory(
                id=20, metric=metric, valid_from=dt(2019, 1, 1), expires_at=dt(2022, 1, 1),
            ),
            GoalFactory(
                id=21, metric=metric, valid_from=dt(2019, 1, 1), expires_at=dt(2022, 1, 1),
            ),
            GoalFactory(
                id=22, metric=metric, valid_from=dt(2019, 1, 1), expires_at=dt(2022, 1, 1),
            ),
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
            res = await self._request(1, 10)
            assert res["data"]["goals"][0]["id"] == 20
            assert res["data"]["goals"][0]["teamGoal"]["value"]["initial"]["int"] == 93

            assert res["data"]["goals"][1]["id"] == 21
            assert res["data"]["goals"][1]["teamGoal"]["value"]["initial"]["int"] == 1

            assert res["data"]["goals"][2]["id"] == 22
            assert res["data"]["goals"][2]["teamGoal"]["value"]["initial"]["int"] == 0

    async def test_pr_metric_complex_filters(self, sdb: Database, mdb_rw: Database) -> None:
        metric = PullRequestMetricID.PR_OPENED
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789, 40020, 40191]),
            TeamFactory(id=11, members=[39789, 40020, 40191], parent_id=10),
            GoalFactory(
                id=20, metric=metric, valid_from=dt(2019, 1, 1), expires_at=dt(2022, 1, 1),
            ),
            GoalFactory(
                id=21, metric=metric, valid_from=dt(2019, 1, 1), expires_at=dt(2022, 1, 1),
            ),
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
            res = await self._request(1, 10)
            assert (goal_20 := res["data"]["goals"][0])["id"] == 20
            assert goal_20["teamGoal"]["team"]["id"] == 10
            # both PRs are selected, priorities is P0, P1
            assert goal_20["teamGoal"]["value"]["initial"]["int"] == 2

            assert goal_20["teamGoal"]["children"][0]["team"]["id"] == 11
            # only 162908 is selected
            assert goal_20["teamGoal"]["children"][0]["value"]["initial"]["int"] == 1

            # only 162901 is selected
            assert (goal_21 := res["data"]["goals"][1])["id"] == 21
            assert goal_21["teamGoal"]["team"]["id"] == 10

            # no PR selected
            assert goal_21["teamGoal"]["children"][0]["team"]["id"] == 11
            assert goal_21["teamGoal"]["children"][0]["value"]["initial"]["int"] == 0

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

            res = await self._request(1, 10)
            assert (goal_20 := res["data"]["goals"][0])["id"] == 20
            assert goal_20["teamGoal"]["team"]["id"] == 10
            # issues 10, 11 and 13 counted
            assert goal_20["teamGoal"]["team"]["id"] == 10
            assert goal_20["teamGoal"]["value"]["current"]["int"] == 3

            # team goal is missing, filters taken from goal; issues 10 and 11 counted
            assert goal_20["teamGoal"]["children"][0]["team"]["id"] == 11
            assert goal_20["teamGoal"]["children"][0]["value"]["current"]["int"] == 2

            assert (goal_21 := res["data"]["goals"][1])["id"] == 21
            assert goal_21["teamGoal"]["team"]["id"] == 10
            # issues 12 and 14 counted
            assert goal_21["teamGoal"]["value"]["current"]["int"] == 2

            assert (goal_22 := res["data"]["goals"][2])["id"] == 22
            # issues 11 and 13 counted
            assert goal_22["teamGoal"]["team"]["id"] == 10
            assert goal_22["teamGoal"]["value"]["current"]["int"] == 2

            assert goal_22["teamGoal"]["children"][0]["team"]["id"] == 11
            # issues 11 counted
            assert goal_22["teamGoal"]["children"][0]["value"]["current"]["int"] == 1

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
        res = await self._request(1, 10)
        assert res["data"]["goals"][0]["teamGoal"]["value"]["initial"]["int"] == 6
        assert res["data"]["goals"][1]["teamGoal"]["value"]["initial"]["int"] == 11

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
