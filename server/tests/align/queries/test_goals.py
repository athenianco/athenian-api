from datetime import datetime, timezone
from typing import Optional, Sequence

import pytest

from athenian.api.db import Database
from athenian.api.models.web import JIRAMetricID, PullRequestMetricID
from tests.align.utils import (
    align_graphql_request,
    assert_extension_error,
    build_fragment,
    build_recursive_fields_structure,
)
from tests.testutils.db import models_insert
from tests.testutils.factory.state import (
    GoalFactory,
    MappedJIRAIdentityFactory,
    TeamFactory,
    TeamGoalFactory,
)
from tests.testutils.requester import Requester


class BaseGoalsTest(Requester):
    _ALL_GOAL_FIELDS = ("id", "name", "metric", "validFrom", "expiresAt")
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
    ) -> dict:
        if repositories is None:
            repositories = {}
        else:
            repositories = {"repositories": repositories}
        body = {
            "query": self._query(goal_fields, value_fields, team_fields, depth=depth),
            "variables": {
                "accountId": account_id,
                "teamId": team_id,
                "onlyWithTargets": only_with_targets,
                **repositories,
            },
        }
        return await align_graphql_request(self.client, headers=self.headers, json=body)


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
