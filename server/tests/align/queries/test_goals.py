from datetime import datetime, timezone
from typing import Sequence

from aiohttp.test_utils import TestClient
import pytest

from athenian.api.db import Database
from tests.align.utils import (
    align_graphql_request,
    assert_extension_error,
    build_fragment,
    build_recursive_fields_structure,
)
from tests.conftest import DEFAULT_HEADERS
from tests.testutils.db import models_insert
from tests.testutils.factory.state import (
    GoalFactory,
    MappedJIRAIdentityFactory,
    TeamFactory,
    TeamGoalFactory,
)


class BaseGoalsTest:
    _ALL_GOAL_FIELDS = ("id", "templateId", "validFrom", "expiresAt")
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
        query ($accountId: Int!, $teamId: Int!) {{
          goals(accountId: $accountId, teamId: $teamId) {{
             {fields}
          }}
        }}
        """

    async def _request(
        self,
        account_id: int,
        team_id: int,
        client: TestClient,
        goal_fields=_ALL_GOAL_FIELDS,
        value_fields=_ALL_VALUE_FIELDS,
        team_fields=_ALL_TEAM_FIELDS,
        depth: int = 6,
    ) -> dict:
        body = {
            "query": self._query(goal_fields, value_fields, team_fields, depth=depth),
            "variables": {"accountId": account_id, "teamId": team_id},
        }
        return await align_graphql_request(client, headers=DEFAULT_HEADERS, json=body)


class TestGoalsErrors(BaseGoalsTest):
    async def test_not_existing_team(self, client: TestClient, sdb: Database) -> None:
        res = await self._request(1, 1, client)
        assert_extension_error(res, "Team 1 not found or access denied")

    async def test_implicit_root_team_no_team_existing(
        self,
        client: TestClient,
        sdb: Database,
    ) -> None:
        res = await self._request(1, 0, client)
        assert_extension_error(res, "Root team not found or access denied")


class TestGoals(BaseGoalsTest):
    async def test_no_goals_for_team(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1),
            TeamFactory(id=2, parent_id=1),
            GoalFactory(id=1),
            TeamGoalFactory(goal_id=1, team_id=1),
        )

        res = await self._request(1, 2, client)
        assert res["data"]["goals"] == []

    async def test_single_team_goal(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, name="A-Team", members=[39789, 40020, 40191]),
            GoalFactory(
                id=20,
                template_id=2,
                valid_from=datetime(2019, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2022, 1, 1, tzinfo=timezone.utc),
            ),
            TeamGoalFactory(goal_id=20, team_id=10, target=1.23),
        )
        res = await self._request(1, 10, client)

        assert len(res["data"]["goals"]) == 1
        goal = res["data"]["goals"][0]

        assert goal["id"] == 20
        assert goal["templateId"] == 2
        assert goal["validFrom"] == "2019-01-01"
        assert goal["expiresAt"] == "2021-12-31"

        team_goal = goal["teamGoal"]
        assert team_goal["team"]["id"] == 10
        assert team_goal["team"]["name"] == "A-Team"
        assert team_goal["children"] == []
        assert team_goal["value"]["target"]["float"] == 1.23
        assert team_goal["value"]["current"]["float"] == pytest.approx(3.7, 0.1)
        assert team_goal["value"]["initial"]["float"] == pytest.approx(11.5, 0.1)

    async def test_two_teams_jira_metric(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, members=[39789]),
            TeamFactory(id=11, members=[40020], parent_id=10),
            GoalFactory(
                id=20,
                template_id=3,
                valid_from=datetime(2019, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2022, 1, 1, tzinfo=timezone.utc),
            ),
            GoalFactory(
                id=21,
                template_id=6,
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
        res = await self._request(1, 10, client)

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

    async def test_multiple_teams(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1),
            TeamFactory(id=10, parent_id=1),
            TeamFactory(id=11, parent_id=10),
            TeamFactory(id=12, parent_id=10),
            TeamFactory(id=13, parent_id=11),
            GoalFactory(id=20),
            TeamGoalFactory(goal_id=20, team_id=1, target=1),
            TeamGoalFactory(goal_id=20, team_id=10, target=10.1),
            TeamGoalFactory(goal_id=20, team_id=11, target="foo"),
            TeamGoalFactory(goal_id=20, team_id=13, target="bar"),
        )
        res = await self._request(1, 10, client)
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
        assert team_goal_12["value"] is None

        assert (team_goal_13 := team_goal_11["children"][0])["team"]["id"] == 13
        assert not team_goal_13["children"]
        assert team_goal_13["value"]["target"]["str"] == "bar"

    async def test_multiple_goals(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamFactory(id=11, parent_id=10),
            TeamFactory(id=12, parent_id=10),
            TeamFactory(id=13, parent_id=12),
            GoalFactory(
                id=20,
                template_id=1,
                valid_from=datetime(2000, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2001, 1, 1, tzinfo=timezone.utc),
            ),
            GoalFactory(
                id=21,
                template_id=2,
                valid_from=datetime(2004, 7, 1, tzinfo=timezone.utc),
                expires_at=datetime(2004, 10, 1, tzinfo=timezone.utc),
            ),
            TeamGoalFactory(goal_id=20, team_id=10, target=1),
            TeamGoalFactory(goal_id=20, team_id=12, target=2),
            TeamGoalFactory(goal_id=21, team_id=11, target=3),
            TeamGoalFactory(goal_id=21, team_id=12, target=4),
        )
        res = await self._request(1, 10, client)
        assert len(res["data"]["goals"]) == 2

        assert (goal_20 := res["data"]["goals"][0])["id"] == 20
        assert goal_20["templateId"] == 1
        assert goal_20["validFrom"] == "2000-01-01"
        assert goal_20["expiresAt"] == "2000-12-31"

        assert (team_goal_10 := goal_20["teamGoal"])["team"]["id"] == 10
        assert team_goal_10["value"]["target"]["int"] == 1
        assert len(team_goal_10["children"]) == 2

        assert (team_goal_11 := team_goal_10["children"][0])["team"]["id"] == 11
        assert team_goal_11["value"] is None
        assert not team_goal_11["children"]

        assert (team_goal_12 := team_goal_10["children"][1])["team"]["id"] == 12
        assert team_goal_12["value"]["target"]["int"] == 2
        assert len(team_goal_12["children"]) == 1

        assert (team_goal_13 := team_goal_12["children"][0])["team"]["id"] == 13
        assert team_goal_13["value"] is None
        assert not team_goal_13["children"]

        # same tree is built for goal 21, with different values
        assert (goal_21 := res["data"]["goals"][1])["id"] == 21
        assert goal_21["templateId"] == 2
        assert goal_21["validFrom"] == "2004-07-01"
        assert goal_21["expiresAt"] == "2004-09-30"

        assert (team_goal_10 := goal_21["teamGoal"])["team"]["id"] == 10
        assert team_goal_10["value"] is None
        assert len(team_goal_10["children"]) == 2

        assert (team_goal_11 := team_goal_10["children"][0])["team"]["id"] == 11
        assert team_goal_11["value"]["target"]["int"] == 3
        assert not team_goal_11["children"]

        assert (team_goal_12 := team_goal_10["children"][1])["team"]["id"] == 12
        assert team_goal_12["value"]["target"]["int"] == 4
        assert len(team_goal_12["children"]) == 1

        assert (team_goal_13 := team_goal_12["children"][0])["team"]["id"] == 13
        assert team_goal_13["value"] is None
        assert not team_goal_13["children"]

    async def test_timedelta_metric(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            GoalFactory(
                id=20,
                template_id=4,
                valid_from=datetime(2019, 1, 1, tzinfo=timezone.utc),
                expires_at=datetime(2019, 7, 1, tzinfo=timezone.utc),
            ),
            TeamGoalFactory(goal_id=20, team_id=10, target="20001s"),
        )

        res = await self._request(1, 10, client)
        assert len(res["data"]["goals"]) == 1

        goal = res["data"]["goals"][0]
        assert goal["teamGoal"]["team"]["id"] == 10

        assert goal["teamGoal"]["value"]["target"]["str"] == "20001s"
        assert goal["teamGoal"]["value"]["current"]["str"] == "3727969s"
        assert goal["teamGoal"]["value"]["initial"]["str"] == "2126373s"
