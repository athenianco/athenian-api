from datetime import datetime, timezone
from typing import Any

from aiohttp.test_utils import TestClient
from freezegun import freeze_time
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.models.state.models import Goal, Team, TeamGoal
from tests.align.utils import align_graphql_request, assert_extension_error, get_extension_error
from tests.conftest import DEFAULT_HEADERS
from tests.testutils.db import assert_existing_row, assert_missing_row, db_datetime_equals, \
    model_insert_stmt, models_insert
from tests.testutils.factory.state import GoalFactory, TeamFactory, TeamGoalFactory


class BaseGoalTest:
    @classmethod
    async def _assert_no_goal_exists(cls, sdb: Database) -> None:
        assert await sdb.fetch_one(sa.select(Goal)) is None
        assert await sdb.fetch_one(sa.select(TeamGoal)) is None


class BaseCreateGoalTest(BaseGoalTest):
    async def _request(self, variables: dict, client: TestClient) -> dict:
        body = {"query": self._QUERY, "variables": variables}
        return await align_graphql_request(client, headers=DEFAULT_HEADERS, json=body)

    _QUERY = """
        mutation ($accountId: Int!, $createGoalInput: CreateGoalInput!) {
          createGoal(accountId: $accountId, input: $createGoalInput) {
            goal {
              id
            }
            errors
          }
        }
    """

    @classmethod
    def _mk_input(self, **kwargs: Any) -> dict:
        kwargs.setdefault("templateId", 1)
        kwargs.setdefault("validFrom", "2022-01-01")
        kwargs.setdefault("expiresAt", "2022-03-31")
        kwargs.setdefault("teamGoals", [])
        return kwargs


class TestCreateGoalErrors(BaseCreateGoalTest):
    async def test_invalid_template_id(self, client: TestClient, sdb: Database) -> None:
        variables = {
            "createGoalInput": self._mk_input(templateId=1234),
            "accountId": 1,
        }
        res = await self._request(variables, client)
        assert_extension_error(res, "Invalid templateId 1234")
        await self._assert_no_goal_exists(sdb)

    async def test_invalid_date(self, client: TestClient, sdb: Database) -> None:
        variables = {
            "createGoalInput": self._mk_input(validFrom="not-a-date"),
            "accountId": 1,
        }
        res = await self._request(variables, client)
        assert "not-a-date" in res["errors"][0]["message"]
        await self._assert_no_goal_exists(sdb)

    async def test_missing_date(self, client: TestClient, sdb: Database) -> None:
        variables = {
            "createGoalInput": self._mk_input(expiresAt=None),
            "accountId": 1,
        }
        res = await self._request(variables, client)
        assert "expiresAt" in res["errors"][0]["message"]
        await self._assert_no_goal_exists(sdb)

    async def test_account_mismatch(self, client: TestClient, sdb: Database) -> None:
        variables = {"createGoalInput": self._mk_input(), "accountId": 3}
        res = await self._request(variables, client)
        assert "data" not in res
        await self._assert_no_goal_exists(sdb)

    async def test_no_team_goals(self, client: TestClient, sdb: Database) -> None:
        variables = {"createGoalInput": self._mk_input(teamGoals=[]), "accountId": 1}
        res = await self._request(variables, client)
        assert "data" not in res
        assert_extension_error(res, "At least one teamGoals is required")
        await self._assert_no_goal_exists(sdb)

    async def test_more_goals_for_same_team(self, client: TestClient, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=1, id=10)))

        team_goals = [
            {"teamId": 10, "target": {"int": 42}},
            {"teamId": 10, "target": {"int": 44}},
        ]
        variables = {
            "createGoalInput": self._mk_input(teamGoals=team_goals),
            "accountId": 1,
        }
        res = await self._request(variables, client)
        assert "data" not in res
        assert_extension_error(res, "More than one team goal with the same teamId")
        await self._assert_no_goal_exists(sdb)

    async def test_unexisting_team(self, client: TestClient, sdb: Database) -> None:
        variables = {
            "createGoalInput": self._mk_input(
                teamGoals={"teamId": 10, "target": {"int": 10}},
            ),
            "accountId": 1,
        }
        res = await self._request(variables, client)
        assert "data" not in res
        assert_extension_error(res, "Some teamId-s don't exist or access denied: 10")
        await self._assert_no_goal_exists(sdb)

    async def test_goals_for_other_account_team(self, client: TestClient, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10), TeamFactory(owner_id=2, id=20))

        team_goals = [
            {"teamId": 10, "target": {"int": 42}},
            {"teamId": 20, "target": {"int": 44}},
        ]
        variables = {
            "createGoalInput": self._mk_input(teamGoals=team_goals), "accountId": 1,
        }
        res = await self._request(variables, client)
        assert "data" not in res
        assert_extension_error(res, "Some teamId-s don't exist or access denied: 20")
        await self._assert_no_goal_exists(sdb)

    async def test_inverted_dates(self, client: TestClient, sdb: Database):
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=1, id=10)))

        team_goals = [{"teamId": 10, "target": {"int": 42}}]

        variables = {
            "createGoalInput": self._mk_input(
                validFrom="2022-04-01",
                expiresAt="2022-01-01",
                teamGoals=team_goals,
            ),
            "accountId": 1,
        }
        res = await self._request(variables, client)
        assert "data" not in res
        assert_extension_error(res, "Goal expiresAt cannot precede validFrom")
        await assert_missing_row(sdb, Goal, account_id=1)


class TestCreateGoals(BaseCreateGoalTest):
    async def test_create_single_team_goal(self, client: TestClient, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=1, id=10)))

        team_goals = [{"teamId": 10, "target": {"int": 42}}]

        variables = {
            "createGoalInput": self._mk_input(
                templateId=1,
                validFrom="2022-01-01",
                expiresAt="2022-12-31",
                teamGoals=team_goals,
            ),
            "accountId": 1,
        }
        res = await self._request(variables, client)
        assert "errors" not in res

        new_goal_id = res["data"]["createGoal"]["goal"]["id"]

        goal_row = await sdb.fetch_one(sa.select(Goal).where(Goal.id == new_goal_id))
        assert goal_row is not None
        assert goal_row["account_id"] == 1
        assert goal_row["template_id"] == 1
        assert db_datetime_equals(
            sdb,
            goal_row["valid_from"],
            datetime(2022, 1, 1, tzinfo=timezone.utc),
        )
        # expires_at is moved to the day after the one received in api
        assert db_datetime_equals(
            sdb,
            goal_row["expires_at"],
            datetime(2023, 1, 1, tzinfo=timezone.utc),
        )

        team_goals = await sdb.fetch_all(
            sa.select([TeamGoal]).where(TeamGoal.goal_id == 1),
        )
        assert len(team_goals) == 1

        assert team_goals[0]["team_id"] == 10
        assert team_goals[0]["target"] == 42

    async def test_create_multiple_team_goals(self, client: TestClient, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(owner_id=1, id=10), TeamFactory(owner_id=1, id=20))

        team_goals = [
            {"teamId": 10, "target": {"int": 42}},
            {"teamId": 20, "target": {"str": "foobar"}},
        ]
        variables = {
            "createGoalInput": self._mk_input(teamGoals=team_goals),
            "accountId": 1,
        }
        res = await self._request(variables, client)
        assert "errors" not in res
        assert res["data"]["createGoal"]["errors"] is None

        new_goal_id = res["data"]["createGoal"]["goal"]["id"]

        goal_row = await sdb.fetch_one(sa.select([Goal]).where(Goal.id == new_goal_id))
        assert goal_row is not None
        assert goal_row["account_id"] == 1
        assert goal_row["template_id"] == 1

        team_goals = await sdb.fetch_all(
            sa.select([TeamGoal]).where(TeamGoal.goal_id == 1).order_by(TeamGoal.team_id),
        )
        assert len(team_goals) == 2

        assert team_goals[0]["team_id"] == 10
        assert team_goals[0]["target"] == 42
        assert team_goals[1]["team_id"] == 20
        assert team_goals[1]["target"] == "foobar"

    async def test_datetime_as_date(self, client: TestClient, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=1, id=10)))
        team_goals = [{"teamId": 10, "target": {"int": 42}}]

        variables = {
            "createGoalInput": self._mk_input(
                validFrom="2022-05-04T08:01:32.485897",
                expiresAt="2022-06-01",
                teamGoals=team_goals,
            ),
            "accountId": 1,
        }
        res = await self._request(variables, client)
        assert "errors" not in res
        new_goal_id = res["data"]["createGoal"]["goal"]["id"]

        goal_row = await sdb.fetch_one(sa.select([Goal]).where(Goal.id == new_goal_id))
        assert db_datetime_equals(
            sdb,
            goal_row["valid_from"],
            datetime(2022, 5, 4, tzinfo=timezone.utc),
        )


class BaseRemoveGoalTest(BaseGoalTest):
    _QUERY = """
        mutation ($accountId: Int!, $goalId: Int!) {
          removeGoal(accountId: $accountId, goalId: $goalId) {
            success
          }
        }
    """

    async def _request(self, accountId: int, goalId: int, client: TestClient) -> dict:
        variables = {"accountId": accountId, "goalId": goalId}
        body = {"query": self._QUERY, "variables": variables}
        return await align_graphql_request(client, headers=DEFAULT_HEADERS, json=body)


class TestRemoveGoalErrors(BaseRemoveGoalTest):
    async def test_non_existing_goal(self, client: TestClient) -> None:
        res = await self._request(1, 999, client)
        assert_extension_error(res, "Goal 999 not found")

    async def test_account_mismatch(self, client: TestClient, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(GoalFactory(id=100, account_id=2)))

        res = await self._request(1, 100, client)
        assert_extension_error(res, "Goal 100 not found")
        await assert_existing_row(sdb, Goal, id=100)


class TestRemoveGoal(BaseRemoveGoalTest):
    async def test_remove(self, client: TestClient, sdb: Database):
        await sdb.execute(model_insert_stmt(GoalFactory(id=100, account_id=1)))

        res = await self._request(1, 100, client)
        assert "errors" not in res
        assert res["data"]["removeGoal"]["success"]
        await self._assert_no_goal_exists(sdb)

    async def test_remove_with_team_goals(self, client: TestClient, sdb: Database):
        await models_insert(
            sdb,
            TeamFactory(owner_id=1, id=10),
            TeamFactory(owner_id=1, id=20),
            GoalFactory(id=100, account_id=1),
            GoalFactory(id=200, account_id=1),
            GoalFactory(id=300, account_id=2),
            TeamGoalFactory(team_id=10, goal_id=100),
            TeamGoalFactory(team_id=20, goal_id=100),
            TeamGoalFactory(team_id=20, goal_id=200),
        )

        res = await self._request(1, 100, client)
        assert "errors" not in res
        assert res["data"]["removeGoal"]["success"]

        await assert_missing_row(sdb, Goal, id=100, account_id=1)
        await assert_missing_row(sdb, TeamGoal, goal_id=100)

        await assert_existing_row(sdb, Goal, id=200)
        await assert_existing_row(sdb, Goal, id=300)
        await assert_existing_row(sdb, TeamGoal, goal_id=200)


class BaseUpdateGoalTest(BaseGoalTest):
    _QUERY = """
        mutation ($accountId: Int!, $input: UpdateGoalInput!) {
          updateGoal(accountId: $accountId, input: $input) {
            goal {
              id
            }
          }
        }
    """

    async def _request(self, variables: dict, client: TestClient) -> dict:
        body = {"query": self._QUERY, "variables": variables}
        return await align_graphql_request(client, headers=DEFAULT_HEADERS, json=body)


class TestUpdateGoalErrors(BaseUpdateGoalTest):
    async def test_dupl_team_ids(self, client: TestClient, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10))

        team_changes = [
            {"teamId": 10, "target": {"int": 10}}, {"teamId": 10, "remove": True},
        ]
        variables = {"accountId": 1, "input": {"goalId": 100, "teamGoalChanges": team_changes}}
        res = await self._request(variables, client)
        assert_extension_error(res, "Multiple changes for teamId-s: 10")

    async def test_invalid_team_changes(self, client: TestClient, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10), TeamFactory(id=20))

        team_changes = [
            {"teamId": 10, "target": {"int": 10}, "remove": True},
            {"teamId": 20, "remove": False},
        ]
        variables = {"accountId": 1, "input": {"goalId": 100, "teamGoalChanges": team_changes}}
        res = await self._request(variables, client)
        error = get_extension_error(res)
        assert "Both remove and new target present for teamId-s: 10" in error
        assert "Invalid target for teamId-s: 20" in error

    async def test_invalid_team_target(self, client: TestClient, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10))

        team_changes = [{"teamId": 10, "target": {}}]
        variables = {"accountId": 1, "input": {"goalId": 100, "teamGoalChanges": team_changes}}
        res = await self._request(variables, client)
        error = get_extension_error(res)
        assert "Invalid target for teamId-s: 10" in error

    async def test_remove_account_mismatch(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100, account_id=2),
            TeamFactory(id=10, owner_id=2),
            TeamGoalFactory(team_id=10, goal_id=100),
        )

        team_changes = [{"teamId": 10, "remove": True}]
        variables = {"accountId": 1, "input": {"goalId": 100, "teamGoalChanges": team_changes}}
        res = await self._request(variables, client)
        assert_extension_error(res, "TeamGoal-s to remove not found for teams: 10")
        await assert_existing_row(sdb, TeamGoal, team_id=10, goal_id=100)

    async def test_goal_account_mismatch(self, client: TestClient, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100, account_id=2), TeamFactory(id=10))

        team_changes = [{"teamId": 10, "target": {"int": 10}}]
        variables = {"accountId": 1, "input": {"goalId": 100, "teamGoalChanges": team_changes}}
        res = await self._request(variables, client)
        assert_extension_error(res, "Goal 100 doesn't exist or access denied")

    async def test_assign_team_account_mismatch(self, client: TestClient, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100), TeamFactory(id=10, owner_id=2))
        team_changes = [{"teamId": 10, "target": {"int": 10}}]
        variables = {"accountId": 1, "input": {"goalId": 100, "teamGoalChanges": team_changes}}
        res = await self._request(variables, client)
        assert_extension_error(res, "Team-s don't exist or access denied: 10")
        await assert_missing_row(sdb, TeamGoal, team_id=10)

    async def test_team_goals_to_remove_are_missing(
        self, client: TestClient, sdb: Database,
    ) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100),
            TeamFactory(id=10),
            TeamFactory(id=20),
            TeamFactory(id=30),
            TeamFactory(id=40),
            TeamGoalFactory(team_id=10, goal_id=100),
            TeamGoalFactory(team_id=30, goal_id=100, target=9999),
        )

        team_changes = [
            {"teamId": 10, "remove": True},
            {"teamId": 20, "remove": True},
            {"teamId": 30, "target": {"float": 4.2}},
            {"teamId": 40, "target": {"int": 42}},
        ]
        variables = {"accountId": 1, "input": {"goalId": 100, "teamGoalChanges": team_changes}}
        res = await self._request(variables, client)
        assert_extension_error(res, "TeamGoal-s to remove not found for teams: 20")

        await assert_existing_row(sdb, TeamGoal, team_id=10, goal_id=100)
        # goal for team 30 not changed
        team30_row = await assert_existing_row(sdb, TeamGoal, team_id=30, goal_id=100)
        assert team30_row["target"] == 9999
        # goal for team 40 not created
        await assert_missing_row(sdb, TeamGoal, team_id=40, goal_id=100)

    @classmethod
    def _assert_error_response(cls, response: dict, error: str) -> None:
        assert "errors" not in response
        assert response["data"]["updateGoal"]["goal"] is None
        assert response["data"]["updateGoal"]["errors"] == [error]


class TestUpdateGoal(BaseUpdateGoalTest):
    async def test_no_team_goal_changes(self, client: TestClient, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(GoalFactory(id=100, account_id=1)))
        variables = {"accountId": 1, "input": {"goalId": 100}}
        res = await self._request(variables, client)
        assert "errors" not in res

    async def test_deletions(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamFactory(id=20),
            TeamFactory(id=30),
            GoalFactory(id=101),
            TeamGoalFactory(team_id=10, goal_id=101),
            TeamGoalFactory(team_id=20, goal_id=101),
            TeamGoalFactory(team_id=30, goal_id=101),
        )
        team_changes = [{"teamId": 10, "remove": True}, {"teamId": 20, "remove": True}]
        variables = {
            "accountId": 1, "input": {"goalId": 101, "teamGoalChanges": team_changes},
        }
        res = await self._request(variables, client)
        assert res["data"]["updateGoal"]["goal"]["id"] == 101
        await assert_missing_row(sdb, TeamGoal, team_id=10)
        await assert_missing_row(sdb, TeamGoal, team_id=20)
        await assert_existing_row(sdb, Goal, id=101)
        await assert_existing_row(sdb, Team, id=10)
        await assert_existing_row(sdb, Team, id=20)
        await assert_existing_row(sdb, TeamGoal, team_id=30)

    @freeze_time("2022-04-01T09:30:00")
    async def test_some_changes(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=100),
            TeamFactory(id=10),
            TeamFactory(id=20),
            TeamFactory(id=30),
            TeamGoalFactory(team_id=10, goal_id=100),
            TeamGoalFactory(
                team_id=20, goal_id=100, target=9999,
                created_at=datetime(1, 1, 2, tzinfo=timezone.utc),
                updated_at=datetime(1, 1, 3, tzinfo=timezone.utc),
            ),
        )

        team_changes = [
            {"teamId": 10, "remove": True},
            {"teamId": 20, "target": {"int": 8888}},
            {"teamId": 30, "target": {"int": 7777}},
        ]
        variables = {"accountId": 1, "input": {"goalId": 100, "teamGoalChanges": team_changes}}
        res = await self._request(variables, client)
        assert "errors" not in res
        assert res["data"]["updateGoal"]["goal"]["id"] == 100

        await assert_missing_row(sdb, TeamGoal, team_id=10, goal_id=100)

        team_goal_20_row = await assert_existing_row(sdb, TeamGoal, team_id=20, goal_id=100)
        assert team_goal_20_row["target"] == 8888
        expected_created_at = datetime(1, 1, 2, tzinfo=timezone.utc)
        assert db_datetime_equals(sdb, team_goal_20_row["created_at"], expected_created_at)
        expected_updated_at = datetime(2022, 4, 1, 9, 30, tzinfo=timezone.utc)
        assert db_datetime_equals(sdb, team_goal_20_row["updated_at"], expected_updated_at)

        team_goal_30_row = await assert_existing_row(sdb, TeamGoal, team_id=30, goal_id=100)
        assert team_goal_30_row["target"] == 7777
