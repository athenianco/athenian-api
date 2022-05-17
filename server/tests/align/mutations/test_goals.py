from datetime import datetime, timezone
from typing import Any

from aiohttp.test_utils import TestClient
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.models.state.models import Goal, TeamGoal
from tests.testutils.db import assert_existing_row, assert_missing_row, db_datetime_equals, \
    model_insert_stmt
from tests.testutils.factory.state import GoalFactory, TeamFactory, TeamGoalFactory


class BaseGoalTest:
    @classmethod
    async def _assert_no_goal_exists(cls, sdb: Database) -> None:
        assert await sdb.fetch_one(sa.select([Goal])) is None
        assert await sdb.fetch_one(sa.select([TeamGoal])) is None


class BaseCreateGoalTest(BaseGoalTest):
    async def _request(
        self, variables: dict, client: TestClient, headers: dict,
    ) -> dict:
        body = {"query": self._QUERY, "variables": variables}
        response = await client.request(
            method="POST",
            path="/align/graphql",
            headers=headers,
            json=body,
        )
        assert response.status == 200
        return await response.json()

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
    async def test_invalid_template_id(
        self,
        client: TestClient,
        headers: dict,
        sdb: Database,
    ) -> None:
        variables = {
            "createGoalInput": self._mk_input(templateId=1234),
            "accountId": 1,
        }
        res = await self._request(variables, client, headers)
        assert res["errors"][0]["extensions"]["detail"] == "Invalid templateId 1234"
        await self._assert_no_goal_exists(sdb)

    async def test_invalid_date(
        self, client: TestClient, headers: dict, sdb: Database,
    ) -> None:
        variables = {
            "createGoalInput": self._mk_input(validFrom="not-a-date"),
            "accountId": 1,
        }
        res = await self._request(variables, client, headers)
        assert "not-a-date" in res["errors"][0]["message"]
        await self._assert_no_goal_exists(sdb)

    async def test_missing_date(
        self, client: TestClient, headers: dict, sdb: Database,
    ) -> None:
        variables = {
            "createGoalInput": self._mk_input(expiresAt=None),
            "accountId": 1,
        }
        res = await self._request(variables, client, headers)
        assert "expiresAt" in res["errors"][0]["message"]
        await self._assert_no_goal_exists(sdb)

    async def test_account_mismatch(
        self,
        client: TestClient,
        headers: dict,
        sdb: Database,
    ) -> None:
        variables = {"createGoalInput": self._mk_input(), "accountId": 3}
        res = await self._request(variables, client, headers)
        assert "data" not in res
        await self._assert_no_goal_exists(sdb)

    async def test_no_team_goals(
        self,
        client: TestClient,
        headers: dict,
        sdb: Database,
    ) -> None:
        variables = {"createGoalInput": self._mk_input(teamGoals=[]), "accountId": 1}
        res = await self._request(variables, client, headers)
        assert "data" not in res
        error = res["errors"][0]["extensions"]["detail"]
        assert error == "At least one teamGoals is required"
        await self._assert_no_goal_exists(sdb)

    async def test_more_goals_for_same_team(
        self,
        client: TestClient,
        headers: dict,
        sdb: Database,
    ) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=1, id=10)))

        team_goals = [
            {"teamId": 10, "target": {"int": 42}},
            {"teamId": 10, "target": {"int": 44}},
        ]
        variables = {
            "createGoalInput": self._mk_input(teamGoals=team_goals),
            "accountId": 1,
        }
        res = await self._request(variables, client, headers)
        assert "data" not in res
        error = res["errors"][0]["extensions"]["detail"]
        assert error == "More than one team goal with the same teamId"
        await self._assert_no_goal_exists(sdb)

    async def test_unexisting_team(
        self, client: TestClient, headers: dict, sdb: Database,
    ) -> None:
        variables = {
            "createGoalInput": self._mk_input(
                teamGoals={"teamId": 10, "target": {"int": 10}},
            ),
            "accountId": 1,
        }
        res = await self._request(variables, client, headers)
        assert "data" not in res
        error = res["errors"][0]["extensions"]["detail"]
        assert error == "Some teamId-s don't exist or access denied: 10"
        await self._assert_no_goal_exists(sdb)

    async def test_goals_for_other_account_team(
        self,
        client: TestClient,
        headers: dict,
        sdb: Database,
    ) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=1, id=10)))
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=2, id=20)))

        team_goals = [
            {"teamId": 10, "target": {"int": 42}},
            {"teamId": 20, "target": {"int": 44}},
        ]
        variables = {
            "createGoalInput": self._mk_input(teamGoals=team_goals), "accountId": 1,
        }
        res = await self._request(variables, client, headers)
        assert "data" not in res
        error = res["errors"][0]["extensions"]["detail"]
        assert error == "Some teamId-s don't exist or access denied: 20"
        await self._assert_no_goal_exists(sdb)


class TestCreateGoals(BaseCreateGoalTest):
    async def test_create_single_team_goal(
        self,
        client: TestClient,
        headers: dict,
        sdb: Database,
    ) -> None:
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
        res = await self._request(variables, client, headers)
        assert "errors" not in res

        new_goal_id = res["data"]["createGoal"]["goal"]["id"]

        goal_row = await sdb.fetch_one(sa.select([Goal]).where(Goal.id == new_goal_id))
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

    async def test_create_multiple_team_goals(
        self,
        client: TestClient,
        headers: dict,
        sdb: Database,
    ) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=1, id=10)))
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=1, id=20)))

        team_goals = [
            {"teamId": 10, "target": {"int": 42}},
            {"teamId": 20, "target": {"str": "foobar"}},
        ]
        variables = {
            "createGoalInput": self._mk_input(teamGoals=team_goals),
            "accountId": 1,
        }
        res = await self._request(variables, client, headers)
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

    async def test_datetime_as_date(
        self,
        client: TestClient,
        headers: dict,
        sdb: Database,
    ) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(owner_id=1, id=10)))
        team_goals = [{"teamId": 10, "target": {"int": 42}}]

        variables = {
            "createGoalInput": self._mk_input(
                validFrom="2022-05-04T08:01:32.485897", teamGoals=team_goals,
            ),
            "accountId": 1,
        }
        res = await self._request(variables, client, headers)
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
          removeGoal(accountId: $accountId, id: $goalId) {
            success
          }
        }
    """

    async def _request(
        self, accountId: int, goalId: int, client: TestClient, headers: dict,
    ) -> dict:
        variables = {"accountId": accountId, "goalId": goalId}
        body = {"query": self._QUERY, "variables": variables}
        response = await client.request(
            method="POST",
            path="/align/graphql",
            headers=headers,
            json=body,
        )
        assert response.status == 200
        return await response.json()


class TestRemoveGoalErrors(BaseRemoveGoalTest):
    async def test_non_existing_goal(self, client: TestClient, headers: dict) -> None:
        res = await self._request(1, 999, client, headers)
        assert res["errors"][0]["extensions"]["detail"] == "Goal 999 not found"

    async def test_account_mismatch(
        self, client: TestClient, headers: dict, sdb: Database,
    ) -> None:
        await sdb.execute(model_insert_stmt(GoalFactory(id=100, account_id=2)))

        res = await self._request(1, 100, client, headers)
        assert res["errors"][0]["extensions"]["detail"] == "Goal 100 not found"
        await assert_existing_row(sdb, Goal, id=100)


class TestRemoveGoal(BaseRemoveGoalTest):
    async def test_remove(self, client: TestClient, headers: dict, sdb: Database):
        await sdb.execute(model_insert_stmt(GoalFactory(id=100, account_id=1)))

        res = await self._request(1, 100, client, headers)
        assert "errors" not in res
        assert res["data"]["removeGoal"]["success"]
        await self._assert_no_goal_exists(sdb)

    async def test_remove_with_team_goals(
        self, client: TestClient, headers: dict, sdb: Database,
    ):
        for model in (
            TeamFactory(owner_id=1, id=10),
            TeamFactory(owner_id=1, id=20),
            GoalFactory(id=100, account_id=1),
            GoalFactory(id=200, account_id=1),
            GoalFactory(id=300, account_id=2),
            TeamGoalFactory(team_id=10, goal_id=100),
            TeamGoalFactory(team_id=20, goal_id=100),
            TeamGoalFactory(team_id=20, goal_id=200),
        ):
            await sdb.execute(model_insert_stmt(model))

        res = await self._request(1, 100, client, headers)
        assert "errors" not in res
        assert res["data"]["removeGoal"]["success"]

        await assert_missing_row(sdb, Goal, id=100, account_id=1)
        await assert_missing_row(sdb, TeamGoal, goal_id=100)

        await assert_existing_row(sdb, Goal, id=200)
        await assert_existing_row(sdb, Goal, id=300)
        await assert_existing_row(sdb, TeamGoal, goal_id=200)
