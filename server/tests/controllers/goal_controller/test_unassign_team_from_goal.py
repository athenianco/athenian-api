from typing import Any

from aiohttp import ClientResponse
import pytest

from athenian.api.db import Database
from athenian.api.models.state.models import Goal, Team, TeamGoal
from tests.testutils.auth import force_request_auth
from tests.testutils.db import assert_existing_row, assert_missing_row, models_insert
from tests.testutils.factory.state import (
    GoalFactory,
    TeamFactory,
    TeamGoalFactory,
    UserAccountFactory,
)
from tests.testutils.requester import Requester

_USER_ID = "github|1"


class BaseUnassignTeamFromGoalTest(Requester):
    path = "/private/align/goal/{id}/unassign_team"

    @pytest.fixture(scope="function", autouse=True)
    async def _create_user(self, sdb):
        await models_insert(sdb, UserAccountFactory(user_id=_USER_ID))

    async def post(
        self,
        goal_id: int,
        *args: Any,
        user_id: str | None = _USER_ID,
        **kwargs: Any,
    ) -> ClientResponse:
        path_kwargs = {"id": goal_id}
        with force_request_auth(user_id, self.headers) as headers:
            return await super().post(*args, headers=headers, path_kwargs=path_kwargs, **kwargs)


class TestUnassignTeamFromGoalErrors(BaseUnassignTeamFromGoalTest):
    async def test_goal_not_found(self, sdb: Database) -> None:
        res = await self.post_json(1, json={"team": 2}, assert_status=404)
        assert "Goal 1 not found" in res["detail"]

    async def test_team_not_found(self, sdb: Database) -> None:
        await models_insert(
            sdb, GoalFactory(id=1), TeamFactory(id=9), TeamGoalFactory(goal_id=1, team_id=9),
        )
        res = await self.post_json(1, json={"team": 10}, assert_status=404)
        assert "Team 10 not assigned" in res["detail"]

    async def test_team_not_assigned(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=1), TeamFactory(id=9))
        res = await self.post_json(1, json={"team": 9}, assert_status=404)
        assert "Team 9 not assigned" in res["detail"]

    async def test_goal_account_mismatch(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=1, account_id=3),
            TeamFactory(id=9, owner_id=3),
            TeamGoalFactory(goal_id=1, team_id=9),
        )
        res = await self.post_json(1, json={"team": 9}, assert_status=404)
        assert "Goal 1 not found" in res["detail"]

    async def test_recursive_goal_account_mismatch(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=1, account_id=3),
            TeamFactory(id=9, owner_id=3),
            TeamGoalFactory(goal_id=1, team_id=9),
        )
        res = await self.post_json(1, json={"team": 9, "recursive": True}, assert_status=404)
        assert "Goal 1 not found" in res["detail"]

    async def test_default_user_forbidden(self, sdb: Database) -> None:
        await models_insert(
            sdb, GoalFactory(id=1), TeamFactory(id=9), TeamGoalFactory(goal_id=1, team_id=9),
        )
        res = await self.post_json(1, json={"team": 9}, user_id=None, assert_status=403)
        assert "is the default user" in res["detail"]


class TestUnassignTeamFromGoal(BaseUnassignTeamFromGoalTest):
    async def test_non_recursive(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=1),
            TeamFactory(id=8),
            TeamFactory(id=9),
            TeamGoalFactory(goal_id=1, team_id=8),
            TeamGoalFactory(goal_id=1, team_id=9),
        )
        res = await self.post_json(1, json={"team": 9})
        assert res == {"id": 1}

        await assert_existing_row(sdb, Goal, id=1)
        await assert_missing_row(sdb, TeamGoal, team_id=9)
        await assert_existing_row(sdb, TeamGoal, team_id=8)
        await assert_existing_row(sdb, Team, id=8)
        await assert_existing_row(sdb, Team, id=9)

    async def test_non_recursive_last_team_unassigned(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=1),
            TeamFactory(id=8),
            TeamGoalFactory(goal_id=1, team_id=8),
        )
        await self.post(1, json={"team": 8}, assert_status=204)

        await assert_missing_row(sdb, Goal, id=1)
        await assert_missing_row(sdb, TeamGoal, team_id=8)
        await assert_existing_row(sdb, Team, id=8)

    async def test_recursive(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=1),
            TeamFactory(id=8),
            TeamFactory(id=9, parent_id=8),
            TeamFactory(id=10, parent_id=9),
            TeamFactory(id=11, parent_id=9),
            TeamFactory(id=12, parent_id=11),
            *[TeamGoalFactory(goal_id=1, team_id=t) for t in [8, 10, 11, 12]],
        )
        res = await self.post_json(1, json={"team": 11, "recursive": True})
        assert res == {"id": 1}

        await assert_existing_row(sdb, Goal, id=1)
        for team in (11, 12):
            await assert_missing_row(sdb, TeamGoal, goal_id=1, team_id=team)
            await assert_existing_row(sdb, Team, id=team)
        for team in (8, 10):
            await assert_existing_row(sdb, TeamGoal, goal_id=1, team_id=team)

    async def test_recursive_last_team_unassigned(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            GoalFactory(id=1),
            TeamFactory(id=8),
            TeamFactory(id=9, parent_id=8),
            TeamFactory(id=10, parent_id=9),
            *[TeamGoalFactory(goal_id=1, team_id=t) for t in [9, 10]],
        )
        await self.post(1, json={"team": 9, "recursive": True}, assert_status=204)

        await assert_missing_row(sdb, Goal, id=1)
        for team in (9, 10):
            await assert_missing_row(sdb, TeamGoal, goal_id=1, team_id=team)
            await assert_existing_row(sdb, Team, id=team)

    async def test_recursive_single_team(self, sdb: Database) -> None:
        await models_insert(
            sdb, GoalFactory(id=1), TeamFactory(id=8), TeamGoalFactory(goal_id=1, team_id=8),
        )
        await self.post(1, json={"team": 8, "recursive": True}, assert_status=204)
        await assert_missing_row(sdb, Goal, id=1)
        await assert_missing_row(sdb, TeamGoal, goal_id=1, team_id=8)
        await assert_existing_row(sdb, Team, id=8)
