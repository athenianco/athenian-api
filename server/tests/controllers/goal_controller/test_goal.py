"""Tests for the goal CRUD controllers."""

from typing import Any, Optional

import pytest
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.models.state.models import Goal, TeamGoal
from tests.testutils.auth import force_request_auth
from tests.testutils.db import assert_existing_row, assert_missing_row, models_insert
from tests.testutils.factory.state import (
    AccountFactory,
    GoalFactory,
    TeamFactory,
    TeamGoalFactory,
    UserAccountFactory,
)
from tests.testutils.requester import Requester

_USER_ID = "github|1"


class BaseDeleteGoalTest(Requester):
    async def _request(
        self,
        goal_id: int,
        assert_status: int = 204,
        user_id: Optional[str] = _USER_ID,
        **kwargs: Any,
    ) -> Optional[dict]:
        path = f"/private/align/goal/{goal_id}"
        with force_request_auth(user_id, self.headers) as headers:
            response = await self.client.request(
                method="DELETE", path=path, headers=headers, **kwargs,
            )
        assert response.status == assert_status
        if assert_status == 204:
            return None
        return await response.json()

    @classmethod
    async def _assert_no_goal_exists(cls, sdb: Database) -> None:
        assert await sdb.fetch_one(sa.select(Goal)) is None
        assert await sdb.fetch_one(sa.select(TeamGoal)) is None

    @pytest.fixture(scope="function", autouse=True)
    async def _create_user(self, sdb):
        await models_insert(sdb, UserAccountFactory(user_id=_USER_ID))


class TestRemoveGoalErrors(BaseDeleteGoalTest):
    async def test_non_existing_goal(self) -> None:
        res = await self._request(999, 404)
        assert res is not None
        assert res["detail"] == "Goal 999 not found or access denied."

    async def test_account_mismatch(self, sdb: Database) -> None:
        await models_insert(sdb, AccountFactory(id=99), GoalFactory(id=100, account_id=99))
        await self._request(100, 404)
        await assert_existing_row(sdb, Goal, id=100)

    async def test_default_user_forbidden(self, sdb: Database) -> None:
        await models_insert(sdb, GoalFactory(id=100, account_id=2))
        await self._request(100, 403, user_id=None)
        await assert_existing_row(sdb, Goal, id=100)


class TestDeleteGoal(BaseDeleteGoalTest):
    async def test_delete(self, sdb: Database):
        await models_insert(sdb, GoalFactory(id=100))
        await self._request(100)
        await self._assert_no_goal_exists(sdb)

    async def test_remove_with_team_goals(self, sdb: Database):
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

        await self._request(100)
        await assert_missing_row(sdb, Goal, id=100, account_id=1)
        await assert_missing_row(sdb, TeamGoal, goal_id=100)

        await assert_existing_row(sdb, Goal, id=200)
        await assert_existing_row(sdb, Goal, id=300)
        await assert_existing_row(sdb, TeamGoal, goal_id=200)
