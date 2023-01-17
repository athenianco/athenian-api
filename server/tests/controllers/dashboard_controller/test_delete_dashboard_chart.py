from typing import Any

import pytest

from athenian.api.db import Database
from athenian.api.models.state.models import DashboardChart, TeamDashboard
from tests.testutils.auth import force_request_auth
from tests.testutils.db import assert_existing_row, assert_missing_row, models_insert
from tests.testutils.factory.state import (
    DashboardChartFactory,
    TeamDashboardFactory,
    TeamFactory,
    UserAccountFactory,
)
from tests.testutils.requester import Requester

_USER_ID = "github|1"


class BaseCreateDashboardChartTest(Requester):
    path = "/private/team/{team_id}/dashboard/{dashboard_id}/chart/{chart_id}"

    @pytest.fixture(scope="function", autouse=True)
    async def _create_user(self, sdb):
        await models_insert(sdb, UserAccountFactory(user_id=_USER_ID))

    async def delete(
        self,
        team_id: int,
        dashboard_id: int,
        chart_id: int,
        *args: Any,
        user_id: str | None = _USER_ID,
        **kwargs: Any,
    ):
        path_kwargs = {"team_id": team_id, "dashboard_id": dashboard_id, "chart_id": chart_id}
        with force_request_auth(user_id, self.headers) as headers:
            return await super().delete(*args, path_kwargs=path_kwargs, headers=headers, **kwargs)


class TestDeleteDashboardChartErrors(BaseCreateDashboardChartTest):
    async def test_team_not_found(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=1, team_id=10),
            DashboardChartFactory(dashboard_id=1, id=3),
        )
        await self.delete(20, 1, 3, 404)
        await assert_existing_row(sdb, DashboardChart, id=3)

    async def test_dashboard_not_found(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=1, team_id=10),
            DashboardChartFactory(dashboard_id=1, id=5),
        )
        await self.delete(10, 2, 5, 404)
        await assert_existing_row(sdb, DashboardChart, id=5)

    async def test_chart_not_found(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        await self.delete(10, 1, 5, 404)

    async def test_dashboard_mismatch(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=1, team_id=10),
            TeamDashboardFactory(id=2, team_id=10),
            DashboardChartFactory(dashboard_id=1, id=5),
        )
        await self.delete(10, 2, 5, 404)
        await assert_existing_row(sdb, DashboardChart, id=5)

    async def test_team_account_mismatch(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10, owner_id=3),
            TeamDashboardFactory(id=1, team_id=10),
            DashboardChartFactory(dashboard_id=1, id=5),
        )
        await self.delete(10, 1, 5, 404)
        await assert_existing_row(sdb, DashboardChart, id=5)

    async def test_default_user_forbidden(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=1, team_id=10),
            DashboardChartFactory(dashboard_id=1, id=5),
        )
        await self.delete(10, 1, 5, 403, user_id=None)


class TestDeleteDashboardChart(BaseCreateDashboardChartTest):
    async def test_delete(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=1, team_id=10),
            DashboardChartFactory(dashboard_id=1, id=5),
        )
        await self.delete(10, 1, 5)
        await assert_missing_row(sdb, DashboardChart, id=5)
        await assert_existing_row(sdb, TeamDashboard, id=1)
