from typing import Any, Sequence

from aiohttp import ClientResponse
import pytest
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.models.state.models import DashboardChart, TeamDashboard
from athenian.api.models.web import DashboardUpdateRequest
from athenian.api.models.web.dashboard import _DashboardUpdateChart
from tests.testutils.auth import force_request_auth
from tests.testutils.db import assert_existing_row, models_insert
from tests.testutils.factory.state import (
    DashboardChartFactory,
    TeamDashboardFactory,
    TeamFactory,
    UserAccountFactory,
)
from tests.testutils.requester import Requester

_USER_ID = "github|1"


class BaseUpdateDashboardTest(Requester):
    path = "/private/team/{team_id}/dashboard/{dashboard_id}"

    @pytest.fixture(scope="function", autouse=True)
    async def _create_user(self, sdb):
        await models_insert(sdb, UserAccountFactory(user_id=_USER_ID))

    async def put(
        self,
        team_id: int,
        dashboard_id: int,
        *args: Any,
        user_id: str | None = _USER_ID,
        **kwargs: Any,
    ) -> ClientResponse:
        path_kwargs = {"team_id": team_id, "dashboard_id": dashboard_id}
        with force_request_auth(user_id, self.headers) as headers:
            return await super().put(*args, path_kwargs=path_kwargs, headers=headers, **kwargs)

    @classmethod
    def _body(cls, chart_ids: Sequence[int]) -> dict:
        model = DashboardUpdateRequest(charts=[_DashboardUpdateChart(id=id_) for id_ in chart_ids])
        return model.to_dict()


class TestUpdateDashboardErrors(BaseUpdateDashboardTest):
    async def test_team_account_mismatch(self, sdb: Database) -> None:
        await models_insert(
            sdb, TeamFactory(id=10, owner_id=3), TeamDashboardFactory(id=1, team_id=10),
        )
        await self.put(10, 1, 404, json=self._body([]))

    async def test_dashboard_not_found(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=1, team_id=10),
            DashboardChartFactory(id=3, dashboard_id=1, position=0),
            DashboardChartFactory(id=7, dashboard_id=1, position=1),
        )
        await self.put(10, 2, 404, json=self._body([7, 3]))
        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 3), (1, 7)]

    async def test_wrong_chart_ids_order(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=1, team_id=10),
            DashboardChartFactory(id=3, dashboard_id=1, position=0),
            DashboardChartFactory(id=7, dashboard_id=1, position=1),
        )
        await self.put(10, 1, 400, json=self._body([7, 3, 2]))
        await self.put(10, 1, 400, json=self._body([3, 3]))
        await self.put(10, 1, 400, json=self._body([7]))

        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 3), (1, 7)]

    async def test_default_user_forbidden(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=1, team_id=10),
            DashboardChartFactory(id=1, dashboard_id=1),
        )
        await self.put(10, 1, 403, user_id=None, json=self._body([1]))


class TestUpdateDashboard(BaseUpdateDashboardTest):
    async def test_reorder(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=1, team_id=10),
            DashboardChartFactory(id=1, name="A", dashboard_id=1, position=0),
            DashboardChartFactory(id=2, name="B", dashboard_id=1, position=1),
            DashboardChartFactory(id=3, name="C", dashboard_id=1, position=2),
            DashboardChartFactory(id=4, name="D", dashboard_id=1, position=3),
        )
        res = await self.put_json(10, 1, json=self._body([1, 2, 3, 4]))
        await assert_existing_row(sdb, TeamDashboard, id=1)
        assert res["id"] == 1
        assert res["team"] == 10
        assert [chart["id"] for chart in res["charts"]] == [1, 2, 3, 4]
        assert [chart["name"] for chart in res["charts"]] == ["A", "B", "C", "D"]
        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 1), (1, 2), (2, 3), (3, 4)]

        res = await self.put_json(10, 1, json=self._body([3, 4, 2, 1]))
        assert [chart["id"] for chart in res["charts"]] == [3, 4, 2, 1]
        assert [chart["name"] for chart in res["charts"]] == ["C", "D", "B", "A"]
        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 3), (1, 4), (2, 2), (3, 1)]
