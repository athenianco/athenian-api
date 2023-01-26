from datetime import date
from typing import Any

import pytest

from athenian.api.db import Database, ensure_db_datetime_tz
from athenian.api.models.state.models import (
    Base as BaseModel,
    DashboardChart,
    DashboardChartGroupBy,
)
from athenian.api.models.web import (
    DashboardChartFilters as WebDashboardChartFilters,
    DashboardChartGroupBy as WebDashboardChartGroupBy,
    DashboardChartUpdateRequest,
)
from tests.testutils.auth import force_request_auth
from tests.testutils.db import DBCleaner, assert_existing_row, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.state import (
    DashboardChartFactory,
    DashboardChartGroupByFactory,
    TeamDashboardFactory,
    TeamFactory,
    UserAccountFactory,
)
from tests.testutils.requester import Requester
from tests.testutils.time import dt

_USER_ID = "github|1"


class BaseUpdateDashboardChartTest(Requester):
    path = "/private/team/{team_id}/dashboard/{dashboard_id}/chart/{chart_id}"

    @pytest.fixture(scope="function", autouse=True)
    async def _create_user(self, sdb):
        await models_insert(sdb, UserAccountFactory(user_id=_USER_ID))

    async def post(
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
            return await super().post(*args, headers=headers, path_kwargs=path_kwargs, **kwargs)

    @classmethod
    def _body(
        cls,
        *,
        name: str = "chart name",
        filters: dict | None = None,
        group_by: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        if not {"time_interval", "date_from", "date_to"}.intersection(kwargs):
            kwargs["time_interval"] = "P3M"

        req = DashboardChartUpdateRequest(
            name=name,
            filters=WebDashboardChartFilters(**filters) if filters else None,
            group_by=WebDashboardChartGroupBy(**group_by) if group_by else None,
            **kwargs,
        )
        body = req.to_dict()
        if body.get("date_from"):
            body["date_from"] = body["date_from"].isoformat()
        if body.get("date_to"):
            body["date_to"] = body["date_to"].isoformat()
        return body

    @classmethod
    def _base_models(cls, **chart_kwargs: Any) -> list[BaseModel]:
        return [
            TeamFactory(id=9),
            TeamDashboardFactory(id=1, team_id=9),
            DashboardChartFactory(id=4, dashboard_id=1, **chart_kwargs),
        ]


class TestUpdateDashboardChartErrors(BaseUpdateDashboardChartTest):
    async def test_not_found(self, sdb: Database) -> None:
        await models_insert(sdb, *self._base_models())
        await self.post(9, 1, 3, 404, json=self._body())
        await self.post(9, 2, 4, 404, json=self._body())
        await self.post(8, 1, 4, 404, json=self._body())

    async def test_account_mismatch(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=9, owner_id=3),
            TeamDashboardFactory(id=1, team_id=9),
            DashboardChartFactory(id=4, dashboard_id=1),
        )
        await self.post(9, 1, 4, 404, json=self._body())

    async def test_invalid_repositories_filter(self, sdb: Database) -> None:
        await models_insert(sdb, *self._base_models())
        body = self._body(filters={"repositories": ["github.com/org/foo"]})
        res = await self.post_json(9, 1, 4, 400, json=body)
        assert res["detail"] == "Unknown repository github.com/org/foo"

    async def test_both_relative_and_static_interval(self, sdb: Database) -> None:
        await models_insert(sdb, *self._base_models())
        body = self._body(
            time_interval="P1Y", date_from=date(2001, 1, 1), date_to=date(2001, 2, 1),
        )
        res = await self.post_json(9, 1, 4, 400, json=body)
        assert "time_interval" in res["detail"] and "date_from" in res["detail"]

    async def test_default_user_forbidden(self, sdb: Database) -> None:
        await models_insert(sdb, *self._base_models())
        res = await self.post_json(9, 1, 4, 403, user_id=None, json=self._body())
        assert "is the default user" in res["detail"]

    async def test_invalid_time_interval(self, sdb: Database) -> None:
        await models_insert(sdb, *self._base_models())
        body = self._body(time_interval="P3MXXX")
        res = await self.post_json(9, 1, 3, 400, json=body)
        assert "P3MXXX" in res["detail"]

    async def test_group_by_multiple_fields_forbidden(self, sdb: Database) -> None:
        await models_insert(sdb, *self._base_models())

        body = self._body(group_by={"teams": [9], "jira_labels": ["l0"]})
        res = await self.post_json(9, 1, 3, 400, json=body)
        assert "too many properties" in res["detail"]


class TestUpdateDashboardChart(BaseUpdateDashboardChartTest):
    async def test_update(self, sdb: Database) -> None:
        await models_insert(
            sdb, *self._base_models(name="n", time_interval="P2M", jira_labels=["l0"]),
        )
        body = self._body(
            name="new name",
            group_by={"jira_priorities": ["high", "medium"]},
            filters={"jira": {"labels_include": ["l1", "l2"]}},
            date_from=date(2001, 1, 1),
            date_to=date(2001, 1, 31),
        )
        await self.post_json(9, 1, 4, json=body)

        row = await assert_existing_row(sdb, DashboardChart, id=4, name="new name")
        assert ensure_db_datetime_tz(row[DashboardChart.time_from.name], sdb) == dt(2001, 1, 1)
        assert ensure_db_datetime_tz(row[DashboardChart.time_to.name], sdb) == dt(2001, 2, 1)
        assert row[DashboardChart.time_interval.name] is None
        assert row[DashboardChart.jira_labels.name] == ["l1", "l2"]

        group_by_row = await assert_existing_row(sdb, DashboardChartGroupBy, chart_id=4)
        assert group_by_row[DashboardChartGroupBy.jira_priorities.name] == ["high", "medium"]
        assert group_by_row[DashboardChartGroupBy.repositories.name] is None

    async def test_set_static_time_interval(self, sdb: Database) -> None:
        await models_insert(
            sdb, *self._base_models(time_from=dt(2021, 1, 1), time_to=dt(2021, 2, 1)),
        )
        body = self._body(time_interval="P3M")
        await self.post_json(9, 1, 4, json=body)
        row = await assert_existing_row(sdb, DashboardChart, id=4)
        assert row[DashboardChart.time_interval.name] == "P3M"

    async def test_repositories(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(sdb, *self._base_models())
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_models = [md_factory.RepositoryFactory(node_id=2, full_name="o/r")]
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            body = self._body(
                filters={"repositories": ["github.com/o/r", "github.com/o/r/l"]},
                group_by={"repositories": ["github.com/o/r", "github.com/o/r/l"]},
            )
            await self.post_json(9, 1, 4, json=body)

        row = await assert_existing_row(sdb, DashboardChart, id=4)
        assert row[DashboardChartGroupBy.repositories.name] == [[2, ""], [2, "l"]]
        group_by_row = await assert_existing_row(sdb, DashboardChartGroupBy, chart_id=4)
        assert group_by_row[DashboardChartGroupBy.repositories.name] == [[2, ""], [2, "l"]]

    async def test_update_group_by(self, sdb: Database) -> None:
        await models_insert(sdb, *self._base_models(jira_labels=["l0"]))

        body = self._body(group_by={"teams": [9]})

        await self.post_json(9, 1, 4, json=body)
        await assert_existing_row(sdb, DashboardChart, id=4)
        group_by_row = await assert_existing_row(sdb, DashboardChartGroupBy, chart_id=4)
        assert group_by_row[DashboardChartGroupBy.jira_labels.name] is None
        assert group_by_row[DashboardChartGroupBy.teams.name] == [9]

    async def test_unset_group_by(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            *self._base_models(),
            DashboardChartGroupByFactory(chart_id=4, repositories=[[1, ""], [2, ""]]),
        )

        body = self._body()

        await self.post_json(9, 1, 4, json=body)
        await assert_existing_row(sdb, DashboardChart, id=4)
        group_by_row = await assert_existing_row(sdb, DashboardChartGroupBy, chart_id=4)
        assert group_by_row[DashboardChartGroupBy.repositories.name] is None
