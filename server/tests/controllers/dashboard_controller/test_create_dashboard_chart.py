from datetime import date
from typing import Any

import sqlalchemy as sa

from athenian.api.db import Database, ensure_db_datetime_tz
from athenian.api.models.state.models import DashboardChart, TeamDashboard
from athenian.api.models.web import (
    DashboardChartCreateRequest,
    DashboardChartFilters,
    PullRequestMetricID,
)
from tests.testutils.db import DBCleaner, assert_existing_row, assert_missing_row, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.state import DashboardChartFactory, TeamDashboardFactory, TeamFactory
from tests.testutils.requester import Requester
from tests.testutils.time import dt


class BaseCreateDashboardChartTest(Requester):
    path = "/private/team/{team_id}/dashboard/{dashboard_id}/chart/create"

    async def post(self, team_id: int, dashboard_id: int, *args: Any, **kwargs: Any):
        path_kwargs = {"team_id": team_id, "dashboard_id": dashboard_id}
        res = await super().post(*args, path_kwargs=path_kwargs, **kwargs)
        return res

    @classmethod
    def _body(
        cls,
        *,
        description: str = "chart desc",
        metric: str = PullRequestMetricID.PR_SIZE,
        name: str = "chart name",
        filters: dict | None = None,
        **kwargs: Any,
    ) -> dict:
        if not {"time_interval", "date_from", "date_to"}.intersection(kwargs):
            kwargs["time_interval"] = "P3M"

        if filters:
            filters = DashboardChartFilters(**filters)
        else:
            filters = None

        req = DashboardChartCreateRequest(
            description=description, metric=metric, name=name, filters=filters, **kwargs,
        )
        body = req.to_dict()
        if body.get("date_from"):
            body["date_from"] = body["date_from"].isoformat()
        if body.get("date_to"):
            body["date_to"] = body["date_to"].isoformat()
        return body


class TestCreateDashboardChartErrors(BaseCreateDashboardChartTest):
    async def test_team_not_found(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        await self.post(20, 1, 404, json=self._body())
        await assert_missing_row(sdb, DashboardChart)

    async def test_dashboard_not_found(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        await self.post(10, 2, 404, json=self._body())
        await assert_missing_row(sdb, DashboardChart)

    async def test_team_account_mismatch(self, sdb: Database) -> None:
        await models_insert(
            sdb, TeamFactory(id=10, owner_id=3), TeamDashboardFactory(id=1, team_id=10),
        )
        await self.post(10, 1, 404, json=self._body())
        await assert_missing_row(sdb, DashboardChart)

    async def test_invalid_time_interval(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        body = self._body(time_interval="1 year")
        await self.post_json(10, 1, 400, json=body)
        await assert_missing_row(sdb, DashboardChart)

    async def test_both_relative_and_static_interval(self, sdb: Database) -> None:
        body = self._body(
            time_interval="P2Y", date_from=date(2022, 1, 1), date_to=date(2022, 12, 31),
        )
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        await self.post_json(10, 1, 400, json=body)
        await assert_missing_row(sdb, DashboardChart)

    async def test_missing_intervals(self, sdb: Database) -> None:
        body = self._body()
        body.pop("time_interval")
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        await self.post_json(10, 1, 400, json=body)
        await assert_missing_row(sdb, DashboardChart)

    async def test_negative_position(self, sdb: Database) -> None:
        body = self._body(position=-1)
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        await self.post_json(10, 1, 400, json=body)

    async def test_invalid_repositories_filter(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        body = self._body(filters={"repositories": ["github.com/org/repo"]})
        res = await self.post_json(10, 1, 400, json=body)
        assert res["detail"] == "Unknown repository github.com/org/repo"


class TestCreateDashboardChart(BaseCreateDashboardChartTest):
    async def test_first_chart(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        body = self._body(
            name="My chart",
            description="My chart description",
            metric=PullRequestMetricID.PR_REVIEW_PENDING_COUNT,
            time_interval="P1Y",
        )
        res = await self.post_json(10, 1, json=body)

        await assert_existing_row(
            sdb,
            DashboardChart,
            id=res["id"],
            position=0,
            name="My chart",
            description="My chart description",
            time_interval="P1Y",
            time_from=None,
            time_to=None,
            metric=PullRequestMetricID.PR_REVIEW_PENDING_COUNT,
            dashboard_id=1,
            repositories=None,
            environments=None,
        )

    async def test_static_interval(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        body = self._body(date_from=date(2022, 4, 1), date_to=date(2022, 4, 30))

        res = await self.post_json(10, 1, json=body)
        row = await assert_existing_row(sdb, DashboardChart, id=res["id"], time_interval=None)

        assert ensure_db_datetime_tz(row[DashboardChart.time_from.name], sdb) == dt(2022, 4, 1)
        assert ensure_db_datetime_tz(row[DashboardChart.time_to.name], sdb) == dt(2022, 5, 1)

    async def test_append_to_existing_charts(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(id=1, team_id=10),
            DashboardChartFactory(id=3, dashboard_id=1, position=10),
            DashboardChartFactory(id=7, dashboard_id=1, position=11),
            DashboardChartFactory(id=5, dashboard_id=1, position=12),
        )
        body = self._body()
        res = await self.post_json(10, 1, json=body)
        # new chart takes the first free position
        await assert_existing_row(sdb, DashboardChart, id=res["id"], position=13)

        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(10, 3), (11, 7), (12, 5), (13, res["id"])]

    async def test_position_prepend(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=9),
            TeamDashboardFactory(id=1, team_id=9),
            DashboardChartFactory(id=5, dashboard_id=1, position=0),
        )
        res = await self.post_json(9, 1, json=self._body(position=0))
        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, res["id"]), (1, 5)]

    async def test_position(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=9),
            TeamDashboardFactory(id=1, team_id=9),
            DashboardChartFactory(id=5, dashboard_id=1, position=0),
            DashboardChartFactory(id=6, dashboard_id=1, position=1),
            DashboardChartFactory(id=7, dashboard_id=1, position=2),
        )

        r = await self.post_json(9, 1, json=self._body(position=1))
        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 5), (1, r["id"]), (2, 6), (3, 7)]

        r2 = await self.post_json(9, 1, json=self._body(position=3))
        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 5), (1, r["id"]), (2, 6), (3, r2["id"]), (4, 7)]

        r3 = await self.post_json(9, 1, json=self._body(position=0))
        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, r3["id"]), (1, 5), (2, r["id"]), (3, 6), (4, r2["id"]), (5, 7)]

    async def test_position_after_last(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=9),
            TeamDashboardFactory(id=1, team_id=9),
            DashboardChartFactory(id=5, dashboard_id=1, position=0),
            DashboardChartFactory(id=6, dashboard_id=1, position=1),
        )
        res = await self.post_json(9, 1, json=self._body(position=4))
        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 5), (1, 6), (2, res["id"])]

    async def test_default_dashboard(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=9))
        res = await self.post_json(9, 0, json=self._body())

        dashboard = await assert_existing_row(sdb, TeamDashboard, team_id=9)
        await assert_existing_row(
            sdb, DashboardChart, dashboard_id=dashboard[TeamDashboard.id.name], id=res["id"],
        )

        res2 = await self.post_json(9, 0, json=self._body())
        await assert_existing_row(
            sdb, DashboardChart, dashboard_id=dashboard[TeamDashboard.id.name], id=res2["id"],
        )

        res3 = await self.post_json(9, 0, json=self._body(position=1))
        await assert_existing_row(
            sdb, DashboardChart, dashboard_id=dashboard[TeamDashboard.id.name], id=res3["id"],
        )

        chart_id_rows = await sdb.fetch_all(
            sa.select(DashboardChart.id).order_by(DashboardChart.position),
        )
        assert [r[0] for r in chart_id_rows] == [res["id"], res3["id"], res2["id"]]


class TestCreateWithFilters(BaseCreateDashboardChartTest):
    async def test_repositories(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(sdb, TeamFactory(id=9))
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            mdb_models = [
                md_factory.RepositoryFactory(node_id=200, full_name="org/a-repo"),
            ]
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb_rw, *mdb_models)

            body = self._body(
                filters={
                    "repositories": ["github.com/org/a-repo", "github.com/org/a-repo/logical"],
                },
            )
            res = await self.post_json(9, 0, json=body)

        chart = await assert_existing_row(sdb, DashboardChart, id=res["id"])
        assert chart[DashboardChart.repositories.name] == [[200, ""], [200, "logical"]]

    async def test_environments(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(sdb, TeamFactory(id=9), TeamDashboardFactory(team_id=9, id=10))
        body = self._body(filters={"environments": ["production"]})
        res = await self.post_json(9, 10, json=body)

        chart = await assert_existing_row(sdb, DashboardChart, id=res["id"])
        assert chart[DashboardChart.environments.name] == ["production"]
