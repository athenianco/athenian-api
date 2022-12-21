import asyncio
from datetime import date
from typing import Any

import pytest
import sqlalchemy as sa

from athenian.api.db import Database, ensure_db_datetime_tz, is_postgresql
from athenian.api.internal.dashboard import (
    MultipleTeamDashboardsError,
    TeamDashboardNotFoundError,
    create_dashboard_chart,
    get_dashboard,
    get_team_default_dashboard,
)
from athenian.api.models.state.models import DashboardChart, TeamDashboard
from athenian.api.models.web import DashboardChartCreateRequest, PullRequestMetricID
from tests.testutils.db import assert_existing_row, models_insert, transaction_conn
from tests.testutils.factory.state import DashboardChartFactory, TeamDashboardFactory, TeamFactory
from tests.testutils.time import dt


class TestGetDashboard:
    async def test_get(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=100, team_id=10))
        dashboard = await get_dashboard(100, sdb)
        assert dashboard[TeamDashboard.id.name] == 100
        assert dashboard[TeamDashboard.team_id.name] == 10

    async def test_not_found(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=2, team_id=10))
        with pytest.raises(TeamDashboardNotFoundError):
            await get_dashboard(1, sdb)


class TestGetTeamDefaultDashboard:
    async def test_get_existing(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10), TeamDashboardFactory(id=1, team_id=10))
        async with sdb.connection() as sdb_conn:
            dashboard = await get_team_default_dashboard(10, sdb_conn)
        assert dashboard[TeamDashboard.id.name] == 1
        assert dashboard[TeamDashboard.team_id.name] == 10

    async def test_get_not_existing(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10))
        async with sdb.connection() as sdb_conn:
            dashboard = await get_team_default_dashboard(10, sdb_conn)

        assert dashboard[TeamDashboard.team_id.name] == 10
        await assert_existing_row(
            sdb, TeamDashboard, id=dashboard[TeamDashboard.id.name], team_id=10,
        )

    async def test_get_multiple_existing(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=10),
            TeamDashboardFactory(team_id=10),
            TeamDashboardFactory(team_id=10),
        )
        async with sdb.connection() as sdb_conn:
            with pytest.raises(MultipleTeamDashboardsError):
                await get_team_default_dashboard(10, sdb_conn)

    async def test_concurrent_creation(self, sdb: Database) -> None:
        if not await is_postgresql(sdb):
            pytest.skip("manual table locking not available in sqlite ")
        await models_insert(sdb, TeamFactory(id=10))

        async def _get():
            async with sdb.connection() as sdb_conn:
                dashboard = await get_team_default_dashboard(10, sdb_conn)
                return dashboard[TeamDashboard.id.name]

        tasks = [_get() for _ in range(10)]
        dashboard_ids = await asyncio.gather(*tasks)

        # only one row has been created
        assert len(set(dashboard_ids)) == 1
        await assert_existing_row(sdb, TeamDashboard, team_id=10, id=dashboard_ids[0])


class TestCreateDashboardChart:
    async def test_create_first(self, sdb: Database) -> None:
        create_req = self._create_request()
        await models_insert(sdb, TeamFactory(id=6), TeamDashboardFactory(id=1, team_id=6))
        async with transaction_conn(sdb) as sdb_conn:
            chart_id = await create_dashboard_chart(1, create_req, sdb_conn)

        row = await assert_existing_row(
            sdb, DashboardChart, dashboard_id=1, id=chart_id, name="n", description="d",
        )
        assert row[DashboardChart.time_interval.name] == "P1Y"
        assert row[DashboardChart.time_from.name] is None
        assert row[DashboardChart.time_to.name] is None
        assert row[DashboardChart.position.name] == 0

    async def test_static_time_interval(self, sdb: Database) -> None:
        create_req = self._create_request(
            time_interval=None, date_from=date(2001, 1, 1), date_to=date(2001, 1, 31),
        )
        await models_insert(sdb, TeamFactory(id=6), TeamDashboardFactory(id=1, team_id=6))
        async with transaction_conn(sdb) as sdb_conn:
            chart_id = await create_dashboard_chart(1, create_req, sdb_conn)

        row = await assert_existing_row(sdb, DashboardChart, dashboard_id=1, id=chart_id)
        assert row[DashboardChart.time_interval.name] is None
        assert ensure_db_datetime_tz(row[DashboardChart.time_from.name], sdb) == dt(2001, 1, 1)
        assert ensure_db_datetime_tz(row[DashboardChart.time_to.name], sdb) == dt(2001, 2, 1)

    async def test_append_to_existing_charts(self, sdb: Database) -> None:
        create_req = self._create_request()
        await models_insert(
            sdb,
            TeamFactory(id=6),
            TeamDashboardFactory(id=1, team_id=6),
            DashboardChartFactory(id=6, position=1, dashboard_id=1),
            DashboardChartFactory(id=7, position=0, dashboard_id=1),
        )
        async with transaction_conn(sdb) as sdb_conn:
            chart_id = await create_dashboard_chart(1, create_req, sdb_conn)

        row = await assert_existing_row(sdb, DashboardChart, dashboard_id=1, id=chart_id)
        assert row[DashboardChart.position.name] == 2

    async def test_insert_custom_position(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=6),
            TeamDashboardFactory(id=1, team_id=6),
            *[
                DashboardChartFactory(id=_id, position=pos, dashboard_id=1)
                for _id, pos in ((6, 3), (7, 0), (8, 2))
            ],
        )
        create_req = self._create_request(position=1)
        async with transaction_conn(sdb) as sdb_conn:
            chart_id = await create_dashboard_chart(1, create_req, sdb_conn)

        row = await assert_existing_row(sdb, DashboardChart, dashboard_id=1, id=chart_id)
        assert row[DashboardChart.position.name] == 2

        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 7), (2, chart_id), (3, 8), (4, 6)]

    @classmethod
    def _create_request(cls, **kwargs: Any) -> DashboardChartCreateRequest:
        kwargs.setdefault("time_interval", "P1Y")
        kwargs.setdefault("description", "d")
        kwargs.setdefault("name", "n")
        kwargs.setdefault("metric", PullRequestMetricID.PR_REJECTED)

        return DashboardChartCreateRequest(**kwargs)
