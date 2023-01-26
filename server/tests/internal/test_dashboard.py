import asyncio
from datetime import date
from typing import Any

import pytest
import sqlalchemy as sa

from athenian.api.db import Database, ensure_db_datetime_tz, is_postgresql
from athenian.api.internal.dashboard import (
    DashboardChartNotFoundError,
    InvalidDashboardChartOrder,
    MultipleTeamDashboardsError,
    TeamDashboardNotFoundError,
    create_dashboard_chart,
    delete_dashboard_chart,
    get_dashboard,
    get_team_default_dashboard,
    reorder_dashboard_charts,
    update_dashboard_chart,
)
from athenian.api.models.state.models import DashboardChart, DashboardChartGroupBy, TeamDashboard
from athenian.api.models.web import (
    DashboardChartCreateRequest,
    DashboardChartUpdateRequest,
    PullRequestMetricID,
)
from tests.testutils.db import (
    assert_existing_row,
    assert_missing_row,
    models_insert,
    transaction_conn,
)
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
            chart_id = await create_dashboard_chart(1, create_req, {}, {}, sdb_conn)

        row = await assert_existing_row(
            sdb, DashboardChart, dashboard_id=1, id=chart_id, name="n", description="d",
        )
        assert row[DashboardChart.time_interval.name] == "P1Y"
        assert row[DashboardChart.time_from.name] is None
        assert row[DashboardChart.time_to.name] is None
        assert row[DashboardChart.position.name] == 0

        await assert_missing_row(sdb, DashboardChartGroupBy, chart_id=chart_id)

    async def test_static_time_interval(self, sdb: Database) -> None:
        create_req = self._create_request(
            time_interval=None, date_from=date(2001, 1, 1), date_to=date(2001, 1, 31),
        )
        await models_insert(sdb, TeamFactory(id=6), TeamDashboardFactory(id=1, team_id=6))
        async with transaction_conn(sdb) as sdb_conn:
            chart_id = await create_dashboard_chart(1, create_req, {}, {}, sdb_conn)

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
            chart_id = await create_dashboard_chart(1, create_req, {}, {}, sdb_conn)

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
            chart_id = await create_dashboard_chart(1, create_req, {}, {}, sdb_conn)

        row = await assert_existing_row(sdb, DashboardChart, dashboard_id=1, id=chart_id)
        assert row[DashboardChart.position.name] == 2

        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 7), (2, chart_id), (3, 8), (4, 6)]

    async def test_extra_values(self, sdb: Database) -> None:
        create_req = self._create_request()
        await models_insert(sdb, TeamFactory(id=6), TeamDashboardFactory(id=1, team_id=6))
        extra_values = {
            DashboardChart.repositories: [[1, ""]],
            DashboardChart.jira_labels: ["labelA", "labelB"],
        }

        async with transaction_conn(sdb) as sdb_conn:
            chart_id = await create_dashboard_chart(1, create_req, extra_values, {}, sdb_conn)

        row = await assert_existing_row(sdb, DashboardChart, dashboard_id=1, id=chart_id)
        assert row[DashboardChart.repositories.name] == [[1, ""]]
        assert row[DashboardChart.jira_labels.name] == ["labelA", "labelB"]

    async def test_group_by(self, sdb: Database) -> None:
        create_req = self._create_request()
        await models_insert(sdb, TeamFactory(id=6), TeamDashboardFactory(id=1, team_id=6))
        group_by_values = {DashboardChartGroupBy.teams: [1, 2]}

        async with transaction_conn(sdb) as sdb_conn:
            chart_id = await create_dashboard_chart(1, create_req, {}, group_by_values, sdb_conn)

        await assert_existing_row(sdb, DashboardChart, dashboard_id=1, id=chart_id)
        group_by = await assert_existing_row(sdb, DashboardChartGroupBy, chart_id=chart_id)

        assert group_by[DashboardChartGroupBy.teams.name] == [1, 2]
        assert group_by[DashboardChartGroupBy.repositories.name] is None

    @classmethod
    def _create_request(cls, **kwargs: Any) -> DashboardChartCreateRequest:
        kwargs.setdefault("time_interval", "P1Y")
        kwargs.setdefault("description", "d")
        kwargs.setdefault("name", "n")
        kwargs.setdefault("metric", PullRequestMetricID.PR_REJECTED)

        return DashboardChartCreateRequest(**kwargs)


class TestDeleteDashboardChart:
    async def test_delete(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=6),
            TeamDashboardFactory(id=1, team_id=6),
            DashboardChartFactory(id=10, dashboard_id=1),
        )
        await delete_dashboard_chart(1, 10, sdb)
        await assert_missing_row(sdb, DashboardChart, id=10)

    async def test_not_found(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=6),
            TeamDashboardFactory(id=1, team_id=6),
            DashboardChartFactory(id=10, dashboard_id=1),
        )
        with pytest.raises(DashboardChartNotFoundError):
            await delete_dashboard_chart(1, 11, sdb)

        with pytest.raises(DashboardChartNotFoundError):
            await delete_dashboard_chart(2, 10, sdb)


class TestUpdateDashboardChart:
    async def test_not_found(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=6),
            TeamDashboardFactory(id=1, team_id=6),
            DashboardChartFactory(id=10, dashboard_id=1),
        )
        req = self._update_request()
        async with transaction_conn(sdb) as sdb_conn:
            with pytest.raises(DashboardChartNotFoundError):
                await update_dashboard_chart(1, 11, req, {}, {}, sdb_conn)

        async with transaction_conn(sdb) as sdb_conn:
            with pytest.raises(DashboardChartNotFoundError):
                await update_dashboard_chart(2, 10, req, {}, {}, sdb_conn)

    async def test_base(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=6),
            TeamDashboardFactory(id=1, team_id=6),
            DashboardChartFactory(
                id=10,
                dashboard_id=1,
                name="old",
                time_from=dt(2022, 3, 1),
                time_to=dt(2022, 3, 30),
                jira_issue_types=["bug"],
            ),
        )
        req = self._update_request(name="new", time_interval="P1M")
        extra_values = {DashboardChart.jira_labels: ["l0"], DashboardChart.jira_issue_types: None}
        async with transaction_conn(sdb) as sdb_conn:
            await update_dashboard_chart(1, 10, req, extra_values, {}, sdb_conn)

        row = await assert_existing_row(sdb, DashboardChart, id=10, name="new")
        assert row[DashboardChart.jira_labels.name] == ["l0"]
        assert row[DashboardChart.name.name] == "new"
        assert row[DashboardChart.jira_issue_types.name] is None
        assert row[DashboardChart.time_from.name] is None
        assert row[DashboardChart.time_to.name] is None

    async def test_with_group_by(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=5),
            TeamDashboardFactory(id=2, team_id=5),
            DashboardChartFactory(id=9, dashboard_id=2),
        )
        req = self._update_request()
        group_by: dict = {DashboardChartGroupBy.repositories: [[1, ""]]}
        async with transaction_conn(sdb) as sdb_conn:
            await update_dashboard_chart(2, 9, req, {}, group_by, sdb_conn)

        await assert_existing_row(sdb, DashboardChart, id=9)
        group_by_row = await assert_existing_row(sdb, DashboardChartGroupBy, chart_id=9)
        assert group_by_row[DashboardChartGroupBy.repositories.name] == [[1, ""]]

        group_by = {
            DashboardChartGroupBy.repositories: None,
            DashboardChartGroupBy.jira_priorities: ["h", "l"],
        }
        async with transaction_conn(sdb) as sdb_conn:
            await update_dashboard_chart(2, 9, req, {}, group_by, sdb_conn)
        group_by_row = await assert_existing_row(sdb, DashboardChartGroupBy, chart_id=9)
        assert group_by_row[DashboardChartGroupBy.repositories.name] is None
        assert group_by_row[DashboardChartGroupBy.jira_priorities.name] == ["h", "l"]

    @classmethod
    def _update_request(cls, **kwargs: Any) -> DashboardChartCreateRequest:
        kwargs.setdefault("time_interval", "P1Y")
        kwargs.setdefault("name", "n")
        return DashboardChartUpdateRequest(**kwargs)


class TestReorderDashboardCharts:
    async def test_missing_charts(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=6),
            TeamDashboardFactory(id=1, team_id=6),
            DashboardChartFactory(id=1, position=1, dashboard_id=1),
            DashboardChartFactory(id=2, position=0, dashboard_id=1),
            DashboardChartFactory(id=3, position=2, dashboard_id=1),
        )

        async with transaction_conn(sdb) as sdb_conn:
            with pytest.raises(InvalidDashboardChartOrder):
                await reorder_dashboard_charts(1, [3, 1], sdb_conn)

            with pytest.raises(InvalidDashboardChartOrder):
                await reorder_dashboard_charts(1, [], sdb_conn)

    async def test_unknown_chart_ids(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=6),
            TeamDashboardFactory(id=1, team_id=6),
            DashboardChartFactory(id=1, position=1, dashboard_id=1),
            DashboardChartFactory(id=2, position=0, dashboard_id=1),
        )

        async with transaction_conn(sdb) as sdb_conn:
            with pytest.raises(InvalidDashboardChartOrder):
                await reorder_dashboard_charts(1, [1, 2, 3], sdb_conn)

            with pytest.raises(InvalidDashboardChartOrder):
                await reorder_dashboard_charts(1, [3], sdb_conn)

    async def test_chart_id_duplicate(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=6),
            TeamDashboardFactory(id=1, team_id=6),
            DashboardChartFactory(id=1, position=1, dashboard_id=1),
            DashboardChartFactory(id=2, position=0, dashboard_id=1),
        )

        async with transaction_conn(sdb) as sdb_conn:
            with pytest.raises(InvalidDashboardChartOrder):
                await reorder_dashboard_charts(1, [1, 2, 2], sdb_conn)

    async def test_reorder(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=6),
            TeamDashboardFactory(id=1, team_id=6),
            DashboardChartFactory(id=1, position=1, dashboard_id=1),
            DashboardChartFactory(id=2, position=0, dashboard_id=1),
            DashboardChartFactory(id=3, position=4, dashboard_id=1),
            DashboardChartFactory(id=4, position=2, dashboard_id=1),
        )

        async with transaction_conn(sdb) as sdb_conn:
            await reorder_dashboard_charts(1, [4, 1, 2, 3], sdb_conn)
        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 4), (1, 1), (2, 2), (3, 3)]

        async with transaction_conn(sdb) as sdb_conn:
            await reorder_dashboard_charts(1, [2, 3, 4, 1], sdb_conn)
        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 2), (1, 3), (2, 4), (3, 1)]

    async def test_no_charts(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=6), TeamDashboardFactory(id=1, team_id=6))
        async with transaction_conn(sdb) as sdb_conn:
            await reorder_dashboard_charts(1, [], sdb_conn)

    async def test_reorder_single_chart(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=6),
            TeamDashboardFactory(id=1, team_id=6),
            DashboardChartFactory(id=1, position=1, dashboard_id=1),
        )
        async with transaction_conn(sdb) as sdb_conn:
            await reorder_dashboard_charts(1, [1], sdb_conn)
        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 1)]

        async with transaction_conn(sdb) as sdb_conn:
            await reorder_dashboard_charts(1, [1], sdb_conn)
        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 1)]

    async def test_reorder_same_order(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=6),
            TeamDashboardFactory(id=1, team_id=6),
            DashboardChartFactory(id=1, position=1, dashboard_id=1),
            DashboardChartFactory(id=2, position=0, dashboard_id=1),
            DashboardChartFactory(id=3, position=2, dashboard_id=1),
        )
        async with transaction_conn(sdb) as sdb_conn:
            await reorder_dashboard_charts(1, [2, 1, 3], sdb_conn)
        rows = await sdb.fetch_all(sa.select(DashboardChart.position, DashboardChart.id))
        assert sorted(rows) == [(0, 2), (1, 1), (2, 3)]
