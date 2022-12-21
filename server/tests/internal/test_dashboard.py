import asyncio

import pytest

from athenian.api.db import Database, is_postgresql
from athenian.api.internal.dashboard import (
    MultipleTeamDashboardsError,
    TeamDashboardNotFoundError,
    get_dashboard,
    get_team_default_dashboard,
)
from athenian.api.models.state.models import TeamDashboard
from tests.testutils.db import assert_existing_row, models_insert
from tests.testutils.factory.state import TeamDashboardFactory, TeamFactory


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
