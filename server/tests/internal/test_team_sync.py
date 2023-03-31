from freezegun import freeze_time
import pytest

from athenian.api.db import Database, ensure_db_datetime_tz
from athenian.api.internal.team_sync import SyncTeamsError, sync_teams
from athenian.api.models.state.models import Goal, Team, TeamGoal
from tests.testutils.db import (
    DBCleaner,
    assert_existing_row,
    assert_missing_row,
    models_insert,
    models_insert_auto_pk,
)
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_ACCOUNT_ID, DEFAULT_MD_ACCOUNT_ID
from tests.testutils.factory.state import GoalFactory, TeamFactory, TeamGoalFactory
from tests.testutils.time import dt


class BaseTestSyncTeams:
    @classmethod
    async def _mk_root_team(cls, sdb: Database, name: str = "Root") -> int:
        return (await models_insert_auto_pk(sdb, TeamFactory(name=name)))[0]


class TestSyncTeams(BaseTestSyncTeams):
    async def test_create_missing(self, sdb: Database, mdb_rw: Database) -> None:
        root_team_id = await self._mk_root_team(sdb)

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = (
                md_factory.TeamFactory(node_id=100, name="teamA"),
                md_factory.TeamFactory(node_id=101, parent_team_id=100, name="teamB"),
                md_factory.TeamMemberFactory(parent_id=100, child_id=200),
                md_factory.TeamMemberFactory(parent_id=100, child_id=201),
                md_factory.TeamMemberFactory(parent_id=101, child_id=200),
            )
            await models_insert(mdb_rw, *models)
            mdb_cleaner.add_models(*models)

            await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw)

        team_a = await assert_existing_row(sdb, Team, name="teamA")
        team_b = await assert_existing_row(sdb, Team, name="teamB")

        assert team_a[Team.parent_id.name] == root_team_id
        assert team_a[Team.members.name] == [200, 201]
        assert team_b[Team.parent_id.name] == team_a[Team.id.name]
        assert team_b[Team.members.name] == [200]

    async def test_create_child_of_existing(self, sdb: Database, mdb_rw: Database) -> None:
        root_team_id = await self._mk_root_team(sdb)
        (team_a_id,) = await models_insert_auto_pk(
            sdb, TeamFactory(name="teamA", origin_node_id=100, parent_id=root_team_id),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = (
                md_factory.TeamFactory(node_id=100, name="teamA"),
                md_factory.TeamFactory(node_id=101, parent_team_id=100, name="teamB"),
            )
            await models_insert(mdb_rw, *models)
            mdb_cleaner.add_models(*models)

            await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw)

        team_b = await assert_existing_row(sdb, Team, name="teamB")

        assert team_b[Team.parent_id.name] == team_a_id

    async def test_create_name_clash(self, sdb: Database, mdb_rw: Database) -> None:
        root_team_id = await self._mk_root_team(sdb, name="root")

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = (
                md_factory.TeamFactory(node_id=100, name="root"),
                md_factory.TeamFactory(node_id=101, parent_team_id=100, name=Team.BOTS),
                md_factory.TeamMemberFactory(parent_id=100, child_id=200),
                md_factory.TeamMemberFactory(parent_id=100, child_id=201),
                md_factory.TeamMemberFactory(parent_id=101, child_id=200),
            )
            await models_insert(mdb_rw, *models)
            mdb_cleaner.add_models(*models)

            await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw)

        team_root = await assert_existing_row(sdb, Team, name="root")
        team_a = await assert_existing_row(sdb, Team, name="root.github")
        team_b = await assert_existing_row(sdb, Team, name=f"{Team.BOTS}.github")
        await assert_missing_row(sdb, Team, name=Team.BOTS)

        assert team_root[Team.members.name] == []
        assert team_a[Team.parent_id.name] == root_team_id
        assert team_a[Team.members.name] == [200, 201]
        assert team_b[Team.parent_id.name] == team_a[Team.id.name]
        assert team_b[Team.members.name] == [200]

    @freeze_time("2021-01-01")
    async def test_update_smoke(self, sdb: Database, mdb_rw: Database) -> None:
        root_team_id = await self._mk_root_team(sdb)
        team_a_id, team_b_id = await models_insert_auto_pk(
            sdb,
            TeamFactory(
                name="teamA",
                origin_node_id=100,
                parent_id=root_team_id,
                members=[],
                updated_at=dt(2020, 1, 1),
            ),
            TeamFactory(
                name="teamB",
                origin_node_id=101,
                parent_id=root_team_id,
                members=[200],
                updated_at=dt(2020, 1, 1),
            ),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = (
                md_factory.TeamFactory(node_id=100, name="teamA-new", parent_team_id=101),
                md_factory.TeamFactory(node_id=101, name="teamB"),
                md_factory.TeamMemberFactory(parent_id=100, child_id=200),
                md_factory.TeamMemberFactory(parent_id=100, child_id=201),
                md_factory.TeamMemberFactory(parent_id=101, child_id=200),
            )
            await models_insert(mdb_rw, *models)
            mdb_cleaner.add_models(*models)
            await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw)

        team_a = await assert_existing_row(sdb, Team, name="teamA-new")
        assert team_a[Team.parent_id.name] == team_b_id
        assert team_a[Team.members.name] == [200, 201]
        # teamA was updated
        assert ensure_db_datetime_tz(team_a[Team.updated_at.name], sdb) == dt(2021, 1, 1)

        team_b = await assert_existing_row(sdb, Team, name="teamB")
        assert team_b[Team.parent_id.name] == root_team_id
        assert team_b[Team.members.name] == [200]
        # teamB was not updated
        assert ensure_db_datetime_tz(team_b[Team.updated_at.name], sdb) == dt(2020, 1, 1)

    @pytest.mark.parametrize("name", ["root", Team.BOTS])
    @freeze_time("2021-01-01")
    async def test_update_name_clash(self, sdb: Database, mdb_rw: Database, name: str) -> None:
        root_team_id = await self._mk_root_team(sdb, "root")
        team_a_id, team_b_id = await models_insert_auto_pk(
            sdb,
            TeamFactory(
                name=f"{name}.github",
                origin_node_id=100,
                parent_id=root_team_id,
                members=[],
                updated_at=dt(2020, 1, 1),
            ),
            TeamFactory(
                name="teamA",
                origin_node_id=101,
                parent_id=root_team_id,
                members=[200],
                updated_at=dt(2020, 1, 1),
            ),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = (
                md_factory.TeamFactory(node_id=100, name=name, parent_team_id=101),
                md_factory.TeamFactory(node_id=101, name="teamB"),
                md_factory.TeamMemberFactory(parent_id=100, child_id=200),
                md_factory.TeamMemberFactory(parent_id=100, child_id=201),
                md_factory.TeamMemberFactory(parent_id=101, child_id=200),
            )
            await models_insert(mdb_rw, *models)
            mdb_cleaner.add_models(*models)
            await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw)

        team_a = await assert_existing_row(sdb, Team, name=f"{name}.github")
        assert team_a[Team.parent_id.name] == team_b_id
        assert team_a[Team.members.name] == [200, 201]
        # {name} was updated
        assert ensure_db_datetime_tz(team_a[Team.updated_at.name], sdb) == dt(2021, 1, 1)

        team_b = await assert_existing_row(sdb, Team, name="teamB")
        assert team_b[Team.parent_id.name] == root_team_id
        assert team_b[Team.members.name] == [200]
        assert ensure_db_datetime_tz(team_b[Team.updated_at.name], sdb) > dt(2020, 1, 1)

    async def test_invert_two_team_names(self, sdb: Database, mdb_rw: Database) -> None:
        root_team_id = await self._mk_root_team(sdb)
        team_a_id, team_b_id = await models_insert_auto_pk(
            sdb,
            TeamFactory(name="A", parent_id=root_team_id, origin_node_id=100),
            TeamFactory(name="B", parent_id=root_team_id, origin_node_id=101),
        )
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = (
                md_factory.TeamFactory(node_id=100, name="B"),
                md_factory.TeamFactory(node_id=101, name="A"),
            )
            await models_insert(mdb_rw, *models)
            mdb_cleaner.add_models(*models)
            await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw)

        await assert_existing_row(sdb, Team, name="B", id=team_a_id, origin_node_id=100)
        await assert_existing_row(sdb, Team, name="A", id=team_b_id, origin_node_id=101)

    async def test_deletion(self, sdb: Database, mdb_rw: Database) -> None:
        root_team_id = await self._mk_root_team(sdb)
        await models_insert_auto_pk(
            sdb,
            TeamFactory(name="teamA", origin_node_id=100, parent_id=root_team_id),
            TeamFactory(name="teamB", origin_node_id=101, parent_id=root_team_id),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [md_factory.TeamFactory(node_id=100, name="teamA")]
            await models_insert(mdb_rw, *models)
            mdb_cleaner.add_models(*models)
            await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw)

        await assert_existing_row(sdb, Team, parent_id=None, name="Root")
        await assert_existing_row(sdb, Team, name="teamA")
        await assert_missing_row(sdb, Team, name="teamB")

    async def test_delete_previous_parent(self, sdb: Database, mdb_rw: Database) -> None:
        root_team_id = await self._mk_root_team(sdb)
        (team_a_id,) = await models_insert_auto_pk(
            sdb,
            TeamFactory(name="A", origin_node_id=100, parent_id=root_team_id),
        )
        team_b_id, team_c_id = await models_insert_auto_pk(
            sdb,
            TeamFactory(name="B", origin_node_id=101, parent_id=team_a_id),
            TeamFactory(name="C", origin_node_id=102, parent_id=root_team_id),
        )

        # A is now deleted and was B's parent, but now B parent is C
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = (
                md_factory.TeamFactory(node_id=101, name="B", parent_team_id=102),
                md_factory.TeamFactory(node_id=102, name="C"),
            )
            await models_insert(mdb_rw, *models)
            mdb_cleaner.add_models(*models)
            await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw)

        await assert_missing_row(sdb, Team, name="A")
        team_b = await assert_existing_row(sdb, Team, name="B")
        assert team_b[Team.parent_id.name] == team_c_id
        team_c = await assert_existing_row(sdb, Team, name="C")
        assert team_c[Team.parent_id.name] == root_team_id

    async def test_complex_scenario(self, sdb: Database, mdb_rw: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=50, parent_id=None),
            TeamFactory(name="A", id=51, parent_id=50, origin_node_id=100),
            TeamFactory(name="B", id=52, parent_id=51, origin_node_id=101, members=[201]),
            TeamFactory(name="C", id=53, parent_id=51, origin_node_id=102, members=[200]),
            TeamFactory(name="D", id=54, parent_id=53, origin_node_id=103),
            TeamFactory(name="E", id=55, parent_id=54, origin_node_id=104),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = (
                md_factory.TeamFactory(node_id=101, name="A", parent_team_id=None),  # was B
                md_factory.TeamFactory(node_id=102, name="newC", parent_team_id=101),
                # D-E relationship has been inverted
                md_factory.TeamFactory(node_id=104, name="E", parent_team_id=102),
                md_factory.TeamFactory(node_id=103, name="D", parent_team_id=104),
                md_factory.TeamFactory(node_id=105, name="Ω", parent_team_id=103),
                md_factory.TeamMemberFactory(parent_id=101, child_id=200),
                md_factory.TeamMemberFactory(parent_id=101, child_id=201),
            )
            await models_insert(mdb_rw, *models)
            mdb_cleaner.add_models(*models)
            await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw)

        await assert_missing_row(sdb, Team, id=51)
        team_b = await assert_existing_row(sdb, Team, id=52, name="A", parent_id=50)
        assert team_b[Team.members.name] == [200, 201]

        team_c = await assert_existing_row(sdb, Team, id=53, name="newC", parent_id=52)
        assert team_c[Team.members.name] == []

        await assert_existing_row(sdb, Team, id=55, name="E", parent_id=53)
        await assert_existing_row(sdb, Team, id=54, name="D", parent_id=55)
        await assert_existing_row(sdb, Team, name="Ω", parent_id=54, origin_node_id=105)

    async def test_deletion_empty_goals_removed(self, sdb: Database, mdb_rw: Database) -> None:
        root_team_id = await self._mk_root_team(sdb)
        teams = await models_insert_auto_pk(
            sdb,
            TeamFactory(parent_id=root_team_id, origin_node_id=100),
            TeamFactory(parent_id=root_team_id, origin_node_id=101),
        )
        await models_insert(
            sdb,
            GoalFactory(id=20),
            GoalFactory(id=21),
            TeamGoalFactory(goal_id=20, team_id=2),
            TeamGoalFactory(goal_id=21, team_id=3),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [md_factory.TeamFactory(node_id=101)]
            await models_insert(mdb_rw, *models)
            mdb_cleaner.add_models(*models)
            await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw)

        # goal 20 was only linked to teams[0] which has been deleted, so it's been deleted too
        await assert_missing_row(sdb, Team, id=teams[0])
        await assert_existing_row(sdb, Team, id=teams[1])
        await assert_missing_row(sdb, Goal, id=20)
        await assert_existing_row(sdb, Goal, id=21)
        await assert_missing_row(sdb, TeamGoal, team_id=teams[0])
        await assert_existing_row(sdb, TeamGoal, team_id=teams[1], goal_id=21)

    async def test_invalid_meta_team_parent(self, sdb: Database, mdb_rw: Database) -> None:
        root_team_id = await self._mk_root_team(sdb)

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.TeamFactory(node_id=101),
                md_factory.TeamFactory(node_id=102, parent_team_id=101),
                md_factory.TeamFactory(node_id=103, parent_team_id=1010000),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)
            await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw)

        team_101 = await assert_existing_row(sdb, Team, origin_node_id=101, parent_id=root_team_id)
        await assert_existing_row(sdb, Team, origin_node_id=102, parent_id=team_101[Team.id.name])
        # team 103 parent is root since 1010000 does not exist
        await assert_existing_row(sdb, Team, origin_node_id=103, parent_id=root_team_id)

    async def test_updated_members_are_sorted(self, sdb: Database, mdb_rw: Database) -> None:
        root_team_id = await self._mk_root_team(sdb)
        await models_insert_auto_pk(
            sdb,
            TeamFactory(parent_id=root_team_id, origin_node_id=100, members=[200]),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.TeamFactory(node_id=100),
                md_factory.TeamMemberFactory(parent_id=100, child_id=202),
                md_factory.TeamMemberFactory(parent_id=100, child_id=200),
                md_factory.TeamMemberFactory(parent_id=100, child_id=201),
                md_factory.TeamMemberFactory(parent_id=100, child_id=199),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)
            await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw)

        team_row = await assert_existing_row(sdb, Team, origin_node_id=100)
        assert team_row[Team.members.name] == [199, 200, 201, 202]


class TestSyncTeamsErrors(BaseTestSyncTeams):
    async def test_fail_with_unmapped_teams(self, sdb: Database, mdb_rw: Database) -> None:
        root_team_id = await self._mk_root_team(sdb)
        await models_insert(
            sdb,
            TeamFactory(parent_id=root_team_id, origin_node_id=None),
        )

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [md_factory.TeamFactory(node_id=101, name="A")]
            await models_insert(mdb_rw, *models)
            mdb_cleaner.add_models(*models)
            with pytest.raises(SyncTeamsError):
                await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw)
        await assert_missing_row(sdb, Team, origin_node_id=101)


class TestSyncTeamsDryRun(BaseTestSyncTeams):
    async def test_create_missing(self, sdb: Database, mdb_rw: Database) -> None:
        await self._mk_root_team(sdb)

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = (md_factory.TeamFactory(node_id=100, name="teamA"),)
            await models_insert(mdb_rw, *models)
            mdb_cleaner.add_models(*models)

            await sync_teams(
                DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb_rw, dry_run=True,
            )

        await assert_missing_row(sdb, Team, name="teamA")


class TestSyncTeamsForce(BaseTestSyncTeams):
    async def test_delete_unmapped_teams(self, sdb: Database, mdb: Database) -> None:
        root_team_id = await self._mk_root_team(sdb)
        await models_insert_auto_pk(
            sdb,
            TeamFactory(parent_id=root_team_id, origin_node_id=123),
        )
        await sync_teams(DEFAULT_ACCOUNT_ID, [DEFAULT_MD_ACCOUNT_ID], sdb, mdb, force=True)
        await assert_missing_row(sdb, Team, origin_node_id=123)
