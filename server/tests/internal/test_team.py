from operator import itemgetter

import pytest
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.internal.team import (
    MultipleRootTeamsError,
    RootTeamNotFoundError,
    TeamNotFoundError,
    delete_team,
    fetch_team_members_recursively,
    fetch_teams_recursively,
    get_bots_team,
    get_root_team,
    get_team_from_db,
    sync_team_members,
)
from athenian.api.models.state.models import Goal, Team, TeamGoal
from tests.testutils.db import (
    assert_existing_row,
    assert_missing_row,
    models_insert,
    transaction_conn,
)
from tests.testutils.factory.state import GoalFactory, TeamFactory, TeamGoalFactory


class TestGetRootTeam:
    async def test_existing(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=99, name="ROOT"))
        root_team = await get_root_team(1, sdb)
        assert root_team["id"] == 99
        assert root_team["name"] == "ROOT"

    async def test_not_existing(self, sdb: Database) -> None:
        with pytest.raises(RootTeamNotFoundError):
            await get_root_team(1, sdb)

    async def test_multiple_root_teams(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(), TeamFactory())
        with pytest.raises(MultipleRootTeamsError):
            await get_root_team(1, sdb)


class TestGetTeamFromDB:
    async def test_found(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=99, name="TEAM 99"))
        team = await get_team_from_db(1, 99, sdb)
        assert team["name"] == "TEAM 99"

    async def test_not_found(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=99, owner_id=2))
        with pytest.raises(TeamNotFoundError):
            await get_team_from_db(1, 99, sdb)


class TestGetBotsTeam:
    async def test_found(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=99, name=Team.BOTS, owner_id=1))
        team = await get_bots_team(1, sdb)
        assert team is not None
        assert team[Team.id.name] == 99

    async def test_not_found(self, sdb: Database) -> None:
        team = await get_bots_team(1, sdb)
        assert team is None


class TestFetchTeamsRecursively:
    async def test_no_teams(self, sdb: Database) -> None:
        res = await fetch_teams_recursively(1, sdb)
        assert res == []

    async def test_some_teams(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1), TeamFactory(id=2, parent_id=1))

        rows = sorted((await fetch_teams_recursively(1, sdb)), key=itemgetter("id"))
        assert len(rows) == 2

        assert rows[0][Team.id.name] == 1
        assert rows[0]["root_id"] == 1
        assert rows[1][Team.id.name] == 2
        assert rows[1]["root_id"] == 1

    async def test_multiple_roots(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1),
            TeamFactory(id=2, parent_id=1),
            TeamFactory(id=3, parent_id=1),
            TeamFactory(id=4, parent_id=None),
            TeamFactory(id=5, parent_id=4),
        )

        rows = sorted((await fetch_teams_recursively(1, sdb)), key=itemgetter("id"))

        assert len(rows) == 5
        assert rows[0] == (1, 1)
        assert rows[1] == (2, 1)
        assert rows[2] == (3, 1)
        assert rows[3] == (4, 4)
        assert rows[4] == (5, 4)

    async def test_explicit_root_team_ids(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1),
            TeamFactory(id=2, parent_id=1),
            TeamFactory(id=3),
            TeamFactory(id=4, parent_id=3),
            TeamFactory(id=5),
        )

        unsorted_rows = await fetch_teams_recursively(1, sdb, root_team_ids=(1, 4))
        rows = sorted(unsorted_rows, key=itemgetter("root_id", "id"))
        assert len(rows) == 3

        assert rows[0] == (1, 1)
        assert rows[1] == (2, 1)
        assert rows[2] == (4, 4)


class TestFetchTeamMembersecursively:
    async def test_team_not_found(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1))
        with pytest.raises(TeamNotFoundError):
            await fetch_team_members_recursively(1, sdb, 2)

    async def test_leaf_team(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=2, members=[1]), TeamFactory(id=3, members=[2, 3]))
        members = await fetch_team_members_recursively(1, sdb, 3)
        assert sorted(members) == [2, 3]

    async def test_base(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=2, members=[1]),
            TeamFactory(id=3, parent_id=2, members=[2, 3]),
            TeamFactory(id=4, parent_id=2, members=[1, 2, 4]),
        )
        members = await fetch_team_members_recursively(1, sdb, 2)
        assert sorted(members) == [1, 2, 3, 4]


class TestDeleteTeam:
    async def test_not_in_transaction(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1))
        team_row = await sdb.fetch_one(sa.select(Team).where(Team.id == 1))

        async with sdb.connection() as sdb_conn:
            with pytest.raises(AssertionError):
                await delete_team(team_row, sdb_conn)

        await assert_existing_row(sdb, Team, id=1)

    async def test_goal_left_empty_is_removed(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1),
            TeamFactory(id=10, parent_id=1),
            TeamFactory(id=11, parent_id=1),
            GoalFactory(id=20),
            GoalFactory(id=21),
            TeamGoalFactory(team_id=10, goal_id=20),
            TeamGoalFactory(team_id=11, goal_id=21),
        )

        team_row = await sdb.fetch_one(sa.select(Team).where(Team.id == 10))

        async with transaction_conn(sdb) as sdb_conn:
            await delete_team(team_row, sdb_conn)

        await assert_missing_row(sdb, Team, id=10)
        await assert_missing_row(sdb, TeamGoal, team_id=10)
        await assert_missing_row(sdb, Goal, id=20)

        await assert_existing_row(sdb, Team, id=1)
        await assert_existing_row(sdb, Team, id=11)
        await assert_existing_row(sdb, Team, id=1)
        await assert_existing_row(sdb, Goal, id=21)
        await assert_existing_row(sdb, TeamGoal, team_id=11, goal_id=21)


class TestSyncTeamMembers:
    async def test_no_update_needed(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=99, members=[1, 2, 3]))
        team_row = await assert_existing_row(sdb, Team, id=99)
        async with transaction_conn(sdb) as sdb_conn:
            added = await sync_team_members(team_row, [2, 3], sdb_conn)

        assert not added
        team_row = await assert_existing_row(sdb, Team, id=99)
        assert team_row[Team.members.name] == [1, 2, 3]

    async def test_update_done(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=99, members=[1, 2, 3]))
        team_row = await assert_existing_row(sdb, Team, id=99)
        async with transaction_conn(sdb) as sdb_conn:
            added = await sync_team_members(team_row, [5, 4, 3], sdb_conn)

        assert added == [4, 5]
        team_row = await assert_existing_row(sdb, Team, id=99)
        assert team_row[Team.members.name] == [1, 2, 3, 4, 5]
