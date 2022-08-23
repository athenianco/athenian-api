import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.internal.account import copy_teams_as_needed, get_multiple_metadata_account_ids
from athenian.api.models.metadata.github import Team as MetaTeam
from athenian.api.models.state.models import Team
from tests.testutils.db import DBCleaner, assert_existing_row, models_insert, models_insert_auto_pk
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.state import AccountFactory, AccountGitHubAccountFactory, TeamFactory


class TestGetMultipleMetadataMetadataIds:
    async def test(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            AccountFactory(id=10),
            AccountFactory(id=11),
            AccountFactory(id=12),
            AccountGitHubAccountFactory(account_id=10, id=20),
            AccountGitHubAccountFactory(account_id=10, id=21),
            AccountGitHubAccountFactory(account_id=11, id=22),
        )

        accounts_meta_ids = await get_multiple_metadata_account_ids([10, 11, 12, 13], sdb, None)

        assert sorted(accounts_meta_ids[10]) == [20, 21]
        assert sorted(accounts_meta_ids[11]) == [22]
        assert sorted(accounts_meta_ids[12]) == []
        assert sorted(accounts_meta_ids[13]) == []


class TestCopyTeamsAsNeeded:
    async def test_base(self, sdb: Database, mdb: Database) -> None:
        (root_team_id,) = await models_insert_auto_pk(sdb, TeamFactory(name=Team.ROOT))

        created_teams, n = await copy_teams_as_needed(1, (6366825,), root_team_id, sdb, mdb, None)
        loaded_team_rows = await sdb.fetch_all(sa.select(Team).where(Team.id != root_team_id))
        loaded_teams = {t[Team.name.name]: t for t in loaded_team_rows}

        assert len(created_teams) == len(loaded_teams) == n
        assert loaded_teams.keys() == {
            "team",
            "engineering",
            "business",
            "operations",
            "product",
            "admin",
            "automation",
        }

        assert loaded_teams["product"][Team.members.name] == [29, 39936]
        assert loaded_teams["product"][Team.parent_id.name] == loaded_teams["team"][Team.id.name]
        # team "team" hasn't a real parent team, so its parent team becomes root_team_id
        assert loaded_teams["team"][Team.parent_id.name] == root_team_id

        for team_name in ("team", "business", "admin"):
            assert loaded_teams[team_name][Team.origin_node_id.name] == await mdb.fetch_val(
                sa.select(MetaTeam.id).where(MetaTeam.name == team_name),
            )

        assert not any(team[Team.parent_id.name] is None for team in loaded_teams.values())

        created_teams, n = await copy_teams_as_needed(1, (6366825,), root_team_id, sdb, mdb, None)
        assert created_teams == []
        assert n == len(loaded_teams)

    async def test_meta_team_invalid_parent(self, sdb: Database, mdb_rw: Database) -> None:
        (root_team_id,) = await models_insert_auto_pk(sdb, TeamFactory(name=Team.ROOT))

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = (md_factory.TeamFactory(node_id=101, parent_team_id=1010110101, name="T"),)
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)
            await copy_teams_as_needed(1, (6366825,), root_team_id, sdb, mdb_rw, None)

        await assert_existing_row(sdb, Team, name="T", origin_node_id=101, parent_id=root_team_id)
