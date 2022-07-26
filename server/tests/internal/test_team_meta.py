from athenian.api.db import Database
from athenian.api.internal.team_meta import get_meta_teams_members
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_MD_ACCOUNT_ID


class TestGetMetaTeamsMembers:
    async def test_base(self, mdb: Database) -> None:
        async with DBCleaner(mdb) as mdb_cleaner:
            models = (
                md_factory.TeamFactory(id=200),
                md_factory.TeamFactory(id=201),
                md_factory.TeamFactory(id=202),
                md_factory.TeamFactory(id=203),
                md_factory.TeamMemberFactory(parent_id=200, child_id=10),
                md_factory.TeamMemberFactory(parent_id=200, child_id=11),
                md_factory.TeamMemberFactory(parent_id=201, child_id=12),
                md_factory.TeamMemberFactory(parent_id=203, child_id=13),
            )
            mdb_cleaner.add_models(*models)
            await models_insert(mdb, *models)

            members = await get_meta_teams_members([200, 201, 202], [DEFAULT_MD_ACCOUNT_ID], mdb)
            assert members[200] == [10, 11]
            assert members[201] == [12]
            assert members[202] == []
            assert members[203] == []
