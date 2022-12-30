from athenian.api.db import Database
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.miners.participation import PRParticipationKind
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.team import fetch_teams_recursively
from athenian.api.internal.with_ import flatten_teams, resolve_withgroups
from athenian.api.models.state.models import Team
from athenian.api.models.web import PullRequestWith
from tests.testutils.db import model_insert_stmt, models_insert
from tests.testutils.factory.state import TeamFactory


class TestResolveWithGroups:
    async def test_base(self, sdb: Database, mdb: Database) -> None:
        for model in (
            TeamFactory(id=1, members=[1, 2, 3]),
            TeamFactory(id=2, members=[3, 4], parent_id=1),
            TeamFactory(id=3, members=[4, 5]),
        ):
            await sdb.execute(model_insert_stmt(model))
        withgroup = PullRequestWith(author=["{1}"])

        meta_ids = await get_metadata_account_ids(1, sdb, None)
        prefixer = await Prefixer.load(meta_ids, mdb, None)

        res = await resolve_withgroups(
            [withgroup], PRParticipationKind, True, 1, None, "", prefixer, sdb,
        )

        assert list(res[0][PRParticipationKind.AUTHOR]) == [1, 2, 3, 4]


class TestFlattenTeams:
    async def test(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[]),
            TeamFactory(id=2, members=[3, 4], parent_id=1),
            TeamFactory(id=3, members=[4, 5], parent_id=1),
            TeamFactory(id=4, members=[6], parent_id=3),
            TeamFactory(id=5, members=[10], parent_id=3),
            TeamFactory(id=6, members=[2, 9], parent_id=1),
            TeamFactory(id=7, members=[], parent_id=5),
        )

        team_rows = await fetch_teams_recursively(
            1, sdb, select_entities=(Team.id, Team.parent_id, Team.members),
        )

        res = flatten_teams(team_rows)

        assert sorted(res) == [1, 2, 3, 4, 5, 6, 7]
        assert res[1] == [2, 3, 4, 5, 6, 9, 10]
        assert res[2] == [3, 4]
        assert res[3] == [4, 5, 6, 10]
        assert res[4] == [6]
        assert res[5] == [10]
        assert res[6] == [2, 9]
        assert res[7] == []
