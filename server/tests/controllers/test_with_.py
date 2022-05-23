from athenian.api.controllers.account import get_metadata_account_ids
from athenian.api.controllers.miners.types import PRParticipationKind
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.with_ import resolve_withgroups
from athenian.api.db import Connection
from athenian.api.models.web import PullRequestWith
from tests.testutils.db import model_insert_stmt
from tests.testutils.factory.state import TeamFactory


class TestResolveWithGroups:
    async def test_base(self, sdb: Connection, mdb: Connection) -> None:
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
