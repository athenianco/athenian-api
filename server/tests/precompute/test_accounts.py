from argparse import Namespace
from unittest import mock

import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.prefixer import Prefixer
from athenian.api.models.state.models import RepositorySet, Team
from athenian.api.precompute.accounts import _ensure_bot_team, _ensure_root_team, main, \
    precompute_reposet
from tests.testutils.db import assert_existing_row, model_insert_stmt
from tests.testutils.factory.state import AccountFactory, RepositorySetFactory, TeamFactory

from .conftest import build_context, clear_all_accounts


async def test_reposets_grouping(sdb, mdb, tqdm_disable) -> None:
    await clear_all_accounts(sdb)
    for model in (
        AccountFactory(id=11),
        RepositorySetFactory(owner_id=11),
        AccountFactory(id=12),
        RepositorySetFactory(owner_id=12),
        RepositorySetFactory(owner_id=12, name="another-reposet"),
    ):
        await sdb.execute(model_insert_stmt(model))

    ctx = build_context(sdb=sdb, mdb=mdb)
    namespace = Namespace(skip_jira_identity_map=True, account=["11", "12"])

    with mock.patch(
        f"{main.__module__}.precompute_reposet",
        wraps=precompute_reposet,
    ) as precompute_mock:
        await main(ctx, namespace, isolate=False)

    assert precompute_mock.call_count == 2
    # real precompute_reposet calls order is undefined, order calls by account id
    call_args_list = sorted(
        precompute_mock.call_args_list,
        key=lambda call: call[0][0].owner_id,
    )
    assert call_args_list[0][0][0].owner_id == 11
    assert call_args_list[0][0][0].name == RepositorySet.ALL
    assert call_args_list[1][0][0].owner_id == 12
    assert call_args_list[1][0][0].name == RepositorySet.ALL


class TestEnsureBotTeam:
    async def test_ensure_bot_team_already_existing(
        self, sdb: Database, mdb: Database,
    ) -> None:
        await sdb.execute(
            model_insert_stmt(TeamFactory(name=Team.BOTS, members=[1, 2, 3])),
        )
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        prefixer = await Prefixer.load(meta_ids, mdb, None)
        assert await _ensure_bot_team(1, meta_ids, set(), prefixer, sdb, mdb) == 3

    async def test_ensure_bot_team_not_existing(
        self, sdb: Database, mdb: Database,
    ) -> None:
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        prefixer = await Prefixer.load(meta_ids, mdb, None)
        bot_logins = {"gkwillie"}
        assert await _ensure_bot_team(1, meta_ids, bot_logins, prefixer, sdb, mdb) == 1
        bot_team = await sdb.fetch_one(sa.select(Team).where(Team.name == "Bots"))
        assert bot_team[Team.owner_id.name] == 1
        assert len(bot_team[Team.members.name]) == 1

    async def test_ensure_bot_team_unknown_user(
        self, sdb: Database, mdb: Database,
    ) -> None:
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        prefixer = await Prefixer.load(meta_ids, mdb, None)
        bot_logins = {"gkwillie", "not-existing-gh-user"}
        assert await _ensure_bot_team(1, meta_ids, bot_logins, prefixer, sdb, mdb) == 1
        bot_team = await sdb.fetch_one(sa.select(Team).where(Team.name == "Bots"))
        assert bot_team[Team.owner_id.name] == 1
        assert len(bot_team[Team.members.name]) == 1


class TestEnsureROotTeam:
    # tests for private function _ensure_root_team
    async def test_already_existing(self, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(parent_id=None, name=Team.ROOT, id=97)))
        assert await _ensure_root_team(1, sdb) == 97

    async def test_not_existing(self, sdb: Database) -> None:
        root_team_id = await _ensure_root_team(1, sdb)
        import pprint; pprint.pprint(root_team_id)
        root_team_row = await assert_existing_row(
            sdb, Team, id=root_team_id, name=Team.ROOT, parent_id=None,
        )
        assert root_team_row[Team.members.name] == []
