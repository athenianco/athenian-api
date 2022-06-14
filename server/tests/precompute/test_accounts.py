from argparse import Namespace
import contextlib
import logging
from typing import Any
from unittest import mock

from freezegun import freeze_time
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.defer import with_defer
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.prefixer import Prefixer
from athenian.api.models.state.models import AccountGitHubAccount, RepositorySet, Team
from athenian.api.precompute.accounts import (
    _DurationTracker,
    _ensure_bot_team,
    _ensure_root_team,
    create_teams,
    main,
    precompute_reposet,
)
from tests.testutils.db import assert_existing_row, model_insert_stmt, models_insert
from tests.testutils.factory.state import AccountFactory, RepositorySetFactory, TeamFactory

from .conftest import build_context, clear_all_accounts


class TestMain:
    @with_defer
    async def test_reposets_grouping(self, sdb, mdb, pdb, rdb, tqdm_disable) -> None:
        await clear_all_accounts(sdb)
        await models_insert(
            sdb,
            AccountFactory(id=11),
            RepositorySetFactory(owner_id=11),
            AccountFactory(id=12),
            RepositorySetFactory(owner_id=12),
            RepositorySetFactory(owner_id=12, name="another-reposet"),
            AccountGitHubAccount(id=1011, account_id=11),
            AccountGitHubAccount(id=1012, account_id=12),
        )

        ctx = build_context(sdb=sdb, mdb=mdb, pdb=pdb, rdb=rdb)
        namespace = self._namespace(account=["11", "12"])

        with self._wrap_precompute() as precompute_mock:
            await main(ctx, namespace, isolate=False)

        assert precompute_mock.call_count == 2
        # real precompute_reposet calls order is undefined, order calls by account id
        call_args_list = sorted(
            precompute_mock.call_args_list,
            key=lambda call: call[0][0].owner_id,
        )
        assert call_args_list[0][0][0].owner_id == 11
        assert call_args_list[0][0][0].name == RepositorySet.ALL
        assert call_args_list[0][0][1] == (1011,)
        assert call_args_list[1][0][0].owner_id == 12
        assert call_args_list[1][0][0].name == RepositorySet.ALL
        assert call_args_list[1][0][1] == (1012,)

    @with_defer
    async def test_not_installed_account(self, sdb, mdb, pdb, rdb, tqdm_disable) -> None:
        await clear_all_accounts(sdb)
        await models_insert(
            sdb,
            AccountFactory(id=11),
            RepositorySetFactory(owner_id=11),
            AccountFactory(id=12),
            RepositorySetFactory(owner_id=12),
            AccountGitHubAccount(id=1011, account_id=11),
        )
        ctx = build_context(sdb=sdb, mdb=mdb, pdb=pdb, rdb=rdb)
        namespace = self._namespace(account=["11", "12"])

        with self._wrap_precompute() as precompute_mock:
            await main(ctx, namespace, isolate=False)

        # precompute is not called on account 12
        assert precompute_mock.call_count == 1
        assert precompute_mock.call_args[0][0].owner_id == 11
        assert precompute_mock.call_args[0][0].name == RepositorySet.ALL
        assert precompute_mock.call_args[0][1] == (1011,)

    @with_defer
    async def test_multiple_gh_accounts(self, sdb, mdb, pdb, rdb, tqdm_disable) -> None:
        await clear_all_accounts(sdb)
        await models_insert(
            sdb,
            AccountFactory(id=11),
            RepositorySetFactory(owner_id=11),
            AccountGitHubAccount(id=1011, account_id=11),
            AccountGitHubAccount(id=2011, account_id=11),
        )
        ctx = build_context(sdb=sdb, mdb=mdb, pdb=pdb, rdb=rdb)
        namespace = self._namespace(account=["11"])

        with self._wrap_precompute() as precompute_mock:
            await main(ctx, namespace, isolate=False)

        assert precompute_mock.call_count == 1
        assert precompute_mock.call_args[0][0].owner_id == 11
        assert precompute_mock.call_args[0][0].name == RepositorySet.ALL
        assert precompute_mock.call_args[0][1] == (1011, 2011)

    def _namespace(self, **kwargs: Any) -> Namespace:
        kwargs.setdefault("skip_jira_identity_map", True)
        kwargs.setdefault("prometheus_pushgateway", None)
        return Namespace(**kwargs)

    def _wrap_precompute(self) -> mock._patch:
        mock_path = f"{main.__module__}.precompute_reposet"
        return mock.patch(mock_path, wraps=precompute_reposet)


class TestCreateTeams:
    async def test_create_all(self, sdb: Database, mdb: Database) -> None:
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        prefixer = await Prefixer.load(meta_ids, mdb, None)
        n_imported, _ = await create_teams(1, meta_ids, set(), prefixer, sdb, mdb, None)

        root_teams = await sdb.fetch_all(sa.select(Team).where(Team.parent_id.is_(None)))
        assert len(root_teams) == 1
        root_team = root_teams[0]

        await assert_existing_row(sdb, Team, name=Team.BOTS, parent_id=root_team[Team.id.name])

        all_child_teams = await sdb.fetch_all(
            sa.select(Team).where(sa.and_(Team.owner_id == 1, Team.parent_id.isnot(None))),
        )
        assert n_imported == len(all_child_teams) - 1


class TestEnsureBotTeam:
    async def test_ensure_bot_team_already_existing(self, sdb: Database, mdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1),
            TeamFactory(id=2, name=Team.BOTS, members=[1, 2, 3], parent_id=1),
        )
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        prefixer = await Prefixer.load(meta_ids, mdb, None)
        assert await _ensure_bot_team(1, meta_ids, set(), 1, prefixer, sdb, mdb) == 3

    async def test_ensure_bot_team_not_existing(self, sdb: Database, mdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10))
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        prefixer = await Prefixer.load(meta_ids, mdb, None)
        bot_logins = {"gkwillie"}
        assert await _ensure_bot_team(1, meta_ids, bot_logins, 10, prefixer, sdb, mdb) == 1

        bot_team = await assert_existing_row(sdb, Team, name="Bots", owner_id=1, parent_id=10)
        assert len(bot_team[Team.members.name]) == 1

    async def test_ensure_bot_team_unknown_user(self, sdb: Database, mdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=10))

        meta_ids = await get_metadata_account_ids(1, sdb, None)
        prefixer = await Prefixer.load(meta_ids, mdb, None)
        bot_logins = {"gkwillie", "not-existing-gh-user"}
        assert await _ensure_bot_team(1, meta_ids, bot_logins, 10, prefixer, sdb, mdb) == 1
        bot_team = await sdb.fetch_one(sa.select(Team).where(Team.name == "Bots"))
        assert bot_team[Team.owner_id.name] == 1
        assert len(bot_team[Team.members.name]) == 1
        assert bot_team[Team.parent_id.name] == 10


class TestEnsureRootTeam:
    # tests for private function _ensure_root_team
    async def test_already_existing(self, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(parent_id=None, name=Team.ROOT, id=97)))
        assert await _ensure_root_team(1, sdb) == 97

    async def test_not_existing(self, sdb: Database) -> None:
        root_team_id = await _ensure_root_team(1, sdb)
        root_team_row = await assert_existing_row(
            sdb, Team, id=root_team_id, name=Team.ROOT, parent_id=None,
        )
        assert root_team_row[Team.members.name] == []


class TestDurationTracker:
    # tests for private class _DurationTracker
    def test_gateway_available(self) -> None:
        with freeze_time("2021-01-10T10:10:10"):
            tracker = _DurationTracker("host:9000", logging.getLogger(__name__))

        with freeze_time("2021-01-10T10:10:52"):
            with self._mock_prometheus_push_handler() as push_handler:
                tracker.track(1, [3, 4], False)

        push_handler.assert_called_once()
        data = push_handler.call_args_list[0][1]["data"].decode("utf-8")
        assert "TYPE precompute_account_seconds histogram" in data
        assert (
            "precompute_account_seconds_count"
            '{account="1",github_account="3,4",is_fresh="False"} 1.0'
            in data
        )

    def test_gateway_not_available(self) -> None:
        tracker = _DurationTracker(None, logging.getLogger(__name__))

        with self._mock_prometheus_push_handler() as push_handler:
            tracker.track(1, [3, 4], False)

        push_handler.assert_not_called()

    @contextlib.contextmanager
    def _mock_prometheus_push_handler(self):
        from prometheus_client.exposition import push_to_gateway

        # change the handler= default argument of push_to_gateway with a mocked handler
        handler_mock = mock.Mock()
        orig_defaults = tuple(push_to_gateway.__defaults__)
        mocked_defaults = orig_defaults[:2] + (handler_mock,)
        with mock.patch.object(push_to_gateway, "__defaults__", mocked_defaults):
            yield handler_mock
