import asyncio
import dataclasses
from datetime import datetime, timezone
from typing import Any
from unittest import mock

import pytest
import sqlalchemy as sa

from athenian.api.db import Database
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.reposet import (
    RepoIdentitiesMapper,
    RepoIdentitiesMapperFactory,
    RepoIdentity,
    RepoName,
    load_account_reposets,
    load_account_state,
    refresh_repository_names,
)
from athenian.api.models.metadata.github import FetchProgress
from athenian.api.models.state.models import RepositorySet
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_MD_ACCOUNT_ID
from tests.testutils.factory.state import LogicalRepositoryFactory


class TestLoadAccountReposets:
    async def test_transaction(self, sdb, mdb_rw):
        mdb = mdb_rw
        await sdb.execute(sa.delete(RepositorySet))

        async def load():
            return await load_account_reposets(
                1, self.login, [RepositorySet], sdb, mdb, None, None,
            )

        items = await asyncio.gather(*(load() for _ in range(10)), return_exceptions=True)
        errors = sum(isinstance(item, ResponseError) for item in items)
        assert errors > 0
        items = [{**i[0]} for i in items if not isinstance(i, ResponseError)]
        assert len(items) > 0
        for i in items[1:]:
            assert i == items[0]

    async def test_load_account_reposets_calmness(self, sdb, mdb_rw):
        await sdb.execute(sa.delete(RepositorySet))
        await mdb_rw.execute(
            sa.update(FetchProgress).values(
                {FetchProgress.updated_at: datetime.now(timezone.utc)},
            ),
        )

        try:
            with pytest.raises(ResponseError):
                await load_account_reposets(
                    1, self.login, [RepositorySet], sdb, mdb_rw, None, None,
                )
        finally:
            await mdb_rw.execute(
                sa.update(FetchProgress).values(
                    {
                        FetchProgress.updated_at: datetime(
                            2020, 3, 10, 14, 39, 19, tzinfo=timezone.utc,
                        ),
                    },
                ),
            )

    async def test_existing_logical_repos_are_preserved(self, sdb, mdb) -> None:
        await sdb.execute(sa.delete(RepositorySet))
        await models_insert(
            sdb,
            LogicalRepositoryFactory(name="my-logical-repo", repository_id=40550),
            LogicalRepositoryFactory(name="my-logical-repo-2", repository_id=40550),
            LogicalRepositoryFactory(name="my-logical-repo-3", repository_id=39652699),
        )

        loaded = await load_account_reposets(1, self.login, [RepositorySet], sdb, mdb, None, None)
        all_reposet = loaded[0]
        expected_items = [
            ["github.com/src-d/gitbase", 39652699],
            ["github.com/src-d/gitbase/my-logical-repo-3", 39652699],
            ["github.com/src-d/go-git", 40550],
            ["github.com/src-d/go-git/my-logical-repo", 40550],
            ["github.com/src-d/go-git/my-logical-repo-2", 40550],
        ]
        assert all_reposet[RepositorySet.items.name] == expected_items

    async def test_invalid_logical_repos_are_ignored(self, sdb, mdb) -> None:
        await sdb.execute(sa.delete(RepositorySet))
        await models_insert(
            sdb,
            # 999 does not exist
            LogicalRepositoryFactory(name="my-logical-repo", repository_id=999),
        )

        loaded = await load_account_reposets(1, self.login, [RepositorySet], sdb, mdb, None, None)
        all_reposet = loaded[0]
        expected_items = [
            ["github.com/src-d/gitbase", 39652699],
            ["github.com/src-d/go-git", 40550],
        ]
        assert all_reposet[RepositorySet.items.name] == expected_items

    async def login(self) -> str:
        return "2793551"


async def test_load_account_state_no_reposet(sdb, mdb):
    await sdb.execute(sa.delete(RepositorySet))
    state = await load_account_state(1, sdb, mdb, None, None)
    assert state is not None


async def test_refresh_repository_names_smoke(sdb, mdb):
    await sdb.execute(
        sa.update(RepositorySet)
        .where(RepositorySet.id == 1)
        .values(
            {
                RepositorySet.items: [
                    ["xxx", 40550],
                    ["github.com/src-d/go-git/alpha", 40550],
                    ["github.com/src-d/zzz/beta", 40550],
                    ["yyy", 100500],
                    ["github.com/src-d/gitbase", 39652699],
                ],
                RepositorySet.updated_at: datetime.now(timezone.utc),
                RepositorySet.updates_count: RepositorySet.updates_count + 1,
            },
        ),
    )
    items = await refresh_repository_names(1, (6366825,), sdb, mdb, None)
    assert items == [
        "github.com/src-d/gitbase",
        "github.com/src-d/go-git",
        "github.com/src-d/go-git/alpha",
        "github.com/src-d/go-git/beta",
    ]
    items = await sdb.fetch_val(sa.select(RepositorySet.items).where(RepositorySet.id == 1))
    assert items == [
        ["github.com/src-d/gitbase", 39652699],
        ["github.com/src-d/go-git", 40550],
        ["github.com/src-d/go-git/alpha", 40550],
        ["github.com/src-d/go-git/beta", 40550],
    ]


class TestRepoName:
    def test_from_prefixed_wrong_value(self) -> None:
        for bad_value in ("org/repo", "org/repo/logic"):
            with pytest.raises(ValueError):
                RepoName.from_prefixed(bad_value)

    def test_from_prefixed(self) -> None:
        name = RepoName.from_prefixed("github.com/org/repo")
        assert name.prefix == "github.com"
        assert name.organization == "org"
        assert name.physical == "repo"
        assert name.logical is None
        assert not name.is_logical
        assert str(name) == "github.com/org/repo"

    def test_from_prefixed_logical(self) -> None:
        name = RepoName.from_prefixed("gitlab.com/org/repo/logical")
        assert name.prefix == "gitlab.com"
        assert name.organization == "org"
        assert name.physical == "repo"
        assert name.logical == "logical"
        assert name.is_logical
        assert str(name) == "gitlab.com/org/repo/logical"

    def test_with_logical(self) -> None:
        name = RepoName.from_prefixed("gitlab.com/org/repo").with_logical("l")
        assert str(name) == "gitlab.com/org/repo/l"


class TestRepoIdentitiesMapper:
    def test_prefixed_names_to_identities(self) -> None:
        prefixer = self._mk_prefixer(repo_name_to_node={"org/repo1": 1, "org/repo2": 2})
        mapper = RepoIdentitiesMapper(prefixer)
        idents = mapper.prefixed_names_to_identities(
            ["github.com/org/repo1", "github.com/org/repo2/log"],
        )
        assert idents == [RepoIdentity(1, None), RepoIdentity(2, "log")]

    def test_prefixed_names_to_identities_invalid_repo(self) -> None:
        prefixer = self._mk_prefixer(repo_name_to_node={"org/repo1": 1})
        mapper = RepoIdentitiesMapper(prefixer)
        with pytest.raises(ResponseError):
            mapper.prefixed_names_to_identities(["github.com/org/repo2"])

    def test_identities_to_prefixed_names(self) -> None:
        prefixer = self._mk_prefixer(
            repo_node_to_prefixed_name={1: "github.com/org/repo1", 2: "github.com/org/repo2"},
        )
        mapper = RepoIdentitiesMapper(prefixer)
        names = mapper.identities_to_prefixed_names([RepoIdentity(1, None), RepoIdentity(1, "l")])
        assert names == ["github.com/org/repo1", "github.com/org/repo1/l"]

    def test_identities_to_prefixed_names_invalid_identities(self) -> None:
        prefixer = self._mk_prefixer(repo_node_to_prefixed_name={1: "github.com/org/repo1"})
        mapper = RepoIdentitiesMapper(prefixer)
        with pytest.raises(ValueError):
            mapper.identities_to_prefixed_names([RepoIdentity(2, None)])

    async def test_with_real_prefixer(self, mdb: Database) -> None:
        async with DBCleaner(mdb) as mdb_cleaner:
            mdb_models = [
                md_factory.RepositoryFactory(node_id=1, full_name="athenianco/repo-A"),
                md_factory.RepositoryFactory(node_id=2, full_name="athenianco/repo-B"),
            ]
            mdb_cleaner.add_models(*mdb_models)
            await models_insert(mdb, *mdb_models)
            prefixer = await Prefixer.load((DEFAULT_MD_ACCOUNT_ID,), mdb, None)

        mapper = RepoIdentitiesMapper(prefixer)

        names = mapper.identities_to_prefixed_names(
            [RepoIdentity(1, None), RepoIdentity(2, "l1"), RepoIdentity(2, "l2")],
        )
        assert names == [
            "github.com/athenianco/repo-A",
            "github.com/athenianco/repo-B/l1",
            "github.com/athenianco/repo-B/l2",
        ]

        identities = mapper.prefixed_names_to_identities(
            ["github.com/athenianco/repo-A/l", "github.com/athenianco/repo-B"],
        )
        assert identities == [RepoIdentity(1, "l"), RepoIdentity(2, None)]

    @classmethod
    def _mk_prefixer(cls, **kwargs: Any) -> Prefixer:
        for field in dataclasses.fields(Prefixer):
            if field.name != "do_not_construct_me_directly":
                kwargs.setdefault(field.name, {})
        kwargs["do_not_construct_me_directly"] = None
        return Prefixer(**kwargs)


class TestRepoIdentitiesMapperFactory:
    async def test_base(self, sdb: Database, mdb: Database) -> None:
        request = mock.Mock()
        request.sdb = sdb
        request.mdb = mdb
        request.cache = None
        factory = RepoIdentitiesMapperFactory(1, request)

        with mock.patch.object(
            RepoIdentitiesMapper, "from_request", wraps=RepoIdentitiesMapper.from_request,
        ) as from_request_mock:
            first_mapper = await factory()
            second_mapper = await factory()

        assert first_mapper is second_mapper
        from_request_mock.assert_called_once()
