import asyncio
from datetime import datetime, timezone
import logging

import pytest
import sqlalchemy as sa

from athenian.api.internal.account import RepositoryReference
from athenian.api.internal.data_health_metrics import DataHealthMetrics
from athenian.api.internal.prefixer import RepositoryName
from athenian.api.internal.reposet import load_account_reposets, load_account_state
from athenian.api.models.metadata.github import FetchProgress
from athenian.api.models.state.models import RepositorySet
from athenian.api.precompute.accounts import insert_new_repositories, refresh_reposet
from athenian.api.response import ResponseError
from tests.testutils.db import models_insert
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
            ["github.com", 40550, ""],
            ["github.com", 40550, "my-logical-repo"],
            ["github.com", 40550, "my-logical-repo-2"],
            ["github.com", 39652699, ""],
            ["github.com", 39652699, "my-logical-repo-3"],
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
            ["github.com", 40550, ""],
            ["github.com", 39652699, ""],
        ]
        assert all_reposet[RepositorySet.items.name] == expected_items

    async def login(self) -> str:
        return "2793551"


async def test_load_account_state_no_reposet(sdb, mdb):
    await sdb.execute(sa.delete(RepositorySet))
    state = await load_account_state(1, sdb, mdb, None, None)
    assert state is not None


@pytest.mark.parametrize(
    "tracking_re, names, refs",
    [
        (
            ".*",
            [
                RepositoryName("github.com", "src-d", "go-git", "alpha"),
                RepositoryName("github.com", "src-d", "gitbase", ""),
            ],
            [
                RepositoryReference("github.com", 40550, "alpha"),
                RepositoryReference("github.com", 39652699, ""),
            ],
        ),
        (
            "(?!src-d/gitbase).*",
            [RepositoryName("github.com", "src-d", "go-git", "alpha")],
            [RepositoryReference("github.com", 40550, "alpha")],
        ),
        (
            "src-d/gitbase",
            [
                RepositoryName("github.com", "src-d", "go-git", "alpha"),
                RepositoryName("github.com", "src-d", "gitbase", ""),
            ],
            [
                RepositoryReference("github.com", 40550, "alpha"),
                RepositoryReference("github.com", 39652699, ""),
            ],
        ),
    ],
)
async def test_insert_new_repositories_smoke(mdb, tracking_re, names, refs):
    new_names, new_refs = await insert_new_repositories(
        [RepositoryName("github.com", "src-d", "go-git", "alpha")],
        [RepositoryReference("github.com", 40550, "alpha")],
        tracking_re,
        (6366825,),
        mdb,
    )
    assert new_names == names
    assert new_refs == refs


async def test_refresh_reposet_smoke(sdb, mdb, prefixer, meta_ids):
    reposet = RepositorySet(
        id=1,
        tracking_re=".*",
        items=[
            RepositoryReference("github.com", 40550, ""),
            RepositoryReference("github.com", 100500, ""),
        ],
    )
    metrics = DataHealthMetrics.empty()
    deref_items = await refresh_reposet(
        reposet, prefixer, meta_ids, sdb, mdb, logging.getLogger(), metrics,
    )
    assert deref_items == [
        RepositoryName("github.com", "src-d", "go-git", ""),
        RepositoryName("github.com", "src-d", "gitbase", ""),
    ]
    saved_items = await sdb.fetch_val(sa.select(RepositorySet.items).where(RepositorySet.id == 1))
    assert saved_items == [
        RepositoryReference("github.com", 40550, ""),
        RepositoryReference("github.com", 39652699, ""),
    ]
