import asyncio
from contextvars import ContextVar

from sqlalchemy import delete

from athenian.api import ResponseError
from athenian.api.controllers.reposet import load_account_reposets
from athenian.api.models.state.models import RepositorySet


async def test_load_account_reposets_transaction(sdb, mdb):
    await sdb.execute(delete(RepositorySet))
    sdb._connection_context = ContextVar("connection_context")

    async def load():
        return await load_account_reposets(1, "2793551", [RepositorySet], sdb, mdb, None, None)

    items = await asyncio.gather(*(load() for _ in range(10)), return_exceptions=True)
    errors = sum(isinstance(item, ResponseError) for item in items)
    assert errors == len(items) - 1, str(items)


async def test_load_account_reposets_single_tenant(sdb, mdb):
    await sdb.execute(delete(RepositorySet))
    reposets = await load_account_reposets(1, None, [RepositorySet], sdb, mdb, None, None)
    assert len(reposets) == 1
    assert reposets[0]["items"] == [
        "github.com/src-d/gitbase", "github.com/src-d/go-git",
    ]
