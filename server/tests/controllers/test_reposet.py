import asyncio
from contextvars import ContextVar

from sqlalchemy import delete

from athenian.api.controllers.reposet import load_account_reposets, load_account_state
from athenian.api.models.state.models import RepositorySet
from athenian.api.response import ResponseError


async def test_load_account_reposets_transaction(sdb, mdb_rw):
    mdb = mdb_rw
    await sdb.execute(delete(RepositorySet))
    sdb._connection_context = ContextVar("connection_context")

    async def login():
        return "2793551"

    async def load():
        return await load_account_reposets(
            1, login, [RepositorySet], sdb, mdb, None, None)

    items = await asyncio.gather(*(load() for _ in range(10)), return_exceptions=True)
    errors = sum(isinstance(item, ResponseError) for item in items)
    assert errors > 0
    items = [{**i[0]} for i in items if not isinstance(i, ResponseError)]
    assert len(items) > 0
    for i in items[1:]:
        assert i == items[0]


async def test_load_account_state_no_reposet(sdb, mdb):
    await sdb.execute(delete(RepositorySet))
    state = await load_account_state(1, sdb, mdb, None, None)
    assert state is not None
