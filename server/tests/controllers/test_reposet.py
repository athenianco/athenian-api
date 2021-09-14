import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy import delete, update

from athenian.api.controllers.reposet import load_account_reposets, load_account_state
from athenian.api.models.metadata.github import FetchProgress
from athenian.api.models.state.models import RepositorySet
from athenian.api.response import ResponseError


async def test_load_account_reposets_transaction(sdb, mdb_rw):
    mdb = mdb_rw
    await sdb.execute(delete(RepositorySet))

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


async def test_load_account_reposets_calmness(sdb, mdb_rw):
    await sdb.execute(delete(RepositorySet))
    await mdb_rw.execute(update(FetchProgress).values({
        FetchProgress.updated_at: datetime.now(timezone.utc),
    }))

    async def login():
        return "2793551"

    try:
        with pytest.raises(ResponseError):
            await load_account_reposets(1, login, [RepositorySet], sdb, mdb_rw, None, None)
    finally:
        await mdb_rw.execute(update(FetchProgress).values({
            FetchProgress.updated_at: datetime(2020, 3, 10, 14, 39, 19, tzinfo=timezone.utc),
        }))


async def test_load_account_state_no_reposet(sdb, mdb):
    await sdb.execute(delete(RepositorySet))
    state = await load_account_state(1, sdb, mdb, None, None)
    assert state is not None
