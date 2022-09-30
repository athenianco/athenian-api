import pytest
from sqlalchemy import delete

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.account import (
    get_metadata_account_ids,
    get_user_account_status,
    match_metadata_installation,
)
from athenian.api.models.state.models import AccountGitHubAccount
from athenian.api.models.web import User
from athenian.api.response import ResponseError


@with_defer
async def test_get_metadata_account_id_cache(sdb, cache):
    assert await get_metadata_account_ids(1, sdb, cache) == (6366825,)
    await wait_deferred()
    assert len(cache.mem) == 1
    # use the cache
    assert await get_metadata_account_ids(1, sdb, cache) == (6366825,)
    await wait_deferred()
    assert len(cache.mem) == 1


async def test_get_metadata_account_id_conn(sdb):
    async with sdb.connection() as sdb_conn:
        assert await get_metadata_account_ids(1, sdb_conn, None) == (6366825,)


async def test_get_metadata_account_id_no_cache(sdb):
    assert await get_metadata_account_ids(1, sdb, None) == (6366825,)


async def test_get_metadata_account_id_error(sdb):
    with pytest.raises(ResponseError):
        await get_metadata_account_ids(2, sdb, None)


@with_defer
async def test_match_metadata_installation(sdb, mdb, slack):
    await sdb.execute(delete(AccountGitHubAccount))
    with pytest.raises(AssertionError):
        async with sdb.connection() as sdb_conn:
            async with sdb_conn.transaction():
                async with mdb.connection() as mdb_conn:
                    await match_metadata_installation(
                        1, "vmarkovtsev", sdb_conn, mdb_conn, mdb_conn, slack,
                    )
    async with sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            async with mdb.connection() as mdb_conn:
                meta_ids = await match_metadata_installation(
                    1, "vmarkovtsev", sdb_conn, mdb_conn, mdb, slack,
                )
    assert meta_ids == {6366825}


@with_defer
async def test_get_user_account_status_slack(sdb, mdb, slack):
    async def user():
        return User(
            id="github||60340680", native_id="i", login="gkwillie", email="bot@athenian.co",
        )

    with pytest.raises(ResponseError):
        await get_user_account_status("github|60340680", 1, sdb, mdb, user, slack, None)
