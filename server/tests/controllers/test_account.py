import pytest
from sqlalchemy import delete, select

from athenian.api.controllers.account import copy_teams_as_needed, get_metadata_account_ids, \
    match_metadata_installation
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.state.models import AccountGitHubAccount, Team
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
                        1, "vmarkovtsev", sdb_conn, mdb_conn, mdb_conn, slack)
    async with sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            async with mdb.connection() as mdb_conn:
                meta_ids = await match_metadata_installation(
                    1, "vmarkovtsev", sdb_conn, mdb_conn, mdb, slack)
    assert meta_ids == {6366825}


async def test_copy_teams_as_needed(sdb, mdb):
    created_teams = await copy_teams_as_needed(1, (6366825,), sdb, mdb, None)
    loaded_teams = {t[Team.name.name]: t for t in await sdb.fetch_all(select([Team]))}
    assert len(created_teams) == len(loaded_teams)
    assert loaded_teams.keys() == {
        "team", "engineering", "business", "operations", "product", "admin", "automation",
    }
    assert loaded_teams["product"][Team.members.name] == ["github.com/eiso", "github.com/warenlg"]
    assert loaded_teams["product"][Team.members_count.name] == 2
    assert loaded_teams["product"][Team.parent_id.name] == 1
