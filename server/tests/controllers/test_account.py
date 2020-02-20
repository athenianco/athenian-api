import pytest

from athenian.api import ResponseError
from athenian.api.controllers.account import get_installation_id


async def test_get_installation_id_cache(sdb, cache):
    assert await get_installation_id(1, sdb, cache) == 6366825
    assert len(cache.mem) == 1
    # use the cache
    assert await get_installation_id(1, sdb, cache) == 6366825
    assert len(cache.mem) == 1


async def test_get_installation_id_conn(sdb):
    async with sdb.connection() as sdb_conn:
        assert await get_installation_id(1, sdb_conn, None) == 6366825


async def test_get_installation_id_no_cache(sdb):
    assert await get_installation_id(1, sdb, None) == 6366825


async def test_get_installation_id_error(sdb):
    with pytest.raises(ResponseError):
        await get_installation_id(2, sdb, None)
