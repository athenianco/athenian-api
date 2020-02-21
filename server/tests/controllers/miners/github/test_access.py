import pytest

from athenian.api import ResponseError
from athenian.api.controllers.miners.github.access import AccessChecker


@pytest.mark.parametrize("has_cache", [True, False])
async def test_check_access_normal(mdb, sdb, cache, has_cache):
    cache = cache if has_cache else None
    checker = AccessChecker(1, sdb, mdb, cache)
    await checker.load()
    assert await checker.check({"github.com/src-d/go-git"}) == set()
    if has_cache:
        assert len(cache.mem) == 2
    assert await checker.check({"github.com/src-d/go-buck"}) == {"github.com/src-d/go-buck"}


async def test_check_access_connections(mdb, sdb):
    async with sdb.connection() as sdb_conn:
        async with mdb.connection() as mdb_conn:
            checker = AccessChecker(1, sdb_conn, mdb_conn, None)
            await checker.load()
            assert await checker.check({"github.com/src-d/go-git"}) == set()


async def test_check_access_error(mdb, sdb):
    checker = AccessChecker(2, sdb, mdb, None)
    with pytest.raises(ResponseError):
        await checker.load()
