import pytest

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.miners.github.access import GitHubAccessChecker


@pytest.mark.parametrize("has_cache", [True, False])
@with_defer
async def test_check_access_normal(mdb, sdb, cache, has_cache):
    cache = cache if has_cache else None
    checker = await GitHubAccessChecker(1, (6366825,), sdb, mdb, cache).load()
    assert await checker.check({"src-d/go-git"}) == set()
    await wait_deferred()
    if has_cache:
        assert len(cache.mem) == 1
    assert await checker.check({"src-d/go-buck"}) == {"src-d/go-buck"}


async def test_check_access_connections(mdb, sdb):
    async with sdb.connection() as sdb_conn:
        async with mdb.connection() as mdb_conn:
            checker = await GitHubAccessChecker(1, (6366825,), sdb_conn, mdb_conn, None).load()
            assert await checker.check({"src-d/go-git"}) == set()


async def test_check_access_empty(mdb, sdb):
    with pytest.raises(AssertionError):
        GitHubAccessChecker(1, (), sdb, mdb, None)
