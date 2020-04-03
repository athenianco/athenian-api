import asyncio
import marshal
from typing import Optional

import aiomcache
import pytest

from athenian.api.cache import cached, gen_cache_key
from tests.conftest import has_memcached


@pytest.mark.parametrize("fmt,args", [("text", []),
                                      ("", []),
                                      ("1", []),
                                      ("%s", [""]),
                                      ("xxx %s %d yyy", ["y", 2]),
                                      ("x" * 100500, [])])
def test_gen_cache_key_formats(fmt, args):
    key = gen_cache_key(fmt, *args)
    assert key
    aiomcache.Client._validate_key(aiomcache.Client, key)
    for _ in range(100):
        # check that there is no randomness
        assert key == gen_cache_key(fmt, *args)


def test_gen_cache_key_distinct():
    key1 = gen_cache_key("a" * 10000)
    key2 = gen_cache_key("a" * 9999 + "b")
    assert key1 != key2
    key2 = gen_cache_key("b" + "a" * 9999)
    assert key1 != key2


@cached(
    exptime=1,
    serialize=marshal.dumps,
    deserialize=marshal.loads,
    key=lambda number, **_: (number,),
)
async def add_one(eval_notify: callable, number: int, cache: Optional[aiomcache.Client]) -> int:
    eval_notify()
    return number + 1


@pytest.mark.skipif(not has_memcached, reason="memcached is unavailable")
async def test_cached(memcached):
    evaluated = 0

    def inc_evaluated():
        nonlocal evaluated
        evaluated += 1

    assert await add_one(inc_evaluated, 1, memcached) == 2
    assert await add_one(inc_evaluated, 1, memcached) == 2
    assert evaluated == 1
    await asyncio.sleep(1)
    assert await add_one(inc_evaluated, 1, memcached) == 2
    assert evaluated == 2
