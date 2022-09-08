import asyncio
import marshal
import pickle
import time
from typing import Callable, Optional
from unittest import mock

import aiomcache
from freezegun import freeze_time
import pytest

from athenian.api.cache import cached, cached_methods, gen_cache_key
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.tracing import sentry_span
from tests.conftest import build_fake_cache, has_memcached


@pytest.mark.parametrize(
    "fmt,args",
    [
        ("text", []),
        ("", []),
        ("1", []),
        ("%s", [""]),
        ("xxx %s %d yyy", ["y", 2]),
        ("x" * 100500, []),
        ("%s", []),
    ],
)
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


class TestCached:
    @pytest.mark.flaky(reruns=3, reruns_delay=1)
    @pytest.mark.skipif(not has_memcached, reason="memcached is unavailable")
    @with_defer
    async def test_cached(self, memcached):
        @cached(
            exptime=1,
            serialize=marshal.dumps,
            deserialize=marshal.loads,
            key=lambda number, **_: (number,),
        )
        async def add_one(
            eval_notify: Callable,
            number: int,
            cache: Optional[aiomcache.Client],
        ) -> int:
            eval_notify()
            return number + 1

        evaluated = 0

        def inc_evaluated():
            nonlocal evaluated
            evaluated += 1

        assert await add_one(inc_evaluated, 1, memcached) == 2
        await wait_deferred()
        assert await add_one(inc_evaluated, 1, memcached) == 2
        await wait_deferred()
        assert evaluated == 1
        await asyncio.sleep(1.1)
        assert await add_one(inc_evaluated, 1, memcached) == 2
        await wait_deferred()
        assert evaluated == 2

    @with_defer
    async def test_serialization_errors(self, cache):
        def crash(*_):
            raise ValueError

        @cached(exptime=1, serialize=crash, deserialize=pickle.loads, key=lambda **_: ())
        async def test(cache):
            return 1

        await test(cache)
        await wait_deferred()

        @cached(exptime=1, serialize=pickle.dumps, deserialize=crash, key=lambda **_: ())
        async def test(cache):
            return 1

        await test(cache)
        await wait_deferred()
        await test(cache)

    @with_defer
    async def test_preprocess(self) -> None:
        def preprocess(result, **kw: list[str]) -> list[str]:
            return list(reversed(result))

        preprocess = mock.Mock(wraps=preprocess)

        def postprocess(result, **kw: list[str]) -> list[str]:
            return list(reversed(result))

        postprocess = mock.Mock(wraps=postprocess)

        n_calls = 0

        @cached(
            2 * 16,
            preprocess=preprocess,
            postprocess=postprocess,
            serialize=pickle.dumps,
            deserialize=pickle.loads,
            key=lambda **kw: ("a",),
        )
        async def func(cache) -> list[str]:
            nonlocal n_calls
            n_calls += 1
            return ["a", "b", "c"]

        cache = build_fake_cache()
        assert (await func(cache)) == ["a", "b", "c"]
        assert n_calls == 1
        preprocess.assert_called_once()

        # wait cache write to finish
        await wait_deferred()

        assert (await func(cache)) == ["a", "b", "c"]
        assert n_calls == 1
        postprocess.assert_called_once()


@cached_methods
class Foo:
    def bar_postprocess(self, result, arg_used, **_) -> str:
        return f"cache is used {arg_used}"

    @cached(
        exptime=100500,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda arg_used, **_: (arg_used,),
        postprocess=bar_postprocess,
    )
    async def bar(self, arg_used, arg_unused, cache) -> str:
        return f"cache is not used {arg_used} {arg_unused}"

    @sentry_span
    @cached(
        exptime=100500,
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda arg_used, **_: (arg_used,),
        postprocess=bar_postprocess,
    )
    async def bar_traced(self, arg_used, arg_unused, cache) -> str:
        return f"cache is not used {arg_used} {arg_unused}"


@pytest.mark.parametrize("met_name", ("bar", "bar_traced"))
@with_defer
async def test_cached_methods(cache, met_name):
    foo = Foo()

    met = getattr(foo, met_name)
    r = await met("yes", "whatever", cache)
    assert r == "cache is not used yes whatever"
    await wait_deferred()
    r = await met("yes", "no", cache)
    assert r == "cache is used yes"
    assert met.__cached__
    await met.reset_cache("yes", "whatever", cache)
    assert isinstance(met.cache_key("yes", "whatever", cache), bytes)


# TODO: fix response validation against the schema
@pytest.mark.app_validate_responses(False)
@freeze_time("2022-01-01")
async def test_expires_header(client, headers, client_cache):
    body = {
        "account": 1,
        "repositories": ["{1}"],
    }
    response = await client.request(
        method="POST", path="/v1/filter/labels", headers=headers, json=body,
    )
    assert response.status == 200
    exp1 = response.headers["expires"]
    assert exp1 == "Sat, 01 Jan 2022 01:00:00 GMT"  # + middle_term_exptime = 1 hour
    time.sleep(1)
    response = await client.request(
        method="POST", path="/v1/filter/labels", headers=headers, json=body,
    )
    exp2 = response.headers["expires"]
    assert exp1 == exp2


class TestPickleDumpsPatch:
    def test(self) -> None:
        # importing module should patch pickle.dumps to default to the last protocol version
        import athenian.api.cache  # noqa

        buf = pickle.dumps(object())
        protocol_version = int(buf[1])
        assert protocol_version == pickle.HIGHEST_PROTOCOL

        # protocol can still be manually overridden
        buf = pickle.dumps(object(), 2)
        protocol_version = int(buf[1])
        assert protocol_version == 2

        buf = pickle.dumps(object(), protocol=3)
        protocol_version = int(buf[1])
        assert protocol_version == 3
