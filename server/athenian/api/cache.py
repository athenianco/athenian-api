from contextvars import ContextVar
import functools
import inspect
import logging
import pickle
import time
from typing import Any, ByteString, Callable, Coroutine, Optional, Tuple, Union

from aiohttp import web
import aiomcache
from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.utils import INF
from xxhash import xxh64_hexdigest

from athenian.api import metadata
from athenian.api.metadata import __package__, __version__
from athenian.api.typing_utils import wraps

pickle.dumps = functools.partial(pickle.dumps, protocol=-1)
max_exptime = 30 * 24 * 3600  # 30 days according to the docs


class CancelCache(Exception):
    """Raised in cached.postprocess() to indicate that the cache should be ignored."""


def gen_cache_key(fmt: str, *args) -> bytes:
    """Compose a memcached-friendly cache key from a printf-like."""
    full_key = (fmt % args).encode()
    first_half = xxh64_hexdigest(full_key[:len(full_key) // 2])
    second_half = xxh64_hexdigest(full_key[len(full_key) // 2:])
    return (first_half + second_half).encode()


def cached(exptime: Union[int, Callable[..., int]],
           serialize: Callable[[Any], ByteString],
           deserialize: Callable[[ByteString], Any],
           key: Callable[..., Tuple],
           cache: Optional[Callable[..., Optional[aiomcache.Client]]] = None,
           refresh_on_access=False,
           postprocess: Optional[Callable[..., Any]] = None,
           version: int = 1,
           ) -> Callable[[Callable[..., Coroutine]], Callable[..., Coroutine]]:
    """
    Return factory that creates decorators that cache function call results if possible.

    :param exptime: Cache item expiration time delta in seconds. Can be a callable the decorated \
                    function's arguments converted to **kwargs and joined with the function's \
                    call result as "result".
    :param serialize: Call result serializer.
    :param deserialize: Cached binary deserializer to the result type.
    :param key: Cache key selector. The decorated function's arguments are converted to **kwargs.
    :param cache: Cache client extractor. The decorated function's arguments are converted to \
                  **kwargs. If is None, the client is assigned to the function's "cache" argument.
    :param refresh_on_access: Reset the cache item's expiration period on each access.
    :param postprocess: Execute an arbitrary function on the deserialized "result" with \
                        the arguments passed to the wrapped function.
    :param version: Version of the cache payload format.
    :return: Decorator that cache function call results if possible.
    """
    def wrapper_cached(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        """Decorate a function to return the cached result if possible."""
        log = logging.getLogger("%s.cache" % metadata.__package__)
        if exptime == max_exptime and not refresh_on_access:
            log.warning("%s will stay cached for max_exptime but will not refresh on access, "
                        "consider setting refresh_on_access=True", func.__name__)
        if cache is None:
            def discover_cache(**kwargs) -> Optional[aiomcache.Client]:
                try:
                    return kwargs["cache"]
                except KeyError:
                    raise AssertionError(
                        '"cache" is not one of %s arguments, you must explicitly define it: '
                        '@cached(cache=...)' % func.__qualname__)  # noqa: Q000
        elif callable(cache):
            discover_cache = cache
        else:
            def discover_cache(**kwargs):
                return cache
        signature = inspect.signature(func)
        full_name = func.__module__ + "." + func.__qualname__

        def _gen_cache_key(args_dict: dict) -> bytes:
            props = key(**args_dict)
            assert isinstance(props, tuple), "key() must return a tuple in %s" % full_name
            return gen_cache_key(
                full_name + "|" + str(version) + "|" + "|".join(str(p) for p in props))

        # no functool.wraps() shit here! It discards the coroutine status and aiohttp notices that
        async def wrapped_cached(*args, **kwargs) -> Any:
            start_time = time.time()
            args_dict = signature.bind(*args, **kwargs).arguments
            client = discover_cache(**args_dict)
            cache_key = None
            if client is not None:
                cache_key = _gen_cache_key(args_dict)
                try:
                    buffer = await client.get(cache_key)
                except aiomcache.exceptions.ClientException:
                    log.exception("Failed to fetch cached %s/%s", full_name, cache_key.decode())
                    buffer = None
                if buffer is not None:
                    try:
                        result = deserialize(buffer)
                    except Exception as e:
                        log.error("Failed to deserialize cached %s/%s: %s: %s",
                                  full_name, cache_key.decode(), type(e).__name__, e)
                    else:
                        t = exptime(result=result, **args_dict) if callable(exptime) else exptime
                        if refresh_on_access:
                            await client.touch(cache_key, t)
                        ignore = False
                        if postprocess is not None:
                            try:
                                result = postprocess(result=result, **args_dict)
                            except CancelCache:
                                log.info("%s/%s was ignored", full_name, cache_key.decode())
                                client.metrics["ignored"].labels(
                                    __package__, __version__, full_name).inc()
                                client.metrics["context"]["ignores"].get()[full_name] += 1
                                ignore = True
                        if not ignore:
                            client.metrics["hits"] \
                                .labels(__package__, __version__, full_name) \
                                .inc()
                            client.metrics["hit_latency"] \
                                .labels(__package__, __version__, full_name) \
                                .observe(time.time() - start_time)
                            client.metrics["context"]["hits"].get()[full_name] += 1
                            return result
            result = await func(*args, **kwargs)
            if client is not None:
                t = exptime(result=result, **args_dict) if callable(exptime) else exptime
                try:
                    payload = serialize(result)
                except Exception as e:
                    log.error("Failed to serialize %s/%s: %s: %s",
                              full_name, cache_key.decode(), type(e).__name__, e)
                else:
                    try:
                        await client.set(cache_key, payload, exptime=t)
                    except aiomcache.exceptions.ClientException:
                        log.exception("Failed to put %d bytes in memcached for %s/%s",
                                      len(payload), full_name, cache_key.decode())
                    else:
                        client.metrics["misses"].labels(__package__, __version__, full_name).inc()
                        client.metrics["miss_latency"] \
                            .labels(__package__, __version__, full_name) \
                            .observe(time.time() - start_time)
                        client.metrics["size"] \
                            .labels(__package__, __version__, full_name) \
                            .observe(len(payload))
                        client.metrics["context"]["misses"].get()[full_name] += 1
            return result

        async def reset_cache(*args, **kwargs) -> bool:
            args_dict = signature.bind(*args, **kwargs).arguments
            client = discover_cache(**args_dict)
            if client is None:
                return False
            props = key(**args_dict)
            assert isinstance(props, tuple), "key() must return a tuple in %s" % full_name
            cache_key = gen_cache_key(
                full_name + "|" + str(version) + "|" + "|".join(str(p) for p in props))
            try:
                return await client.delete(cache_key)
            except aiomcache.exceptions.ClientException:
                log.exception("Failed to delete %s/%s in memcached", full_name, cache_key.decode())

        def cache_key(*args, **kwargs) -> bytes:
            args_dict = signature.bind(*args, **kwargs).arguments
            return _gen_cache_key(args_dict)

        wrapped_cached.reset_cache = reset_cache
        wrapped_cached.cache_key = cache_key
        return wraps(wrapped_cached, func)

    return wrapper_cached


def setup_cache_metrics(cache: Optional[aiomcache.Client],
                        app: web.Application,
                        registry: CollectorRegistry) -> None:
    """Initialize the Prometheus metrics for tracking the cache interoperability."""
    if cache is None:
        app["cache_context"] = {}
        return
    app.setdefault("cache_context", {}).update({
        "misses": ContextVar("cache_misses", default=None),
        "ignores": ContextVar("cache_ignores", default=None),
        "hits": ContextVar("cache_hits", default=None),
    })
    cache.metrics = {
        "hits": Counter(
            "cache_hits", "Number of times the cache was useful",
            ["app_name", "version", "func"],
            registry=registry,
        ),
        "ignored": Counter(
            "cache_ignored", "Number of times the cache was ignored",
            ["app_name", "version", "func"],
            registry=registry,
        ),
        "misses": Counter(
            "cache_misses", "Number of times the cache was useless",
            ["app_name", "version", "func"],
            registry=registry,
        ),
        "hit_latency": Histogram(
            "cache_hit_latency", "Elapsed time to retrieve items from the cache",
            ["app_name", "version", "func"],
            registry=registry,
        ),
        "miss_latency": Histogram(
            "cache_miss_latency", "Elapsed time to retrieve items bypassing the cache",
            ["app_name", "version", "func"],
            buckets=[0.05, 0.1, 0.25, 0.5, 0.75, 1.0,
                     1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                     12.0, 15.0, 20.0, 25.0, 30.0,
                     45.0, 60.0, 120.0, 180.0, 240.0, INF],
            registry=registry,
        ),
        "size": Histogram(
            "cache_size", "Cached object size",
            ["app_name", "version", "func"],
            buckets=[10, 100, 1000, 5000, 10000, 25000, 50000, 75000,
                     100000, 200000, 300000, 400000, 500000, 750000,
                     1000000, 2000000, 3000000, 4000000, 5000000, 7500000,
                     10000000, INF],
            registry=registry,
        ),
        "context": app["cache_context"],
    }
