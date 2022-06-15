from asyncio import IncompleteReadError
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
import functools
import inspect
import logging
import pickle
import struct
import time
from typing import Any, Callable, Coroutine, Mapping, Optional, Tuple, Union
from wsgiref.handlers import format_date_time

from aiohttp import web
import aiomcache
import lz4.frame
from prometheus_client import Counter, Histogram
from prometheus_client.utils import INF
import sentry_sdk
from xxhash import xxh64_hexdigest

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.defer import defer
from athenian.api.metadata import __package__, __version__
from athenian.api.prometheus import PROMETHEUS_REGISTRY_VAR_NAME
from athenian.api.request import AthenianWebRequest
from athenian.api.typing_utils import serialize_mutable_fields_in_dataclasses, wraps

pickle.dumps = functools.partial(pickle.dumps, protocol=-1)
max_exptime = 30 * 24 * 3600  # 30 days according to the docs
short_term_exptime = 5 * 60  # 5 minutes
middle_term_exptime = 60 * 60  # 1 hour


class CancelCache(Exception):
    """Raised in cached.postprocess() to indicate that the cache should be ignored."""


def gen_cache_key(fmt: str, *args) -> bytes:
    """Compose a memcached-friendly cache key from a printf-like."""
    if args:
        full_key = (fmt % args).encode()
    else:
        full_key = fmt
    first_half = xxh64_hexdigest(full_key[: len(full_key) // 2])
    second_half = xxh64_hexdigest(full_key[len(full_key) // 2 :])
    return (first_half + second_half).encode()


def cached(
    exptime: Union[int, Callable[..., int]],
    serialize: Callable[[Any], bytes],
    deserialize: Callable[[bytes], Any],
    key: Callable[..., Optional[Tuple]],
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
    :param key: Cache key selector. The decorated function's arguments are converted to **kwargs. \
                If it returns None then the cache is not used.
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
            log.warning(
                "%s will stay cached for max_exptime but will not refresh on access, "
                "consider setting refresh_on_access=True",
                func.__name__,
            )
        if cache is None:

            def discover_cache(**kwargs) -> Optional[aiomcache.Client]:
                try:
                    return kwargs["cache"]
                except KeyError:
                    raise AssertionError(
                        '"cache" is not one of %s arguments, you must explicitly define it: '
                        "@cached(cache=...)"
                        % func.__qualname__,
                    )  # noqa: Q000

        elif callable(cache):
            discover_cache = cache
        else:

            def discover_cache(**kwargs):
                return cache

        signature = inspect.signature(func)
        full_name = func.__module__ + "." + func.__qualname__
        if full_name.startswith(__package__):
            short_name = full_name[len(__package__) + 1 :]
        else:
            short_name = full_name

        def _gen_cache_key(args_dict: dict) -> Optional[bytes]:
            props = key(**args_dict)
            if props is None:
                return None
            assert isinstance(props, tuple), "key() must return a tuple in %s" % full_name
            props = "|".join(str(p).replace("|", "/") for p in props)
            return gen_cache_key(full_name + "|" + str(version) + "|" + props)

        # no functool.wraps() shit here! It discards the coroutine status and aiohttp notices that
        async def wrapped_cached(*args, **kwargs) -> Any:
            start_time = time.time()
            __tracebackhide__ = True  # noqa: F841
            args_dict = signature.bind(*args, **kwargs).arguments
            client = discover_cache(**args_dict)
            cache_key = None
            if client is not None:
                if (cache_key := _gen_cache_key(args_dict)) is not None:
                    try:
                        with sentry_sdk.start_span(op="get " + cache_key.hex()):
                            buffer = await client.get(cache_key)
                    except (aiomcache.exceptions.ClientException, IncompleteReadError) as e:
                        log.exception(
                            "Failed to fetch cached %s/%s: %s: %s",
                            full_name,
                            cache_key.decode(),
                            type(e).__name__,
                            e,
                        )
                        buffer = None
                else:
                    buffer = None
                if buffer is not None:
                    try:
                        with sentry_sdk.start_span(op="deserialize", description=str(len(buffer))):
                            result = deserialize(lz4.frame.decompress(buffer))
                    except Exception as e:
                        log.error(
                            "Failed to deserialize cached %s/%s: %s: %s",
                            full_name,
                            cache_key.decode(),
                            type(e).__name__,
                            e,
                        )
                    else:
                        t = exptime(result=result, **args_dict) if callable(exptime) else exptime
                        if refresh_on_access:
                            await client.touch(cache_key, t)
                        ignore = False
                        if postprocess is not None:
                            try:
                                with sentry_sdk.start_span(op="postprocess"):
                                    result = postprocess(result=result, **args_dict)
                            except CancelCache:
                                log.info("%s/%s was ignored", full_name, cache_key.decode())
                                client.metrics["ignored"].labels(
                                    __package__, __version__, short_name,
                                ).inc()
                                client.metrics["context"]["ignores"].get()[short_name] += 1
                                ignore = True
                            except Exception:
                                log.exception("failed to postprocess cached %s", full_name)
                                ignore = True
                            else:
                                log.debug("%s/postprocess passed OK", full_name)
                        if not ignore:
                            log.debug("%s cache hit", full_name)
                            client.metrics["hits"].labels(
                                __package__, __version__, short_name,
                            ).inc()
                            client.metrics["hit_latency"].labels(
                                __package__, __version__, short_name,
                            ).observe(time.time() - start_time)
                            client.metrics["context"]["hits"].get()[short_name] += 1
                            return result
            log.debug("%s cache miss", full_name)
            result = await func(*args, **kwargs)
            if cache_key is not None:
                t = exptime(result=result, **args_dict) if callable(exptime) else exptime
                try:
                    with sentry_sdk.start_span(op="serialize") as span:
                        with serialize_mutable_fields_in_dataclasses():
                            payload = serialize(result)
                        uncompressed_payload_size = len(payload)
                        span.description = str(uncompressed_payload_size)
                except Exception as e:
                    log.error(
                        "Failed to serialize %s/%s: %s: %s",
                        full_name,
                        cache_key.decode(),
                        type(e).__name__,
                        e,
                    )
                else:

                    async def set_cache_item():
                        nonlocal payload
                        with sentry_sdk.start_span(op="compress") as span:
                            payload = lz4.frame.compress(
                                payload,
                                block_size=lz4.frame.BLOCKSIZE_MAX1MB,
                                compression_level=9,
                            )
                            span.description = "%d -> %d" % (
                                uncompressed_payload_size,
                                len(payload),
                            )
                        try:
                            await client.set(cache_key, payload, exptime=t)
                        except aiomcache.exceptions.ClientException:
                            log.exception(
                                "Failed to put %d bytes in memcached for %s/%s",
                                len(payload),
                                full_name,
                                cache_key.decode(),
                            )

                    await defer(
                        set_cache_item(),
                        "set_cache_items(%s, %d)" % (func.__qualname__, len(payload)),
                    )
                    client.metrics["misses"].labels(__package__, __version__, short_name).inc()
                    client.metrics["miss_latency"].labels(
                        __package__, __version__, short_name,
                    ).observe(time.time() - start_time)
                    client.metrics["size"].labels(__package__, __version__, short_name).observe(
                        uncompressed_payload_size,
                    )
                    client.metrics["context"]["misses"].get()[short_name] += 1
            return result

        async def reset_cache(*args, **kwargs) -> bool:
            args_dict = signature.bind(*args, **kwargs).arguments
            client = discover_cache(**args_dict)
            if client is None:
                return False
            props = key(**args_dict)
            assert isinstance(props, tuple), "key() must return a tuple in %s" % full_name
            cache_key = gen_cache_key(
                full_name + "|" + str(version) + "|" + "|".join(str(p) for p in props),
            )
            try:
                return await client.delete(cache_key)
            except aiomcache.exceptions.ClientException:
                log.exception("Failed to delete %s/%s in memcached", full_name, cache_key.decode())

        def cache_key(*args, **kwargs) -> bytes:
            args_dict = signature.bind(*args, **kwargs).arguments
            return _gen_cache_key(args_dict)

        wrapped_cached.__cached__ = True
        wrapped_cached.reset_cache = reset_cache
        wrapped_cached.cache_key = cache_key
        return wraps(wrapped_cached, func)

    return wrapper_cached


def cached_methods(cls):
    """Decorate class to properly support cached instance methods."""
    for name, func in inspect.getmembers(cls, predicate=inspect.isfunction):
        outer_func = func

        while not getattr(func, "__cached__", False) or getattr(func, "__cached_method__", False):
            # This is needed in case the `@cached` decorator is not the outermost one
            if not (func := getattr(func, "__wrapped__", None)):
                break

        if not func:
            continue

        def generate_cls_method_shim(f):
            def cls_method_shim(self):
                async def cached_wrapper(*args, **kwargs):
                    __tracebackhide__ = True  # noqa: F841
                    return await f(self, *args, **kwargs)

                async def reset_cache(*args, **kwargs):
                    return await f.reset_cache(self, *args, **kwargs)

                @functools.wraps(f.cache_key)
                def cache_key(*args, **kwargs):
                    return f.cache_key(self, *args, **kwargs)

                cached_wrapper.__cached__ = True
                cached_wrapper.__cached_method__ = True  # do not screw inheritance
                cached_wrapper.reset_cache = wraps(reset_cache, f.reset_cache)
                cached_wrapper.cache_key = cache_key
                return wraps(cached_wrapper, f)

            return cls_method_shim

        setattr(cls, name, property(generate_cls_method_shim(outer_func)))
    return cls


CACHE_VAR_NAME = "cache"


def setup_cache_metrics(app: Union[web.Application, Mapping]) -> None:
    """Initialize the Prometheus metrics for tracking the cache interoperability."""
    cache = app[CACHE_VAR_NAME]
    registry = app[PROMETHEUS_REGISTRY_VAR_NAME]
    if cache is None:
        app["cache_context"] = {}
        return
    app.setdefault("cache_context", {}).update(
        {
            "misses": ContextVar("cache_misses", default=None),
            "ignores": ContextVar("cache_ignores", default=None),
            "hits": ContextVar("cache_hits", default=None),
        },
    )
    cache.metrics = {
        "hits": Counter(
            "cache_hits",
            "Number of times the cache was useful",
            ["app_name", "version", "func"],
            registry=registry,
        ),
        "ignored": Counter(
            "cache_ignored",
            "Number of times the cache was ignored",
            ["app_name", "version", "func"],
            registry=registry,
        ),
        "misses": Counter(
            "cache_misses",
            "Number of times the cache was useless",
            ["app_name", "version", "func"],
            registry=registry,
        ),
        "hit_latency": Histogram(
            "cache_hit_latency",
            "Elapsed time to retrieve items from the cache",
            ["app_name", "version", "func"],
            registry=registry,
        ),
        "miss_latency": Histogram(
            "cache_miss_latency",
            "Elapsed time to retrieve items bypassing the cache",
            ["app_name", "version", "func"],
            buckets=[
                0.05,
                0.1,
                0.25,
                0.5,
                0.75,
                1.0,
                1.5,
                2.0,
                2.5,
                3.0,
                4.0,
                5.0,
                6.0,
                7.0,
                8.0,
                9.0,
                10.0,
                12.0,
                15.0,
                20.0,
                25.0,
                30.0,
                45.0,
                60.0,
                120.0,
                180.0,
                240.0,
                INF,
            ],
            registry=registry,
        ),
        "size": Histogram(
            "cache_size",
            "Cached object size",
            ["app_name", "version", "func"],
            buckets=[
                10,
                100,
                1000,
                5000,
                10000,
                25000,
                50000,
                75000,
                100000,
                200000,
                300000,
                400000,
                500000,
                750000,
                1000000,
                2000000,
                3000000,
                4000000,
                5000000,
                7500000,
                10000000,
                INF,
            ],
            registry=registry,
        ),
        "context": app["cache_context"],
    }


@cached(
    exptime=lambda duration, **_: duration,
    serialize=lambda dt: struct.pack("!Q", int(dt.timestamp())),
    deserialize=lambda s: datetime.fromtimestamp(struct.unpack("!Q", s)[0], timezone.utc),
    cache=lambda request, **_: request.cache,
    key=lambda request, duration, **kwargs: (request.method, request.path, kwargs),
)
async def _fetch_endpoint_expiration(
    request: AthenianWebRequest,
    duration: int,
    **kwargs,
) -> datetime:
    return datetime.now(timezone.utc) + timedelta(seconds=duration)


def expires_header(duration: int):
    """Append "Expires" header to the response according to `duration` in seconds."""

    def cached_header_decorator(fn):
        async def wrapped_cached_header(request: AthenianWebRequest, **kwargs) -> web.Response:
            response, expires = await gather(
                fn(request, **kwargs),
                _fetch_endpoint_expiration(request, duration, **kwargs),
            )
            response.headers.add("Expires", format_date_time(expires.timestamp()))
            return response

        return wraps(wrapped_cached_header, fn)

    return cached_header_decorator
