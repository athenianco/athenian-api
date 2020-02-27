import functools
import inspect
import pickle
from typing import Any, ByteString, Callable, Coroutine, Optional, Tuple, Union

import aiomcache
from xxhash import xxh64_hexdigest


pickle.dumps = functools.partial(pickle.dumps, protocol=-1)


def _gen_cache_key(fmt: str, *args) -> bytes:
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
    :return: Decorator that cache function call results if possible.
    """
    def wrapper_cached(func: Callable[..., Coroutine]) -> Callable[..., Coroutine]:
        """Decorate a function to return the cached result if possible."""
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

        # no functool.wraps() shit here! It discards the coroutine status and aiohttp notices that
        async def wrapped_cached(*args, **kwargs):
            args_dict = inspect.signature(func).bind(*args, **kwargs).arguments
            client = discover_cache(**args_dict)
            cache_key = None
            if client is not None:
                props = key(**args_dict)
                assert isinstance(props, tuple), "key() must return a tuple"
                cache_key = _gen_cache_key(
                    "%s.%s|%s",
                    func.__module__,
                    func.__qualname__,
                    "|".join([str(p) for p in props]),
                )
                buffer = await client.get(cache_key)
                if buffer is not None:
                    return deserialize(buffer)
            result = await func(*args, **kwargs)
            if client is not None:
                t = exptime(result=result, **args_dict) if callable(exptime) else exptime
                await client.set(cache_key, serialize(result), exptime=t)
            return result

        wrapped_cached.__name__ = func.__name__
        wrapped_cached.__qualname__ = func.__qualname__
        wrapped_cached.__module__ = func.__module__
        wrapped_cached.__doc__ = func.__doc__
        wrapped_cached.__annotations__ = func.__annotations__
        wrapped_cached.__wrapped__ = func
        return wrapped_cached

    return wrapper_cached
