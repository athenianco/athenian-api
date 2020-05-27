import asyncio
import functools

import sentry_sdk

from athenian.api import typing_utils


def sentry_span(func):
    """Wrap the function in a Sentry span to trace the elapsed time."""
    if asyncio.iscoroutinefunction(func):
        async def wrapped_async_sentry_span(*args, **kwargs):
            with sentry_sdk.start_span(op=func.__qualname__):
                return await func(*args, **kwargs)

        # forward the @cached service sub-routines
        reset_cache = getattr(func, "reset_cache", None)
        if reset_cache is not None:
            wrapped_async_sentry_span.reset_cache = reset_cache
            wrapped_async_sentry_span.cache_key = func.cache_key

        return typing_utils.wraps(wrapped_async_sentry_span, func)

    @functools.wraps(func)
    def wrapped_sync_sentry_span(*args, **kwargs):
        with sentry_sdk.start_span(op=func.__qualname__):
            return func(*args, **kwargs)

    return wrapped_sync_sentry_span
