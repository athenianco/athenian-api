from asyncio import Condition, ensure_future, shield
from contextvars import ContextVar
import logging
from typing import Coroutine

from sentry_sdk import Hub
from sentry_sdk.tracing import Transaction

from athenian.api import metadata
from athenian.api.typing_utils import wraps

_defer_sync = ContextVar("defer_sync")
_defer_counter = ContextVar("defer_counter")
_log = logging.getLogger("%s.defer" % metadata.__package__)


def enable_defer() -> None:
    """Allow deferred couroutines in the current context."""
    _defer_sync.set(Condition())
    _defer_counter.set([0])


async def defer(coroutine: Coroutine, name: str) -> None:
    """Schedule coroutine in parallel with the main control flow and return immediately."""
    sync = _defer_sync.get()
    counter = _defer_counter.get()
    async with sync:
        counter[0] += 1
    span = Hub.current.scope.span
    if span is None:
        span = Transaction(sampled=False)

    async def wrapped_defer():
        try:
            with span.start_child(op="defer %s" % name):
                await coroutine
        except BaseException:
            _log.exception("Unhandled exception in deferred function %s", name)
        finally:
            async with sync:
                value = counter[0] - 1
                counter[0] = value
                if value == 0:
                    sync.notify_all()

    ensure_future(shield(wrapped_defer()))


async def wait_deferred() -> None:
    """Wait for all the deferred coroutines to finish."""
    sync = _defer_sync.get()
    counter = _defer_counter.get()
    async with sync:
        value = counter[0]
        if value > 0:
            _log.info("Waiting for %d deferred tasks...", value)
            await sync.wait()


def with_defer(func):
    """Decorate a coroutine to enable defer()."""
    async def wrapped_with_defer(*args, **kwargs):
        enable_defer()
        try:
            await func(*args, **kwargs)
        finally:
            await wait_deferred()

    wraps(wrapped_with_defer, func)
    return wrapped_with_defer
