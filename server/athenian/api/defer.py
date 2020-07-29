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


async def defer(coroutine: Coroutine) -> None:
    """Schedule coroutine in parallel with the main control flow and return immediately."""
    sync = _defer_sync.get()
    counter = _defer_counter.get()
    async with sync:
        counter[0] += 1
    code = coroutine.cr_frame.f_code
    func_name = code.co_name
    if func_name.startswith("wrapped_"):
        func = coroutine.cr_frame.f_locals["func"]
        while hasattr(func, "__wrapped__"):
            func = func.__wrapped__
        func_name = func.__name__
        code = func.__code__
    span = Hub.current.scope.span
    op = span.op if span is not None else "unknown"
    transaction = Hub.current.scope.transaction
    if transaction is None:
        transaction = Transaction(sampled=False)

    async def wrapped_defer():
        try:
            with transaction.start_child(op="defer", description=op):
                await coroutine
        except BaseException:
            _log.exception("Unhandled exception in deferred function %s:%d %s",
                           code.co_filename, code.co_firstlineno, func_name)
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
