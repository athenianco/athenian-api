from asyncio import Condition, ensure_future, Event, shield, wait
from contextvars import ContextVar
import logging
from typing import Coroutine

from sentry_sdk import Hub
from sentry_sdk.tracing import Transaction

from athenian.api import metadata
from athenian.api.typing_utils import wraps


_defer_sync = None
_defer_counter = None
_log = logging.getLogger("%s.defer" % metadata.__package__)


class GlobalVar:
    """Mock the interface of ContextVar."""

    def __init__(self, value):
        """Assign an object at initialization time."""
        self.value = value

    def get(self):
        """Return the encapsulated object."""
        return self.value


def setup_defer(global_scope: bool) -> None:
    """
    Initialize the deferred subsystem.

    This function is a no-op if called multiple times.

    :param global_scope: Indicates whether the deferred tasks must register in the global context \
                         or in the current coroutine context.
    """
    global _defer_sync, _defer_counter
    if _defer_sync is not None:
        assert global_scope == isinstance(_defer_sync, GlobalVar)
        return
    if global_scope:
        _defer_sync = GlobalVar(Condition())
        _defer_counter = GlobalVar([0])
    else:
        _defer_sync = ContextVar("defer_sync")
        _defer_counter = ContextVar("defer_counter")


def enable_defer() -> None:
    """Allow deferred couroutines in the current context."""
    if isinstance(_defer_sync, GlobalVar):
        return
    _defer_sync.set(Condition())
    _defer_counter.set([0])


async def defer(coroutine: Coroutine, name: str) -> None:
    """Schedule coroutine in parallel with the main control flow and return immediately."""
    sync = _defer_sync.get()
    counter = _defer_counter.get()
    async with sync:
        counter[0] += 1
    hub = Hub.current
    span = hub.scope.span
    if span is None:
        span = Transaction(sampled=False)

    async def wrapped_defer():
        try:
            with Hub(hub):
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


class AllEvents:
    """Wait for multiple asyncio Event-s to happen."""

    def __init__(self, *events: Event):
        """Register several events to wait."""
        self.events = events

    async def wait(self) -> None:
        """Block until all the events happen."""
        await wait([e.wait() for e in self.events])
