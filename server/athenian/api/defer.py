from asyncio import Event, current_task, ensure_future, gather, shield, sleep
from contextvars import ContextVar
import logging
from typing import Awaitable, Coroutine

from aiohttp import web
import asyncpg
import morcilla
from sentry_sdk import Hub, push_scope, start_transaction
from sentry_sdk.tracing import Transaction

from athenian.api import metadata
from athenian.api.typing_utils import wraps

_defer_launch_event = ContextVar("defer_launch_event")
_defer_sync = ContextVar("defer_sync")
_defer_counter = ContextVar("defer_counter")
_defer_explicit = ContextVar("defer_explicit")
_global_defer_counter = 0
_global_defer_sync = Event()
_defer_transaction = ContextVar("defer_transaction")
_log = logging.getLogger("%s.defer" % metadata.__package__)


def set_defer_loop() -> None:
    """Bind asyncio.Event to the current global event loop."""
    global _global_defer_sync
    _global_defer_sync = Event()


def enable_defer(explicit_launch: bool) -> None:
    """Allow deferred couroutines in the current context."""
    _defer_launch_event.set(Event())
    _defer_sync.set(Event())
    _defer_transaction.set([None])
    _defer_counter.set([0])
    _defer_explicit.set(explicit_launch)
    if not explicit_launch:
        launch_defer(0, "enable_defer")


def launch_defer(delay: float, name: str, detached: bool = False) -> None:
    """Allow the deferred coroutines to execute after a certain delay (in seconds)."""
    log = logging.getLogger(f"{metadata.__package__}.launch_defer")
    try:
        launch_event = _defer_launch_event.get()  # type: Event
        if launch_event.is_set():
            log.warning("launch_defer() called multiple times")
            return
        transaction_ptr: list[Transaction] = _defer_transaction.get()
        deferred_count_ptr = _defer_counter.get()
        explicit_launch = _defer_explicit.get()
    except LookupError:
        log.exception("%s\nPossible reason: called wait_deferred() in an endpoint.", name)
        return
    if detached:
        explicit_launch = False
        _defer_explicit.set(False)

    if (parent := Hub.current.scope.transaction) is None:

        def transaction():
            return start_transaction(name="defer", sampled=False)

    else:

        def transaction():
            return start_transaction(
                name="defer " + parent.name,
                sampled=parent.sampled,
                op="defer" + ((" " + parent.op) if parent.op else ""),
                description="%d tasks" % deferred_count_ptr[0],
            )

    def launch():
        _log.debug("launching %d deferred tasks %r", deferred_count_ptr[0], launch_event)
        if deferred_count_ptr[0] > 0 or not explicit_launch:
            transaction_ptr[0] = transaction()
        launch_event.set()

    if delay == 0:
        launch()
    else:

        async def delayer():
            global _global_defer_counter
            _global_defer_counter += 1
            try:
                current_task().set_name("launch_defer.delayer " + name)
                await sleep(delay)
                launch()
            finally:
                _global_defer_counter -= 1
                if _global_defer_counter == 0:
                    _global_defer_sync.set()

        ensure_future(delayer())


def launch_defer_from_request(
    request: web.Request,
    delay: float = 0,
    detached: bool = False,
) -> None:
    """
    Allow the deferred coroutines to execute after a certain delay (in seconds).

    We set the name to the method and the path of the HTTP request.
    """
    return launch_defer(delay, "%s %s" % (request.method, request.path), detached=detached)


async def defer(coroutine: Awaitable, name: str) -> None:
    """Schedule coroutine in parallel with the main control flow and return immediately."""
    if isinstance(coroutine, Coroutine):
        for key, val in coroutine.cr_frame.f_locals.items():
            try:
                # because we will use this connection in the deferred task but it's already closed
                assert not isinstance(
                    val, morcilla.core.Connection,
                ), f"{key} must not be visible to {coroutine.__qualname__}"
            except AssertionError as e:
                coroutine.close()
                raise e from None
    sync: Event = _defer_sync.get()
    sync.clear()
    counter_ptr: list[int] = _defer_counter.get()
    counter_ptr[0] += 1
    _global_defer_sync.clear()
    global _global_defer_counter
    _global_defer_counter += 1
    launch_event: Event = _defer_launch_event.get()
    transaction_ptr = _defer_transaction.get()
    explicit_launch = _defer_explicit.get()
    _log.debug("planned %s %d %r", name, counter_ptr[0], launch_event)

    async def wrapped_defer():
        current_task().set_name("defer[wait] " + name)
        await launch_event.wait()
        _log.debug("%d exec %s", counter_ptr[0], name)
        current_task().set_name("defer " + name)
        transaction = transaction_ptr[0]  # type: Transaction
        try:
            if transaction is not None:
                with transaction.start_child(op=name):
                    with push_scope() as scope:
                        scope.fingerprint = ["{{ default }}", "defer"]
                        await coroutine
            else:
                _log.error("empty Sentry transaction in a deferred task")
                await coroutine
        except (asyncpg.DeadlockDetectedError, asyncpg.InterfaceError) as e:
            # 1. Our DB transaction lost, but that's fine for a deferred task.
            # 2. Failed to rollback a transaction.
            _log.warning("defer %s: %s: %s", name, type(e).__name__, e)
            return
        except Exception:
            _log.exception("Unhandled exception in deferred function %s", name)
        finally:
            global _global_defer_counter
            _global_defer_counter -= 1
            if _global_defer_counter == 0:
                _global_defer_sync.set()
            value = counter_ptr[0] - 1
            counter_ptr[0] = value
            _log.debug("_defer_counter %d", value)
            if value == 0:
                sync.set()
                if explicit_launch:
                    if transaction is not None:
                        transaction.finish()
                    launch_event.clear()

    ensure_future(shield(wrapped_defer()))


async def wait_deferred(final: bool = False) -> None:
    """Wait for the deferred coroutines in the current context to finish."""
    launch_event = _defer_launch_event.get()  # type: Event
    if not launch_event.is_set():
        _log.warning("called wait_deferred() before launch_defer(), can deadlock")
    sync = _defer_sync.get()
    counter = _defer_counter.get()
    if (value := counter[0]) > 0:
        _log.info("Waiting for %d deferred tasks...", value)
        await sync.wait()
    assert counter[0] == 0
    if final and not _defer_explicit.get():
        _defer_transaction.get()[0].finish()
        launch_event.clear()


async def wait_all_deferred() -> None:
    """Wait for all the deferred coroutines to finish."""
    if _global_defer_counter > 0:
        _log.info("Waiting for %d deferred tasks...", _global_defer_counter)
        await _global_defer_sync.wait()
    assert _global_defer_counter == 0
    _global_defer_sync.clear()


def with_defer(func):
    """Decorate a coroutine to enable defer()."""

    async def wrapped_with_defer(*args, **kwargs):
        enable_defer(False)
        try:
            return await func(*args, **kwargs)
        finally:
            await wait_deferred()

    wraps(wrapped_with_defer, func)
    return wrapped_with_defer


def with_explicit_defer(func):
    """Decorate a coroutine to enable defer() - but the user must call launch_defer()."""

    async def wrapped_with_defer(*args, **kwargs):
        enable_defer(True)
        try:
            await func(*args, **kwargs)
        finally:
            launch_defer(0, "with_explicit_defer")
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
        await gather(*(e.wait() for e in self.events), return_exceptions=True)
