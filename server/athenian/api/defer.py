from asyncio import ensure_future, Event, shield, sleep, wait
from contextvars import ContextVar
import logging
from typing import Coroutine, List, Optional

from sentry_sdk import Hub
from sentry_sdk.tracing import Transaction

from athenian.api import metadata
from athenian.api.typing_utils import wraps

_defer_launch_event = None
_defer_sync = None
_defer_counter = None
_global_defer_counter = 0
_global_defer_sync = None  # type: Optional[Event]
_defer_transaction = None
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
    global _defer_sync, _defer_counter, _defer_launch_event, _defer_transaction
    if _defer_sync is not None:
        # do not allow to reinitialize with a different scope
        assert global_scope == isinstance(_defer_sync, GlobalVar)
    if global_scope:
        _defer_launch_event = GlobalVar(Event())
        _defer_sync = GlobalVar(Event())
        _defer_counter = GlobalVar([0])
        _defer_transaction = GlobalVar([None])
    else:
        _defer_launch_event = ContextVar("defer_launch_event")
        _defer_sync = ContextVar("defer_sync")
        _defer_counter = ContextVar("defer_counter")
        _defer_transaction = ContextVar("defer_transaction")
    global _global_defer_counter, _global_defer_sync
    _global_defer_counter = 0
    _global_defer_sync = Event()


def enable_defer(explicit_launch: bool) -> None:
    """Allow deferred couroutines in the current context."""
    if isinstance(_defer_sync, GlobalVar):
        return
    _defer_launch_event.set(Event())
    _defer_sync.set(Event())
    _defer_counter.set([0])
    _defer_transaction.set([None])
    if not explicit_launch:
        launch_defer(0)


def launch_defer(delay: float) -> None:
    """Allow the deferred coroutines to execute after a certain delay (in seconds)."""
    transaction_ptr = _defer_transaction.get()  # type: List[Transaction]
    launch_event = _defer_launch_event.get()  # type: Event

    def launch():
        if Hub.current.scope.span is None:
            transaction = Transaction(name="defer", sampled=False, hub=Hub.current)
        else:
            if (parent := Hub.current.scope.transaction) is None:
                transaction = Transaction(name="defer", sampled=False, hub=Hub.current)
            else:
                transaction = Transaction(
                    name="defer " + parent.name,
                    sampled=parent.sampled,
                    op=parent.op,
                    description=parent.description,
                    hub=Hub.current,
                )
        transaction_ptr[0] = transaction
        launch_event.set()

    if delay == 0:
        launch()
    else:

        async def delayer():
            await sleep(delay)
            launch()

        ensure_future(delayer())


async def defer(coroutine: Coroutine, name: str) -> None:
    """Schedule coroutine in parallel with the main control flow and return immediately."""
    sync = _defer_sync.get()  # type: Event
    sync.clear()
    counter_ptr = _defer_counter.get()  # type: List[int]
    counter_ptr[0] += 1
    _global_defer_sync.clear()
    global _global_defer_counter
    _global_defer_counter += 1
    launch_event = _defer_launch_event.get()  # type: Event
    transaction_ptr = _defer_transaction.get()

    async def wrapped_defer():
        await launch_event.wait()
        transaction = transaction_ptr[0]  # type: Transaction
        try:
            with transaction.start_child(op=name):
                await coroutine
        except BaseException:
            _log.exception("Unhandled exception in deferred function %s", name)
        finally:
            global _global_defer_counter
            _global_defer_counter -= 1
            if _global_defer_counter == 0:
                _global_defer_sync.set()
            value = counter_ptr[0] - 1
            counter_ptr[0] = value
            if value == 0:
                sync.set()
                transaction.finish()

    ensure_future(shield(wrapped_defer()))


async def wait_deferred() -> None:
    """Wait for the deferred coroutines in the current context to finish."""
    sync = _defer_sync.get()
    counter = _defer_counter.get()
    value = counter[0]
    if value > 0:
        _log.info("Waiting for %d deferred tasks...", value)
        await sync.wait()


async def wait_all_deferred() -> None:
    """Wait for all the deferred coroutines to finish."""
    if _global_defer_counter > 0:
        _log.info("Waiting for %d deferred tasks...", _global_defer_counter)
        await _global_defer_sync.wait()


def with_defer(func):
    """Decorate a coroutine to enable defer()."""

    async def wrapped_with_defer(*args, **kwargs):
        enable_defer(False)
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
