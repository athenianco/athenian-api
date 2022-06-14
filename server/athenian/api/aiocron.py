"""Based on https://github.com/gawel/aiocron."""
import asyncio
from datetime import datetime, timezone
import functools
import random
import time

from croniter.croniter import croniter
import sentry_sdk

from athenian.api.typing_utils import wraps


async def _null_callback(*args):
    return args


def wrap_func(func):
    """Wrap function in a coroutine."""
    if not asyncio.iscoroutinefunction(func):

        async def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wraps(wrapper, func)
    return func


class Cron:
    """Utility for dealing with cron tasks using asyncio."""

    def __init__(self, spec, func=None, args=(), start=False):
        """Initialize a Cron object."""
        self.spec = spec
        if func is not None:
            if not args:
                self.func = func
            else:
                partial_func = functools.partial(func, *args)
                functools.update_wrapper(partial_func, func)
                self.func = partial_func
        else:
            self.func = _null_callback
        self.cron = wrap_func(self.func)
        self.auto_start = start
        self.handle = self.future = self.croniter = None
        if start and self.func is not _null_callback:
            self.handle = asyncio.get_event_loop().call_soon_threadsafe(self.start)

    def start(self):
        """Start scheduling."""
        self.stop()
        self._initialize()
        self.handle = asyncio.get_event_loop().call_at(self._get_next(), self._call_next)

    def stop(self):
        """Stop scheduling."""
        if self.handle is not None:
            self.handle.cancel()
        self.handle = self.future = self.croniter = None

    async def next(self, *args):
        """Yield from next."""
        self._initialize()
        self.future = asyncio.Future()
        self.handle = asyncio.get_event_loop().call_at(self._get_next(), self._call_func, *args)
        return await self.future

    def _initialize(self):
        """Initialize croniter and related times."""
        if self.croniter is None:
            self.time = time.time()
            self.datetime = datetime.now(timezone.utc)
            self.loop_time = asyncio.get_event_loop().time()
            self.croniter = croniter(self.spec, start_time=self.datetime)

    def _get_next(self):
        """Return next iteration time related to loop time."""
        return self.loop_time + (self.croniter._get_next(float) - self.time)

    def _call_next(self):
        """Set next hop in the loop. Call task."""
        if self.handle is not None:
            self.handle.cancel()
        next_time = self._get_next()
        self.handle = asyncio.get_event_loop().call_at(next_time, self._call_next)
        self._call_func()

    def _call_func(self, *args, **kwargs):
        """Call and take care of exceptions using gather."""
        asyncio.gather(self.cron(*args, **kwargs), return_exceptions=True).add_done_callback(
            self._set_result
        )

    def _set_result(self, result):
        """Set future's result if needed (can be an exception), else raise if needed."""
        result = result.result()[0]
        if self.future is not None:
            if isinstance(result, Exception):
                sentry_sdk.capture_exception(result)
                self.future.set_exception(result)
            else:
                self.future.set_result(result)
            self.future = None
        elif isinstance(result, Exception):
            sentry_sdk.capture_exception(result)
            raise result from None

    def __call__(self, func):
        """Use as a decorator."""
        self.func = func
        self.cron = wrap_func(func)
        if self.auto_start:
            asyncio.get_event_loop().call_soon_threadsafe(self.start)
        return self

    def __str__(self):
        """Return a friendly string representation of object."""
        return f"{self.spec} {self.func}"

    def __repr__(self):
        """Return an unambiguous string representation of object."""
        return f"<Cron {self.spec} {self.func}>"


class EarlyExpirationCron(Cron):
    """Like Cron, but with probabilistic early expiration."""

    def __init__(self, spec, func=None, args=(), start=False, max_early_expiration_seconds=0):
        """Initialize an EarlyExpirationCron object.

        A value of 0 for `max_early_expiration_seconds` makes it behave as `Cron`.
        """
        super(EarlyExpirationCron, self).__init__(spec, func=func, args=args, start=start)
        self.max_early_expiration_seconds = max_early_expiration_seconds

    def get_next(self):
        """Return next iteration time related to loop time."""
        early_expiration = random.random() * self.max_early_expiration_seconds
        return super(EarlyExpirationCron, self).get_next() - early_expiration


def crontab(spec, func=None, args=(), start=True, max_early_expiration_seconds=0):
    """Entrypoint for using Cron utility."""
    return EarlyExpirationCron(
        spec,
        func=func,
        args=args,
        start=start,
        max_early_expiration_seconds=max_early_expiration_seconds,
    )
