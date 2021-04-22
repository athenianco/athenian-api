import asyncio
from datetime import timezone
from typing import Any, Iterable, List

import pytz
import sentry_sdk


# Workaround https://github.com/pandas-dev/pandas/issues/32619
pytz.UTC = pytz.utc = timezone.utc

# Allow other coroutines to execute every Nth iteration in long loops
COROUTINE_YIELD_EVERY_ITER = 250


async def list_with_yield(iterable: Iterable[Any], sentry_op: str) -> List[Any]:
    """Drain an iterable to a list, tracing the loop in Sentry and respecting other coroutines."""
    with sentry_sdk.start_span(op=sentry_op) as span:
        things = []
        for i, thing in enumerate(iterable):
            if (i + 1) % COROUTINE_YIELD_EVERY_ITER == 0:
                await asyncio.sleep(0)
            things.append(thing)
        try:
            span.description = str(i)
        except UnboundLocalError:
            pass
    return things
