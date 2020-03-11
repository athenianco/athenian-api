import asyncio
import inspect
from typing import Callable, Coroutine, Optional

from aiohttp import web
import aiomcache
import databases

from athenian.api import Auth0
from athenian.api.models.web.user import User


class AthenianWebRequest(web.Request):
    """Type hint for any API HTTP request."""

    mdb: databases.Database
    sdb: databases.Database
    cache: Optional[aiomcache.Client]  # can be None, yes
    auth: Auth0
    user: lambda: User
    uid: str
    native_uid: str


class GatheredError(Exception):
    """Several exceptions joined together."""

    def __init__(self, message: str, *errors: Exception):
        """
        Initialize a new instance of GatheredError.

        :param message: Summary of the errors.
        :param errors: Upstream exceptions.
        """
        self.message = message
        self.args = errors


def with_conn_pool(db_getter: Callable[..., databases.Database], name="acquire_conn"):
    """
    Provide a scoped DB connection pool to the decorated function.

    :param db_getter: Function that returns the database instance from **kwargs equivalent to \
                      the passed arguments of the decorated function.
    :param name: Name of the augmenting DB connection getter argument.
    """
    def with_conn_pool_decorator(func):
        signature = inspect.signature(func)

        # no functool.wraps() shit here! It discards the coroutine status and aiohttp notices that
        async def wrapped_with_conn_pool(*args, **kwargs) -> web.Response:
            db = db_getter(**signature.bind(*args, **kwargs, **{name: None}).arguments)
            pool = []

            async def acquire_conn():
                conn = await db.connection().__aenter__()
                pool.append(conn)
                return conn

            pool_kwargs = {name: acquire_conn}

            try:
                return await func(*args, **kwargs, **pool_kwargs)
            finally:
                errors = await asyncio.gather(
                    *[conn.__aexit__(None, None, None) for conn in pool],
                    return_exceptions=True)
                if any(errors):
                    raise GatheredError("Hit errors while releasing %d connections." % len(pool),
                                        *[e for e in errors if e is not None])

        wrapped_with_conn_pool.__name__ = func.__name__
        wrapped_with_conn_pool.__qualname__ = func.__qualname__
        wrapped_with_conn_pool.__module__ = func.__module__
        wrapped_with_conn_pool.__doc__ = func.__doc__
        annotations = func.__annotations__.copy()
        try:
            del annotations[name]
        except KeyError:
            pass
        wrapped_with_conn_pool.__annotations__ = annotations
        wrapped_with_conn_pool.__wrapped__ = func
        return wrapped_with_conn_pool

    return with_conn_pool_decorator


acquire_conn_type = Callable[[], Coroutine[None, None, databases.core.Connection]]
