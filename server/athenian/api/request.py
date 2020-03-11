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


def with_conn_pool(db_attr):
    """Provide a scoped DB connection pool to the decorated function."""
    def with_conn_pool_decorator(func):
        # no functool.wraps() shit here! It discards the coroutine status and aiohttp notices that
        async def wrapped_with_conn_pool(
                request: AthenianWebRequest, *args, **kwargs) -> web.Response:
            pool = []

            async def acquire_conn():
                conn = await getattr(request, db_attr).connection().__aenter__()
                pool.append(conn)
                return conn

            pool_kwargs = {"acquire_%s_conn" % db_attr: acquire_conn}

            try:
                return await func(request, *args, **kwargs, **pool_kwargs)
            finally:
                for conn in pool:
                    await conn.__aexit__(None, None, None)

        wrapped_with_conn_pool.__name__ = func.__name__
        wrapped_with_conn_pool.__qualname__ = func.__qualname__
        wrapped_with_conn_pool.__module__ = func.__module__
        wrapped_with_conn_pool.__doc__ = func.__doc__
        annotations = func.__annotations__.copy()
        try:
            del annotations["%s_acquire_conn" % db_attr]
        except KeyError:
            pass
        wrapped_with_conn_pool.__annotations__ = annotations
        wrapped_with_conn_pool.__wrapped__ = func
        return wrapped_with_conn_pool

    return with_conn_pool_decorator


acquire_conn_type = Callable[[], Coroutine[None, None, databases.core.Connection]]
