from typing import Optional

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
