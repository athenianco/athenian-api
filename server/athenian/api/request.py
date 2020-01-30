from aiohttp import web
import databases

from athenian.api import Auth0
from athenian.api.models.web.user import User


class AthenianWebRequest(web.Request):
    """Type hint for any API HTTP request."""

    mdb: databases.Database
    sdb: databases.Database
    auth: Auth0
    user: lambda: User
    uid: lambda: str
