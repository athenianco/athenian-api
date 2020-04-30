from typing import Optional

from aiohttp import web
import aiomcache
import databases

from athenian.api import Auth0
from athenian.api.models.web.user import User


class AthenianWebRequest(web.Request):
    """
    Type hint for any API HTTP request.

    Attributes:
        * mdb           Global metadata DB instance.
        * sdb           Global state DB instance.
        * pdb           Global precomputed objects DB instance.
        * cache         Global memcached client. Can be None.
        * auth          Global authorization abstraction class instance. It is used in very few \
                        places where we need to fetch user details from Auth0.
        * uid           Requesting user Auth0 ID, e.g. "github|60340680".
        * native_uid    Requesting user "native" ID, e.g. "60340680". We only support GutHub atm, \
                        so it matches the GitHub user ID (do not confuse with the login @name).
    """

    mdb: databases.Database
    sdb: databases.Database
    pdb: databases.Database
    cache: Optional[aiomcache.Client]  # can be None, yes
    auth: Auth0
    user: lambda: User
    uid: str
    native_uid: str
