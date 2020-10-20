from typing import Callable, Coroutine, Optional

from aiohttp import web
import aiomcache
import databases

from athenian.api.models.web.user import User


class AthenianWebRequest(web.Request):
    """
    Type hint for any API HTTP request.

    Attributes:
        * mdb              Global metadata DB instance.
        * sdb              Global state DB instance.
        * pdb              Global precomputed objects DB instance.
        * cache            Global memcached client. Can be None.
        * uid              Requesting user Auth0 ID, e.g. "github|60340680".
        * user             Coroutine to load the full user profile.
        * is_default_user  Value indicating whether the user is a default user, for example, \
                           @gkwillie. Requests on behalf of the default user are considered \
                           public and do not require any authentication.
    """

    mdb: databases.Database
    sdb: databases.Database
    pdb: databases.Database
    cache: Optional[aiomcache.Client]  # can be None
    user: Callable[[], Coroutine[None, None, User]]
    uid: str
    is_default_user: bool
