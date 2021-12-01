from typing import Callable, Coroutine, Optional

from aiohttp import web
import aiomcache

from athenian.api.db import Database
from athenian.api.models.web.user import User


class AthenianWebRequest(web.Request):
    """
    Type hint for any API HTTP request.

    Attributes:
        * mdb              Global metadata DB instance.
        * sdb              Global state DB instance.
        * pdb              Global precomputed objects DB instance.
        * rdb              Global persistentdata DB instance.
        * cache            Global memcached client. Can be None.
        * uid              Requesting user Auth0 ID, e.g. "github|60340680".
        * account          Requesting user's account ID, exists only with APIKey auth.
        * user             Coroutine to load the full user profile.
        * is_default_user  Value indicating whether the user is a default user, for example, \
                           @gkwillie. Requests on behalf of the default user are considered \
                           public and do not require any authentication.
    """

    mdb: Database
    sdb: Database
    pdb: Database
    rdb: Database
    cache: Optional[aiomcache.Client]  # can be None
    user: Callable[[], Coroutine[None, None, User]]
    uid: str
    account: Optional[int]
    is_default_user: bool
