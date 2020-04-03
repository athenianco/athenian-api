import struct
from typing import Optional

import aiomcache
from sqlalchemy import and_, select

from athenian.api.cache import cached, max_exptime
from athenian.api.models.state.models import Account, UserAccount
from athenian.api.models.web import NoSourceDataError, NotFoundError
from athenian.api.response import ResponseError
from athenian.api.typing_utils import DatabaseLike


@cached(
    # the TTL is huge because this relation will never change and is requested frequently
    exptime=max_exptime,
    serialize=lambda iid: struct.pack("!q", iid),
    deserialize=lambda buf: struct.unpack("!q", buf)[0],
    key=lambda account, **_: (account,),
    refresh_on_access=True,
)
async def get_installation_id(account: int,
                              sdb_conn: DatabaseLike,
                              cache: Optional[aiomcache.Client],
                              ) -> int:
    """Fetch the Athenian metadata installation ID for the given account ID."""
    iid = await sdb_conn.fetch_val(select([Account.installation_id]).where(Account.id == account))
    if iid is None:
        raise ResponseError(NoSourceDataError(
            detail="The account installation has not finished yet."))
    return iid


@cached(
    exptime=60,
    serialize=lambda is_admin: b"1" if is_admin else b"0",
    deserialize=lambda buf: buf == b"1",
    key=lambda user, account, **_: (user, account),
)
async def get_user_account_status(user: str,
                                  account: int,
                                  sdb_conn: DatabaseLike,
                                  cache: Optional[aiomcache.Client],
                                  ) -> bool:
    """Return the value indicating whether the given user is an admin of the given account."""
    status = await sdb_conn.fetch_val(
        select([UserAccount.is_admin])
        .where(and_(UserAccount.user_id == user, UserAccount.account_id == account)))
    if status is None:
        raise ResponseError(NotFoundError(
            detail="Account %d does not exist or user %s is not a member." % (account, user)))
    return status
