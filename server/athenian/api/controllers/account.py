import struct
from typing import Optional, Tuple

import aiomcache
from sqlalchemy import and_, select

from athenian.api.cache import cached, max_exptime
from athenian.api.models.state.models import Account, AccountGitHubInstallation, UserAccount
from athenian.api.models.web import NoSourceDataError, NotFoundError
from athenian.api.response import ResponseError
from athenian.api.typing_utils import DatabaseLike


@cached(
    # the TTL is huge because this relation will never change and is requested frequently
    exptime=max_exptime,
    serialize=lambda iids: struct.pack("!" + "q" * len(iids), *iids),
    deserialize=lambda buf: struct.unpack("!" + "q" * (len(buf) // 8), buf),
    key=lambda account, **_: (account,),
    refresh_on_access=True,
)
async def get_github_installation_ids(account: int,
                                      sdb_conn: DatabaseLike,
                                      cache: Optional[aiomcache.Client],
                                      ) -> Tuple[int, ...]:
    """Fetch the Athenian metadata installation ID for the given account ID."""
    iids = await sdb_conn.fetch_all(select([AccountGitHubInstallation.id])
                                    .where(AccountGitHubInstallation.account_id == account))
    if len(iids) == 0:
        acc_exists = await sdb_conn.fetch_val(select([Account.id]).where(Account.id == account))
        if not acc_exists:
            raise ResponseError(NotFoundError(detail="Account %d does not exist" % account))
        raise ResponseError(NoSourceDataError(
            detail="The installation of account %d has not finished yet." % account))
    return tuple(r[0] for r in iids)


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
