import pickle
import struct
from typing import List, Optional, Tuple

import aiomcache
from sqlalchemy import and_, select

from athenian.api.cache import cached, max_exptime
from athenian.api.models.metadata.github import Organization
from athenian.api.models.state.models import Account, AccountGitHubAccount, RepositorySet, \
    UserAccount
from athenian.api.models.web import NoSourceDataError, NotFoundError
from athenian.api.response import ResponseError
from athenian.api.typing_utils import DatabaseLike


@cached(
    # the TTL is huge because this relation will never change and is requested frequently
    exptime=max_exptime,
    serialize=lambda ids: struct.pack("!" + "q" * len(ids), *ids),
    deserialize=lambda buf: struct.unpack("!" + "q" * (len(buf) // 8), buf),
    key=lambda account, **_: (account,),
    refresh_on_access=True,
)
async def get_metadata_account_ids(account: int,
                                   sdb: DatabaseLike,
                                   cache: Optional[aiomcache.Client],
                                   ) -> Tuple[int, ...]:
    """Fetch the metadata account IDs for the given API account ID."""
    ids = await sdb.fetch_all(select([AccountGitHubAccount.id])
                              .where(AccountGitHubAccount.account_id == account))
    if len(ids) == 0:
        acc_exists = await sdb.fetch_val(select([Account.id]).where(Account.id == account))
        if not acc_exists:
            raise ResponseError(NotFoundError(detail="Account %d does not exist" % account))
        raise ResponseError(NoSourceDataError(
            detail="The installation of account %d has not finished yet." % account))
    return tuple(r[0] for r in ids)


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


async def get_account_repositories(account: int,
                                   sdb: DatabaseLike,
                                   with_prefix=True) -> List[str]:
    """Fetch all the repositories belonging to the account."""
    repos = await sdb.fetch_one(select([RepositorySet.items]).where(and_(
        RepositorySet.owner_id == account,
        RepositorySet.name == RepositorySet.ALL,
    )))
    if repos is None:
        raise ResponseError(NoSourceDataError(
            detail="The installation of account %d has not finished yet." % account))
    repos = repos[0]
    if not with_prefix:
        repos = [r.split("/", 1)[1] for r in repos]
    return repos


@cached(
    exptime=24 * 60 * 60,  # 1 day
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda account, **_: (account,),
)
async def get_account_organizations(account: int,
                                    sdb: DatabaseLike,
                                    mdb: DatabaseLike,
                                    cache: Optional[aiomcache.Client],
                                    ) -> List[Organization]:
    """Fetch the list of GitHub organizations installed for the account."""
    ghids = await get_metadata_account_ids(account, sdb, cache)
    rows = await mdb.fetch_all(select([Organization]).where(Organization.acc_id.in_(ghids)))
    return [Organization(**r) for r in rows]
