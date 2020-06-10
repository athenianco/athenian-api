import asyncio
from itertools import chain
import logging
from sqlite3 import IntegrityError, OperationalError
from typing import List, Mapping, Optional, Sequence, Set, Tuple, Type, Union

import aiomcache
from asyncpg import UniqueViolationError
import databases.core
import slack
from sqlalchemy import and_, insert, select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.controllers.account import get_installation_ids, get_user_account_status
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import InstallationOwner, InstallationRepo
from athenian.api.models.state.models import Installation, RepositorySet, UserAccount
from athenian.api.models.web import ForbiddenError, InvalidRequestError, NoSourceDataError, \
    NotFoundError
from athenian.api.models.web.generic_error import DatabaseConflict
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import DatabaseLike


async def resolve_reposet(repo: str,
                          pointer: str,
                          uid: str,
                          account: int,
                          db: Union[databases.core.Connection, databases.Database],
                          cache: Optional[aiomcache.Client],
                          ) -> List[str]:
    """
    Dereference the repository sets.

    If `repo` is a regular repository, return `[repo]`. Otherwise, return the list of \
    repositories by the parsed ID from the database.
    """
    if not repo.startswith("{"):
        return [repo]
    if not repo.endswith("}"):
        raise ResponseError(InvalidRequestError(
            detail="repository set format is invalid: %s" % repo,
            pointer=pointer,
        ))
    try:
        set_id = int(repo[1:-1])
    except ValueError:
        raise ResponseError(InvalidRequestError(
            detail="repository set identifier is invalid: %s" % repo,
            pointer=pointer,
        ))
    rs, _ = await fetch_reposet(set_id, [RepositorySet.items], uid, db, cache)
    if rs.owner != account:
        raise ResponseError(ForbiddenError(
            detail="User %s is not allowed to reference reposet %d in this query" %
                   (uid, set_id)))
    return rs.items


@sentry_span
async def fetch_reposet(
    id: int,
    columns: Union[Sequence[Type[RepositorySet]], Sequence[InstrumentedAttribute]],
    uid: str,
    sdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> Tuple[RepositorySet, bool]:
    """
    Retrieve a repository set by ID and check the access for the given user.

    :return: Loaded RepositorySet and `is_admin` flag that indicates whether the user has \
             RW access to that set.
    """
    if not columns or columns[0] is not RepositorySet:
        for col in columns:
            if col is RepositorySet.owner:
                break
        else:
            columns = list(columns)
            columns.append(RepositorySet.owner)
    rs = await sdb.fetch_one(select(columns).where(RepositorySet.id == id))
    if rs is None or len(rs) == 0:
        raise ResponseError(NotFoundError(detail="Repository set %d does not exist" % id))
    account = rs[RepositorySet.owner.key]
    adm = await get_user_account_status(uid, account, sdb, cache)
    return RepositorySet(**rs), adm


@sentry_span
async def resolve_repos(repositories: List[str],
                        account: int,
                        uid: str,
                        native_uid: str,
                        sdb_conn: Union[databases.core.Connection, databases.Database],
                        mdb_conn: Union[databases.core.Connection, databases.Database],
                        cache: Optional[aiomcache.Client],
                        slack: Optional[slack.WebClient],
                        strip_prefix=True,
                        ) -> Set[str]:
    """Dereference all the reposets and produce the joint list of all mentioned repos."""
    status = await sdb_conn.fetch_one(
        select([UserAccount.is_admin]).where(and_(UserAccount.user_id == uid,
                                                  UserAccount.account_id == account)))
    if status is None:
        raise ResponseError(ForbiddenError(
            detail="User %s is forbidden to access account %d" % (uid, account)))
    if not repositories:
        rss = await load_account_reposets(
            account, native_uid, [RepositorySet.id], sdb_conn, mdb_conn, cache, slack)
        repositories = ["{%d}" % rss[0][RepositorySet.id.key]]
    repos = set(chain.from_iterable(
        await asyncio.gather(*[
            resolve_reposet(r, ".in[%d]" % i, uid, account, sdb_conn, cache)
            for i, r in enumerate(repositories)])))
    prefix = PREFIXES["github"]
    checked_repos = {r[r.startswith(prefix) and len(prefix):] for r in repos}
    checker = await access_classes["github"](account, sdb_conn, mdb_conn, cache).load()
    denied = await checker.check(checked_repos)
    if denied:
        raise ResponseError(ForbiddenError(
            detail="the following repositories are access denied for %s: %s" % (prefix, denied),
        ))
    if strip_prefix:
        repos = checked_repos
    return repos


@sentry_span
async def load_account_reposets(account: int,
                                native_uid: str,
                                fields: list,
                                sdb_conn: DatabaseLike,
                                mdb_conn: DatabaseLike,
                                cache: Optional[aiomcache.Client],
                                slack: Optional[slack.WebClient],
                                ) -> List[Mapping]:
    """
    Load the account's repository sets and create one if no exists.

    :param sdb_conn: Connection to the state DB.
    :param mdb_conn: Connection to the metadata DB, needed only if no reposet exists.
    :param cache: memcached Client.
    :param account: Owner of the loaded reposets.
    :param native_uid: Native user ID, needed only if no reposet exists.
    :param fields: Which columns to fetch for each RepositorySet.
    :return: List of DB rows or __dict__-s representing the loaded RepositorySets.
    """
    async def nested(_sdb_conn):
        if isinstance(mdb_conn, databases.Database):
            async with mdb_conn.connection() as _mdb_conn:
                return await _load_account_reposets(
                    account, native_uid, fields, _sdb_conn, _mdb_conn, cache, slack)
        return await _load_account_reposets(
            account, native_uid, fields, _sdb_conn, mdb_conn, cache, slack)

    if isinstance(sdb_conn, databases.Database):
        async with sdb_conn.connection() as _sdb_conn:
            return await nested(_sdb_conn)
    return await nested(sdb_conn)


async def _load_account_reposets(account: int,
                                 native_uid: str,
                                 fields: list,
                                 sdb_conn: databases.core.Connection,
                                 mdb_conn: databases.core.Connection,
                                 cache: Optional[aiomcache.Client],
                                 slack: Optional[slack.WebClient],
                                 ) -> List[Mapping]:
    assert isinstance(sdb_conn, databases.core.Connection)
    assert isinstance(mdb_conn, databases.core.Connection)
    rss = await sdb_conn.fetch_all(select(fields)
                                   .where(RepositorySet.owner == account)
                                   .order_by(RepositorySet.created_at))
    if rss:
        return rss

    log = logging.getLogger("%s.load_account_reposets" % metadata.__package__)

    def raise_no_source_data():
        raise ResponseError(NoSourceDataError(
            detail="The metadata installation has not registered yet."))

    try:
        async with sdb_conn.transaction():
            # new account, discover their repos from the installation and create the first reposet
            try:
                iids = await get_installation_ids(account, sdb_conn, cache)
            except ResponseError:
                iids = await mdb_conn.fetch_all(
                    select([InstallationOwner.install_id])
                    .where(InstallationOwner.user_id == int(native_uid)))
                iids = {r[0] for r in iids}
                if not iids:
                    raise_no_source_data()
                owned_iids = await sdb_conn.fetch_all(select([Installation.id])
                                                      .where(Installation.id.in_(iids)))
                owned_iids = {r[0] for r in owned_iids}
                iids -= owned_iids
                if not iids:
                    raise_no_source_data()
                for iid in iids:
                    # we don't expect many installations for the same account so don't go parallel
                    values = Installation(id=iid, account_id=account).explode(
                        with_primary_keys=True)
                    await sdb_conn.execute(insert(Installation).values(values))
            repos = await mdb_conn.fetch_all(select([InstallationRepo.repo_full_name])
                                             .where(InstallationRepo.install_id.in_(iids)))
            prefix = PREFIXES["github"]
            repos = [(prefix + r[0]) for r in repos]
            rs = RepositorySet(owner=account, items=repos).create_defaults()
            rs.id = await sdb_conn.execute(insert(RepositorySet).values(rs.explode()))
            log.info(
                "Created the first reposet %d for account %d with %d repos on behalf of %s",
                rs.id, account, len(repos), native_uid,
            )
            if slack is not None:
                await slack.post(
                    "new_installation.jinja2", account=account, repos=repos, iids=iids)

            return [rs.explode(with_primary_keys=True)]
    except (UniqueViolationError, IntegrityError, OperationalError) as e:
        log.error("%s: %s", type(e).__name__, e)
        raise ResponseError(DatabaseConflict(
            detail="concurrent or duplicate initial reposet creation")) from None
