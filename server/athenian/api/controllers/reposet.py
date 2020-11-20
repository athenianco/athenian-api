import asyncio
from itertools import chain
import logging
from sqlite3 import IntegrityError, OperationalError
from typing import Callable, Coroutine, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union

import aiomcache
from asyncpg import UniqueViolationError
import databases.core
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
from sqlalchemy import and_, insert, select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.controllers.account import get_metadata_account_ids, get_user_account_status
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import Account, AccountRepository, OrganizationMember, \
    User
from athenian.api.models.state.models import AccountGitHubAccount, RepositorySet, UserAccount
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
    if rs.owner_id != account:
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
            if col is RepositorySet.owner_id:
                break
        else:
            columns = list(columns)
            columns.append(RepositorySet.owner_id)
    rs = await sdb.fetch_one(select(columns).where(RepositorySet.id == id))
    if rs is None or len(rs) == 0:
        raise ResponseError(NotFoundError(detail="Repository set %d does not exist" % id))
    account = rs[RepositorySet.owner_id.key]
    adm = await get_user_account_status(uid, account, sdb, cache)
    return RepositorySet(**rs), adm


@sentry_span
async def resolve_repos(repositories: List[str],
                        account: int,
                        uid: str,
                        login: Callable[[], Coroutine[None, None, str]],
                        sdb_conn: Union[databases.core.Connection, databases.Database],
                        mdb_conn: Union[databases.core.Connection, databases.Database],
                        cache: Optional[aiomcache.Client],
                        slack: Optional[SlackWebClient],
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
            account, login, [RepositorySet.id], sdb_conn, mdb_conn, cache, slack)
        repositories = ["{%d}" % rss[0][RepositorySet.id.key]]
    repos = set(chain.from_iterable(
        await gather(*[
            resolve_reposet(r, ".in[%d]" % i, uid, account, sdb_conn, cache)
            for i, r in enumerate(repositories)], op="resolve_reposet-s")))
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
                                login: Callable[[], Coroutine[None, None, str]],
                                fields: list,
                                sdb_conn: DatabaseLike,
                                mdb_conn: DatabaseLike,
                                cache: Optional[aiomcache.Client],
                                slack: Optional[SlackWebClient],
                                ) -> List[Mapping]:
    """
    Load the account's repository sets and create one if no exists.

    :param sdb_conn: Connection to the state DB.
    :param mdb_conn: Connection to the metadata DB, needed only if no reposet exists.
    :param cache: memcached Client.
    :param account: Owner of the loaded reposets.
    :param login: Coroutine to load the contextual user's login.
    :param fields: Which columns to fetch for each RepositorySet.
    :return: List of DB rows or __dict__-s representing the loaded RepositorySets.
    """
    async def nested(_sdb_conn):
        if isinstance(mdb_conn, databases.Database):
            async with mdb_conn.connection() as _mdb_conn:
                return await _load_account_reposets(
                    account, login, fields, _sdb_conn, _mdb_conn, cache, slack)
        return await _load_account_reposets(
            account, login, fields, _sdb_conn, mdb_conn, cache, slack)

    if isinstance(sdb_conn, databases.Database):
        async with sdb_conn.connection() as _sdb_conn:
            return await nested(_sdb_conn)
    return await nested(sdb_conn)


async def _load_account_reposets(account: int,
                                 login: Callable[[], Coroutine[None, None, str]],
                                 fields: list,
                                 sdb_conn: databases.core.Connection,
                                 mdb_conn: databases.core.Connection,
                                 cache: Optional[aiomcache.Client],
                                 slack: Optional[SlackWebClient],
                                 ) -> List[Mapping]:
    assert isinstance(sdb_conn, databases.core.Connection)
    assert isinstance(mdb_conn, databases.core.Connection)
    rss = await sdb_conn.fetch_all(select(fields)
                                   .where(RepositorySet.owner_id == account)
                                   .order_by(RepositorySet.created_at))
    if rss:
        return rss

    log = logging.getLogger("%s.load_account_reposets" % metadata.__package__)

    def raise_no_source_data():
        raise ResponseError(NoSourceDataError(
            detail="The primary metadata application has not been installed yet."))

    try:
        async with sdb_conn.transaction():
            # new account, discover their repos from the installation and create the first reposet
            tasks = [
                get_metadata_account_ids(account, sdb_conn, cache),
                login(),
            ]
            metadata_ids, login = await asyncio.gather(*tasks, return_exceptions=True)
            if isinstance(login, Exception):
                raise ResponseError(ForbiddenError(detail=str(login)))
            if isinstance(metadata_ids, Exception):
                if login is None:
                    # single tenant mode, fetch all installations
                    # if this happens on a regular cloud instance, everybody is deeply fucked
                    cond = True
                else:
                    members = await _load_organization_member_logins(login, mdb_conn)
                    cond = Account.owner_login.in_(members)
                metadata_ids = {r[0] for r in await mdb_conn.fetch_all(select([Account.id])
                                                                       .where(cond))}
                if not metadata_ids:
                    raise_no_source_data()
                owned_accounts = {r[0] for r in await sdb_conn.fetch_all(
                    select([AccountGitHubAccount.id])
                    .where(AccountGitHubAccount.id.in_(metadata_ids)))}
                metadata_ids -= owned_accounts
                if not metadata_ids:
                    raise_no_source_data()
                for acc_id in metadata_ids:
                    # we don't expect many installations for the same account so don't go parallel
                    values = AccountGitHubAccount(id=acc_id, account_id=account).explode(
                        with_primary_keys=True)
                    await sdb_conn.execute(insert(AccountGitHubAccount).values(values))
            repos = await mdb_conn.fetch_all(
                select([AccountRepository.repo_full_name])
                .where(and_(AccountRepository.acc_id.in_(metadata_ids),
                            AccountRepository.enabled))
                .order_by(AccountRepository.repo_full_name))
            prefix = PREFIXES["github"]
            repos = [(prefix + r[0]) for r in repos]
            rs = RepositorySet(
                name=RepositorySet.ALL, owner_id=account, items=repos,
            ).create_defaults()
            rs.id = await sdb_conn.execute(insert(RepositorySet).values(rs.explode()))
            log.info(
                "Created the first reposet %d for account %d with %d repos on behalf of %s",
                rs.id, account, len(repos), login,
            )
            if slack is not None:
                metadata_accounts = [(r[0], r[1]) for r in await mdb_conn.fetch_all(
                    select([Account.id, Account.owner_login]).where(Account.id.in_(metadata_ids)))]
                prefixes = {r.split("/", 2)[1] for r in repos}
                await slack.post("new_installation.jinja2",
                                 account=account,
                                 repos=len(repos),
                                 prefixes=prefixes,
                                 all_reposet_name=RepositorySet.ALL,
                                 reposet=rs.id,
                                 metadata_accounts=metadata_accounts,
                                 login=login)

            return [rs.explode(with_primary_keys=True)]
    except (UniqueViolationError, IntegrityError, OperationalError) as e:
        log.error("%s: %s", type(e).__name__, e)
        raise ResponseError(DatabaseConflict(
            detail="concurrent or duplicate initial reposet creation")) from None


async def _load_organization_member_logins(seed_login: str, mdb: DatabaseLike) -> List[str]:
    seed_node_id = await mdb.fetch_val(select([User.node_id]).where(User.login == seed_login))
    if seed_node_id is None:
        return [seed_login]
    orgs = [r[0] for r in await mdb.fetch_all(select([OrganizationMember.parent_id])
                                              .where(OrganizationMember.child_id == seed_node_id))]
    if not orgs:
        return [seed_login]
    members = [r[0] for r in await mdb.fetch_all(select([OrganizationMember.child_id])
                                                 .where(OrganizationMember.parent_id.in_(orgs)))]
    logins = [r[0] for r in await mdb.fetch_all(select([User.login])
                                                .where(User.node_id.in_(members)))]
    return logins
