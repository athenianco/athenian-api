import asyncio
from itertools import chain
import logging
from sqlite3 import IntegrityError, OperationalError
from typing import Callable, Coroutine, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union

import aiomcache
import asyncpg
from asyncpg import UniqueViolationError
import databases
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
from sqlalchemy import and_, func, insert, select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.controllers.account import fetch_github_installation_progress, \
    get_metadata_account_ids, get_user_account_status, match_metadata_installation
from athenian.api.controllers.miners.access_classes import access_classes
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.db import DatabaseLike, FastConnection, ParallelDatabase
from athenian.api.defer import defer
from athenian.api.models.metadata.github import AccountRepository
from athenian.api.models.state.models import RepositorySet, UserAccount
from athenian.api.models.web import ForbiddenError, InvalidRequestError, NoSourceDataError, \
    NotFoundError
from athenian.api.models.web.generic_error import DatabaseConflict
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span


async def resolve_reposet(repo: str,
                          pointer: str,
                          uid: str,
                          account: int,
                          db: Union[FastConnection, ParallelDatabase],
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
                        sdb: DatabaseLike,
                        mdb: DatabaseLike,
                        cache: Optional[aiomcache.Client],
                        slack: Optional[SlackWebClient],
                        strip_prefix=True,
                        ) -> Tuple[Set[str], Tuple[int, ...]]:
    """
    Dereference all the reposets and produce the joint list of all mentioned repos.

    :return: (Union of all the mentioned repo names, metadata (GitHub) account IDs).
    """
    status = await sdb.fetch_one(
        select([UserAccount.is_admin]).where(and_(UserAccount.user_id == uid,
                                                  UserAccount.account_id == account)))
    if status is None:
        raise ResponseError(ForbiddenError(
            detail="User %s is forbidden to access account %d" % (uid, account)))
    if not repositories:
        rss = await load_account_reposets(
            account, login, [RepositorySet.id], sdb, mdb, cache, slack)
        repositories = ["{%d}" % rss[0][RepositorySet.id.key]]
    tasks = [get_metadata_account_ids(account, sdb, cache)] + [
        resolve_reposet(r, ".in[%d]" % i, uid, account, sdb, cache)
        for i, r in enumerate(repositories)]
    task_results = await gather(*tasks, op="resolve_reposet-s + meta_ids")
    repos, meta_ids = set(chain.from_iterable(task_results[1:])), task_results[0]
    checked_repos = {r.split("/", 1)[1] for r in repos}
    checker = await access_classes["github"](account, meta_ids, sdb, mdb, cache).load()
    if denied := await checker.check(checked_repos):
        raise ResponseError(ForbiddenError(
            detail="The following repositories are access denied for account %d: %s" %
                   (account, denied),
        ))
    if strip_prefix:
        repos = checked_repos
    return repos, meta_ids


@sentry_span
async def load_account_reposets(account: int,
                                login: Callable[[], Coroutine[None, None, str]],
                                fields: list,
                                sdb: DatabaseLike,
                                mdb: DatabaseLike,
                                cache: Optional[aiomcache.Client],
                                slack: Optional[SlackWebClient],
                                ) -> List[Mapping]:
    """
    Load the account's repository sets and create one if no exists.

    :param sdb: Connection to the state DB.
    :param mdb: Connection to the metadata DB, needed only if no reposet exists.
    :param cache: memcached Client.
    :param account: Owner of the loaded reposets.
    :param login: Coroutine to load the contextual user's login.
    :param fields: Which columns to fetch for each RepositorySet.
    :return: List of DB rows or __dict__-s representing the loaded RepositorySets.
    """
    async def nested(_sdb_conn):
        if isinstance(mdb, databases.Database):
            async with mdb.connection() as _mdb_conn:
                return await _load_account_reposets(
                    account, login, fields, _sdb_conn, _mdb_conn, cache, slack)
        return await _load_account_reposets(
            account, login, fields, _sdb_conn, mdb, cache, slack)

    if isinstance(sdb, databases.Database):
        async with sdb.connection() as _sdb_conn:
            return await nested(_sdb_conn)
    return await nested(sdb)


async def _load_account_reposets(account: int,
                                 login: Callable[[], Coroutine[None, None, str]],
                                 fields: list,
                                 sdb_conn: FastConnection,
                                 mdb_conn: FastConnection,
                                 cache: Optional[aiomcache.Client],
                                 slack: Optional[SlackWebClient],
                                 ) -> List[Mapping]:
    assert isinstance(sdb_conn, FastConnection)
    assert isinstance(mdb_conn, FastConnection)
    rss = await sdb_conn.fetch_all_safe(
        select(fields)
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

            async def load_prefixer():
                meta_ids = await get_metadata_account_ids(account, sdb_conn, cache)
                prefixer = await Prefixer.load(meta_ids, mdb_conn, cache)
                return prefixer, meta_ids

            prefixer_meta_ids, login = await asyncio.gather(
                load_prefixer(), login(), return_exceptions=True)
            if isinstance(login, Exception):
                raise ResponseError(ForbiddenError(detail=str(login)))
            if isinstance(prefixer_meta_ids, Exception):
                if not isinstance(prefixer_meta_ids, ResponseError):
                    raise prefixer_meta_ids from None
                if not (meta_ids := await match_metadata_installation(
                        account, login, sdb_conn, mdb_conn, slack)):
                    raise_no_source_data()
                prefixer = await Prefixer.load(meta_ids, mdb_conn, cache)
            else:
                prefixer, meta_ids = prefixer_meta_ids
            progress = await fetch_github_installation_progress(
                account, sdb_conn, mdb_conn, cache)
            if progress.finished_date is None:
                raise_no_source_data()
            ar = AccountRepository
            updated_col = (ar.updated_at == func.max(ar.updated_at).over(
                partition_by=ar.repo_node_id,
            )).label("latest")
            window_query = (
                select([ar.repo_node_id, ar.enabled, updated_col])
                .where(ar.acc_id.in_(meta_ids))
            ).alias("w")
            if isinstance(sdb_conn.raw_connection, asyncpg.Connection):
                and_func = func.bool_and
            else:
                and_func = func.max
            query = (
                select([window_query.c.repo_node_id])
                .select_from(window_query)
                .where(window_query.c.latest)
                .group_by(window_query.c.repo_node_id)
                .having(and_func(window_query.c.enabled))
            )
            repo_node_ids = await mdb_conn.fetch_all(query)
            repos = []
            missing = []
            for r in repo_node_ids:
                try:
                    repos.append(prefixer.repo_node_map[r[0]])
                except KeyError:
                    missing.append(r[0])
            if missing:
                log.error("account_repos_log does not agree with api_repositories: %s", missing)
            if not repos:
                raise_no_source_data()
            rs = RepositorySet(
                name=RepositorySet.ALL, owner_id=account, items=repos,
            ).create_defaults()
            rs.id = await sdb_conn.execute(insert(RepositorySet).values(rs.explode()))
            log.info("Created the first reposet %d for account %d with %d repos",
                     rs.id, account, len(repos))
            if slack is not None:
                prefixes = {r.split("/", 2)[1] for r in repos}
                await defer(slack.post("installation_goes_on.jinja2",
                                       account=account,
                                       repos=len(repos),
                                       prefixes=prefixes,
                                       all_reposet_name=RepositorySet.ALL,
                                       reposet=rs.id,
                                       ),
                            "report_installation_goes_on")
            return [rs.explode(with_primary_keys=True)]
    except (UniqueViolationError, IntegrityError, OperationalError) as e:
        log.error("%s: %s", type(e).__name__, e)
        raise ResponseError(DatabaseConflict(
            detail="concurrent or duplicate initial reposet creation")) from None
