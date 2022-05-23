import asyncio
from datetime import datetime, timezone
from http import HTTPStatus
import logging
from sqlite3 import IntegrityError, OperationalError
from typing import Any, Callable, Collection, Coroutine, Dict, List, Mapping, Optional, Sequence, \
    Set, Tuple, Type, Union

import aiomcache
import asyncpg
from asyncpg import UniqueViolationError
import sentry_sdk
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
from sqlalchemy import and_, desc, func, insert, select, update
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.db import Connection, Database, DatabaseLike
from athenian.api.defer import defer
from athenian.api.internal.account import fetch_github_installation_progress, \
    get_metadata_account_ids, match_metadata_installation
from athenian.api.internal.logical_repos import coerce_logical_repos, extract_logical_repo
from athenian.api.internal.miners.access import AccessChecker
from athenian.api.internal.miners.access_classes import access_classes
from athenian.api.internal.prefixer import Prefixer, strip_proto
from athenian.api.models.metadata.github import AccountRepository, NodeRepository, NodeUser
from athenian.api.models.state.models import RepositorySet, UserAccount
from athenian.api.models.web import ForbiddenError, InstallationProgress, InvalidRequestError, \
    NoSourceDataError, NotFoundError
from athenian.api.models.web.generic_error import BadRequestError, DatabaseConflict
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span


async def resolve_reposet(repo: str,
                          pointer: str,
                          uid: str,
                          account: int,
                          logical_settings,  # LogicalRepositorySettings
                          sdb: Union[Connection, Database],
                          cache: Optional[aiomcache.Client],
                          ) -> Collection[str]:
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
    rs = await fetch_reposet(set_id, [RepositorySet.items, RepositorySet.tracking_re], sdb)
    if rs.owner_id != account:
        raise ResponseError(ForbiddenError(
            detail="User %s is not allowed to reference reposet %d in this query" %
                   (uid, set_id)))
    return [r[0] for r in rs.items]


@sentry_span
async def fetch_reposet(
    id: int,
    columns: Union[Sequence[Type[RepositorySet]], Sequence[InstrumentedAttribute]],
    sdb: DatabaseLike,
) -> RepositorySet:
    """
    Retrieve a repository set by ID and check the access for the given user.

    :return: Loaded RepositorySet with `columns`.
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
        raise ResponseError(NotFoundError(detail=f"Repository set {id} does not exist"))
    return RepositorySet(**rs)


@sentry_span
async def load_all_reposet(account: int,
                           login: Callable[[], Coroutine[None, None, str]],
                           logical_settings,  # LogicalRepositorySettings
                           sdb: Database,
                           mdb: Database,
                           cache: Optional[aiomcache.Client],
                           slack: Optional[SlackWebClient],
                           ) -> List[Collection[str]]:
    """Fetch the contents (items) of the main reposet with all the repositories to consider."""
    rss = await load_account_reposets(
        account, login,
        [RepositorySet.id, RepositorySet.name, RepositorySet.items, RepositorySet.tracking_re],
        sdb, mdb, cache, slack)
    for rs in rss:
        if rs[RepositorySet.name.name] == RepositorySet.ALL:
            return [[r[0] for r in rs[RepositorySet.items.name]]]
    raise ResponseError(NoSourceDataError(detail=f'No "{RepositorySet.ALL}" reposet exists.'))


@sentry_span
async def resolve_repos(repositories: List[str],
                        account: int,
                        uid: str,
                        login: Callable[[], Coroutine[None, None, str]],
                        logical_settings,  # LogicalRepositorySettings
                        meta_ids: Optional[Tuple[int, ...]],
                        sdb: Database,
                        mdb: Database,
                        cache: Optional[aiomcache.Client],
                        slack: Optional[SlackWebClient],
                        strip_prefix=True,
                        separate=False,
                        checkers: Optional[Dict[str, AccessChecker]] = None,
                        pointer: Optional[str] = "?",
                        ) -> Tuple[Union[Set[str], List[Set[str]]], str]:
    """
    Dereference all the reposets and produce the joint list of all mentioned repos.

    We don't check the user's access! That should happen automatically in Auth0 code.

    If `repositories` is empty, we load the "ALL" reposet.

    :param separate: Value indicating whether to return each reposet separately.
    :return: (Union of all the mentioned repo names, service prefix).
    """
    if not repositories:
        # this may initialize meta_ids, so execute serialized
        reposets = await load_all_reposet(account, login, logical_settings, sdb, mdb, cache, slack)
        if meta_ids is None:
            meta_ids = await get_metadata_account_ids(account, sdb, cache)
    else:
        assert meta_ids is not None
        tasks = [resolve_reposet(r, f"{pointer}[{i}]", uid, account, logical_settings, sdb, cache)
                 for i, r in enumerate(repositories)]
        reposets = await gather(*tasks, op="resolve_reposet-s + meta_ids")
    repos = []
    checked_repos = set()
    wrong_format = set()
    prefix = None
    for reposet in reposets:
        resolved = set()
        for r in reposet:
            try:
                repo_prefix, repo = r.split("/", 1)
            except ValueError:
                wrong_format.add(r)
                continue
            if prefix is None:
                prefix = repo_prefix
            elif prefix != repo_prefix:
                raise ResponseError(InvalidRequestError(
                    detail=f'Mixed services are not allowed: "{prefix}" vs. "{r}"',
                    pointer=pointer))
            checked_repos.add(repo)
            if strip_prefix:
                resolved.add(repo)
            else:
                resolved.add(r)
        if separate:
            repos.append(resolved)
        else:
            repos.extend(resolved)
    if prefix is None:
        raise ResponseError(InvalidRequestError(
            detail="The service prefix may not be empty.",
            pointer=pointer))
    if wrong_format:
        raise ResponseError(BadRequestError(
            'The following repositories are malformed (missing "github.com/your_org/" prefix?): %s'
            % wrong_format))
    checkers = checkers or {}
    if (checker := checkers.get(prefix)) is None:
        checkers[prefix] = checker = await access_classes["github"](
            account, meta_ids, sdb, mdb, cache).load()
    if denied := await checker.check(coerce_logical_repos(checked_repos).keys()):
        log = logging.getLogger(f"{metadata.__package__}.resolve_repos")
        log.warning("access denied account %d%s: user sent %s we've got %s",
                    account, meta_ids, denied, list(checker.installed_repos))
        raise ResponseError(ForbiddenError(
            detail='The following repositories are access denied for account %d (missing "'
                   'github.com/" prefix?): %s' % (account, denied),
        ))
    if not separate:
        if strip_prefix:
            repos = checked_repos
        else:
            repos = set(repos)
    return repos, prefix + "/"


@sentry_span
async def load_account_reposets(account: int,
                                login: Callable[[], Coroutine[None, None, str]],
                                fields: list,
                                sdb: Database,
                                mdb: Database,
                                cache: Optional[aiomcache.Client],
                                slack: Optional[SlackWebClient],
                                check_progress: bool = True,
                                ) -> List[Mapping[str, Any]]:
    """
    Load the account's repository sets and create one if no exists.

    :param sdb: Connection to the state DB.
    :param mdb: Connection to the metadata DB, needed only if no reposet exists.
    :param cache: memcached Client.
    :param account: Owner of the loaded reposets.
    :param login: Coroutine to load the contextual user's login.
    :param fields: Which columns to fetch for each RepositorySet.
    :param check_progress: Check that we've already fetched all the repositories in mdb.
    :return: List of DB rows or __dict__-s representing the loaded RepositorySets.
    """
    assert isinstance(sdb, Database)
    assert isinstance(mdb, Database)
    async with sdb.connection() as sdb_conn:
        async with mdb.connection() as mdb_conn:
            return await _load_account_reposets(
                account, login, fields, check_progress, sdb_conn, mdb_conn, mdb, cache, slack)


async def _load_account_reposets(account: int,
                                 login: Callable[[], Coroutine[None, None, str]],
                                 fields: list,
                                 check_progress: bool,
                                 sdb_conn: Connection,
                                 mdb_conn: Connection,
                                 mdb: Database,
                                 cache: Optional[aiomcache.Client],
                                 slack: Optional[SlackWebClient],
                                 ) -> List[Mapping]:
    assert isinstance(sdb_conn, Connection)
    assert isinstance(mdb_conn, Connection)
    rss = await sdb_conn.fetch_all(
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
            if isinstance(prefixer_meta_ids, Exception):
                if not isinstance(prefixer_meta_ids, ResponseError):
                    raise prefixer_meta_ids from None
                if isinstance(login, Exception):
                    raise ResponseError(ForbiddenError(detail=str(login)))
                if not (meta_ids := await match_metadata_installation(
                        account, login, sdb_conn, mdb_conn, mdb, slack)):
                    raise_no_source_data()
                prefixer = await Prefixer.load(meta_ids, mdb_conn, cache)
            else:
                prefixer, meta_ids = prefixer_meta_ids

        async with sdb_conn.transaction():
            if check_progress:
                progress = await fetch_github_installation_progress(
                    account, sdb_conn, mdb_conn, cache)
                if progress.finished_date is None:
                    raise_no_source_data()
            ar = AccountRepository
            updated_col = (ar.updated_at == func.max(ar.updated_at).over(
                partition_by=ar.repo_graph_id,
            )).label("latest")
            window_query = (
                select([ar.repo_graph_id, ar.enabled, updated_col])
                .where(ar.acc_id.in_(meta_ids))
            ).alias("w")
            async with sdb_conn.raw_connection() as raw_connection:
                if isinstance(raw_connection, asyncpg.Connection):
                    and_func = func.bool_and
                else:
                    and_func = func.max
            query = (
                select([window_query.c.repo_graph_id])
                .select_from(window_query)
                .where(window_query.c.latest)
                .group_by(window_query.c.repo_graph_id)
                .having(and_func(window_query.c.enabled))
            )
            repo_node_ids = await mdb_conn.fetch_all(query)
            repos = []
            missing = []
            for r in repo_node_ids:
                try:
                    repos.append([prefixer.repo_node_to_prefixed_name[r[0]], r[0]])
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
            prefixes = {r[0].split("/", 2)[1] for r in repos}
            await defer(slack.post_install("installation_goes_on.jinja2",
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


async def load_account_state(account: int,
                             sdb: Database,
                             mdb: Database,
                             cache: Optional[aiomcache.Client],
                             slack: Optional[SlackWebClient],
                             log: Optional[logging.Logger] = None,
                             ) -> Optional[InstallationProgress]:
    """
    Decide on the account's installation progress.

    :return: If None, the account is not installed, otherwise the caller must analyze \
             `InstallationProgress` to figure out whether it's 100% or not.
    """
    if log is None:
        log = logging.getLogger(f"{metadata.__package__}.load_account_state")
    try:
        await get_metadata_account_ids(account, sdb, cache)
    except ResponseError:
        # Load the login
        auth0_id = await sdb.fetch_val(select([UserAccount.user_id]).where(and_(
            UserAccount.account_id == account,
        )).order_by(desc(UserAccount.is_admin)))
        if auth0_id is None:
            log.warning("There are no users in the account %d", account)
            return None
        try:
            db_id = int(auth0_id.split("|")[-1])
        except ValueError:
            log.error("Unable to match user %s with metadata installations, "
                      "you have to hack DB manually for account %d",
                      auth0_id, account)
            return None
        login = await mdb.fetch_val(select([NodeUser.login]).where(NodeUser.database_id == db_id))
        if login is None:
            log.warning("Could not find the user login of %s", auth0_id)
            return None
        async with sdb.connection() as sdb_conn:
            async with sdb_conn.transaction():
                if not await match_metadata_installation(
                        account, login, sdb_conn, mdb, mdb, slack):
                    log.warning("Did not match installations to account %d as %s",
                                account, auth0_id)
                    return None
    try:
        progress = await fetch_github_installation_progress(account, sdb, mdb, cache)
    except ResponseError as e1:
        if e1.response.status != HTTPStatus.UNPROCESSABLE_ENTITY:
            sentry_sdk.capture_exception(e1)
        log.warning("account %d: fetch_github_installation_progress ResponseError: %s",
                    account, e1.response)
        return None
    except Exception as e2:
        sentry_sdk.capture_exception(e2)
        log.warning("account %d: fetch_github_installation_progress %s: %s",
                    account, type(e2).__name__, e2)
        return None
    if progress.finished_date is None:
        return progress

    async def load_login() -> str:
        raise AssertionError("This should never be called at this point")

    try:
        reposets = await load_account_reposets(
            account, load_login, [RepositorySet.name], sdb, mdb, cache, slack)
    except ResponseError as e3:
        log.warning("account %d: load_account_reposets ResponseError: %s",
                    account, e3.response)
    except Exception as e4:
        sentry_sdk.capture_exception(e4)
    else:
        if reposets:
            return progress
    return None


async def refresh_repository_names(account: int,
                                   meta_ids: Tuple[int, ...],
                                   sdb: DatabaseLike,
                                   mdb: DatabaseLike) -> List[str]:
    """
    Update repository names in the account's reposets according to github.node_repository.

    github.account_repos updates faster but we don't care if we are missing some rename during one
    hour.

    :return: List of repository names belonging to the "ALL" reposet.
    """
    log = logging.getLogger(f"{metadata.__package__}.refresh_repository_names")
    reposet_rows = await sdb.fetch_all(
        select([RepositorySet.id, RepositorySet.items, RepositorySet.name])
        .where(RepositorySet.owner_id == account))
    repo_ids = set()
    for row in reposet_rows:
        repo_ids.update(r[1] for r in row[RepositorySet.items.name])
    name_rows = await mdb.fetch_all(
        select([NodeRepository.node_id, NodeRepository.url])
        .where(NodeRepository.acc_id.in_(meta_ids),
               NodeRepository.node_id.in_(repo_ids),
               NodeRepository.url.isnot(None)))
    name_map = {r[0]: strip_proto(r[1]) for r in name_rows}
    updates = []
    all_reposet_names = []
    for row in reposet_rows:
        dirty = False
        new_items = []
        is_all = row[RepositorySet.name.name] == RepositorySet.ALL
        for old_name, node_id in row[RepositorySet.items.name]:
            try:
                new_name = name_map[node_id]
            except KeyError:
                new_name = old_name
            if logical := extract_logical_repo(old_name, 3):
                new_name += "/" + logical
            if old_name != new_name:
                dirty = True
                log.info("[%d] rename %s -> %s", account, old_name, new_name)
            new_items.append([new_name, node_id])
            if is_all:
                all_reposet_names.append(new_name)
        if dirty:
            new_items.sort()
            updates.append(sdb.execute(
                update(RepositorySet)
                .where(RepositorySet.id == row[RepositorySet.id.name])
                .values({
                    RepositorySet.items: new_items,
                    RepositorySet.updates_count: RepositorySet.updates_count + 1,
                    RepositorySet.updated_at: datetime.now(timezone.utc),
                })))
    await gather(*updates)
    all_reposet_names.sort()
    return all_reposet_names
