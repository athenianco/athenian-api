from __future__ import annotations

import asyncio
from http import HTTPStatus
import logging
from sqlite3 import IntegrityError, OperationalError
from typing import Any, Callable, Coroutine, Iterator, Mapping, Optional, Sequence, Type

import aiomcache
from asyncpg import UniqueViolationError
import sentry_sdk
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
from sqlalchemy import and_, desc, insert, select
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.db import Connection, Database, DatabaseLike
from athenian.api.defer import defer
from athenian.api.internal.account import (
    RepositoryReference,
    fetch_github_installation_progress,
    get_account_name,
    get_metadata_account_ids,
    match_metadata_installation,
)
from athenian.api.internal.logical_repos import coerce_logical_repos
from athenian.api.internal.miners.access import AccessChecker
from athenian.api.internal.miners.access_classes import access_classes
from athenian.api.internal.prefixer import Prefixer, RepositoryName
from athenian.api.models.metadata.github import AccountRepository, NodeUser
from athenian.api.models.state.models import LogicalRepository, RepositorySet, UserAccount
from athenian.api.models.web import (
    ForbiddenError,
    InstallationProgress,
    InvalidRequestError,
    NoSourceDataError,
    NotFoundError,
)
from athenian.api.models.web.generic_error import BadRequestError, DatabaseConflict
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span


def reposet_items_to_refs(items: list[tuple[int, str, str]]) -> Iterator[RepositoryReference]:
    """Convert the raw DB repository tuples to RepositoryReference."""
    for i in items:
        yield RepositoryReference(*i)


async def resolve_reposet(
    repo: str,
    wrong_format: set[str],
    pointer: str,
    uid: str,
    account: int,
    prefixer: Prefixer,
    sdb: DatabaseLike,
) -> list[RepositoryName]:
    """
    Dereference the repository sets.

    If `repo` is a regular repository, return `[repo]`. Otherwise, return the list of \
    repositories by the parsed ID from the database.
    """
    if not repo.startswith("{"):
        try:
            return [RepositoryName.from_prefixed(repo)]
        except ValueError:
            wrong_format.add(repo)
            return []
    if not repo.endswith("}"):
        raise ResponseError(
            InvalidRequestError(
                detail=f"repository set format is invalid: {repo}",
                pointer=pointer,
            ),
        )
    try:
        set_id = int(repo[1:-1])
    except ValueError:
        raise ResponseError(
            InvalidRequestError(
                detail=f"repository set identifier is invalid: {repo}",
                pointer=pointer,
            ),
        )
    rs = await fetch_reposet(set_id, [RepositorySet.items, RepositorySet.tracking_re], sdb)
    if rs.owner_id != account:
        raise ResponseError(
            ForbiddenError(
                detail=f"User {uid} is not allowed to reference reposet {set_id} in this query",
            ),
        )
    return prefixer.dereference_repositories(reposet_items_to_refs(rs.items))


@sentry_span
async def fetch_reposet(
    id: int,
    columns: Sequence[Type[RepositorySet]] | Sequence[InstrumentedAttribute],
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
async def resolve_repos_with_request(
    repositories: list[str],
    account: int,
    request: AthenianWebRequest,
    separate=False,
    meta_ids: Optional[tuple[int, ...]] = None,
    prefixer: Optional[Prefixer] = None,
    checkers: Optional[dict[str, AccessChecker]] = None,
    pointer: Optional[str] = "?",
) -> tuple[set[RepositoryName] | list[set[RepositoryName]], str]:
    """
    Dereference all the reposets and produce the joint list of all mentioned repos.

    Alternative to the lower-level resolve_repos().

    :param separate: Value indicating whether to return each reposet separately.
    :return: (Union of all the mentioned repo names, service prefix).
    """

    async def login_loader() -> str:
        return (await request.user()).login

    return await resolve_repos(
        repositories,
        account,
        request.uid,
        login_loader,
        prefixer,
        meta_ids,
        request.sdb,
        request.mdb,
        request.cache,
        request.app["slack"],
        separate=separate,
        checkers=checkers,
        pointer=pointer,
    )


@sentry_span
async def resolve_repos(
    repositories: list[str],
    account: int,
    uid: str,
    login: Callable[[], Coroutine[None, None, str]],
    prefixer: Optional[Prefixer],
    meta_ids: Optional[tuple[int, ...]],
    sdb: Database,
    mdb: Database,
    cache: Optional[aiomcache.Client],
    slack: Optional[SlackWebClient],
    separate=False,
    checkers: Optional[dict[str, AccessChecker]] = None,
    pointer: Optional[str] = "?",
) -> tuple[set[RepositoryName] | list[set[RepositoryName]], str]:
    """
    Dereference all the reposets and produce the joint list of all mentioned repos.

    We don't check the user's access! That should happen automatically in Auth0 code.

    If `repositories` is empty, we load the "ALL" reposet.

    :param separate: Value indicating whether to return each reposet separately.
    :return: (Union of all the mentioned repo names, service prefix).
    """
    wrong_format = set()
    if not repositories:
        # this may initialize meta_ids, so must execute serialized
        reposets = await load_account_reposets(
            account,
            login,
            [RepositorySet.id, RepositorySet.name, RepositorySet.items, RepositorySet.tracking_re],
            sdb,
            mdb,
            cache,
            slack,
        )
        if meta_ids is None:
            meta_ids = await get_metadata_account_ids(account, sdb, cache)
        if prefixer is None:
            prefixer = await Prefixer.load(meta_ids, mdb, cache)
        for reposet in reposets:
            if reposet[RepositorySet.name.name] == RepositorySet.ALL:
                reposets = [
                    prefixer.dereference_repositories(
                        reposet_items_to_refs(reposet[RepositorySet.items.name]),
                    ),
                ]
                break
        else:
            raise ResponseError(
                NoSourceDataError(detail=f'No "{RepositorySet.ALL}" reposet exists.'),
            )
    else:
        assert meta_ids is not None
        if prefixer is None:
            prefixer = await Prefixer.load(meta_ids, mdb, cache)
        tasks = [
            resolve_reposet(r, wrong_format, f"{pointer}[{i}]", uid, account, prefixer, sdb)
            for i, r in enumerate(repositories)
        ]
        reposets = await gather(*tasks, op="resolve_reposet-s + meta_ids")
    repos = []
    checked_repos = set()
    prefix = None
    for reposet in reposets:
        resolved = set()
        for repo_name in reposet:
            repo_prefix, repo = repo_name.prefix, repo_name.unprefixed
            if prefix is None:
                prefix = repo_prefix
            elif prefix != repo_prefix:
                raise ResponseError(
                    InvalidRequestError(
                        detail=f'Mixed services are not allowed: "{prefix}" vs. "{repo_prefix}"',
                        pointer=pointer,
                    ),
                )
            checked_repos.add(repo)
            resolved.add(repo_name)
        if separate:
            repos.append(resolved)
        else:
            repos.extend(resolved)
    if prefix is None:
        raise ResponseError(
            InvalidRequestError(detail="The service prefix may not be empty.", pointer=pointer),
        )
    if wrong_format:
        raise ResponseError(
            BadRequestError(
                'The following repositories are malformed (missing "github.com/your_org/"'
                " prefix?): %s" % wrong_format,
            ),
        )
    checkers = checkers or {}
    if (checker := checkers.get(prefix)) is None:
        checkers[prefix] = checker = await access_classes["github.com"](
            account, meta_ids, sdb, mdb, cache,
        ).load()
    if denied := await checker.check(coerce_logical_repos(checked_repos).keys()):
        log = logging.getLogger(f"{metadata.__package__}.resolve_repos")
        log.warning(
            "access denied account %d%s: user sent %s we've got %s",
            account,
            meta_ids,
            denied,
            list(checker.installed_repos),
        )
        raise ResponseError(
            ForbiddenError(
                detail=(
                    'The following repositories are access denied for account %d (missing "'
                    'github.com/" prefix?): %s'
                )
                % (account, denied),
            ),
        )
    if not separate:
        repos = set(repos)
    return repos, prefix + "/"


@sentry_span
async def load_account_reposets(
    account: int,
    login: Callable[[], Coroutine[None, None, str]],
    fields: list,
    sdb: Database,
    mdb: Database,
    cache: Optional[aiomcache.Client],
    slack: Optional[SlackWebClient],
    check_progress: bool = True,
) -> list[Mapping[str, Any]]:
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
                account, login, fields, check_progress, sdb_conn, mdb_conn, mdb, cache, slack,
            )


async def _load_account_reposets(
    account: int,
    login: Callable[[], Coroutine[None, None, str]],
    fields: list,
    check_progress: bool,
    sdb_conn: Connection,
    mdb_conn: Connection,
    mdb: Database,
    cache: Optional[aiomcache.Client],
    slack: Optional[SlackWebClient],
) -> list[Mapping]:
    assert isinstance(sdb_conn, Connection)
    assert isinstance(mdb_conn, Connection)
    rss = await sdb_conn.fetch_all(
        select(fields).where(RepositorySet.owner_id == account).order_by(RepositorySet.created_at),
    )
    if rss:
        return rss

    log = logging.getLogger("%s.load_account_reposets" % metadata.__package__)

    def raise_no_source_data():
        raise ResponseError(
            NoSourceDataError(
                detail="The primary metadata application has not been installed yet.",
            ),
        )

    try:
        async with sdb_conn.transaction():
            # new account, discover their repos from the installation and create the first reposet

            async def load_prefixer():
                meta_ids = await get_metadata_account_ids(account, sdb_conn, cache)
                prefixer = await Prefixer.load(meta_ids, mdb_conn, cache)
                return prefixer, meta_ids

            prefixer_meta_ids, login = await asyncio.gather(
                load_prefixer(), login(), return_exceptions=True,
            )
            if isinstance(prefixer_meta_ids, Exception):
                if not isinstance(prefixer_meta_ids, ResponseError):
                    raise prefixer_meta_ids from None
                if isinstance(login, Exception):
                    raise ResponseError(ForbiddenError(detail=str(login)))
                if not (
                    meta_ids := await match_metadata_installation(
                        account, login, sdb_conn, mdb_conn, mdb, slack,
                    )
                ):
                    raise_no_source_data()
                prefixer = await Prefixer.load(meta_ids, mdb_conn, cache)
            else:
                prefixer, meta_ids = prefixer_meta_ids

        async with sdb_conn.transaction():
            if check_progress:
                progress = await fetch_github_installation_progress(
                    account, sdb_conn, mdb_conn, cache,
                )
                if progress.finished_date is None:
                    raise_no_source_data()
            repo_node_ids = await mdb_conn.fetch_all(
                select(AccountRepository.repo_graph_id).where(
                    AccountRepository.acc_id.in_(meta_ids),
                ),
            )
            repos = []
            missing = []
            for r in repo_node_ids:
                if r[0] in prefixer.repo_node_to_prefixed_name:
                    repos.append(RepositoryReference("github.com", r[0], ""))
                else:
                    missing.append(r[0])
            if missing:
                log.error("account_repos does not agree with api_repositories: %s", missing)
            if not repos:
                raise_no_source_data()

            # add the existing logical repositories as items of the reposet
            logical_repos = await sdb_conn.fetch_all(
                select(LogicalRepository.name, LogicalRepository.repository_id).where(
                    LogicalRepository.account_id == account,
                ),
            )

            wrong_references = []
            for name, physical_repo_id in logical_repos:
                ref = RepositoryReference("github.com", physical_repo_id, name)
                if physical_repo_id in prefixer.repo_node_to_prefixed_name:
                    repos.append(ref)
                else:
                    wrong_references.append(ref)
            if wrong_references:
                wrong_refs_repr = "; ".join(
                    f"{r.logical_name} => {r.node_id}" for r in wrong_references
                )
                log.error("logical repos point to not existing repos: %s", wrong_refs_repr)

            # sort repositories by node ID, then logical name
            repos.sort()

            rs = RepositorySet(
                name=RepositorySet.ALL, owner_id=account, items=repos,
            ).create_defaults()
            rs.id = await sdb_conn.execute(insert(RepositorySet).values(rs.explode()))
            log.info(
                "Created the first reposet %d for account %d with %d repos",
                rs.id,
                account,
                len(repos),
            )
        if slack is not None:
            name = await get_account_name(account, sdb_conn, mdb, cache, meta_ids=meta_ids)
            await defer(
                slack.post_install(
                    "installation_goes_on.jinja2",
                    account=account,
                    repos=len(repos),
                    name=name,
                    all_reposet_name=RepositorySet.ALL,
                    reposet=rs.id,
                ),
                "report_installation_goes_on",
            )
        return [rs.explode(with_primary_keys=True)]
    except (UniqueViolationError, IntegrityError, OperationalError) as e:
        log.warning("%s: %s", type(e).__name__, e)
        raise ResponseError(
            DatabaseConflict(detail="concurrent or duplicate initial reposet creation"),
        ) from None


async def load_account_state(
    account: int,
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
        auth0_id = await sdb.fetch_val(
            select([UserAccount.user_id])
            .where(and_(UserAccount.account_id == account))
            .order_by(desc(UserAccount.is_admin)),
        )
        if auth0_id is None:
            log.warning("There are no users in the account %d", account)
            return None
        try:
            db_id = int(auth0_id.split("|")[-1])
        except ValueError:
            log.error(
                "Unable to match user %s with metadata installations, "
                "you have to hack DB manually for account %d",
                auth0_id,
                account,
            )
            return None
        login = await mdb.fetch_val(select([NodeUser.login]).where(NodeUser.database_id == db_id))
        if login is None:
            log.warning("Could not find the user login of %s", auth0_id)
            return None
        async with sdb.connection() as sdb_conn:
            async with sdb_conn.transaction():
                if not await match_metadata_installation(
                    account, login, sdb_conn, mdb, mdb, slack,
                ):
                    log.warning(
                        "Did not match installations to account %d as %s", account, auth0_id,
                    )
                    return None
    try:
        progress = await fetch_github_installation_progress(account, sdb, mdb, cache)
    except ResponseError as e1:
        if e1.response.status != HTTPStatus.UNPROCESSABLE_ENTITY:
            sentry_sdk.capture_exception(e1)
        log.warning(
            "account %d: fetch_github_installation_progress ResponseError: %s",
            account,
            e1.response,
        )
        return None
    except Exception as e2:
        sentry_sdk.capture_exception(e2)
        log.warning(
            "account %d: fetch_github_installation_progress %s: %s",
            account,
            type(e2).__name__,
            e2,
        )
        return None
    if progress.finished_date is None:
        return progress

    async def load_login() -> str:
        raise AssertionError("This should never be called at this point")

    try:
        reposets = await load_account_reposets(
            account, load_login, [RepositorySet.name], sdb, mdb, cache, slack,
        )
    except ResponseError as e3:
        log.warning("account %d: load_account_reposets ResponseError: %s", account, e3.response)
    except Exception as e4:
        sentry_sdk.capture_exception(e4)
    else:
        if reposets:
            return progress
    return None


async def get_account_repositories(
    account: int,
    prefixer: Prefixer,
    sdb: DatabaseLike,
) -> list[RepositoryName]:
    """Fetch all the repositories belonging to the account."""
    repos = await sdb.fetch_val(
        select(RepositorySet.items).where(
            RepositorySet.owner_id == account,
            RepositorySet.name == RepositorySet.ALL,
        ),
    )
    if repos is None:
        raise ResponseError(
            NoSourceDataError(
                detail="The installation of account %d has not finished yet." % account,
            ),
        )
    return prefixer.dereference_repositories(reposet_items_to_refs(repos))
