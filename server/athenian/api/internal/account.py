from datetime import datetime, timedelta, timezone
from graphlib import CycleError
from itertools import chain
import logging
import marshal
import os
import pickle
from sqlite3 import IntegrityError
import struct
import time
from typing import Any, Callable, Collection, Coroutine, Mapping, Optional, Sequence

from aiohttp import web
import aiomcache
import aiosqlite
from asyncpg import UniqueViolationError
import morcilla.core
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
import sqlalchemy as sa
from sqlalchemy import and_, func, insert, select

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.cache import cached, max_exptime, middle_term_exptime, short_term_exptime
from athenian.api.db import Connection, Database, DatabaseLike, Row
from athenian.api.defer import defer
from athenian.api.internal.account_feature import is_feature_enabled
from athenian.api.internal.team_meta import (
    get_meta_teams_members,
    get_meta_teams_topological_order,
)
from athenian.api.models.metadata.github import (
    Account as MetadataAccount,
    AccountRepository,
    FetchProgress,
    NodeUser,
    Organization,
    Team as MetadataTeam,
)
from athenian.api.models.state.models import (
    Account,
    AccountGitHubAccount,
    Feature,
    RepositorySet,
    Team as StateTeam,
    UserAccount,
)
from athenian.api.models.web import (
    ForbiddenError,
    InstallationProgress,
    NoSourceDataError,
    NotFoundError,
    TableFetchingProgress,
    User,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError
from athenian.api.typing_utils import wraps

jira_url_template = os.getenv("ATHENIAN_JIRA_INSTALLATION_URL_TEMPLATE")


@cached(
    # the TTL is huge because this relation will never change and is requested frequently
    exptime=max_exptime,
    serialize=lambda ids: struct.pack("!" + "q" * len(ids), *ids),
    deserialize=lambda buf: struct.unpack("!" + "q" * (len(buf) // 8), buf),
    key=lambda account, **_: (account,),
    refresh_on_access=True,
)
async def get_metadata_account_ids(
    account: int,
    sdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> tuple[int, ...]:
    """Fetch the metadata account IDs for the given API account ID."""
    ids = await sdb.fetch_all(
        select([AccountGitHubAccount.id]).where(AccountGitHubAccount.account_id == account),
    )
    if len(ids) == 0:
        acc_exists = await sdb.fetch_val(select([Account.id]).where(Account.id == account))
        if not acc_exists:
            raise ResponseError(
                NotFoundError(
                    detail=f"Account {account} does not exist", type_="/errors/AccountNotFound",
                ),
            )
        raise ResponseError(
            NoSourceDataError(
                detail="The installation of account %d has not finished yet." % account,
            ),
        )
    return tuple(r[0] for r in ids)


async def get_metadata_account_ids_or_empty(
    account: int,
    sdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> tuple[int, ...]:
    """
    Fetch the metadata account IDs for the given API account ID.

    Return None if nothing was found.
    """
    try:
        return await get_metadata_account_ids(account, sdb, cache)
    except ResponseError:
        return ()


@cached(
    exptime=middle_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda accounts, **_: (sorted(accounts),),
)
async def get_multiple_metadata_account_ids(
    accounts: Sequence[int],
    sdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> dict[int, list[int]]:
    """Fetch the metadata account IDs for many accounts."""
    query = sa.select(AccountGitHubAccount.id, AccountGitHubAccount.account_id).where(
        AccountGitHubAccount.account_id.in_(accounts),
    )
    accounts_meta_ids: dict[int, list[int]] = {account: [] for account in accounts}
    for row in await sdb.fetch_all(query):
        accounts_meta_ids[row[AccountGitHubAccount.account_id.name]].append(
            row[AccountGitHubAccount.id.name],
        )
    return accounts_meta_ids


async def match_metadata_installation(
    account: int,
    login: str,
    sdb_conn: Connection,
    mdb_conn: Connection,
    mdb: Database,
    slack: Optional[SlackWebClient],
) -> Collection[int]:
    """Discover new metadata installations for the given state DB account.

    sdb_conn must be in a transaction!
    """
    log = logging.getLogger(f"{metadata.__package__}.match_metadata_installation")
    user_rows, owner_rows = await gather(
        mdb_conn.fetch_all(select([NodeUser.acc_id]).where(NodeUser.login == login)),
        mdb.fetch_all(select([MetadataAccount.id]).where(MetadataAccount.owner_login == login)),
    )
    meta_ids = set(chain((r[0] for r in user_rows), (r[0] for r in owner_rows)))
    del user_rows, owner_rows
    if not meta_ids:
        log.warning("account %d: no installations found for %s", account, login)
        return ()
    owned_accounts = {
        r[0]
        for r in await sdb_conn.fetch_all(
            select([AccountGitHubAccount.id]).where(AccountGitHubAccount.id.in_(meta_ids)),
        )
    }
    meta_ids -= owned_accounts
    if not meta_ids:
        log.warning(
            "account %d: no new installations for %s among %d",
            account,
            login,
            len(owned_accounts),
        )
        return ()
    inserted = [
        AccountGitHubAccount(id=acc_id, account_id=account)
        .create_defaults()
        .explode(with_primary_keys=True)
        for acc_id in meta_ids
    ]
    await sdb_conn.execute_many(insert(AccountGitHubAccount), inserted)
    log.info("account %d: installed %s for %s", account, meta_ids, login)
    if slack is not None:

        async def report_new_installation():
            metadata_accounts = [
                (r[0], r[1])
                for r in await mdb.fetch_all(
                    select([MetadataAccount.id, MetadataAccount.owner_login]).where(
                        MetadataAccount.id.in_(meta_ids),
                    ),
                )
            ]
            await slack.post_install(
                "new_installation.jinja2",
                account=account,
                all_reposet_name=RepositorySet.ALL,
                metadata_accounts=metadata_accounts,
                login=login,
            )

        await defer(report_new_installation(), "report_new_installation")
    return meta_ids


@cached(
    exptime=max_exptime,
    serialize=lambda name: name.encode(),
    deserialize=lambda name: name.decode(),
    key=lambda account, **_: (account,),
    refresh_on_access=True,
)
async def get_account_name(
    account: int,
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
    meta_ids: Optional[tuple[int, ...]] = None,
) -> str:
    """Load the human-readable name of the account."""
    if meta_ids is None:
        meta_ids = await get_metadata_account_ids(account, sdb, cache)
    rows = await mdb.fetch_all(
        select([MetadataAccount.name]).where(MetadataAccount.id.in_(meta_ids)),
    )
    return ", ".join(r[0] for r in rows)


async def get_account_name_or_stub(
    account: int,
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
    meta_ids: Optional[tuple[int, ...]] = None,
) -> str:
    """Load the human-readable name of the account or a placeholder if no name exists."""
    try:
        return await get_account_name(account, sdb, mdb, cache, meta_ids)
    except ResponseError as e:
        return f"N/A ({int(e.response.status)})"


@cached(
    exptime=60,
    serialize=lambda _: b"1",
    deserialize=lambda _: False,
    key=lambda user, account, **_: (user, account),
)
async def _report_user_rejected(
    user: str,
    user_info: Optional[Callable[..., Coroutine]],
    account: int,
    context: str,
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    slack: SlackWebClient,
    cache: Optional[aiomcache.Client],
) -> bool:
    async def dummy_user():
        return User(login="N/A")

    name, user_info = await gather(
        get_account_name_or_stub(account, sdb, mdb, cache),
        user_info() if user_info is not None else dummy_user(),
    )
    await slack.post_account(
        "user_rejected.jinja2",
        user=user,
        user_name=user_info.login,
        user_email=user_info.email,
        account=account,
        account_name=name,
        context=context,
    )
    return True


async def get_user_account_status_from_request(request: AthenianWebRequest, account: int) -> bool:
    """Return the value indicating whether the requesting user is an admin of the given account."""
    return await get_user_account_status(
        request.uid,
        account,
        request.sdb,
        request.mdb,
        request.user,
        request.app["slack"],
        request.cache,
        context=f"{request.method} {request.path}",
        is_god=hasattr(request, "god_id"),
    )


@cached(
    exptime=60,
    serialize=lambda is_admin: b"1" if is_admin else b"0",
    deserialize=lambda buf: buf == b"1",
    key=lambda user, account, **_: (user, account),
)
async def get_user_account_status(
    user: str,
    account: int,
    sdb: Database,
    mdb: Optional[Database],
    user_info: Optional[Callable[..., Coroutine]],
    slack: Optional[SlackWebClient],
    cache: Optional[aiomcache.Client],
    context: str = "",
    is_god: bool = False,
) -> bool:
    """
    Return the value indicating whether the given user is an admin of the given account.

    `mdb` must exist if `slack` exists. We await `user_info()` only if it exists.
    `context` is an optional string to pass in the user rejection Slack message.
    """
    status = await sdb.fetch_val(
        select([UserAccount.is_admin]).where(
            and_(UserAccount.user_id == user, UserAccount.account_id == account),
        ),
    )
    if status is None:
        if slack is not None and not is_god:
            await defer(
                _report_user_rejected(user, user_info, account, context, sdb, mdb, slack, cache),
                "report_user_rejected_to_slack",
            )
        raise ResponseError(
            NotFoundError(
                detail="Account %d does not exist or user %s is not a member." % (account, user),
                type_="/errors/AccountNotFound",
            ),
        )
    return status


def only_admin(func):
    """Enforce the admin access level to an API handler."""

    async def wrapped_only_admin(request: AthenianWebRequest, body: dict) -> web.Response:
        account = body["account"]
        if not await get_user_account_status_from_request(request, account):
            raise ResponseError(
                ForbiddenError(f'User "{request.uid}" must be an admin of account {account}'),
            )
        return await func(request, body)

    return wraps(wrapped_only_admin, func)


def only_god(func):
    """Enforce the god access level to an API handler."""

    async def wrapped_only_god(request: AthenianWebRequest, **kwargs) -> web.Response:
        if not hasattr(request, "god_id"):
            raise ResponseError(ForbiddenError(detail=f"User {request.uid} must be a god"))
        return await func(request, **kwargs)

    return wraps(wrapped_only_god, func)


async def get_account_repositories(
    account: int,
    with_prefix: bool,
    sdb: DatabaseLike,
) -> list[str]:
    """Fetch all the repositories belonging to the account."""
    repos = await sdb.fetch_val(
        select([RepositorySet.items]).where(
            and_(
                RepositorySet.owner_id == account,
                RepositorySet.name == RepositorySet.ALL,
            ),
        ),
    )
    if repos is None:
        raise ResponseError(
            NoSourceDataError(
                detail="The installation of account %d has not finished yet." % account,
            ),
        )
    if not with_prefix:
        repos = [r[0].split("/", 1)[1] for r in repos]
    else:
        repos = [r[0] for r in repos]
    return repos


@cached(
    exptime=24 * 60 * 60,  # 1 day
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda account, **_: (account,),
)
async def get_account_organizations(
    account: int,
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> list[Organization]:
    """Fetch the list of GitHub organizations installed for the account."""
    ghids = await get_metadata_account_ids(account, sdb, cache)
    rows = await mdb.fetch_all(select([Organization]).where(Organization.acc_id.in_(ghids)))
    return [Organization(**r) for r in rows]


async def copy_teams_as_needed(
    account: int,
    meta_ids: tuple[int, ...],
    root_team_id: int,
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> tuple[list[Mapping[str, Any]], int]:
    """
    Copy the teams from GitHub organization if none exist yet.

    :return: <list of created teams if nothing exists>, <final number of teams>.
    """
    log = logging.getLogger("%s.create_teams_as_needed" % metadata.__package__)
    existing = await sdb.fetch_val(
        select(func.count(StateTeam.id)).where(
            StateTeam.owner_id == account,
            StateTeam.name != StateTeam.BOTS,
            StateTeam.id != root_team_id,
        ),
    )
    if existing > 0:
        log.info("Found %d existing teams for account %d, no-op", existing, account)
        return [], existing
    orgs = [org.id for org in await get_account_organizations(account, sdb, mdb, cache)]
    team_rows = await mdb.fetch_all(
        select(MetadataTeam).where(
            and_(MetadataTeam.organization_id.in_(orgs), MetadataTeam.acc_id.in_(meta_ids)),
        ),
    )
    if not team_rows:
        log.warning("Found 0 metadata teams for account %d", account)
        return [], 0
    try:
        teams_topological_order = get_meta_teams_topological_order(team_rows)
    except CycleError as e:
        log.error("Found a metadata parent-child team reference cycle: %s", e)
        return [], 0
    teams: dict[int, Row] = {t[MetadataTeam.id.name]: t for t in team_rows}
    members = await get_meta_teams_members(teams, meta_ids, mdb)
    db_ids = {}
    created_teams = []
    for node_id in teams_topological_order:
        team = teams[node_id]

        parent_id = root_team_id
        if (github_parent_id := team[MetadataTeam.parent_team_id.name]) is not None:
            if (parent := teams.get(github_parent_id)) is not None:
                parent_id = db_ids[parent[MetadataTeam.id.name]]
        # we remain with parent_id = root_team_id either when the team hasn't got a real parent
        # or its parent failed to create

        team = (
            StateTeam(
                owner_id=account,
                name=team[MetadataTeam.name.name],
                members=sorted(members.get(team[MetadataTeam.id.name], [])),
                parent_id=parent_id,
                origin_node_id=team[MetadataTeam.id.name],
            )
            .create_defaults()
            .explode()
        )
        try:
            db_ids[node_id] = team[StateTeam.id.name] = await sdb.execute(
                insert(StateTeam).values(team),
            )
        except (UniqueViolationError, IntegrityError) as e:
            log.warning(
                'Failed to create team "%s" in account %d: %s',
                team[StateTeam.name.name],
                account,
                e,
            )
            db_ids[node_id] = None
        else:
            created_teams.append(team)
    team_names = [t[MetadataTeam.name.name] for t in team_rows]
    log.info(
        "Created %d out of %d teams in account %d: %s",
        len(created_teams),
        len(team_names),
        account,
        [t[StateTeam.name.name] for t in created_teams],
    )
    return created_teams, len(created_teams)


async def generate_jira_invitation_link(account: int, sdb: DatabaseLike) -> str:
    """Return the JIRA installation URL for the given account."""
    secret = await sdb.fetch_val(select([Account.secret]).where(Account.id == account))
    assert secret not in (None, Account.missing_secret)
    return jira_url_template % secret


@cached(
    exptime=24 * 3600,  # 1 day
    serialize=lambda t: marshal.dumps(t),
    deserialize=lambda buf: marshal.loads(buf),
    key=lambda account, **_: (account,),
)
async def get_installation_event_ids(
    account: int,
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> list[tuple[int, str]]:
    """Load the GitHub account and delivery event IDs for the given sdb account."""
    meta_ids = await get_metadata_account_ids(account, sdb, cache)
    rows = await mdb.fetch_all(
        select([AccountRepository.acc_id, AccountRepository.event_id])
        .where(AccountRepository.acc_id.in_(meta_ids))
        .distinct(),
    )
    if diff := set(meta_ids) - {r[0] for r in rows}:
        raise ResponseError(
            NoSourceDataError(
                detail="Some installation%s missing: %s."
                % ("s are" if len(diff) > 1 else " is", diff),
            ),
        )
    return [(r[0], r[1]) for r in rows]


@cached(
    exptime=max_exptime,
    serialize=lambda s: s.encode(),
    deserialize=lambda b: b.decode(),
    key=lambda metadata_account_id, **_: (metadata_account_id,),
    refresh_on_access=True,
)
async def get_installation_owner(
    metadata_account_id: int,
    mdb_conn: morcilla.core.Connection,
    cache: Optional[aiomcache.Client],
) -> str:
    """Load the native user ID who installed the app."""
    user_login = await mdb_conn.fetch_val(
        select([MetadataAccount.owner_login]).where(MetadataAccount.id == metadata_account_id),
    )
    if user_login is None:
        raise ResponseError(NoSourceDataError(detail="The installation has not started yet."))
    return user_login


async def fetch_github_installation_progress(
    account: int,
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> InstallationProgress:
    """Load the GitHub installation progress for the specified account."""
    return (await _fetch_github_installation_progress_timed(account, sdb, mdb, cache))[0]


@cached(
    exptime=lambda result, **_: 5 if result[1] < 1 else short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda account, **_: (account,),
)
async def _fetch_github_installation_progress_timed(
    account: int,
    sdb: DatabaseLike,
    mdb: DatabaseLike,
    cache: Optional[aiomcache.Client],
) -> tuple[InstallationProgress, float]:
    start_time = time.time()
    if isinstance(mdb, Database):
        async with mdb.connection() as mdb_conn:
            progress = await _fetch_github_installation_progress_db(account, sdb, mdb_conn, cache)
    else:
        progress = await _fetch_github_installation_progress_db(account, sdb, mdb, cache)
    elapsed = time.time() - start_time
    return progress, elapsed


async def _fetch_github_installation_progress_db(
    account: int,
    sdb: DatabaseLike,
    mdb_conn: Connection,
    cache: Optional[aiomcache.Client],
) -> InstallationProgress:
    log = logging.getLogger("%s.fetch_github_installation_progress" % metadata.__package__)
    assert isinstance(mdb_conn, Connection)
    async with mdb_conn.raw_connection() as raw_connection:
        mdb_sqlite = isinstance(raw_connection, aiosqlite.Connection)
    idle_threshold = timedelta(hours=3)
    calm_threshold = timedelta(hours=1, minutes=30)
    event_ids = await get_installation_event_ids(account, sdb, mdb_conn, cache)
    owner = await get_installation_owner(event_ids[0][0], mdb_conn, cache)
    # we don't cache this because the number of repos can dynamically change
    models = []
    for metadata_account_id, event_id in event_ids:
        repositories = await mdb_conn.fetch_val(
            select([func.count(AccountRepository.repo_graph_id)]).where(
                AccountRepository.acc_id == metadata_account_id,
            ),
        )
        rows = await mdb_conn.fetch_all(
            select([FetchProgress]).where(
                and_(
                    FetchProgress.event_id == event_id,
                    FetchProgress.acc_id == metadata_account_id,
                ),
            ),
        )
        if not rows:
            continue
        tables = [
            TableFetchingProgress(
                fetched=r[FetchProgress.nodes_processed.name],
                name=r[FetchProgress.node_type.name],
                total=r[FetchProgress.nodes_total.name],
            )
            for r in rows
        ]
        started_date = min(r[FetchProgress.created_at.name] for r in rows)
        if mdb_sqlite:
            started_date = started_date.replace(tzinfo=timezone.utc)
        finished_date = max(r[FetchProgress.updated_at.name] for r in rows)
        if mdb_sqlite:
            finished_date = finished_date.replace(tzinfo=timezone.utc)
        pending = sum(t.fetched < t.total for t in tables)
        now = datetime.now(tz=timezone.utc)
        if now - finished_date > idle_threshold:
            for table in tables:
                table.total = table.fetched
            if pending:
                log.info(
                    "Overriding the installation progress of %d by the idle time threshold; "
                    "there are %d pending tables, last update on %s",
                    account,
                    pending,
                    finished_date,
                )
                finished_date += idle_threshold  # don't fool the user
        elif pending:
            finished_date = None
        elif now - finished_date < calm_threshold:
            log.warning(
                "Account %d's installation is calming, postponed until %s",
                account,
                finished_date + calm_threshold,
            )
            finished_date = None
        else:
            finished_date += calm_threshold  # don't fool the user
        model = InstallationProgress(
            started_date=started_date,
            finished_date=finished_date,
            owner=owner,
            repositories=repositories,
            tables=tables,
        )
        models.append(model)
    if not models:
        raise ResponseError(
            NoSourceDataError(detail="No installation progress exists for account %d." % account),
        )
    tables = {}
    finished_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for m in models:
        for t in m.tables:
            table = tables.setdefault(
                t.name, TableFetchingProgress(name=t.name, fetched=0, total=0),
            )
            table.fetched += t.fetched
            table.total += t.total
        if model.finished_date is None:
            finished_date = None
        elif finished_date is not None:
            finished_date = max(finished_date, model.finished_date)
    model = InstallationProgress(
        started_date=min(m.started_date for m in models),
        finished_date=finished_date,
        owner=owner,
        repositories=sum(m.repositories for m in models),
        tables=sorted(tables.values()),
    )
    return model


async def is_membership_check_enabled(account: int, sdb: DatabaseLike) -> bool:
    """Check whether the user registration requires the organization membership."""
    return await is_feature_enabled(account, Feature.USER_ORG_MEMBERSHIP_CHECK, sdb)


async def is_github_login_enabled(account: int, sdb: DatabaseLike) -> bool:
    """Check whether we accept invitations for GitHub users."""
    return await is_feature_enabled(account, Feature.GITHUB_LOGIN_ENABLED, sdb)


async def check_account_expired(context: AthenianWebRequest, log: logging.Logger) -> bool:
    """Return the value indicating whether the account's expiration datetime is in the past."""
    expires_at = await context.sdb.fetch_val(
        select([Account.expires_at]).where(Account.id == context.account),
    )
    if getattr(context, "god_id", context.uid) == context.uid and (
        expires_at is None or expires_at < datetime.now(expires_at.tzinfo)
    ):
        if (slack := context.app["slack"]) is not None:
            await defer(
                report_user_account_expired(
                    context.uid,
                    context.account,
                    expires_at,
                    context.sdb,
                    context.mdb,
                    context.user,
                    slack,
                    context.cache,
                ),
                "report_user_account_expired_to_slack",
            )
        log.warning(
            "Attempt to use an expired account %d by user %s", context.account, context.uid,
        )
        return True
    return False


@cached(
    exptime=middle_term_exptime,
    serialize=lambda x: x,
    deserialize=lambda x: x,
    key=lambda user, account, **_: (user, account),
)
async def report_user_account_expired(
    user: str,
    account: int,
    expired_at: datetime,
    sdb: Database,
    mdb: Database,
    user_info: Callable[..., Coroutine],
    slack: Optional[SlackWebClient],
    cache: Optional[aiomcache.Client],
):
    """Send a Slack message about the user who accessed an expired account."""

    async def dummy_user():
        return User(login="N/A")

    name, user_info = await gather(
        get_account_name_or_stub(account, sdb, mdb, cache),
        user_info() if user_info is not None else dummy_user(),
    )
    await slack.post_account(
        "user_account_expired.jinja2",
        user=user,
        user_name=user_info.login,
        user_email=user_info.email,
        account=account,
        account_name=name,
        expired_at=expired_at,
    )
    return b"1"
