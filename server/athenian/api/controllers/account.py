from collections import defaultdict
from datetime import datetime, timedelta, timezone
import logging
import marshal
import os
import pickle
from sqlite3 import IntegrityError
import struct
from typing import Any, Callable, Collection, Coroutine, List, Mapping, Optional, Tuple

from aiohttp import web
import aiomcache
import aiosqlite
from asyncpg import UniqueViolationError
import morcilla.core
import networkx as nx
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient
from sqlalchemy import and_, func, insert, select

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.cache import cached, max_exptime
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.db import Connection, Database, DatabaseLike
from athenian.api.defer import defer
from athenian.api.models.metadata.github import Account as MetadataAccount, AccountRepository, \
    FetchProgress, NodeUser, Organization, Team as MetadataTeam, TeamMember
from athenian.api.models.state.models import Account, AccountGitHubAccount, RepositorySet, \
    Team as StateTeam, UserAccount
from athenian.api.models.web import ForbiddenError, InstallationProgress, NoSourceDataError, \
    NotFoundError, \
    TableFetchingProgress, User
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
            raise ResponseError(NotFoundError(detail=f"Account {account} does not exist"))
        raise ResponseError(NoSourceDataError(
            detail="The installation of account %d has not finished yet." % account))
    return tuple(r[0] for r in ids)


async def get_metadata_account_ids_or_empty(account: int,
                                            sdb: DatabaseLike,
                                            cache: Optional[aiomcache.Client],
                                            ) -> Tuple[int, ...]:
    """
    Fetch the metadata account IDs for the given API account ID.

    Return None if nothing was found.
    """
    try:
        return await get_metadata_account_ids(account, sdb, cache)
    except ResponseError:
        return ()


async def match_metadata_installation(account: int,
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
    meta_ids = {r[0] for r in await mdb_conn.fetch_all(
        select([NodeUser.acc_id]).where(NodeUser.login == login))}
    if not meta_ids:
        log.warning("account %d: no installations found for %s", account, login)
        return ()
    owned_accounts = {r[0] for r in await sdb_conn.fetch_all(
        select([AccountGitHubAccount.id])
        .where(AccountGitHubAccount.id.in_(meta_ids)))}
    meta_ids -= owned_accounts
    if not meta_ids:
        log.warning("account %d: no new installations for %s among %d",
                    account, login, len(owned_accounts))
        return ()
    inserted = [
        AccountGitHubAccount(id=acc_id, account_id=account)
        .create_defaults().explode(with_primary_keys=True)
        for acc_id in meta_ids
    ]
    await sdb_conn.execute_many(insert(AccountGitHubAccount), inserted)
    log.info("account %d: installed %s for %s", account, meta_ids, login)
    if slack is not None:
        async def report_new_installation():
            metadata_accounts = [(r[0], r[1]) for r in await mdb.fetch_all(
                select([MetadataAccount.id, MetadataAccount.owner_login])
                .where(MetadataAccount.id.in_(meta_ids)))]
            await slack.post_install("new_installation.jinja2",
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
async def get_account_name(account: int,
                           sdb: DatabaseLike,
                           mdb: DatabaseLike,
                           cache: Optional[aiomcache.Client],
                           meta_ids: Optional[Tuple[int, ...]] = None,
                           ) -> str:
    """Load the human-readable name of the account."""
    if meta_ids is None:
        meta_ids = await get_metadata_account_ids(account, sdb, cache)
    rows = await mdb.fetch_all(select([MetadataAccount.name])
                               .where(MetadataAccount.id.in_(meta_ids)))
    return ", ".join(r[0] for r in rows)


@cached(
    exptime=60,
    serialize=lambda is_admin: b"1" if is_admin else b"0",
    deserialize=lambda buf: buf == b"1",
    key=lambda user, account, **_: (user, account),
)
async def get_user_account_status(user: str,
                                  account: int,
                                  sdb: Database,
                                  mdb: Optional[Database],
                                  user_info: Optional[Callable[..., Coroutine]],
                                  slack: Optional[SlackWebClient],
                                  cache: Optional[aiomcache.Client],
                                  context: str = "",
                                  ) -> bool:
    """
    Return the value indicating whether the given user is an admin of the given account.

    `mdb` must exist if `slack` exists. We await `user_info()` only if it exists.
    `context` is an optional string to pass in the user rejection Slack message.
    """
    status = await sdb.fetch_val(
        select([UserAccount.is_admin])
        .where(and_(UserAccount.user_id == user, UserAccount.account_id == account)))
    if status is None:
        async def report_user_rejected():
            async def dummy_user():
                return User(login="N/A")

            nonlocal user_info
            name, user_info = await gather(
                get_account_name(account, sdb, mdb, cache),
                user_info() if user_info is not None else dummy_user(),
            )
            await slack.post_account(
                "user_rejected.jinja2",
                user=user,
                user_name=user_info.login,
                user_email=user_info.email if user_info.email != User.EMPTY_EMAIL else "",
                account=account,
                account_name=name,
                context=context)

        if slack is not None:
            await defer(report_user_rejected(), "report_user_rejected_to_slack")
        raise ResponseError(NotFoundError(
            detail="Account %d does not exist or user %s is not a member." % (account, user)))
    return status


def only_admin(func):
    """Enforce the admin access level to an API handler."""
    async def wrapped_only_admin(request: AthenianWebRequest, body: dict) -> web.Response:
        account = body["account"]
        if not await get_user_account_status(
                request.uid, account, request.sdb, request.mdb, request.user,
                request.app["slack"], request.cache):
            raise ResponseError(ForbiddenError(
                f'User "{request.uid}" must be an admin of account {account}'))
        return await func(request, body)
    return wraps(wrapped_only_admin, func)


async def get_account_repositories(account: int,
                                   with_prefix: bool,
                                   sdb: DatabaseLike,
                                   ) -> List[str]:
    """Fetch all the repositories belonging to the account."""
    repos = await sdb.fetch_val(select([RepositorySet.items]).where(and_(
        RepositorySet.owner_id == account,
        RepositorySet.name == RepositorySet.ALL,
    )))
    if repos is None:
        raise ResponseError(NoSourceDataError(
            detail="The installation of account %d has not finished yet." % account))
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


async def copy_teams_as_needed(account: int,
                               meta_ids: Tuple[int, ...],
                               sdb: DatabaseLike,
                               mdb: DatabaseLike,
                               cache: Optional[aiomcache.Client],
                               ) -> Tuple[List[Mapping[str, Any]], int]:
    """
    Copy the teams from GitHub organization if none exist yet.

    :return: <list of created teams if nothing exists>, <final number of teams>.
    """
    log = logging.getLogger("%s.create_teams_as_needed" % metadata.__package__)
    prefixer = await Prefixer.load(meta_ids, mdb, cache)
    existing = await sdb.fetch_val(select([func.count(StateTeam.id)])
                                   .where(and_(StateTeam.owner_id == account,
                                               StateTeam.name != StateTeam.BOTS)))
    if existing > 0:
        log.info("Found %d existing teams for account %d, no-op", existing, account)
        return [], existing
    orgs = [org.id for org in await get_account_organizations(account, sdb, mdb, cache)]
    team_rows = await mdb.fetch_all(select([MetadataTeam])
                                    .where(and_(MetadataTeam.organization_id.in_(orgs),
                                                MetadataTeam.acc_id.in_(meta_ids))))
    if not team_rows:
        log.warning("Found 0 metadata teams for account %d", account)
        return [], 0
    # check for cycles - who knows?
    dig = nx.DiGraph()
    for row in team_rows:
        team_id = row[MetadataTeam.id.name]
        if (parent_id := row[MetadataTeam.parent_team_id.name]) is not None:
            dig.add_edge(team_id, parent_id)
        else:
            dig.add_node(team_id)
    try:
        cycle = nx.find_cycle(dig)
    except nx.NetworkXNoCycle:
        pass
    else:
        log.error("Found a metadata parent-child team reference cycle: %s", cycle)
        return [], 0
    teams = {t[MetadataTeam.id.name]: t for t in team_rows}
    member_rows = await mdb.fetch_all(
        select([TeamMember.parent_id, TeamMember.child_id])
        .where(and_(TeamMember.parent_id.in_(teams), TeamMember.acc_id.in_(meta_ids))))
    members = defaultdict(list)
    for row in member_rows:
        try:
            members[row[TeamMember.parent_id.name]].append(
                prefixer.user_node_to_prefixed_login[row[TeamMember.child_id.name]])
        except KeyError:
            log.error("Could not resolve user %s", row[TeamMember.child_id.name])
    db_ids = {}
    created_teams = []
    for node_id in reversed(list(nx.topological_sort(dig))):
        team = teams[node_id]
        if (parent := teams.get(team[MetadataTeam.parent_team_id.name])) is not None:
            parent = db_ids[parent[MetadataTeam.id.name]]
        team = StateTeam(owner_id=account,
                         name=team[MetadataTeam.name.name],
                         members=sorted(members.get(team[MetadataTeam.id.name], [])),
                         parent_id=parent,
                         ).create_defaults().explode()
        try:
            db_ids[node_id] = team[StateTeam.id.name] = \
                await sdb.execute(insert(StateTeam).values(team))
        except (UniqueViolationError, IntegrityError) as e:
            log.warning('Failed to create team "%s" in account %d: %s',
                        team[StateTeam.name.name], account, e)
            db_ids[node_id] = None
        else:
            created_teams.append(team)
    team_names = [t[MetadataTeam.name.name] for t in team_rows]
    log.info("Created %d out of %d teams in account %d: %s",
             len(created_teams), len(team_names), account,
             [t[StateTeam.name.name] for t in created_teams])
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
async def get_installation_event_ids(account: int,
                                     sdb: DatabaseLike,
                                     mdb: DatabaseLike,
                                     cache: Optional[aiomcache.Client],
                                     ) -> List[Tuple[int, str]]:
    """Load the GitHub account and delivery event IDs for the given sdb account."""
    meta_ids = await get_metadata_account_ids(account, sdb, cache)
    rows = await mdb.fetch_all(
        select([AccountRepository.acc_id, AccountRepository.event_id])
        .where(AccountRepository.acc_id.in_(meta_ids))
        .distinct())
    if diff := set(meta_ids) - {r[0] for r in rows}:
        raise ResponseError(NoSourceDataError(detail="Some installation%s missing: %s." %
                                                     ("s are" if len(diff) > 1 else " is", diff)))
    return [(r[0], r[1]) for r in rows]


@cached(
    exptime=max_exptime,
    serialize=lambda s: s.encode(),
    deserialize=lambda b: b.decode(),
    key=lambda metadata_account_id, **_: (metadata_account_id,),
    refresh_on_access=True,
)
async def get_installation_owner(metadata_account_id: int,
                                 mdb_conn: morcilla.core.Connection,
                                 cache: Optional[aiomcache.Client],
                                 ) -> str:
    """Load the native user ID who installed the app."""
    user_login = await mdb_conn.fetch_val(
        select([MetadataAccount.owner_login])
        .where(MetadataAccount.id == metadata_account_id))
    if user_login is None:
        raise ResponseError(NoSourceDataError(detail="The installation has not started yet."))
    return user_login


@cached(exptime=5,  # matches the webapp poll interval
        serialize=pickle.dumps,
        deserialize=pickle.loads,
        key=lambda account, **_: (account,))
async def fetch_github_installation_progress(account: int,
                                             sdb: DatabaseLike,
                                             mdb_conn: Connection,
                                             cache: Optional[aiomcache.Client],
                                             ) -> InstallationProgress:
    """Load the GitHub installation progress for the specified account."""
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
            select([func.count(AccountRepository.repo_graph_id)])
            .where(AccountRepository.acc_id == metadata_account_id))
        rows = await mdb_conn.fetch_all(
            select([FetchProgress])
            .where(and_(FetchProgress.event_id == event_id,
                        FetchProgress.acc_id == metadata_account_id)))
        if not rows:
            continue
        tables = [TableFetchingProgress(fetched=r[FetchProgress.nodes_processed.name],
                                        name=r[FetchProgress.node_type.name],
                                        total=r[FetchProgress.nodes_total.name])
                  for r in rows]
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
                log.info("Overriding the installation progress of %d by the idle time threshold; "
                         "there are %d pending tables, last update on %s",
                         account, pending, finished_date)
                finished_date += idle_threshold  # don't fool the user
        elif pending:
            finished_date = None
        elif now - finished_date < calm_threshold:
            log.warning("Account %d's installation is calming, postponed until %s",
                        account, finished_date + calm_threshold)
            finished_date = None
        else:
            finished_date += calm_threshold  # don't fool the user
        model = InstallationProgress(started_date=started_date,
                                     finished_date=finished_date,
                                     owner=owner,
                                     repositories=repositories,
                                     tables=tables)
        models.append(model)
    if not models:
        raise ResponseError(NoSourceDataError(
            detail="No installation progress exists for account %d." % account))
    tables = {}
    finished_date = datetime(2020, 1, 1, tzinfo=timezone.utc)
    for m in models:
        for t in m.tables:
            table = tables.setdefault(
                t.name, TableFetchingProgress(name=t.name, fetched=0, total=0))
            table.fetched += t.fetched
            table.total += t.total
        if model.finished_date is None:
            finished_date = None
        elif finished_date is not None:
            finished_date = max(finished_date, model.finished_date)
    model = InstallationProgress(started_date=min(m.started_date for m in models),
                                 finished_date=finished_date,
                                 owner=owner,
                                 repositories=sum(m.repositories for m in models),
                                 tables=sorted(tables.values()))
    return model
