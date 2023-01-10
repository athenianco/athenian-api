import asyncio
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from itertools import chain
import logging
import re
import sqlite3
from typing import Collection, Iterable, Mapping, Optional, Union

from aiohttp import web
import aiomcache
import asyncpg
import pandas as pd
from sqlalchemy import (
    and_,
    delete,
    distinct,
    exists,
    func,
    insert,
    not_,
    select,
    text,
    union,
    union_all,
    update,
)
from xxhash._xxhash import xxh32_intdigest

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.auth import disable_default_user
from athenian.api.balancing import weight
from athenian.api.db import Connection, Database, dialect_specific_insert
from athenian.api.defer import defer, launch_defer_from_request, wait_deferred
from athenian.api.internal.account import get_installation_url_prefix, get_metadata_account_ids
from athenian.api.internal.features.entries import PRFactsCalculator
from athenian.api.internal.miners.access_classes import access_classes
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.bots import bots
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.commit import compose_commit_url
from athenian.api.internal.miners.github.deployment import mine_deployments
from athenian.api.internal.miners.github.release_mine import mine_releases
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.reposet import resolve_repos_with_request
from athenian.api.internal.settings import (
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseSettings,
    Settings,
)
from athenian.api.models.metadata.github import PushCommit, Release, User
from athenian.api.models.persistentdata.models import (
    DeployedComponent,
    DeployedLabel,
    DeploymentNotification,
    ReleaseNotification,
)
from athenian.api.models.precomputed.models import (
    GitHubCommitDeployment,
    GitHubDeploymentFacts,
    GitHubDonePullRequestFacts,
    GitHubMergedPullRequestFacts,
    GitHubPullRequestDeployment,
    GitHubReleaseDeployment,
    GitHubReleaseFacts,
)
from athenian.api.models.web import (
    DatabaseConflict,
    DeleteEventsCacheRequest,
    DeploymentNotification as WebDeploymentNotification,
    ForbiddenError,
    InvalidRequestError,
    ReleaseNotification as WebReleaseNotification,
    ReleaseNotificationStatus,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response
from athenian.api.serialization import ParseError
from athenian.api.tracing import sentry_span

commit_hash_re = re.compile(r"[a-f0-9]{7}([a-f0-9]{33})?")


@disable_default_user
@weight(0)
async def notify_releases(request: AthenianWebRequest, body: list[dict]) -> web.Response:
    """Notify about new releases. The release settings must be set to "notification"."""
    # account is automatically checked at this point
    log = logging.getLogger(f"{metadata.__package__}.notify_releases")
    log.info("%s", body)
    try:
        notifications = [WebReleaseNotification.from_dict(n) for n in body]
    except ParseError as e:
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    account = request.account
    sdb, mdb, rdb = request.sdb, request.mdb, request.rdb
    authors = []
    for n in notifications:
        author = n.author
        if author is not None and "/" in author:
            author = author.rsplit("/", 1)[1]  # remove github.com/ or any other prefix
        authors.append(author)
    unique_authors = set(authors) - {None}
    statuses = [None] * len(notifications)

    async def main_flow():
        repos = set()
        full_commits = set()
        prefixed_commits = defaultdict(set)
        unique_notifications = set()
        for i, n in enumerate(notifications):
            try:
                repos.add(repo := n.repository.split("/", 1)[1])
            except IndexError:
                raise ResponseError(
                    InvalidRequestError(
                        "[%d].repository" % i,
                        detail="repository name is invalid: %s" % n.repository,
                    ),
                )
            if not commit_hash_re.fullmatch(n.commit):
                raise ResponseError(
                    InvalidRequestError("[%d].commit" % i, detail="invalid commit hash"),
                )
            if len(n.commit) == 7:
                prefixed_commits[repo].add(n.commit)
            else:
                full_commits.add(n.commit)
            if (key := (n.name or n.commit, n.repository)) in unique_notifications:
                statuses[i] = ReleaseNotificationStatus.IGNORED_DUPLICATE
                continue
            unique_notifications.add(key)
        meta_ids = await get_metadata_account_ids(account, sdb, request.cache)
        checker = await access_classes["github.com"](
            account, meta_ids, sdb, mdb, request.cache,
        ).load()
        if denied := await checker.check(repos):
            raise ResponseError(
                ForbiddenError(
                    detail="the following repositories are access denied for account %d: %s"
                    % (account, denied),
                ),
            )

        # the commit may not exist yet in the metadata, but let's try to resolve what we can
        commit_rows, user_rows, *url_prefixes = await gather(
            mdb.fetch_all(
                union(
                    select(
                        PushCommit.acc_id,
                        PushCommit.sha,
                        PushCommit.node_id,
                        PushCommit.repository_full_name,
                    ).where(PushCommit.acc_id.in_(meta_ids), PushCommit.sha.in_(full_commits)),
                    *(
                        select(
                            PushCommit.acc_id,
                            PushCommit.sha,
                            PushCommit.node_id,
                            PushCommit.repository_full_name,
                        ).where(
                            PushCommit.acc_id.in_(meta_ids),
                            PushCommit.repository_full_name == repo,
                            func.substr(PushCommit.sha, 1, 7).in_(prefixes),
                        )
                        for repo, prefixes in prefixed_commits.items()
                    ),
                ),
            ),
            mdb.fetch_all(
                select(User.login, User.node_id).where(
                    User.acc_id.in_(meta_ids), User.login.in_(unique_authors),
                ),
            ),
            *(get_installation_url_prefix(meta_id, mdb, request.cache) for meta_id in meta_ids),
        )
        url_prefixes = dict(zip(meta_ids, url_prefixes))
        resolved_prefixed_commits = {}
        resolved_full_commits = {}
        for row in commit_rows:
            commit, repo = row[PushCommit.sha.name], row[PushCommit.repository_full_name.name]
            resolved_prefixed_commits[(commit[:7], repo)] = resolved_full_commits[
                (commit, repo)
            ] = row
        resolved_users = {}
        for row in user_rows:
            resolved_users[row[User.login.name]] = row[User.node_id.name]
        return (
            resolved_full_commits,
            resolved_prefixed_commits,
            checker.installed_repos,
            resolved_users,
            url_prefixes,
            meta_ids,
        )

    user, (
        resolved_full_commits,
        resolved_prefixed_commits,
        installed_repos,
        resolved_users,
        url_prefixes,
        meta_ids,
    ) = await gather(request.user(), main_flow())

    if None in authors:
        resolved_users[None] = await mdb.fetch_val(
            select(User.node_id).where(User.acc_id.in_(meta_ids), User.login == user.login),
        )
    inserted = []
    repos = set()
    now = datetime.now(timezone.utc)
    empty_resolved = {
        PushCommit.acc_id.name: None,
        PushCommit.sha.name: None,
        PushCommit.node_id.name: None,
    }
    for i, (n, author, status) in enumerate(zip(notifications, authors, statuses)):
        if status == ReleaseNotificationStatus.IGNORED_DUPLICATE:
            log.warning("ignored %s", n.to_dict())
            continue
        repos.add(repo := n.repository.split("/", 1)[1])
        try:
            resolved_commits = (
                resolved_full_commits if len(n.commit) == 40 else resolved_prefixed_commits
            )[(n.commit, repo)]
            statuses[i] = ReleaseNotificationStatus.ACCEPTED_RESOLVED
        except KeyError:
            resolved_commits = empty_resolved
            statuses[i] = ReleaseNotificationStatus.ACCEPTED_PENDING
        inserted.append(
            ReleaseNotification(
                account_id=account,
                repository_node_id=installed_repos[repo],
                commit_hash_prefix=(prefix := resolved_commits[PushCommit.sha.name] or n.commit),
                resolved_commit_hash=(sha := resolved_commits[PushCommit.sha.name]),
                resolved_commit_node_id=(cid := resolved_commits[PushCommit.node_id.name]),
                resolved_at=now if cid is not None else None,
                name=n.name or f"{repo}@{prefix}",
                author_node_id=resolved_users.get(author),
                url=n.url
                or (
                    compose_commit_url(
                        url_prefixes[resolved_commits[PushCommit.acc_id.name]], repo, sha,
                    )
                    if sha is not None
                    else None
                ),
                published_at=n.published_at,
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        )
    sql = (await dialect_specific_insert(rdb))(ReleaseNotification)
    sql = sql.on_conflict_do_update(
        index_elements=ReleaseNotification.__table__.primary_key.columns,
        set_={
            ReleaseNotification.name.name: sql.excluded.name,
            ReleaseNotification.author_node_id.name: sql.excluded.author_node_id,
            ReleaseNotification.url.name: sql.excluded.url,
            ReleaseNotification.published_at.name: sql.excluded.published_at,
            ReleaseNotification.updated_at.name: sql.excluded.updated_at,
        },
    )
    if rdb.url.dialect == "sqlite":
        async with rdb.connection() as perdata_conn:
            async with perdata_conn.transaction():
                await perdata_conn.execute_many(sql, inserted)
    else:
        # don't require a transaction in Postgres, executemany() is atomic in new asyncpg
        await rdb.execute_many(sql, inserted)
    if (slack := request.app["slack"]) is not None:

        async def report_new_release_event_to_slack():
            await slack.post_event("new_release_event.jinja2", account=account, repos=repos)

        await defer(report_new_release_event_to_slack(), "report_new_release_event_to_slack")
    return model_response(statuses)


@sentry_span
async def _drop_precomputed_deployments(
    account: int,
    repos: Collection[str],
    prefixer: Prefixer,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    branches: pd.DataFrame,
    default_branches: dict[str, str],
    request: AthenianWebRequest,
    meta_ids: tuple[int, ...],
) -> None:
    pdb, rdb = request.pdb, request.rdb
    repo_name_to_node = prefixer.repo_name_to_node.get
    repo_node_ids = [repo_name_to_node(r, 0) for r in repos]
    deployments_to_kill = await rdb.fetch_all(
        select(distinct(DeployedComponent.deployment_name)).where(
            DeployedComponent.account_id == account,
            DeployedComponent.repository_node_id.in_(repo_node_ids),
        ),
    )
    deployments_to_kill = [r[0] for r in deployments_to_kill]
    await gather(
        pdb.execute(
            delete(GitHubDeploymentFacts).where(
                GitHubDeploymentFacts.acc_id == account,
                GitHubDeploymentFacts.deployment_name.in_(deployments_to_kill),
            ),
        ),
        pdb.execute(
            delete(GitHubReleaseDeployment).where(
                GitHubReleaseDeployment.acc_id == account,
                GitHubReleaseDeployment.deployment_name.in_(deployments_to_kill),
            ),
        ),
        pdb.execute(
            delete(GitHubPullRequestDeployment).where(
                GitHubPullRequestDeployment.acc_id == account,
                GitHubPullRequestDeployment.deployment_name.in_(deployments_to_kill),
            ),
        ),
        pdb.execute(
            delete(GitHubCommitDeployment).where(
                GitHubCommitDeployment.acc_id == account,
                GitHubCommitDeployment.deployment_name.in_(deployments_to_kill),
            ),
        ),
        op="remove %d deployments from pdb" % len(deployments_to_kill),
    )
    today = datetime.now(timezone.utc)
    await mine_deployments(
        repos,
        {},
        today - timedelta(days=730),
        today,
        [],
        [],
        {},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_settings,
        logical_settings,
        branches,
        default_branches,
        prefixer,
        account,
        None,
        meta_ids,
        request.mdb,
        pdb,
        rdb,
        None,
    )


@sentry_span
async def _drop_precomputed_event_releases(
    account: int,
    repos: Collection[str],
    prefixer: Prefixer,
    release_settings: ReleaseSettings,
    logical_settings: LogicalRepositorySettings,
    branches: pd.DataFrame,
    default_branches: dict[str, str],
    request: AthenianWebRequest,
    meta_ids: tuple[int, ...],
) -> None:
    pdb = request.pdb
    bots_task = asyncio.create_task(
        bots(account, meta_ids, request.mdb, request.sdb, request.cache),
        name="_drop_precomputed_event_releases/bots",
    )
    await gather(
        *(
            pdb.execute(
                delete(table).where(
                    table.release_match == ReleaseMatch.event.name,
                    table.repository_full_name.in_(repos),
                    table.acc_id == account,
                ),
            )
            for table in (
                GitHubDonePullRequestFacts,
                GitHubMergedPullRequestFacts,
                GitHubReleaseFacts,
            )
        ),
        op="delete precomputed releases",
    )

    # preheat these repos
    mdb, pdb, rdb = request.mdb, request.pdb, request.rdb
    time_to = datetime.combine(
        date.today() + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc,
    )
    no_time_from = datetime(1970, 1, 1, tzinfo=timezone.utc)
    time_from = time_to - timedelta(days=365 * 2)
    await mine_releases(
        repos,
        {},
        branches,
        default_branches,
        no_time_from,
        time_to,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_settings,
        logical_settings,
        prefixer,
        account,
        meta_ids,
        mdb,
        pdb,
        rdb,
        None,
        force_fresh=True,
        with_avatars=False,
        with_deployments=False,
        with_extended_pr_details=False,
    )
    await gather(wait_deferred(), bots_task)
    await PRFactsCalculator(account, meta_ids, mdb, pdb, rdb, cache=None)(
        time_from,
        time_to,
        set(repos),
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        bots_task.result(),
        release_settings,
        logical_settings,
        prefixer,
        True,
        0,
    )


droppers = {
    "release": _drop_precomputed_event_releases,
    "deployment": _drop_precomputed_deployments,
}


@disable_default_user
@weight(10)
async def clear_precomputed_events(request: AthenianWebRequest, body: dict) -> web.Response:
    """Reset the precomputed data related to the pushed events."""
    launch_defer_from_request(request, detached=True)  # DEV-2798
    model = DeleteEventsCacheRequest.from_dict(body)

    meta_ids = await get_metadata_account_ids(model.account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_request(request, model.account, prefixer)
    (prefixed_repos, _), logical_settings = await gather(
        resolve_repos_with_request(
            model.repositories,
            model.account,
            request,
            meta_ids=meta_ids,
            prefixer=prefixer,
            pointer=".repositories",
        ),
        settings.list_logical_repositories(),
    )
    repos = [r.unprefixed for r in prefixed_repos]
    (branches, default_branches), release_settings = await gather(
        BranchMiner.load_branches(repos, prefixer, meta_ids, request.mdb, request.cache),
        settings.list_release_matches([str(r) for r in prefixed_repos]),
    )
    tasks = [
        droppers[t](
            model.account,
            repos,
            prefixer,
            release_settings,
            logical_settings,
            branches,
            default_branches,
            request,
            meta_ids,
        )
        for t in model.targets
    ]
    await gather(*tasks, op="clear_precomputed_events/gather drops")
    await wait_deferred(final=True)
    return web.json_response({})


@disable_default_user
@weight(0)
async def notify_deployments(request: AthenianWebRequest, body: list[dict]) -> web.Response:
    """Notify about new deployments."""
    # account is automatically checked at this point
    log = logging.getLogger(f"{metadata.__package__}.notify_deployments")
    log.info("%s", body)
    try:
        notifications = [WebDeploymentNotification.from_dict(n) for n in body]
    except ParseError as e:
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e
    account, sdb, mdb, rdb, cache = (
        request.account,
        request.sdb,
        request.mdb,
        request.rdb,
        request.cache,
    )
    meta_ids = await get_metadata_account_ids(account, sdb, cache)
    checker = await access_classes["github.com"](account, meta_ids, sdb, mdb, cache).load()
    for i, notification in enumerate(notifications):
        notification.validate_timestamps()
        try:
            deployed_repos = {c.repository.split("/", 1)[1] for c in notification.components}
        except IndexError:
            raise ResponseError(
                InvalidRequestError(
                    f"[{i}].components",
                    "repository names must be prefixed (missing `github.com/`?):"
                    f" {', '.join(c.repository for c in notification.components if len(c.repository.split('/', 1)) < 2)}",  # noqa
                ),
            ) from None
        if denied_repos := await checker.check(deployed_repos):
            raise ResponseError(
                ForbiddenError(
                    f"Access denied to some repositories in [{i}].components: {denied_repos}",
                ),
            )
        if notification.name is None:
            notification.name = _compose_name(notification)
    resolved = await _resolve_references(
        chain.from_iterable(
            ((c.repository, c.reference) for c in n.components) for n in notifications
        ),
        meta_ids,
        mdb,
        False,
    )
    log.info(
        "persisting %d deployments with %d resolved references", len(notifications), len(resolved),
    )
    tasks = [
        _notify_deployment(notification, account, rdb, resolved, checker.installed_repos)
        for notification in notifications
    ]
    try:
        await gather(*tasks, op=f"_notify_deployment({len(tasks)})")
    except (sqlite3.IntegrityError, asyncpg.IntegrityConstraintViolationError):
        raise ResponseError(DatabaseConflict("Specified deployment(s) already exist.")) from None
    return web.json_response({})


def _normalize_reference(ref: str) -> str:
    if len(ref) == 40:
        ref = ref[:7]
    if not ref.startswith("v"):
        ref = "v" + ref
    return ref


b70chars = "".join(chr(i) for i in range(ord("a"), ord("z") + 1))
b70chars = "0123456789" + b70chars + b70chars.upper() + "@#$%&/+="
assert len(b70chars) == 70


def _compose_name(notification: WebDeploymentNotification) -> str:
    components = sorted(notification.components, key=lambda c: c.repository)
    text = "|".join(f"{c.repository}-{_normalize_reference(c.reference)}" for c in components)
    chash = xxh32_intdigest(text)
    ts = notification.date_started
    secs = int((ts - ts.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds())
    full_hash = (chash << 17) | secs  # 24 * 3600 requires 17 bits
    str_hash = ""
    chars = b70chars
    for _ in range(8):  # 70^8 > 2^(32+17)
        str_hash += chars[full_hash % len(chars)]
        full_hash //= len(chars)
    assert full_hash == 0
    env = notification.environment
    for key, repl in (("production", "prod"), ("staging", "stage"), ("development", "dev")):
        env = env.replace(key, repl)
    return "%s-%d-%02d-%02d-%s" % (env, ts.year, ts.month, ts.day, str_hash)


@sentry_span
async def _resolve_references(
    components: Iterable[tuple[Union[int, str], str]],
    meta_ids: tuple[int, ...],
    mdb: Database,
    repository_node_ids: bool,
) -> dict[str, dict[str, str]]:
    releases = defaultdict(set)
    full_commits = set()
    prefix_commits = defaultdict(set)
    for repository, reference in components:
        repo = repository if repository_node_ids else repository.split("/", 1)[1]
        if len(reference) == 7:
            prefix_commits[repo].add(reference)
        if len(reference) == 40:
            full_commits.add(reference)
        releases[repo].add(reference)
        if reference.startswith("v"):
            releases[repo].add(reference[1:])
        else:
            releases[repo].add("v" + reference)

    def select_repo(model):
        return model.repository_node_id if repository_node_ids else model.repository_full_name

    def filter_repo(model, repo):
        return (
            model.repository_node_id == repo
            if repository_node_ids
            else model.repository_full_name == repo
        )

    queries = [
        select(
            [text("'commit_f'"), PushCommit.sha, PushCommit.node_id, select_repo(PushCommit)],
        ).where(PushCommit.acc_id.in_(meta_ids), PushCommit.sha.in_(full_commits)),
        *(
            select(
                [text("'commit_p'"), PushCommit.sha, PushCommit.node_id, select_repo(PushCommit)],
            ).where(
                and_(
                    PushCommit.acc_id.in_(meta_ids),
                    filter_repo(PushCommit, repo),
                    func.substr(PushCommit.sha, 1, 7).in_(prefixes),
                ),
            )
            for repo, prefixes in prefix_commits.items()
        ),
        *(
            select(
                [text("'release'"), Release.name, Release.commit_id, select_repo(Release)],
            ).where(
                and_(
                    Release.acc_id.in_(meta_ids),
                    filter_repo(Release, repo),
                    Release.name.in_(names),
                ),
            )
            for repo, names in releases.items()
        ),
    ]
    rows = await mdb.fetch_all(union_all(*queries))
    # order: first full commits, then commit prefixes, then releases
    rows = sorted(rows, key=lambda row: row[0])
    result = defaultdict(dict)
    for row in rows:
        type_id, ref, commit, repo = row[0], row[1], row[2], row[3]
        if type_id == "commit_f":
            result[repo][ref] = commit
        elif type_id == "commit_p":
            result[repo][ref[:7]] = commit
        else:
            result[repo][ref] = commit
            if not ref.startswith("v"):
                result[repo]["v" + ref] = commit
            else:
                result[repo][ref[1:]] = commit
    return result


@sentry_span
async def _notify_deployment(
    notification: WebDeploymentNotification,
    account: int,
    rdb: Database,
    resolved_refs: Mapping[str, Mapping[str, str]],
    repo_nodes: Mapping[str, int],
) -> None:
    async with rdb.connection() as rdb_conn:
        async with rdb_conn.transaction():
            await rdb_conn.execute(
                insert(DeploymentNotification).values(
                    DeploymentNotification(
                        account_id=account,
                        conclusion=notification.conclusion,
                        environment=notification.environment,
                        name=notification.name,
                        url=notification.url,
                        started_at=notification.date_started,
                        finished_at=notification.date_finished,
                    )
                    .create_defaults()
                    .explode(with_primary_keys=True),
                ),
            )
            now = datetime.now(timezone.utc)
            ed = {}  # empty dict
            cvalues = [
                DeployedComponent(
                    account_id=account,
                    deployment_name=notification.name,
                    repository_node_id=repo_nodes[(repo := c.repository.split("/", 1)[1])],
                    reference=c.reference,
                    resolved_commit_node_id=(cid := resolved_refs.get(repo, ed).get(c.reference)),
                    resolved_at=now if cid is not None else None,
                )
                .create_defaults()
                .explode(with_primary_keys=True)
                for c in notification.components
            ]
            await rdb_conn.execute_many(insert(DeployedComponent), cvalues)
            if notification.labels:
                lvalues = [
                    DeployedLabel(
                        account_id=account,
                        deployment_name=notification.name,
                        key=str(key),
                        value=value,
                    )
                    .create_defaults()
                    .explode(with_primary_keys=True)
                    for key, value in notification.labels.items()
                ]
                await rdb_conn.execute_many(insert(DeployedLabel), lvalues)


async def resolve_deployed_component_references(
    sdb: Database,
    mdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> None:
    """Resolve the missing deployed component references and remove stale deployments."""
    async with rdb.connection() as rdb_conn:
        async with rdb_conn.transaction():
            return await _resolve_deployed_component_references(sdb, mdb, rdb_conn, cache)


async def _resolve_deployed_component_references(
    sdb: Database,
    mdb: Database,
    rdb: Connection,
    cache: Optional[aiomcache.Client],
) -> None:
    await rdb.execute(
        delete(DeployedComponent).where(
            DeployedComponent.resolved_commit_node_id.is_(None),
            DeployedComponent.created_at < datetime.now(timezone.utc) - timedelta(days=14),
        ),
    )
    discarded_notifications = await rdb.fetch_all(
        select(DeploymentNotification.account_id, DeploymentNotification.name).where(
            not_(
                exists().where(
                    DeploymentNotification.account_id == DeployedComponent.account_id,
                    DeploymentNotification.name == DeployedComponent.deployment_name,
                ),
            ),
        ),
    )
    discarded_by_account = defaultdict(list)
    for row in discarded_notifications:
        discarded_by_account[row[DeploymentNotification.account_id.name]].append(
            row[DeploymentNotification.name.name],
        )
    del discarded_notifications
    tasks = [
        rdb.execute(
            delete(DeployedLabel).where(
                DeployedLabel.account_id == acc,
                DeployedLabel.deployment_name.in_(discarded),
            ),
        )
        for acc, discarded in discarded_by_account.items()
    ] + [
        rdb.execute(
            delete(DeployedComponent).where(
                DeployedComponent.account_id == acc,
                DeployedComponent.deployment_name.in_(discarded),
            ),
        )
        for acc, discarded in discarded_by_account.items()
    ]
    await gather(*tasks, op="resolve_deployed_component_references/delete/dependencies")
    tasks = [
        rdb.execute(
            delete(DeploymentNotification).where(
                and_(
                    DeploymentNotification.account_id == acc,
                    DeploymentNotification.name.in_(discarded),
                ),
            ),
        )
        for acc, discarded in discarded_by_account.items()
    ]
    await gather(*tasks, op="resolve_deployed_component_references/delete/notifications")
    unresolved = await rdb.fetch_all(
        select(
            DeployedComponent.account_id,
            DeployedComponent.repository_node_id,
            DeployedComponent.reference,
        ).where(DeployedComponent.resolved_commit_node_id.is_(None)),
    )
    unresolved_by_account = defaultdict(list)
    for row in unresolved:
        unresolved_by_account[row[DeployedComponent.account_id.name]].append(row)
    del unresolved
    for account, unresolved in unresolved_by_account.items():
        meta_ids = await get_metadata_account_ids(account, sdb, cache)
        resolved = await _resolve_references(
            [
                (r[DeployedComponent.repository_node_id.name], r[DeployedComponent.reference.name])
                for r in unresolved
            ],
            meta_ids,
            mdb,
            True,
        )
        updated = set()
        for row in unresolved:
            repo = row[DeployedComponent.repository_node_id.name]
            ref = row[DeployedComponent.reference.name]
            try:
                rr = resolved[repo][ref]
            except KeyError:
                continue
            updated.add((repo, ref, rr))
        now = datetime.now(timezone.utc)
        tasks = [
            rdb.execute(
                update(DeployedComponent)
                .where(
                    DeployedComponent.account_id == account,
                    DeployedComponent.repository_node_id == u[0],
                    DeployedComponent.reference == u[1],
                )
                .values(
                    {
                        DeployedComponent.resolved_commit_node_id: u[2],
                        DeployedComponent.resolved_at: now,
                    },
                ),
            )
            for u in updated
        ]
        await gather(*tasks, op=f"resolve_deployed_component_references/update/{account}")
