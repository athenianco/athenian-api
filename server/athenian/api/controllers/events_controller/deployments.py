from collections import defaultdict
from datetime import datetime, timedelta, timezone
from itertools import chain
import logging
import sqlite3
from typing import Iterable, Mapping

from aiohttp import web
import aiomcache
import asyncpg
import sqlalchemy as sa
from xxhash._xxhash import xxh32_intdigest

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.auth import disable_default_user
from athenian.api.balancing import weight
from athenian.api.db import Connection, Database
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.miners.access_classes import access_classes
from athenian.api.internal.refetcher import Refetcher
from athenian.api.models.metadata.github import NodeRepository, PushCommit, Release
from athenian.api.models.persistentdata.models import (
    DeployedComponent,
    DeployedLabel,
    DeploymentNotification,
)
from athenian.api.models.web import (
    DatabaseConflict,
    DeploymentNotification as WebDeploymentNotification,
    ForbiddenError,
    InvalidRequestError,
    NotifiedDeployment,
    NotifyDeploymentsResponse,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response
from athenian.api.serialization import ParseError
from athenian.api.tracing import sentry_span


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
    resolved, _ = await _resolve_references(
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
        notifications_resolved = await gather(*tasks, op=f"_notify_deployment({len(tasks)})")
    except (sqlite3.IntegrityError, asyncpg.IntegrityConstraintViolationError):
        raise ResponseError(DatabaseConflict("Specified deployment(s) already exist.")) from None

    notified_deployments = [
        NotifiedDeployment(name=notification.name, resolved=resolved)
        for notification, resolved in zip(notifications, notifications_resolved)
    ]

    return model_response(NotifyDeploymentsResponse(deployments=notified_deployments))


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


def _normalize_reference(ref: str) -> str:
    if len(ref) == 40:
        ref = ref[:7]
    if not ref.startswith("v"):
        ref = "v" + ref
    return ref


@sentry_span
async def _resolve_references(
    components: Iterable[tuple[int | str, str]],
    meta_ids: tuple[int, ...],
    mdb: Database,
    repository_node_ids: bool,
) -> tuple[dict[str, dict[str, str]], dict[str, set[str]]]:
    releases = defaultdict(set)
    full_commits = defaultdict(set)
    prefix_commits = defaultdict(set)
    for repository, reference in components:
        repo = repository if repository_node_ids else repository.split("/", 1)[1]
        if len(reference) == 7:
            prefix_commits[repo].add(reference)
        if len(reference) == 40:
            full_commits[repo].add(reference)
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
        sa.select(
            sa.text("'commit_f'"), PushCommit.sha, PushCommit.node_id, select_repo(PushCommit),
        ).where(
            PushCommit.acc_id.in_(meta_ids),
            PushCommit.sha.in_(list(chain.from_iterable(full_commits.values()))),
        ),
        *(
            sa.select(
                sa.text("'commit_p'"), PushCommit.sha, PushCommit.node_id, select_repo(PushCommit),
            ).where(
                PushCommit.acc_id.in_(meta_ids),
                filter_repo(PushCommit, repo),
                sa.func.substr(PushCommit.sha, 1, 7).in_(prefixes),
            )
            for repo, prefixes in prefix_commits.items()
        ),
        *(
            sa.select(
                sa.text("'release'"), Release.name, Release.commit_id, select_repo(Release),
            ).where(
                Release.acc_id.in_(meta_ids),
                filter_repo(Release, repo),
                Release.name.in_(names),
            )
            for repo, names in releases.items()
        ),
    ]
    rows = await mdb.fetch_all(sa.union_all(*queries))
    # order: first full commits, then commit prefixes, then releases
    rows = sorted(rows, key=lambda row: row[0])
    result = defaultdict(dict)
    for row in rows:
        type_id, ref, commit, repo = row[0], row[1], row[2], row[3]
        if type_id == "commit_f":
            result[repo][ref] = commit
            (acc_full_commits := full_commits[repo]).discard(ref)
            if not acc_full_commits:
                del full_commits[repo]
        elif type_id == "commit_p":
            result[repo][ref[:7]] = commit
        else:
            result[repo][ref] = commit
            if not ref.startswith("v"):
                result[repo]["v" + ref] = commit
            else:
                result[repo][ref[1:]] = commit
    return result, full_commits


@sentry_span
async def _notify_deployment(
    notification: WebDeploymentNotification,
    account: int,
    rdb: Database,
    resolved_refs: Mapping[str, Mapping[str, str]],
    repo_nodes: Mapping[str, int],
) -> bool:
    async with rdb.connection() as rdb_conn:
        async with rdb_conn.transaction():
            await rdb_conn.execute(
                sa.insert(DeploymentNotification).values(
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
            ed: dict[str, int] = {}  # empty dict
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
            all_resolved = all(component["resolved_at"] is not None for component in cvalues)
            await rdb_conn.execute_many(sa.insert(DeployedComponent), cvalues)
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
                await rdb_conn.execute_many(sa.insert(DeployedLabel), lvalues)

    return all_resolved


async def resolve_deployed_component_references(
    refetcher: Refetcher,
    sdb: Database,
    mdb: Database,
    rdb: Database,
    cache: aiomcache.Client | None,
) -> None:
    """Resolve the missing deployed component references and remove stale deployments."""
    async with rdb.connection() as rdb_conn:
        async with rdb_conn.transaction():
            return await _resolve_deployed_component_references(
                refetcher, sdb, mdb, rdb_conn, cache,
            )


async def _resolve_deployed_component_references(
    refetcher: Refetcher,
    sdb: Database,
    mdb: Database,
    rdb: Connection,
    cache: aiomcache.Client | None,
) -> None:
    await rdb.execute(
        sa.delete(DeployedComponent).where(
            DeployedComponent.resolved_commit_node_id.is_(None),
            DeployedComponent.created_at < datetime.now(timezone.utc) - timedelta(days=14),
        ),
    )
    discarded_notifications = await rdb.fetch_all(
        sa.select(DeploymentNotification.account_id, DeploymentNotification.name).where(
            sa.not_(
                sa.exists().where(
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
            sa.delete(DeployedLabel).where(
                DeployedLabel.account_id == acc,
                DeployedLabel.deployment_name.in_(discarded),
            ),
        )
        for acc, discarded in discarded_by_account.items()
    ] + [
        rdb.execute(
            sa.delete(DeployedComponent).where(
                DeployedComponent.account_id == acc,
                DeployedComponent.deployment_name.in_(discarded),
            ),
        )
        for acc, discarded in discarded_by_account.items()
    ]
    await gather(*tasks, op="resolve_deployed_component_references/delete/dependencies")
    tasks = [
        rdb.execute(
            sa.delete(DeploymentNotification).where(
                DeploymentNotification.account_id == acc,
                DeploymentNotification.name.in_(discarded),
            ),
        )
        for acc, discarded in discarded_by_account.items()
    ]
    await gather(*tasks, op="resolve_deployed_component_references/delete/notifications")
    unresolved = await rdb.fetch_all(
        sa.select(
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
        resolved, unresolved_commits = await _resolve_references(
            [
                (r[DeployedComponent.repository_node_id.name], r[DeployedComponent.reference.name])
                for r in unresolved
            ],
            meta_ids,
            mdb,
            True,
        )
        if unresolved_commits:
            repo_id_map = dict(
                await mdb.fetch_all(
                    sa.select(NodeRepository.node_id, NodeRepository.database_id).where(
                        NodeRepository.acc_id.in_(meta_ids),
                        NodeRepository.node_id.in_(unresolved_commits),
                    ),
                ),
            )
            await refetcher.specialize(meta_ids).submit_commit_hashes(
                [
                    (repo_id_map[repo], h)
                    for repo, hashes in unresolved_commits.items()
                    if repo in repo_id_map
                    for h in hashes
                ],
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
                sa.update(DeployedComponent)
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
