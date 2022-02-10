from collections import defaultdict
from datetime import datetime, timedelta, timezone
import pickle
from typing import Collection, Dict, List, Optional, Tuple

import aiomcache
import pandas as pd
from sqlalchemy import and_, func, join, select, union_all

from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import cached, middle_term_exptime, short_term_exptime
from athenian.api.controllers.logical_repos import drop_logical_repo
from athenian.api.controllers.miners.github.logical import split_logical_deployed_components
from athenian.api.controllers.miners.types import DeployedComponent as DeployedComponentDC, \
    Deployment, DeploymentConclusion, Environment
from athenian.api.controllers.prefixer import Prefixer
from athenian.api.controllers.settings import LogicalRepositorySettings
from athenian.api.db import Database
from athenian.api.models.metadata.github import NodeCommit
from athenian.api.models.persistentdata.models import DeployedComponent, DeployedLabel, \
    DeploymentNotification
from athenian.api.tracing import sentry_span


@sentry_span
@cached(
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda names, logical_settings, **_: (",".join(sorted(names)), logical_settings),
    exptime=middle_term_exptime,
    refresh_on_access=True,
)
async def load_included_deployments(names: Collection[str],
                                    logical_settings: LogicalRepositorySettings,
                                    prefixer: Prefixer,
                                    account: int,
                                    meta_ids: Tuple[int, ...],
                                    mdb: Database,
                                    rdb: Database,
                                    cache: Optional[aiomcache.Client],
                                    ) -> Dict[str, Deployment]:
    """
    Fetch brief details about the deployments.

    Compared to `mine_deployments()`, this is much more lightweight and is intended for `include`.
    """
    notifications, components, labels = await gather(
        read_sql_query(
            select([DeploymentNotification])
            .where(and_(DeploymentNotification.account_id == account,
                        DeploymentNotification.name.in_any_values(names))),
            rdb, DeploymentNotification,
        ),
        read_sql_query(
            select([DeployedComponent])
            .where(and_(DeployedComponent.account_id == account,
                        DeployedComponent.deployment_name.in_any_values(names))),
            rdb, DeployedComponent,
        ),
        read_sql_query(
            select([DeployedLabel])
            .where(and_(DeployedLabel.account_id == account,
                        DeployedLabel.deployment_name.in_any_values(names))),
            rdb, DeployedLabel,
        ),
    )
    repo_node_to_name = prefixer.repo_node_to_name.get
    components[DeployedComponent.repository_full_name] = [
        repo_node_to_name(r) for r in components[DeployedComponent.repository_node_id.name].values
    ]
    commit_ids = components[DeployedComponent.resolved_commit_node_id.name].unique()
    if len(commit_ids) and commit_ids[0] is None:
        commit_ids = commit_ids[1:]
    hashes = await mdb.fetch_all(
        select([NodeCommit.sha, NodeCommit.graph_id])
        .where(and_(NodeCommit.acc_id.in_(meta_ids),
                    NodeCommit.graph_id.in_any_values(commit_ids))))
    hashes = {r[NodeCommit.graph_id.name]: r[NodeCommit.sha.name] for r in hashes}
    labels = group_deployed_labels_df(labels)
    labels_by_dep = {
        name: dict(zip(keyvals[DeployedLabel.key.name].values,
                       keyvals[DeployedLabel.value.name].values))
        for name, keyvals in zip(labels.index.values, labels["labels"].values)
    }
    components = split_logical_deployed_components(
        notifications, labels, components,
        logical_settings.with_logical_deployments([]), logical_settings,
    )
    comps_by_dep = {}
    for name, repo, ref, commit_id in zip(
            components[DeployedComponent.deployment_name.name].values,
            components[DeployedComponent.repository_full_name].values,
            components[DeployedComponent.reference.name].values,
            components[DeployedComponent.resolved_commit_node_id.name].values):
        comps_by_dep.setdefault(name, []).append(
            DeployedComponentDC(
                repository_full_name=repo,
                reference=ref,
                sha=hashes.get(commit_id)))
    return {
        name: Deployment(
            name=name,
            conclusion=DeploymentConclusion[conclusion],
            environment=env,
            url=url,
            started_at=started_at,
            finished_at=finished_at,
            components=comps_by_dep.get(name, []),
            labels=labels_by_dep.get(name, None),
        )
        for name, conclusion, env, url, started_at, finished_at in zip(
            notifications[DeploymentNotification.name.name].values,
            notifications[DeploymentNotification.conclusion.name].values,
            notifications[DeploymentNotification.environment.name].values,
            notifications[DeploymentNotification.url.name].values,
            notifications[DeploymentNotification.started_at.name],
            notifications[DeploymentNotification.finished_at.name],
        )
    }


repository_environment_threshold = timedelta(days=60)


@sentry_span
@cached(
    exptime=middle_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, **_: (",".join(sorted(repos)),),
)
async def fetch_repository_environments(repos: Collection[str],
                                        prefixer: Prefixer,
                                        account: int,
                                        rdb: Database,
                                        cache: Optional[aiomcache.Client],
                                        ) -> Dict[str, List[str]]:
    """Map environments to repositories that have deployed there."""
    repo_name_to_node_get = prefixer.repo_name_to_node.get
    repo_ids = {repo_name_to_node_get(r) for r in repos} - {None}
    rows = await rdb.fetch_all(
        select([DeployedComponent.repository_node_id, DeploymentNotification.environment])
        .select_from(join(DeployedComponent, DeploymentNotification, and_(
            DeployedComponent.account_id == DeploymentNotification.account_id,
            DeployedComponent.deployment_name == DeploymentNotification.name,
        )))
        .where(and_(
            DeployedComponent.account_id == account,
            DeployedComponent.repository_node_id.in_(repo_ids),
            DeploymentNotification.finished_at >
            datetime.now(timezone.utc) - repository_environment_threshold,
        ))
        .group_by(DeployedComponent.repository_node_id, DeploymentNotification.environment)
        .distinct(),
    )
    result = defaultdict(list)
    repo_node_to_name = prefixer.repo_node_to_name.__getitem__
    for row in rows:
        repo_id = row[DeployedComponent.repository_node_id.name]
        env = row[DeploymentNotification.environment.name]
        result[env].append(repo_node_to_name(repo_id))
    return result


def group_deployed_labels_df(df: pd.DataFrame) -> pd.DataFrame:
    """Group the DataFrame with key-value labels by deployment name."""
    groups = list(df.groupby(DeployedLabel.deployment_name.name, sort=False))
    grouped_labels = pd.DataFrame({
        "deployment_name": [g[0] for g in groups],
        "labels": [g[1] for g in groups],
    })
    for df in grouped_labels["labels"].values:
        df.reset_index(drop=True, inplace=True)
    grouped_labels.set_index("deployment_name", drop=True, inplace=True)
    return grouped_labels


class NoDeploymentNotificationsError(Exception):
    """Indicate 0 deployment notifications for the account in total."""


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda repos, time_from, time_to, **_: (
        ",".join(sorted(repos if repos else [])),
        time_from.timestamp(),
        time_to.timestamp(),
    ),
)
async def mine_environments(repos: Optional[List[str]],
                            time_from: datetime,
                            time_to: datetime,
                            prefixer: Prefixer,
                            account: int,
                            rdb: Database,
                            cache: Optional[aiomcache.Client],
                            ) -> List[Environment]:
    """
    Fetch unique deployment environments according to the filters.

    The output should be sorted by environment name.
    """
    filters = [
        DeploymentNotification.account_id == account,
        DeploymentNotification.started_at >= time_from,
        DeploymentNotification.finished_at < time_to,
    ]
    if repos:
        repo_name_to_node = prefixer.repo_name_to_node.__getitem__
        repo_node_ids = {repo_name_to_node(drop_logical_repo(r)) for r in repos}
        core = join(DeploymentNotification, DeployedComponent, and_(
            DeploymentNotification.account_id == DeployedComponent.account_id,
            DeploymentNotification.name == DeployedComponent.deployment_name,
        ))
        filters.append(DeployedComponent.repository_node_id.in_(repo_node_ids))
    else:
        core = DeploymentNotification
    query = (
        select([
            DeploymentNotification.environment,
            func.count(DeploymentNotification.name).label("deployments_count"),
            func.max(DeploymentNotification.finished_at).label("latest_finished_at"),
        ])
        .select_from(core)
        .where(and_(*filters))
        .group_by(DeploymentNotification.environment)
        .order_by(DeploymentNotification.environment)
    )
    rows = await rdb.fetch_all(query)
    envs = {r[0]: r for r in rows}
    if not envs:
        has_notifications = await rdb.fetch_val(
            select([func.count(DeploymentNotification.name)])
            .where(DeploymentNotification.account_id == account))
        if not has_notifications:
            raise NoDeploymentNotificationsError()
    queries = [
        select([
            DeploymentNotification.environment,
            DeploymentNotification.conclusion,
        ]).where(and_(
            DeploymentNotification.account_id == account,
            DeploymentNotification.environment == env,
            DeploymentNotification.finished_at == envs[env]["latest_finished_at"],
        ))
        for env in envs
    ]
    query = union_all(*queries) if len(queries) > 1 else queries[0]
    rows = await rdb.fetch_all(query)
    conclusions = {r[0]: r[1] for r in rows}
    return [Environment(name=key,
                        deployments_count=val["deployments_count"],
                        last_conclusion=conclusions[key])
            for key, val in envs.items()]
