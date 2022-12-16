from datetime import datetime, timedelta, timezone
import pickle
from typing import Any, Collection, Dict, List, Mapping, Optional, Tuple

import aiomcache
import numpy as np
import pandas as pd
from sqlalchemy import and_, desc, exists, join, not_, or_, select

from athenian.api.async_utils import gather, read_sql_query
from athenian.api.cache import CancelCache, cached, middle_term_exptime, short_term_exptime
from athenian.api.db import Database
from athenian.api.internal.logical_accelerated import drop_logical_repo
from athenian.api.internal.miners.github.logical import split_logical_deployed_components
from athenian.api.internal.miners.types import (
    DeployedComponent as DeployedComponentDC,
    Deployment,
    DeploymentConclusion,
    Environment,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.models.metadata.github import NodeCommit
from athenian.api.models.persistentdata.models import (
    DeployedComponent,
    DeployedLabel,
    DeploymentNotification,
)
from athenian.api.pandas_io import deserialize_args, serialize_args
from athenian.api.tracing import sentry_span
from athenian.api.unordered_unique import in1d_str, unordered_unique


@sentry_span
@cached(
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda names, logical_settings, **_: (",".join(sorted(names)), logical_settings),
    exptime=middle_term_exptime,
    refresh_on_access=True,
)
async def load_included_deployments(
    names: Collection[str],
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

    :return: Mapping from deployment name to the corresponding details.
    """
    notifications, components, labels = await gather(
        read_sql_query(
            select(DeploymentNotification).where(
                DeploymentNotification.account_id == account,
                DeploymentNotification.name.in_any_values(names),
            ),
            rdb,
            DeploymentNotification,
            index=DeploymentNotification.name.name,
        ),
        read_sql_query(
            select(DeployedComponent).where(
                DeployedComponent.account_id == account,
                DeployedComponent.deployment_name.in_any_values(names),
            ),
            rdb,
            DeployedComponent,
        ),
        read_sql_query(
            select(DeployedLabel).where(
                DeployedLabel.account_id == account,
                DeployedLabel.deployment_name.in_any_values(names),
            ),
            rdb,
            DeployedLabel,
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
        select([NodeCommit.sha, NodeCommit.graph_id]).where(
            and_(NodeCommit.acc_id.in_(meta_ids), NodeCommit.graph_id.in_any_values(commit_ids)),
        ),
    )
    hashes = {r[NodeCommit.graph_id.name]: r[NodeCommit.sha.name] for r in hashes}
    labels = group_deployed_labels_df(labels)
    labels_by_dep = {
        name: dict(
            zip(keyvals[DeployedLabel.key.name].values, keyvals[DeployedLabel.value.name].values),
        )
        for name, keyvals in zip(labels.index.values, labels["labels"].values)
    }
    components = split_logical_deployed_components(
        notifications,
        labels,
        components,
        logical_settings.with_logical_deployments([]),
        logical_settings,
    )
    comps_by_dep = {}
    for name, repo, ref, commit_id in zip(
        components[DeployedComponent.deployment_name.name].values,
        components[DeployedComponent.repository_full_name].values,
        components[DeployedComponent.reference.name].values,
        components[DeployedComponent.resolved_commit_node_id.name].values,
    ):
        comps_by_dep.setdefault(name, []).append(
            DeployedComponentDC(
                repository_full_name=repo, reference=ref, sha=hashes.get(commit_id),
            ),
        )
    return {
        name: Deployment(
            name=name,
            conclusion=DeploymentConclusion[conclusion.decode()],
            environment=env,
            url=url,
            started_at=started_at,
            finished_at=finished_at,
            components=comps_by_dep.get(name, []),
            labels=labels_by_dep.get(name, None),
        )
        for name, conclusion, env, url, started_at, finished_at in zip(
            notifications.index.values,
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
    key=lambda repos, environments, time_from=None, time_to=None, **_: (
        ",".join(sorted(repos)),
        ",".join(sorted(environments if environments is not None else [])),
        time_from.timestamp() if time_from is not None else "",
        time_to.timestamp() if time_to is not None else "",
    ),
)
async def fetch_repository_environments(
    repos: Collection[str],
    environments: Optional[Collection[str]],
    prefixer: Prefixer,
    account: int,
    rdb: Database,
    cache: Optional[aiomcache.Client],
    time_from: Optional[datetime] = None,
    time_to: Optional[datetime] = None,
) -> Dict[str, List[str]]:
    """Map environments to physical repositories that deployed there."""
    repo_name_to_node_get = prefixer.repo_name_to_node.get
    repo_ids = {repo_name_to_node_get(drop_logical_repo(r)) for r in repos} - {None}
    assert (time_from is None) == (time_to is None)
    if time_from is None:
        time_from = datetime.now(timezone.utc)
    filters = [
        DeployedComponent.account_id == account,
        DeployedComponent.repository_node_id.in_(repo_ids),
        DeploymentNotification.finished_at > time_from - repository_environment_threshold
        if time_to is None
        else DeploymentNotification.finished_at.between(
            time_from - repository_environment_threshold,
            time_to + repository_environment_threshold,
        ),
    ]
    if environments is not None:
        filters.append(DeploymentNotification.environment.in_(environments))
    rows = await rdb.fetch_all(
        select([DeployedComponent.repository_node_id, DeploymentNotification.environment])
        .select_from(
            join(
                DeployedComponent,
                DeploymentNotification,
                and_(
                    DeployedComponent.account_id == DeploymentNotification.account_id,
                    DeployedComponent.deployment_name == DeploymentNotification.name,
                ),
            ),
        )
        .where(*filters)
        .group_by(DeployedComponent.repository_node_id, DeploymentNotification.environment)
        .distinct(),
    )
    result = {}
    repo_node_to_name = prefixer.repo_node_to_name.__getitem__
    for row in rows:
        repo_id = row[DeployedComponent.repository_node_id.name]
        env = row[DeploymentNotification.environment.name]
        result.setdefault(env, []).append(repo_node_to_name(repo_id))
    if environments is not None:
        for env in environments:
            result.setdefault(env, [])
    return result


def group_deployed_labels_df(df: pd.DataFrame) -> pd.DataFrame:
    """Group the DataFrame with key-value labels by deployment name."""
    groups = list(df.groupby(DeployedLabel.deployment_name.name, sort=False))
    grouped_labels = pd.DataFrame(
        {
            "deployment_name": [g[0] for g in groups],
            "labels": [g[1] for g in groups],
        },
    )
    for df in grouped_labels["labels"].values:
        df.reset_index(drop=True, inplace=True)
    grouped_labels.set_index("deployment_name", drop=True, inplace=True)
    return grouped_labels


class NoDeploymentNotificationsError(Exception):
    """Indicate 0 deployment notifications for the account in total."""


@sentry_span
async def fetch_components_and_prune_unresolved(
    notifications: pd.DataFrame,
    prefixer: Prefixer,
    account: int,
    rdb: Database,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove deployment notifications with unresolved components. Fetch the components."""
    components = await read_sql_query(
        select(DeployedComponent).where(
            DeployedComponent.account_id == account,
            DeployedComponent.deployment_name.in_any_values(notifications.index.values),
        ),
        rdb,
        DeployedComponent,
    )
    for col in (
        DeployedComponent.account_id,
        DeployedComponent.created_at,
        DeployedComponent.resolved_at,
    ):
        del components[col.name]
    unresolved_names = unordered_unique(
        components[DeployedComponent.deployment_name.name]
        .values[components[DeployedComponent.resolved_commit_node_id.name].values == 0]
        .astype("U"),
    )
    # we drop not yet resolved notifications and rely on
    # _invalidate_precomputed_on_out_of_order_notifications() to correctly compute them
    # in the future
    notifications = notifications.take(
        np.flatnonzero(
            np.in1d(
                notifications.index.values.astype("U"),
                unresolved_names,
                assume_unique=True,
                invert=True,
            ),
        ),
    )
    components = components.take(
        np.flatnonzero(
            np.in1d(
                components[DeployedComponent.deployment_name.name].values.astype("U"),
                unresolved_names,
                assume_unique=True,
                invert=True,
            ),
        ),
    )
    components.set_index(DeployedComponent.deployment_name.name, drop=True, inplace=True)
    repo_node_to_name = prefixer.repo_node_to_name.get
    components[DeployedComponent.repository_full_name] = [
        repo_node_to_name(n, f"unidentified_{n}")
        for n in components[DeployedComponent.repository_node_id.name].values
    ]
    return notifications, components


@sentry_span
async def fetch_labels(
    names: Collection[str],
    account: int,
    rdb: Database,
) -> pd.DataFrame:
    """Fetch the labels corresponding to the deployment notifications."""
    return await read_sql_query(
        select([DeployedLabel]).where(
            and_(DeployedLabel.account_id == account, DeployedLabel.deployment_name.in_(names)),
        ),
        rdb,
        DeployedLabel,
        index=DeployedLabel.deployment_name.name,
    )


async def fetch_deployment_candidates(
    repo_node_ids: Collection[int],
    time_from: datetime,
    time_to: datetime,
    environments: Collection[str],
    conclusions: Collection[DeploymentConclusion],
    with_labels: Mapping[str, Any],
    without_labels: Mapping[str, Any],
    account: int,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> pd.DataFrame:
    """
    Load deployment notifications that satisfy the filters.

    They are sorted by finished_at in descending order.
    """
    __tracebackhide__ = True  # noqa: F841
    return (
        await _fetch_deployment_candidates(
            repo_node_ids,
            time_from,
            time_to,
            environments,
            conclusions,
            with_labels,
            without_labels,
            account,
            rdb,
            cache,
        )
    )[0]


@sentry_span
def _postprocess_fetch_deployment_candidates(
    result: tuple[pd.DataFrame, Collection[str], Collection[DeploymentConclusion]],
    environments: Collection[str],
    conclusions: Collection[DeploymentConclusion],
    **_,
) -> tuple[pd.DataFrame, Collection[str], Collection[DeploymentConclusion]]:
    df, cached_envs, cached_concls = result
    if not cached_envs or (environments and set(environments).issubset(cached_envs)):
        if environments:
            df = df.take(
                np.flatnonzero(
                    in1d_str(
                        df[DeploymentNotification.environment.name].values.astype("U"),
                        np.array(list(environments), dtype="U"),
                    ),
                ),
            )
    else:
        raise CancelCache()
    if not cached_concls or (conclusions and set(conclusions).issubset(cached_concls)):
        if conclusions:
            df = df.take(
                np.flatnonzero(
                    in1d_str(
                        df[DeploymentNotification.conclusion.name].values,
                        np.array([c.name for c in conclusions], dtype="S"),
                    ),
                ),
            )
    else:
        raise CancelCache()
    return df, environments, conclusions


@sentry_span
@cached(
    exptime=short_term_exptime,
    serialize=serialize_args,
    deserialize=deserialize_args,
    postprocess=_postprocess_fetch_deployment_candidates,
    key=lambda repo_node_ids, time_from, time_to, with_labels, without_labels, **_: (
        ",".join(map(str, repo_node_ids)),
        time_from.timestamp(),
        time_to.timestamp(),
        ",".join(f"{k}:{v}" for k, v in sorted(with_labels.items())),
        ",".join(f"{k}:{v}" for k, v in sorted(without_labels.items())),
    ),
)
async def _fetch_deployment_candidates(
    repo_node_ids: Collection[int],
    time_from: datetime,
    time_to: datetime,
    environments: Collection[str],
    conclusions: Collection[DeploymentConclusion],
    with_labels: Mapping[str, Any],
    without_labels: Mapping[str, Any],
    account: int,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> tuple[pd.DataFrame, Collection[str], Collection[DeploymentConclusion]]:
    query = select([DeploymentNotification])
    filters = [
        DeploymentNotification.account_id == account,
        DeploymentNotification.started_at < time_to,
        DeploymentNotification.finished_at >= time_from,
    ]
    if environments:
        filters.append(DeploymentNotification.environment.in_(environments))
    if conclusions:
        filters.append(DeploymentNotification.conclusion.in_([dc.name for dc in conclusions]))
    if repo_node_ids:
        filters.append(
            exists().where(
                and_(
                    DeploymentNotification.account_id == DeployedComponent.account_id,
                    DeploymentNotification.name == DeployedComponent.deployment_name,
                    DeployedComponent.repository_node_id.in_any_values(repo_node_ids),
                ),
            ),
        )
    if without_labels:
        filters.append(
            not_(
                exists().where(
                    and_(
                        DeploymentNotification.account_id == DeployedLabel.account_id,
                        DeploymentNotification.name == DeployedLabel.deployment_name,
                        DeployedLabel.key.in_([k for k, v in without_labels.items() if v is None]),
                    ),
                ),
            ),
        )
        for k, v in without_labels.items():
            if v is None:
                continue
            filters.append(
                not_(
                    exists().where(
                        and_(
                            DeploymentNotification.account_id == DeployedLabel.account_id,
                            DeploymentNotification.name == DeployedLabel.deployment_name,
                            DeployedLabel.key == k,
                            DeployedLabel.value == v,
                        ),
                    ),
                ),
            )
    if with_labels:
        filters.append(
            exists().where(
                and_(
                    DeploymentNotification.account_id == DeployedLabel.account_id,
                    DeploymentNotification.name == DeployedLabel.deployment_name,
                    or_(
                        DeployedLabel.key.in_([k for k, v in with_labels.items() if v is None]),
                        *(
                            and_(DeployedLabel.key == k, DeployedLabel.value == v)
                            for k, v in with_labels.items()
                            if v is not None
                        ),
                    ),
                ),
            ),
        )
    query = query.where(and_(*filters)).order_by(desc(DeploymentNotification.finished_at))
    notifications = await read_sql_query(
        query, rdb, DeploymentNotification, index=DeploymentNotification.name.name,
    )
    del notifications[DeploymentNotification.account_id.name]
    del notifications[DeploymentNotification.created_at.name]
    del notifications[DeploymentNotification.updated_at.name]
    return notifications, environments, conclusions


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
async def mine_environments(
    repos: Optional[List[str]],
    time_from: datetime,
    time_to: datetime,
    prefixer: Prefixer,
    logical_settings: LogicalRepositorySettings,
    account: int,
    rdb: Database,
    cache: Optional[aiomcache.Client],
) -> List[Environment]:
    """
    Fetch unique deployment environments according to the filters.

    The output should be sorted by environment name.
    """
    if repos:
        repo_name_to_node = prefixer.repo_name_to_node.get
        repo_node_ids = {repo_name_to_node(drop_logical_repo(r)) for r in repos} - {None}
    else:
        repo_node_ids = set()
    notifications, has_any_notifications = await gather(
        fetch_deployment_candidates(
            repo_node_ids,
            time_from,
            time_to,
            [],
            [],
            {},
            {},
            account,
            rdb,
            cache,
        ),
        rdb.fetch_val(
            select(DeploymentNotification.name)
            .where(DeploymentNotification.account_id == account)
            .limit(1),
        ),
    )
    if has_any_notifications is None:
        raise NoDeploymentNotificationsError()
    tasks = [fetch_components_and_prune_unresolved(notifications, prefixer, account, rdb)]
    if logical_settings.has_deployments_by_label():
        tasks.append(fetch_labels(notifications.index.values, account, rdb))
    (notifications, components), *labels = await gather(*tasks)
    if not notifications.empty:
        # we must load all logical repositories at once to unambiguously process the residuals
        # (the root repository minus all the logicals)
        components = split_logical_deployed_components(
            notifications,
            labels[0] if labels else pd.DataFrame(),
            components,
            logical_settings.with_logical_deployments(repos or []),
            logical_settings,
        )
    envs_col = notifications[DeploymentNotification.environment.name].values
    unique_envs, first_indexes, env_counts = np.unique(
        envs_col, return_index=True, return_counts=True,
    )
    last_conclusions = notifications[DeploymentNotification.conclusion.name].values[first_indexes]

    components = components[[DeployedComponent.repository_full_name]].join(
        notifications[DeploymentNotification.environment.name],
    )

    repo_col = components[DeployedComponent.repository_full_name].values.astype("U")
    envs_col = components[DeploymentNotification.environment.name].values.astype("U")
    keys = np.char.add(envs_col, repo_col)
    _, first_indexes = np.unique(keys, return_index=True)
    components = components.take(first_indexes)
    envs_col = components[DeploymentNotification.environment.name].values
    order = np.argsort(envs_col)
    repo_unique_envs, repo_group_counts = np.unique(envs_col[order], return_counts=True)
    assert list(repo_unique_envs) == list(unique_envs)
    repo_name_col = components[DeployedComponent.repository_full_name].values
    repo_pos = 0
    result = []
    for env, notifications_count, last_conclusion, repo_group_count in zip(
        unique_envs,
        env_counts,
        last_conclusions,
        repo_group_counts,
    ):
        result.append(
            Environment(
                name=env,
                deployments_count=notifications_count,
                last_conclusion=last_conclusion.decode(),
                repositories=prefixer.prefix_repo_names(
                    repo_name_col[order[repo_pos : repo_pos + repo_group_count]],
                ),
            ),
        )
        repo_pos += repo_group_count
    return result
