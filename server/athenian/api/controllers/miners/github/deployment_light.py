from collections import defaultdict
from datetime import timezone
import pickle
from typing import Collection, Dict, Optional, Tuple

import aiomcache
from sqlalchemy import and_, select

from athenian.api.async_utils import gather
from athenian.api.cache import cached, middle_term_expire
from athenian.api.controllers.miners.types import DeployedComponent as DeployedComponentDC, \
    Deployment, PullRequestFacts
from athenian.api.db import ParallelDatabase
from athenian.api.models.metadata.github import NodeCommit
from athenian.api.models.persistentdata.models import DeployedComponent, DeployedLabel, \
    DeploymentNotification
from athenian.api.models.precomputed.models import GitHubPullRequestDeployment
from athenian.api.tracing import sentry_span


@sentry_span
@cached(
    serialize=pickle.dumps,
    deserialize=pickle.loads,
    key=lambda names, **_: (",".join(sorted(names)),),
    exptime=middle_term_expire,
    refresh_on_access=True,
)
async def load_included_deployments(names: Collection[str],
                                    account: int,
                                    meta_ids: Tuple[int, ...],
                                    mdb: ParallelDatabase,
                                    rdb: ParallelDatabase,
                                    cache: Optional[aiomcache.Client],
                                    ) -> Dict[str, Deployment]:
    """
    Fetch brief details about the deployments.

    Compared to `mine_deployments()`, this is much more lightweight and is intended for `include`.
    """
    notifications, components, labels = await gather(
        rdb.fetch_all(
            select([DeploymentNotification])
            .where(and_(DeploymentNotification.account_id == account,
                        DeploymentNotification.name.in_any_values(names)))),
        rdb.fetch_all(
            select([DeployedComponent])
            .where(and_(DeployedComponent.account_id == account,
                        DeployedComponent.deployment_name.in_any_values(names)))),
        rdb.fetch_all(
            select([DeployedLabel])
            .where(and_(DeployedLabel.account_id == account,
                        DeployedLabel.deployment_name.in_any_values(names)))),
    )
    commit_ids = {c[DeployedComponent.resolved_commit_node_id.name] for c in components} \
        - {None}
    hashes = await mdb.fetch_all(
        select([NodeCommit.sha, NodeCommit.graph_id])
        .where(and_(NodeCommit.acc_id.in_(meta_ids),
                    NodeCommit.graph_id.in_any_values(commit_ids))))
    hashes = {r[NodeCommit.graph_id.name]: r[NodeCommit.sha.name] for r in hashes}
    comps_by_dep = {}
    for row in components:
        comps_by_dep.setdefault(row[DeployedComponent.deployment_name.name], []).append(
            DeployedComponentDC(
                repository_id=row[DeployedComponent.repository_node_id.name],
                reference=row[DeployedComponent.reference.name],
                sha=hashes.get(row[DeployedComponent.resolved_commit_node_id.name])))
    labels_by_dep = {}
    for row in labels:
        labels_by_dep.setdefault(
            row[DeployedLabel.deployment_name.name], {},
        )[row[DeployedLabel.key.name]] = row[DeployedLabel.value.name]
    if rdb.url.dialect == "sqlite":
        notifications = [dict(r) for r in notifications]
        for row in notifications:
            for col in (DeploymentNotification.started_at, DeploymentNotification.finished_at):
                row[col.name] = row[col.name].replace(tzinfo=timezone.utc)
    return {
        (name := row[DeploymentNotification.name.name]): Deployment(
            name=name,
            conclusion=row[DeploymentNotification.conclusion.name],
            environment=row[DeploymentNotification.environment.name],
            url=row[DeploymentNotification.url.name],
            started_at=row[DeploymentNotification.started_at.name],
            finished_at=row[DeploymentNotification.finished_at.name],
            components=comps_by_dep.get(name, []),
            labels=labels_by_dep.get(name, None),
        )
        for row in notifications
    }


@sentry_span
async def append_pr_deployment_mapping(prs: Dict[int, PullRequestFacts],
                                       account: int,
                                       pdb: ParallelDatabase) -> None:
    """Fetch the precomputed deployments for each pull request."""
    ghprd = GitHubPullRequestDeployment
    rows = await pdb.fetch_all(select([ghprd.deployment_name, ghprd.pull_request_id])
                               .where(and_(ghprd.acc_id == account,
                                           ghprd.pull_request_id.in_(prs))))
    depmap = defaultdict(list)
    for row in rows:
        depmap[row[ghprd.pull_request_id.name]].append(row[ghprd.deployment_name.name])
    for key, val in depmap.items():
        prs[key].deployments = val
