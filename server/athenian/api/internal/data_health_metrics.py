import dataclasses
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Collection

from sqlalchemy import insert, select

from athenian.api.db import DatabaseLike
from athenian.api.internal.features.entries import MinePullRequestMetrics
from athenian.api.internal.miners.github.branches import BranchMinerMetrics
from athenian.api.internal.miners.github.deployment import MineDeploymentsMetrics
from athenian.api.internal.miners.github.release_load import MineReleaseMetrics
from athenian.api.internal.reposet import RepositorySetMetrics
from athenian.api.models.persistentdata.models import HealthMetric
from athenian.api.models.web import AccountHealth


@dataclass(frozen=True, slots=True)
class DataHealthMetrics:
    """Collection of data error statistics to report."""

    branches: BranchMinerMetrics | None
    deployments: MineDeploymentsMetrics | None
    prs: MinePullRequestMetrics | None
    releases: MineReleaseMetrics | None
    reposet: RepositorySetMetrics | None

    @classmethod
    def empty(cls) -> "DataHealthMetrics":
        """Initialize a new DataHealthMetrics instance filled with zeros."""
        return DataHealthMetrics(
            **{f.name: f.type.__args__[0].empty() for f in dataclasses.fields(cls)},
        )

    @classmethod
    def skip(cls) -> "DataHealthMetrics":
        """Initialize a new DataHealthMetrics instance filled with None-s, effectively disabling \
        the feature."""
        return DataHealthMetrics(**{f.name: None for f in dataclasses.fields(cls)})

    async def persist(self, account: int, rdb: DatabaseLike) -> None:
        """Insert all the measured metrics to the database."""
        now = datetime.now(timezone.utc)
        values = []
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            for m in v.as_db():
                m.account_id = account
                m.created_at = now
                values.append(m.explode(with_primary_keys=True))
        await rdb.execute_many(insert(HealthMetric), values)


async def measure_accounts_health(
    ids: Collection[int],
    time_points: list[datetime],
    rdb: DatabaseLike,
):
    """Collect all the supported health metrics for each account in `ids`."""
    rows = await rdb.fetch_all(
        select(HealthMetric)
        .where(
            HealthMetric.account_id.in_(ids),
            HealthMetric.created_at >= time_points[0],
            HealthMetric.created_at < time_points[-1] + timedelta(hours=1),
        )
        .order_by(HealthMetric.created_at),
    )
    ltp = len(time_points)
    result = {
        acc: AccountHealth(
            broken_branches=[0] * ltp,
            broken_dags=[0] * ltp,
            deployments=[0] * ltp,
            empty_releases=[0] * ltp,
            endpoint_p50={},
            endpoint_p95={},
            event_releases=[0] * ltp,
            inconsistent_nodes={},
            pending_fetch_branches=[0] * ltp,
            pending_fetch_prs=[0] * ltp,
            prs_count=[0] * ltp,
            released_prs_ratio=[0] * ltp,
            reposet_problems=[0] * ltp,
            repositories_count=[0] * ltp,
            unresolved_deployments=[0] * ltp,
            unresolved_releases=[0] * ltp,
        )
        for acc in ids
    }
    start_time = time_points[0]
    one_hour = timedelta(hours=1)
    acc_col = HealthMetric.account_id.name
    created_col = HealthMetric.created_at.name
    name_col = HealthMetric.name.name
    value_col = HealthMetric.value.name
    for row in rows:
        model = result[row[acc_col]]
        dt = row[created_col]
        pos = (dt - start_time) // one_hour
        val = row[value_col]
        match row[name_col]:
            case "branches_count":
                continue
            case ["branches_empty_count", "branches_no_default"]:
                model.broken_branches[pos] += val
            case ["commits_pristine", "commits_corrupted", "commits_orphaned"]:
                model.broken_dags[pos] += val
            case "deployments_count":
                model.deployments[pos] = val
            case "deployments_unresolved":
                model.unresolved_deployments[pos] = val
            case "releases_unresolved":
                model.unresolved_releases[pos] = val
            case "releases_by_event":
                model.event_releases[pos] = val
            case "releases_empty":
                model.empty_releases[pos] = val
            case "reposet_problems":
                model.empty_releases[pos] = val
            case "reposet_length":
                model.repositories_count[pos] = val
            case "prs_count":
                model.prs_count[pos] = val
            case "prs_done_count":
                model.released_prs_ratio[pos] = val
            case p50 if p50.startswith("p50/"):
                endpoint = p50[4:]
                try:
                    model.endpoint_p50[endpoint][pos] = val
                except KeyError:
                    vals = [None] * ltp
                    vals[pos] = val
                    model.endpoint_p50[endpoint] = vals
            case p95 if p95.startswith("p95/"):
                endpoint = p95[4:]
                try:
                    model.endpoint_p95[endpoint][pos] = val
                except KeyError:
                    vals = [None] * ltp
                    vals[pos] = val
                    model.endpoint_p95[endpoint] = vals
            case node if node.startswith("inconsistency/"):
                node = node[14:]
                try:
                    model.inconsistent_nodes[node][pos] = val
                except KeyError:
                    vals = [0] * ltp
                    vals[pos] = val
                    model.inconsistent_nodes[node] = vals
    for model in result.values():
        released_prs_ratio = model.released_prs_ratio
        for i, (done, count) in enumerate(zip(released_prs_ratio, model.prs_count)):
            if count == 0:
                released_prs_ratio[i] = 0
            else:
                released_prs_ratio[i] = done / count
    return result
