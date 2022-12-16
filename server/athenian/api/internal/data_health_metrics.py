import dataclasses
from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import insert

from athenian.api.db import DatabaseLike
from athenian.api.internal.features.entries import MinePullRequestMetrics
from athenian.api.internal.miners.github.branches import BranchMinerMetrics
from athenian.api.internal.miners.github.deployment import MineDeploymentsMetrics
from athenian.api.internal.miners.github.release_load import MineReleaseMetrics
from athenian.api.internal.reposet import RepositorySetMetrics
from athenian.api.models.persistentdata.models import HealthMetric


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
