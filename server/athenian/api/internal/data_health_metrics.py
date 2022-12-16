from dataclasses import dataclass

from athenian.api.internal.features.entries import MinePullRequestMetrics
from athenian.api.internal.miners.github.branches import BranchMinerMetrics
from athenian.api.internal.miners.github.deployment import MineDeploymentsMetrics
from athenian.api.internal.miners.github.release_load import MineReleaseMetrics
from athenian.api.internal.reposet import RepositorySetMetrics


@dataclass(frozen=True, slots=True)
class DataHealthMetrics:
    """Collection of data error statistics to report."""

    branches: BranchMinerMetrics
    deployments: MineDeploymentsMetrics
    prs: MinePullRequestMetrics
    releases: MineReleaseMetrics
    reposet: RepositorySetMetrics

    @classmethod
    def empty(cls) -> "DataHealthMetrics":
        """Initialize a new DataHealthMetrics instance filled with zeros."""
        return DataHealthMetrics(**{k: v.empty() for k, v in cls.__annotations__.items()})
