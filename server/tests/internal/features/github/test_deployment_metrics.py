import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from athenian.api.internal.features.github.deployment_metrics import (
    group_deployments_by_repositories,
)
from athenian.api.internal.miners.types import DeploymentFacts


class TestGroupDeploymentsByRepositories:
    def test_empty_are_discarded(self) -> None:
        df = self._make_df(
            (["o/r0"], [2]),
            (["o/r0"], [0]),
            (["o/r0", "o/r1"], [0, 2]),
            (["o/r0", "o/r1"], [3, 3]),
            (["o/r1"], [1]),
        )

        groups = group_deployments_by_repositories([["o/r0"]], df)
        assert_array_equal(groups, np.array([[0, 3]]))

        groups = group_deployments_by_repositories([["o/r1"]], df)
        assert_array_equal(groups, np.array([[2, 3, 4]]))

        groups = group_deployments_by_repositories([["o/r0", "o/r1"]], df)
        assert_array_equal(groups, np.array([[0, 2, 3, 4]]))

    @classmethod
    def _make_df(cls, *rows: tuple) -> pd.DataFrame:
        return pd.DataFrame.from_records(
            rows, columns=[DeploymentFacts.f.repositories, DeploymentFacts.f.commits_overall],
        )
