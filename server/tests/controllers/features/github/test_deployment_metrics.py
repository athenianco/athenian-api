import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest

from athenian.api.controllers.features.github.deployment_metrics import \
    group_deployments_by_environments, group_deployments_by_participants, \
    group_deployments_by_repositories
from athenian.api.controllers.miners.types import ReleaseParticipationKind
from athenian.api.models.persistentdata.models import DeployedComponent


@pytest.fixture(scope="module")
def sample_deps() -> pd.DataFrame:
    rnid = DeployedComponent.repository_node_id.name
    return pd.DataFrame.from_dict({
        "components": [pd.DataFrame([{rnid: 1}, {rnid: 2}]),
                       pd.DataFrame([{rnid: 3}, {rnid: 1}]),
                       pd.DataFrame([{rnid: 3}, {rnid: 2}]),
                       pd.DataFrame([{rnid: 1}]),
                       pd.DataFrame([{rnid: 3}])],
        "pr_authors": [[1, 2, 3], [1, 4, 5], [2, 4, 6], [], [3]],
        "commit_authors": [[1, 2, 3], [1, 4, 5, 6], [2, 4, 6], [7], [3]],
        "release_authors": [[], [], [1, 2], [], [7]],
        "environment": ["1", "2", "1", "3", "3"],
    })


def test_group_deployments_by_repositories_smoke(sample_deps):
    assert_array_equal(group_deployments_by_repositories([[1, 2], [2, 3]], sample_deps),
                       [[0, 1, 2, 3], [0, 1, 2, 4]])
    assert_array_equal(
        [arr.tolist() for arr in group_deployments_by_repositories(
            [[1], [2], [1, 2]], sample_deps)],
        [[0, 1, 3], [0, 2], [0, 1, 2, 3]])
    assert_array_equal(group_deployments_by_repositories([], sample_deps),
                       [np.arange(len(sample_deps))])
    assert_array_equal(group_deployments_by_repositories([[1, 2], [2, 3]], pd.DataFrame()),
                       [np.array([], dtype=int)] * 2)
    assert_array_equal(group_deployments_by_repositories([], pd.DataFrame()),
                       [np.array([], dtype=int)])


def test_group_deployments_by_participants_smoke(sample_deps):
    assert_array_equal(
        group_deployments_by_participants(
            [{ReleaseParticipationKind.PR_AUTHOR: [1]}], sample_deps),
        [[0, 1]],
    )
    assert_array_equal(
        group_deployments_by_participants(
            [{ReleaseParticipationKind.COMMIT_AUTHOR: [3]}], sample_deps),
        [[0, 4]],
    )
    assert_array_equal(
        group_deployments_by_participants(
            [{ReleaseParticipationKind.RELEASER: [2]}], sample_deps),
        [[2]],
    )
    assert_array_equal(
        group_deployments_by_participants(
            [{ReleaseParticipationKind.RELEASER: [7]}], sample_deps),
        [[4]],
    )
    assert_array_equal(
        group_deployments_by_participants(
            [{ReleaseParticipationKind.PR_AUTHOR: [1, 3]}], sample_deps),
        [[0, 1, 4]],
    )
    assert_array_equal(
        group_deployments_by_participants(
            [{ReleaseParticipationKind.PR_AUTHOR: [1],
              ReleaseParticipationKind.COMMIT_AUTHOR: [3]}], sample_deps),
        [[0, 1, 4]],
    )
    assert_array_equal(
        group_deployments_by_participants(
            [{ReleaseParticipationKind.PR_AUTHOR: [1, 8],
              ReleaseParticipationKind.COMMIT_AUTHOR: [1]},
             {ReleaseParticipationKind.RELEASER: [2, 7]}], sample_deps),
        [[0, 1], [2, 4]],
    )
    assert_array_equal(group_deployments_by_participants([
        {ReleaseParticipationKind.PR_AUTHOR: [1]},
        {ReleaseParticipationKind.COMMIT_AUTHOR: [1]},
    ], pd.DataFrame()), [np.array([], dtype=int)] * 2)
    assert_array_equal(group_deployments_by_participants([], pd.DataFrame()),
                       [np.array([], dtype=int)])


def test_group_deployments_by_environments_smoke(sample_deps):
    assert [x.tolist() for x in group_deployments_by_environments(
        [["1", "2"], ["1", "3"]], sample_deps)] == \
        [[0, 1, 2], [0, 2, 3, 4]]
    assert [x.tolist() for x in group_deployments_by_environments(
        [["1"], ["3"]], sample_deps)] == \
        [[0, 2], [3, 4]]
    assert [x.tolist() for x in group_deployments_by_environments([], sample_deps)] == \
           [[0, 1, 2, 3, 4]]
    assert [x.tolist() for x in group_deployments_by_environments([["1"]], pd.DataFrame())] == \
           [[]]
    assert [x.tolist() for x in group_deployments_by_environments([], pd.DataFrame())] == [[]]
