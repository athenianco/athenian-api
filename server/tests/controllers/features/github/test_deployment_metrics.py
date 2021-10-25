from datetime import datetime, timedelta, timezone

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest

from athenian.api.controllers.features.entries import MetricEntriesCalculator
from athenian.api.controllers.features.github.deployment_metrics import \
    group_deployments_by_environments, group_deployments_by_participants, \
    group_deployments_by_repositories
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.types import ReleaseParticipationKind
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.persistentdata.models import DeployedComponent
from athenian.api.models.web import DeploymentMetricID


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


@with_defer
async def test_deployment_metrics_calculators_smoke(
        sample_deployments, metrics_calculator_factory, release_match_setting_tag_or_branch,
        prefixer_promise, branches, default_branches):
    for i in range(2):
        calc = metrics_calculator_factory(
            1, (6366825,), with_cache=True)  # type: MetricEntriesCalculator
        if i == 1:
            calc._mdb = None
            calc._rdb = None
            calc._pdb = None
        metrics = await calc.calc_deployment_metrics_line_github(
            list(DeploymentMetricID),
            [[datetime(2015, 1, 1, tzinfo=timezone.utc),
              datetime(2021, 1, 1, tzinfo=timezone.utc)]],
            (0, 1),
            [[40550]],
            {}, [["staging"], ["production"]],
            LabelFilter.empty(), {}, {}, JIRAFilter.empty(),
            release_match_setting_tag_or_branch,
            prefixer_promise,
            branches, default_branches,
            (1, ("10003", "10009")),
        )
        await wait_deferred()
        assert len(metrics) == 1
        assert len(metrics[0]) == 1
        assert len(metrics[0][0]) == 2
        assert len(metrics[0][0][0]) == 1
        assert len(metrics[0][0][1]) == 1
        assert len(metrics[0][0][0][0]) == 1
        assert len(metrics[0][0][1][0]) == 1
        assert metrics[0][0][0][0][0] == metrics[0][0][1][0][0]
        assert dict(zip(DeploymentMetricID, (m.value for m in metrics[0][0][0][0][0]))) == {
            DeploymentMetricID.DEP_JIRA_ISSUES_COUNT: 44,
            DeploymentMetricID.DEP_COMMITS_COUNT: 2342,
            DeploymentMetricID.DEP_SIZE_RELEASES: 7.55555534362793,
            DeploymentMetricID.DEP_JIRA_BUG_FIXES_COUNT: 12,
            DeploymentMetricID.DEP_LINES_COUNT: 416242,
            DeploymentMetricID.DEP_SIZE_COMMITS: 260.22222900390625,
            DeploymentMetricID.DEP_RELEASES_COUNT: 68,
            DeploymentMetricID.DEP_COUNT: 9,
            DeploymentMetricID.DEP_DURATION_ALL: timedelta(seconds=600),
            DeploymentMetricID.DEP_FAILURE_COUNT: 2,
            DeploymentMetricID.DEP_SIZE_LINES: 46249.109375,
            DeploymentMetricID.DEP_SUCCESS_RATIO: 0.7777777910232544,
            DeploymentMetricID.DEP_DURATION_SUCCESSFUL: timedelta(seconds=600),
            DeploymentMetricID.DEP_DURATION_FAILED: timedelta(seconds=600),
            DeploymentMetricID.DEP_SIZE_PRS: 70.22222137451172,
            DeploymentMetricID.DEP_PRS_COUNT: 632,
            DeploymentMetricID.DEP_SUCCESS_COUNT: 7,
        }
