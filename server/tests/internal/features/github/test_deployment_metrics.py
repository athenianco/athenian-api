from datetime import datetime

import medvedi as md
import numpy as np
from numpy.testing import assert_array_equal
import pytest

from athenian.api.internal.features.github.deployment_metrics import (
    CHANGE_FAILURE_LABEL,
    ChangeFailureCounter,
    group_deployments_by_environments,
    group_deployments_by_participants,
    group_deployments_by_repositories,
)
from athenian.api.internal.miners.participation import ReleaseParticipationKind
from athenian.api.internal.miners.types import DeploymentFacts
from athenian.api.models.persistentdata.models import DeployedLabel, DeploymentNotification
from tests.testutils.time import dt, dt64arr_us


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
    def _make_df(cls, *rows: tuple) -> md.DataFrame:
        column_names = [DeploymentFacts.f.repositories, DeploymentFacts.f.commits_overall]
        columns = {k: [] for k in column_names}
        for r in rows:
            for i, v in enumerate(r):
                columns[column_names[i]].append(v)
        return md.DataFrame(columns)


@pytest.fixture(scope="module")
def sample_deps() -> md.DataFrame:
    return md.DataFrame(
        {
            DeploymentFacts.f.pr_authors: [[1, 2, 3], [1, 4, 5], [2, 4, 6], [], [3]],
            DeploymentFacts.f.commit_authors: [[1, 2, 3], [1, 4, 5, 6], [2, 4, 6], [7], [3]],
            DeploymentFacts.f.release_authors: [[], [], [1, 2], [], [7]],
            DeploymentFacts.f.commits_overall: [[1, 1], [1, 1], [0, 1], [1], [1]],
            "environment": ["1", "2", "1", "3", "3"],
            DeploymentFacts.f.repositories: [["1", "2"], ["1", "3"], ["2", "3"], ["1"], ["3"]],
        },
    )


def test_group_deployments_by_repositories_smoke(sample_deps):
    assert_array_equal(
        [
            arr.tolist()
            for arr in group_deployments_by_repositories([["1", "2"], ["2", "3"]], sample_deps)
        ],
        [[0, 1, 3], [0, 1, 2, 4]],
    )
    assert_array_equal(
        [
            arr.tolist()
            for arr in group_deployments_by_repositories([["1"], ["2"], ["1", "2"]], sample_deps)
        ],
        [[0, 1, 3], [0], [0, 1, 3]],
    )
    assert_array_equal(
        group_deployments_by_repositories([], sample_deps), [np.arange(len(sample_deps))],
    )
    assert_array_equal(
        group_deployments_by_repositories([["1", "2"], ["2", "3"]], md.DataFrame()),
        [np.array([], dtype=int)] * 2,
    )
    assert_array_equal(
        group_deployments_by_repositories([], md.DataFrame()), [np.array([], dtype=int)],
    )


def test_group_deployments_by_participants_smoke(sample_deps):
    assert_array_equal(
        group_deployments_by_participants(
            [{ReleaseParticipationKind.PR_AUTHOR: [1]}], sample_deps,
        ),
        [[0, 1]],
    )
    assert_array_equal(
        group_deployments_by_participants(
            [{ReleaseParticipationKind.COMMIT_AUTHOR: [3]}], sample_deps,
        ),
        [[0, 4]],
    )
    assert_array_equal(
        group_deployments_by_participants([{ReleaseParticipationKind.RELEASER: [2]}], sample_deps),
        [[2]],
    )
    assert_array_equal(
        group_deployments_by_participants([{ReleaseParticipationKind.RELEASER: [7]}], sample_deps),
        [[4]],
    )
    assert_array_equal(
        group_deployments_by_participants(
            [{ReleaseParticipationKind.PR_AUTHOR: [1, 3]}], sample_deps,
        ),
        [[0, 1, 4]],
    )
    assert_array_equal(
        group_deployments_by_participants(
            [
                {
                    ReleaseParticipationKind.PR_AUTHOR: [1],
                    ReleaseParticipationKind.COMMIT_AUTHOR: [3],
                },
            ],
            sample_deps,
        ),
        [[0, 1, 4]],
    )
    assert_array_equal(
        group_deployments_by_participants(
            [
                {
                    ReleaseParticipationKind.PR_AUTHOR: [1, 8],
                    ReleaseParticipationKind.COMMIT_AUTHOR: [1],
                },
                {ReleaseParticipationKind.RELEASER: [2, 7]},
            ],
            sample_deps,
        ),
        [[0, 1], [2, 4]],
    )
    assert_array_equal(
        group_deployments_by_participants(
            [
                {ReleaseParticipationKind.PR_AUTHOR: [1]},
                {ReleaseParticipationKind.COMMIT_AUTHOR: [1]},
            ],
            md.DataFrame(),
        ),
        [np.array([], dtype=int)] * 2,
    )
    assert_array_equal(
        group_deployments_by_participants([], md.DataFrame()), [np.array([], dtype=int)],
    )


def test_group_deployments_by_environments_smoke(sample_deps):
    assert [
        x.tolist()
        for x in group_deployments_by_environments([["1", "2"], ["1", "3"]], sample_deps)
    ] == [[0, 1, 2], [0, 2, 3, 4]]
    assert [
        x.tolist() for x in group_deployments_by_environments([["1"], ["3"]], sample_deps)
    ] == [[0, 2], [3, 4]]
    assert [x.tolist() for x in group_deployments_by_environments([], sample_deps)] == [
        [0, 1, 2, 3, 4],
    ]
    assert [x.tolist() for x in group_deployments_by_environments([["1"]], md.DataFrame())] == [[]]
    assert [x.tolist() for x in group_deployments_by_environments([], md.DataFrame())] == [[]]


class TestChangeFailureCounter:
    _SUCCESS = DeploymentNotification.CONCLUSION_SUCCESS
    _FAILURE = DeploymentNotification.CONCLUSION_FAILURE

    def test_base(self) -> None:
        facts = self._gen_facts(
            [self._SUCCESS, datetime(2001, 1, 1), self._labels(("foo", "bar"))],
            [self._SUCCESS, datetime(2001, 1, 1), self._labels((CHANGE_FAILURE_LABEL, "a"))],
            [self._SUCCESS, datetime(2001, 1, 1), self._labels((CHANGE_FAILURE_LABEL, None))],
            [self._SUCCESS, datetime(2001, 1, 1), self._labels()],
            [self._SUCCESS, datetime(2001, 1, 1), md.DataFrame()],
            [
                self._SUCCESS,
                datetime(2001, 1, 1),
                self._labels(("k", "v"), (CHANGE_FAILURE_LABEL, ["a"])),
            ],
            [self._SUCCESS, datetime(2001, 1, 1), self._labels(("k", "v"), ("k2", "v2"))],
            [self._FAILURE, datetime(2001, 1, 1), self._labels((CHANGE_FAILURE_LABEL, "a"))],
            [self._FAILURE, datetime(2003, 1, 1), self._labels((CHANGE_FAILURE_LABEL, "a"))],
        )
        min_times = dt64arr_us(dt(2001, 1, 1))
        max_times = dt64arr_us(dt(2002, 1, 1))
        groups_mask = np.full((1, len(facts)), True, bool)
        calc = ChangeFailureCounter(quantiles=(0, 1))
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == 3

    @classmethod
    def _gen_facts(cls, *values: list) -> md.DataFrame:
        labels = np.empty(dtype=object, shape=len(values))
        labels[:] = [v[2] for v in values]
        return md.DataFrame(
            {
                DeploymentNotification.conclusion.name: np.array([v[0] for v in values]),
                DeploymentNotification.finished_at.name: np.array([v[1] for v in values]),
                "labels": labels,
            },
        )

    @classmethod
    def _labels(cls, *values: tuple) -> md.DataFrame:
        return md.DataFrame(
            {
                DeployedLabel.key.name: np.array([v[0] for v in values], dtype=object),
                DeployedLabel.value.name: np.array([v[1] for v in values], dtype=object),
            },
        )
