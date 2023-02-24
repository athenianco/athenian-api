import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd

from athenian.api.internal.features.metric_calculator import (
    JIRAGrouping,
    group_by_repo,
    group_jira_facts_by_jira,
    group_pr_facts_by_jira,
)
from athenian.api.internal.miners.types import PullRequestFacts
from athenian.api.models.metadata.jira import Issue


def test_group_by_repo_single_repos():
    df = pd.DataFrame({"repo": ["one", "two", "one", "one", "one", "two"]})
    groups = group_by_repo("repo", [["one"], ["two"], ["one", "two"]], df)
    assert len(groups) == 3
    assert_array_equal(groups[0], [0, 2, 3, 4])
    assert_array_equal(groups[1], [1, 5])
    assert_array_equal(groups[2], np.arange(len(df)))


def test_group_by_repo_few_groups():
    df = pd.DataFrame({"repo": ["one", "two", "one", "one", "one", "two"]})
    groups = group_by_repo("repo", [["one"]], df)
    assert len(groups) == 1
    assert_array_equal(groups[0], [0, 2, 3, 4])


class TestGroupPRFactsByJIRA:
    PROJECTS = PullRequestFacts.INDIRECT_FIELDS.JIRA_PROJECTS
    PRIORITIES = PullRequestFacts.INDIRECT_FIELDS.JIRA_PRIORITIES
    TYPES = PullRequestFacts.INDIRECT_FIELDS.JIRA_TYPES
    LABELS = PullRequestFacts.INDIRECT_FIELDS.JIRA_LABELS

    def test_empty_df(self) -> None:
        df = self._make_df()
        res = group_pr_facts_by_jira([JIRAGrouping(["p0"], None, ["t0", "t1"])], df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([], dtype=int))

    def test_total_groups(self) -> None:
        groups = [JIRAGrouping.empty()] * 2
        df = self._make_df(([], [], [], []), ([], [], [], []))
        res = group_pr_facts_by_jira(groups, df)
        assert len(res) == 2
        assert_array_equal(res[0], np.array([0, 1], dtype=int))
        assert_array_equal(res[1], np.array([0, 1], dtype=int))

    def test_empty_group(self) -> None:
        groups = [JIRAGrouping(None, None, [])]
        df = self._make_df(
            ([b"pr0"], [], [b"t0"], []),
            ([b"pr1"], [], [], []),
        )
        res = group_pr_facts_by_jira(groups, df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([], dtype=int))

    def test_single_group(self) -> None:
        df = self._make_df(
            ([b"pr0"], [], [b"t0"], []),
            ([b"pr1"], [], [b"t0"], []),
            ([b"pr0", b"pr1"], [], [], []),
            ([b"pr0", b"pr1"], [], [b"t1"], []),
            ([b"pr1"], [], [b"t1", b"t2"], []),
        )

        res = group_pr_facts_by_jira([JIRAGrouping(["pr0"], None, ["t0", "t1"])], df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([0, 3], dtype=int))
        return

        res = group_pr_facts_by_jira([JIRAGrouping(None, None, ["t0"])], df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([0, 1], dtype=int))

        res = group_pr_facts_by_jira([JIRAGrouping(["pr2"], None, ["t2"])], df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([], dtype=int))

        res = group_pr_facts_by_jira([JIRAGrouping(["pr0"], None, None)], df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([0, 2, 3], dtype=int))

    def test_multiple_groups(self) -> None:
        df = self._make_df(
            ([b"pr0", b"pr1"], [b"pri0"], [b"t0"], []),
            ([b"pr0"], [b"pri1"], [b"t0"], []),
            ([b"pr1"], [b"pri0"], [b"t0", b"t1"], []),
            ([b"pr1"], [], [b"t1"], []),
        )
        groups = [
            JIRAGrouping(["pr0", "pr1"], None, ["t0"]),
            JIRAGrouping(["pr1"], None, ["t0"]),
            JIRAGrouping(["pr0", "pr1"], ["pri1"], ["t0", "t1"]),
        ]
        res = group_pr_facts_by_jira(groups, df)
        assert len(res) == 3
        assert_array_equal(res[0], np.array([0, 1, 2], dtype=int))
        assert_array_equal(res[1], np.array([0, 2], dtype=int))
        assert_array_equal(res[2], np.array([1], dtype=int))

    def test_priority(self) -> None:
        df = self._make_df(
            ([], [b"high"], [], []),
            ([], [], [], []),
            ([], [b"high", b"low", b"medium"], [], []),
            ([], [b"low"], [], []),
        )
        jira_groups = [JIRAGrouping(None, ["high"], None)]
        res = group_pr_facts_by_jira(jira_groups, df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([0, 2]))

        jira_groups = [JIRAGrouping(None, ["low"], None)]
        res = group_pr_facts_by_jira(jira_groups, df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([2, 3]))

    def test_priority_single_match(self) -> None:
        jira_groups = [JIRAGrouping(None, ["high"], None)]
        df = self._make_df(
            ([], [], [], []),
            ([], [b"high"], [], []),
            ([], [], [], []),
        )
        res = group_pr_facts_by_jira(jira_groups, df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([1]))

    def test_labels(self) -> None:
        jira_groups = [JIRAGrouping(labels=["l0"]), JIRAGrouping(labels=["l1", "l0"])]
        df = self._make_df(
            ([], [], [], ["l0"]),
            ([], [b"high"], [], ["l2"]),
            ([], [], [], ["l1"]),
            ([], [], [], ["l1", "l2"]),
        )
        res = group_pr_facts_by_jira(jira_groups, df)
        assert len(res) == 2
        assert_array_equal(res[0], np.array([0]))
        assert_array_equal(res[1], np.array([0, 2, 3]))

    def _make_df(self, *rows: tuple) -> pd.DataFrame:
        dtypes = ("S", "S", "S", "U")
        data = [
            tuple(np.array(field, dtype=dtype) for field, dtype in zip(r, dtypes)) for r in rows
        ]
        return pd.DataFrame.from_records(
            data, columns=[self.PROJECTS, self.PRIORITIES, self.TYPES, self.LABELS],
        )


class TestGroupJIRAFactsByJIRA:
    def test_empty_df(self) -> None:
        df = self._make_df()
        res = group_jira_facts_by_jira([JIRAGrouping(["p0"], None, ["t0", "t1"])], df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([], dtype=int))

    def test_single_group(self) -> None:
        df = self._make_df(
            (b"p0", b"t0", None, None),
            (b"p0", b"t1", b"pr0", None),
            (b"p0", b"t0", b"pr0", None),
            (b"p1", b"t1", None, None),
        )
        jira_groups = [JIRAGrouping(["p0"], types=["t0"])]
        res = group_jira_facts_by_jira(jira_groups, df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([0, 2]))

        jira_groups = [JIRAGrouping(["p0"], priorities=["pr0", "pr1"])]
        res = group_jira_facts_by_jira(jira_groups, df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([1, 2]))

        jira_groups = [JIRAGrouping(types=["t0"], priorities=["pr0", "pr1"])]
        res = group_jira_facts_by_jira(jira_groups, df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([2]))

    def test_multiple_groups(self) -> None:
        df = self._make_df(
            (b"p0", b"t0", None, None),
            (b"p0", b"t1", b"pr0", None),
            (b"p0", b"t0", None, None),
            (b"p0", b"t0", b"pr0", None),
            (b"p0", b"t1", None, None),
            (b"p1", b"t2", b"pr0", None),
            (b"p1", b"t0", b"pr0", None),
            (b"p1", b"t0", b"pr1", None),
        )
        jira_groups = [
            JIRAGrouping(["p1"]),
            JIRAGrouping(types=["t0", "t2"]),
            JIRAGrouping(types=["t0", "t2"], priorities=["pr0"]),
            JIRAGrouping(types=["t2"], priorities=["pr1"]),
            JIRAGrouping(projects=["p0", "p2"], types=["t0", "t2"], priorities=["pr0"]),
            JIRAGrouping.empty(),
        ]
        res = group_jira_facts_by_jira(jira_groups, df)
        assert len(res) == 6
        assert_array_equal(res[0], np.array([5, 6, 7]))
        assert_array_equal(res[1], np.array([0, 2, 3, 5, 6, 7]))
        assert_array_equal(res[2], np.array([3, 5, 6]))
        assert_array_equal(res[3], np.array([]))
        assert_array_equal(res[4], np.array([3]))
        assert_array_equal(res[5], np.arange(8))

    def test_missing_column_for_unused_field(self) -> None:
        jira_groups = [JIRAGrouping(None, ["t0", "t1"], None)]

        df = self._make_df(
            (None, None, b"t0", None),
            (None, b"pr0", b"t1", None),
            (None, None, b"t2", None),
            (None, None, b"t0", None),
        )
        df = df.drop(Issue.project_id.name, axis="columns")
        res = group_jira_facts_by_jira(jira_groups, df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([0, 1, 3]))

    def test_labels(self) -> None:
        jira_groups = [JIRAGrouping(labels=["l0"])]
        df = self._make_df(
            (None, None, None, ["l0"]),
            (None, None, None, []),
            (None, None, None, ["l1"]),
            (None, None, None, ["l0", "l1"]),
        )
        res = group_jira_facts_by_jira(jira_groups, df)
        assert len(res) == 1
        assert_array_equal(res[0], np.array([0, 3]))

    @classmethod
    def _make_df(cls, *rows: tuple) -> pd.DataFrame:
        columns = [Issue.project_id, Issue.type_id, Issue.priority_id, Issue.labels]

        df = pd.DataFrame.from_records(rows, columns=[c.name for c in columns])
        types = {"project_id": "S", "type_id": "S", "priority_id": "S"}
        return df.astype(types)
