from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
import pytest

from athenian.api.controllers.features.github.check_run_metrics_accelerated import \
    mark_check_suite_types
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.check_run import _postprocess_check_runs, \
    _split_duplicate_check_runs, mine_check_runs
from athenian.api.int_to_str import int_to_str
from athenian.api.models.metadata.github import CheckRun


@pytest.mark.parametrize("time_from, time_to, repositories, pushers, labels, jira, size", [
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [], LabelFilter.empty(), JIRAFilter.empty(), 4581),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/hercules"], [], LabelFilter.empty(), JIRAFilter.empty(), 0),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2018, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [], LabelFilter.empty(), JIRAFilter.empty(), 2371),
    (datetime(2018, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [], LabelFilter.empty(), JIRAFilter.empty(), 2213),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], ["mcuadros"], LabelFilter.empty(), JIRAFilter.empty(), 1642),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [], LabelFilter({"bug", "plumbing", "enhancement"}, set()),
     JIRAFilter.empty(), 67),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [], LabelFilter.empty(),
     JIRAFilter(1, ["10003", "10009"], LabelFilter.empty(), set(), {"task"}, False, False), 229),
    (datetime(2015, 10, 10, tzinfo=timezone.utc), datetime(2015, 10, 23, tzinfo=timezone.utc),
     ["src-d/go-git"], [], LabelFilter.empty(), JIRAFilter.empty(), 4),
])
async def test_check_run_smoke(mdb, time_from, time_to, repositories, pushers, labels, jira, size):
    df = await mine_check_runs(
        time_from, time_to, repositories, pushers, labels, jira, False,
        (6366825,), mdb, None)
    assert len(df) == size
    for col in CheckRun.__table__.columns:
        if col.name not in (CheckRun.committed_date_hack.name,):
            assert col.name in df.columns
    assert len(df[CheckRun.check_run_node_id.name].unique()) == len(df)


@pytest.mark.parametrize("time_from, time_to, size", [
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc), 2766),
    (datetime(2018, 1, 1, tzinfo=timezone.utc), datetime(2019, 1, 1, tzinfo=timezone.utc), 1068),
])
async def test_check_run_only_prs(mdb, time_from, time_to, size):
    df = await mine_check_runs(
        time_from, time_to, ["src-d/go-git"], [], LabelFilter.empty(), JIRAFilter.empty(),
        True, (6366825,), mdb, None)
    assert (df[CheckRun.pull_request_node_id.name].values != 0).all()
    assert len(df) == size


def test_mark_check_suite_types_smoke():
    names = np.array(["one", "two", "one", "three", "one", "one", "two"])
    suites = np.array([1, 1, 4, 3, 2, 5, 5])
    suite_indexes, group_ids = mark_check_suite_types(names, suites)
    assert_array_equal(suite_indexes, [0, 4, 3, 2, 5])
    assert_array_equal(group_ids, [2, 1, 0, 1, 2])


def test_mark_check_suite_types_empty():
    suite_indexes, group_ids = mark_check_suite_types(
        np.array([], dtype="U"), np.array([], dtype=int))
    assert len(suite_indexes) == 0
    assert len(group_ids) == 0


@pytest.fixture(scope="module")
def alternative_facts() -> pd.DataFrame:
    df = pd.read_csv(
        Path(__file__).parent.parent.parent / "features" / "github" / "check_runs.csv.gz")
    for col in (CheckRun.started_at,
                CheckRun.completed_at,
                CheckRun.pull_request_created_at,
                CheckRun.pull_request_closed_at,
                CheckRun.committed_date):
        df[col.name] = df[col.name].astype(np.datetime64)
    df = _split_duplicate_check_runs(df)
    _postprocess_check_runs(df)
    return df


def test_mark_check_suite_types_real_world(alternative_facts):
    repos = int_to_str(alternative_facts[CheckRun.repository_node_id.name].values)
    names = np.char.add(
        repos,
        np.char.encode(alternative_facts[CheckRun.name.name].values.astype("U"), "UTF-8"))
    suite_indexes, group_ids = mark_check_suite_types(
        names, alternative_facts[CheckRun.check_suite_node_id.name].values)
    assert (suite_indexes < len(alternative_facts)).all()
    assert (suite_indexes >= 0).all()
    unique_groups, counts = np.unique(group_ids, return_counts=True)
    assert_array_equal(unique_groups, np.arange(21))
    assert_array_equal(
        counts,
        [1, 1, 110, 1, 275, 1, 928, 369, 2, 1472, 8490,
         1, 707, 213, 354, 205, 190, 61, 731, 251, 475])
