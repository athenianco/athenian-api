from datetime import datetime, timedelta
from typing import List, Optional

import pandas as pd

from athenian.api.controllers.features.github.check_run_filter import filter_check_runs
from athenian.api.controllers.miners.filters import JIRAFilter
from athenian.api.controllers.miners.types import CodeCheckRunListItem, CodeCheckRunListStats


def td_list(items: List[Optional[int]]) -> List[timedelta]:
    return [(x if x is None else timedelta(seconds=x)) for x in items]


async def test_filter_check_runs_monthly_quantiles(mdb):
    timeline, items = await filter_check_runs(
        datetime(2015, 1, 1), datetime(2020, 1, 1), ["src-d/go-git"], [], JIRAFilter.empty(),
        [0, 0.95], (6366825,), mdb, None)
    assert len(timeline) == len(set(timeline)) == 61
    assert len(items) == 9
    assert items[0] == CodeCheckRunListItem(
        title="DCO", repository="src-d/go-git",
        last_execution_time=pd.Timestamp("2019-12-30T09:41:10+00").to_pydatetime(),
        last_execution_url="https://github.com/src-d/go-git/runs/367607194",
        size_groups=[1, 3, 4, 5, 6],
        total_stats=CodeCheckRunListStats(
            count=383, successes=361, skips=0, flaky_count=0,
            mean_execution_time=timedelta(seconds=0),
            median_execution_time=timedelta(seconds=1),
            count_timeline=[
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 18, 18, 17, 26, 49, 38, 42, 11, 5, 5, 15, 11,
                59, 17, 9, 18, 12, 0, 3, 6, 4],
            successes_timeline=[
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 17, 17, 16, 24, 48, 37, 34, 10, 5, 5, 15, 11,
                57, 17, 9, 17, 10, 0, 3, 5, 4],
            mean_execution_time_timeline=td_list([
                None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None,
                None, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, None, 1, 1, 1]),
            median_execution_time_timeline=td_list([
                None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, 0., 0., 0., 0., 0., 1., 1., 1., 1., 2., 1., 1.,
                None, 2., 1., 2.]),
        ),
        prs_stats=CodeCheckRunListStats(
            count=319, successes=301, skips=0, flaky_count=0,
            mean_execution_time=timedelta(seconds=0),
            median_execution_time=timedelta(seconds=1),
            count_timeline=[
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10, 18, 13, 15, 39, 26, 37, 10, 5, 5, 14, 11,
                50, 15, 9, 17, 12, 0, 3, 6, 4],
            successes_timeline=[
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 17, 12, 15, 38, 25, 30, 9, 5, 5, 14, 11,
                48, 15, 9, 17, 10, 0, 3, 5, 4],
            mean_execution_time_timeline=td_list([
                None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None, None,
                None, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, None, 1, 1, 1]),
            median_execution_time_timeline=td_list([
                None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, None, None, None, None, None, None,
                None, None, None, None, 0., 0., 0., 0., 0., 1., 1., 1., 1., 2., 1., 1.,
                None, 2., 1., 2.]),
        ),
    )


async def test_filter_check_runs_daily(mdb):
    timeline, items = await filter_check_runs(
        datetime(2018, 2, 1), datetime(2018, 2, 12), ["src-d/go-git"], [], JIRAFilter.empty(),
        [0, 1], (6366825,), mdb, None)
    assert len(timeline) == len(set(timeline)) == 12
    assert len(items) == 7


async def test_filter_check_runs_empty(mdb):
    timeline, items = await filter_check_runs(
        datetime(2018, 2, 1), datetime(2018, 2, 12), ["src-d/go-git"], ["xxx"], JIRAFilter.empty(),
        [0, 1], (6366825,), mdb, None)
    assert len(timeline) == len(set(timeline)) == 12
    assert len(items) == 0
