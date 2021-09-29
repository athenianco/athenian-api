from datetime import datetime, timezone

import pytest

from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.check_run import mine_check_runs
from athenian.api.models.metadata.github import CheckRun


@pytest.mark.parametrize("time_from, time_to, repositories, pushers, labels, jira, size", [
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [], LabelFilter.empty(), JIRAFilter.empty(), 4393),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/hercules"], [], LabelFilter.empty(), JIRAFilter.empty(), 0),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2018, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [], LabelFilter.empty(), JIRAFilter.empty(), 2183),
    (datetime(2018, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [], LabelFilter.empty(), JIRAFilter.empty(), 2213),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], ["mcuadros"], LabelFilter.empty(), JIRAFilter.empty(), 1575),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [], LabelFilter({"bug", "plumbing", "enhancement"}, set()),
     JIRAFilter.empty(), 67),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [], LabelFilter.empty(),
     JIRAFilter(1, ["10003", "10009"], LabelFilter.empty(), set(), {"task"}, False), 229),
    (datetime(2015, 10, 10, tzinfo=timezone.utc), datetime(2015, 10, 23, tzinfo=timezone.utc),
     ["src-d/go-git"], [], LabelFilter.empty(), JIRAFilter.empty(), 4),
])
async def test_check_run_smoke(mdb, time_from, time_to, repositories, pushers, labels, jira, size):
    df = await mine_check_runs(
        time_from, time_to, repositories, pushers, labels, jira, False, (6366825,), mdb, None)
    assert len(df) == size
    for col in CheckRun.__table__.columns:
        if col.name != CheckRun.committed_date_hack.name:
            assert col.name in df.columns
    assert len(df[CheckRun.check_run_node_id.name].unique()) == len(df)


@pytest.mark.parametrize("time_from, time_to, size", [
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc), 2681),
    (datetime(2018, 1, 1, tzinfo=timezone.utc), datetime(2019, 1, 1, tzinfo=timezone.utc), 1068),
])
async def test_check_run_only_prs(mdb, time_from, time_to, size):
    df = await mine_check_runs(
        time_from, time_to, ["src-d/go-git"], [], LabelFilter.empty(), JIRAFilter.empty(), True,
        (6366825,), mdb, None)
    assert (df[CheckRun.pull_request_node_id.name].values != 0).all()
    assert len(df) == size
