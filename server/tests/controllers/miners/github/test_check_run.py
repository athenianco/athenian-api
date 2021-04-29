from datetime import datetime, timezone

import pytest

from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.check_run import mine_check_runs
from athenian.api.models.metadata.github import CheckRun


@pytest.mark.parametrize("time_from, time_to, repositories, commit_authors, jira, size", [
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [], JIRAFilter.empty(), 5323),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/hercules"], [], JIRAFilter.empty(), 0),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2018, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [], JIRAFilter.empty(), 2429),
    (datetime(2018, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [], JIRAFilter.empty(), 2894),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], ["mcuadros"], JIRAFilter.empty(), 1741),
    (datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc),
     ["src-d/go-git"], [],
     JIRAFilter(1, ["10003", "10009"], LabelFilter.empty(), set(), {"task"}, False), 229),
])
async def test_check_run_smoke(mdb, time_from, time_to, repositories, commit_authors, jira, size):
    df = await mine_check_runs(
        time_from, time_to, repositories, commit_authors, jira, (6366825,), mdb, None)
    assert len(df) == size
    for col in CheckRun.__table__.columns:
        assert col.key in df.columns
