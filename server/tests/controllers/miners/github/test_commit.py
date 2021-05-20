from datetime import datetime, timezone

import pytest

from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.commit import extract_commits, FilterCommitsProperty
from athenian.api.defer import with_defer


@pytest.mark.parametrize("property, count", [
    (FilterCommitsProperty.NO_PR_MERGES, 596),
    (FilterCommitsProperty.BYPASSING_PRS, 289),
])
@with_defer
async def test_extract_commits_users(mdb, pdb, property, count):
    commits = await extract_commits(property,
                                    datetime(2015, 1, 1, tzinfo=timezone.utc),
                                    datetime(2020, 6, 1, tzinfo=timezone.utc),
                                    ["src-d/go-git"],
                                    ["mcuadros"],
                                    ["mcuadros"],
                                    BranchMiner(), 1, (6366825,), mdb, pdb, None)
    assert len(commits) == count
