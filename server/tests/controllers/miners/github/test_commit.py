from datetime import datetime, timezone

from athenian.api.controllers.miners.github.commit import extract_commits, FilterCommitsProperty


async def test_extract_commits_users(mdb):
    commits = await extract_commits((6366825,),
                                    FilterCommitsProperty.NO_PR_MERGES,
                                    datetime(2015, 1, 1, tzinfo=timezone.utc),
                                    datetime(2020, 6, 1, tzinfo=timezone.utc),
                                    ["src-d/go-git"],
                                    ["mcuadros"],
                                    ["mcuadros"],
                                    mdb, None)
    assert len(commits) == 750
