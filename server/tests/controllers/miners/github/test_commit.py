from datetime import datetime, timezone

import pytest

from athenian.api.controllers.miners.github.branches import BranchMiner
from athenian.api.controllers.miners.github.commit import extract_commits, FilterCommitsProperty
from athenian.api.defer import wait_deferred, with_defer


@pytest.mark.parametrize("property, count", [
    (FilterCommitsProperty.NO_PR_MERGES, 596),
    (FilterCommitsProperty.BYPASSING_PRS, 289),
])
@with_defer
async def test_extract_commits_users(mdb, pdb, property, count, cache, prefixer):
    """
    date_from: datetime,
                          date_to: datetime,
                          repos: Collection[str],
                          with_author: Optional[Collection[str]],
                          with_committer: Optional[Collection[str]],
                          only_default_branch: bool,
                          branch_miner: Optional[BranchMiner],
                          account: int,
                          meta_ids: Tuple[int, ...],
                          mdb: DatabaseLike,
                          pdb: DatabaseLike,
                          cache: Optional[aiomcache.Client],
    """
    args = dict(
        prop=property,
        date_from=datetime(2015, 1, 1, tzinfo=timezone.utc),
        date_to=datetime(2020, 6, 1, tzinfo=timezone.utc),
        repos=["src-d/go-git"],
        with_author=["mcuadros"],
        with_committer=["mcuadros"],
        only_default_branch=False,
        branch_miner=BranchMiner(),
        prefixer=prefixer,
        account=1,
        meta_ids=(6366825,),
        mdb=mdb,
        pdb=pdb,
        cache=cache,
    )
    commits = await extract_commits(**args)
    assert len(commits) == count
    await wait_deferred()
    args["only_default_branch"] = True
    commits = await extract_commits(**args)
    assert len(commits) < count
