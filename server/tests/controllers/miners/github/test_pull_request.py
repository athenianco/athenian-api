from collections import defaultdict
from datetime import datetime, timedelta

from athenian.api.controllers.miners.github.pull_request import PullRequestMiner


async def test_pr_miner_iter(mdb):
    miner = await PullRequestMiner.mine(
        datetime.now() - timedelta(days=10 * 365),
        datetime.now(),
        ["src-d/go-git"],
        [],
        mdb,
    )
    with_data = defaultdict(int)
    for pr, reviews, review_comments, commits in miner:
        with_data["reviews"] += len(reviews) > 0
        with_data["review_comments"] += len(review_comments) > 0
        with_data["commits"] += len(commits) > 0
    assert with_data["reviews"] > 0
    assert with_data["review_comments"] > 0
    assert with_data["commits"] > 0
    print(dict(with_data))
