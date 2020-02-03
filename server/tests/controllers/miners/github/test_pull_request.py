from collections import defaultdict
from datetime import datetime, timedelta

from athenian.api.controllers.miners.github.pull_request import PullRequestMiner,\
    PullRequestTimesMiner


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


async def test_pr_times_miner(mdb):
    miner = await PullRequestTimesMiner.mine(
        datetime.now() - timedelta(days=10 * 365),
        datetime.now(),
        ["src-d/go-git"],
        [],
        mdb,
    )
    for prt in miner:
        for k, v in vars(prt).items():
            if not v:
                continue
            if k not in ("first_commit", "last_commit", "last_commit_before_first_review"):
                assert prt.created.best <= v.best, k
            assert prt.work_began.best <= v.best, k
            if prt.closed:
                assert prt.closed.best >= v.best
        if prt.first_commit:
            assert prt.last_commit.best >= prt.first_commit.best
            assert prt.last_commit_before_first_review.best >= prt.first_commit.best
            assert prt.last_commit_before_first_review.best <= prt.last_commit.best
        if prt.first_comment_on_first_review:
            assert prt.last_commit_before_first_review.best <= \
                   prt.first_comment_on_first_review.best
        if prt.approved:
            assert prt.first_comment_on_first_review.best <= prt.approved.best
            assert prt.first_review_request.best <= prt.approved.best
            if prt.merged:
                assert prt.approved.best <= prt.merged.best
