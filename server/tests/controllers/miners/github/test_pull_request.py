from collections import defaultdict
import dataclasses
from datetime import date, timedelta

from pandas.testing import assert_frame_equal
import pytest

from athenian.api.controllers.miners.github.pull_request import PullRequestMiner, \
    PullRequestTimes, PullRequestTimesMiner
from tests.conftest import has_memcached


async def test_pr_miner_iter(mdb, release_match_setting_tag):
    miner = await PullRequestMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        None,
    )
    with_data = defaultdict(int)
    size = 0
    for pr in miner:
        size += 1
        with_data["prs"] += bool(pr.pr)
        with_data["reviews"] += not pr.reviews.empty
        with_data["review_comments"] += not pr.review_comments.empty
        with_data["review_requests"] += not pr.review_requests.empty
        with_data["comments"] += not pr.comments.empty
        with_data["commits"] += not pr.commits.empty
        with_data["releases"] += bool(pr.release)
    for k, v in with_data.items():
        assert v > 0, k
    assert with_data["prs"] == size


@pytest.mark.parametrize("with_memcached", [False] + ([True] if has_memcached else []))
async def test_pr_miner_iter_cache(mdb, cache, memcached, release_match_setting_tag,
                                   with_memcached):
    if with_memcached:
        cache = memcached
    miner = await PullRequestMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        cache,
    )
    if not with_memcached:
        assert len(cache.mem) > 0
    first_data = list(miner)
    miner = await PullRequestMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        None,
        cache,
    )
    second_data = list(miner)
    for first, second in zip(first_data, second_data):
        for fv, sv in zip(dataclasses.astuple(first), dataclasses.astuple(second)):
            if isinstance(fv, dict):
                assert str(fv) == str(sv)
            else:
                assert_frame_equal(fv.reset_index(), sv.reset_index())


def validate_pull_request_times(prt: PullRequestTimes):
    for k, v in vars(prt).items():
        if not v:
            continue
        if k not in ("first_commit", "last_commit", "last_commit_before_first_review"):
            assert prt.created <= v, k
        assert prt.work_began <= v, k
        if prt.closed and k != "released":
            assert prt.closed >= v
        if prt.released:
            assert prt.released >= v
    if prt.first_commit:
        assert prt.last_commit >= prt.first_commit
    else:
        assert not prt.last_commit
    if prt.first_comment_on_first_review:
        assert prt.last_commit_before_first_review >= prt.first_commit
        assert prt.last_commit_before_first_review <= prt.last_commit
        assert prt.last_commit_before_first_review <= prt.first_comment_on_first_review
        assert prt.first_review_request <= prt.first_comment_on_first_review
        if prt.last_review:
            # There may be a regular comment that counts for `first_comment_on_first_review`
            # but no actual review submission.
            assert prt.last_review >= prt.first_comment_on_first_review
        assert prt.first_review_request <= prt.first_comment_on_first_review
    else:
        assert not prt.last_review
        assert not prt.last_commit_before_first_review
    if prt.approved:
        assert prt.first_comment_on_first_review <= prt.approved
        assert prt.first_review_request <= prt.approved
        if prt.last_review:
            assert prt.last_review >= prt.approved
        if prt.merged:
            assert prt.approved <= prt.merged
            assert prt.closed


async def test_pr_times_miner(mdb, release_match_setting_tag):
    miner = await PullRequestTimesMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        None,
    )
    for prt in miner:
        validate_pull_request_times(prt)


async def test_pr_times_miner_empty_review_comments(mdb, release_match_setting_tag):
    miner = await PullRequestTimesMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        None,
    )
    miner._review_comments = miner._review_comments.iloc[0:0]
    for prt in miner:
        validate_pull_request_times(prt)


async def test_pr_times_miner_empty_commits(mdb, release_match_setting_tag):
    miner = await PullRequestTimesMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        None,
    )
    miner._commits = miner._commits.iloc[0:0]
    for prt in miner:
        validate_pull_request_times(prt)


async def test_pr_times_miner_bug_less_timestamp_float(mdb, release_match_setting_tag):
    miner = await PullRequestTimesMiner.mine(
        date(2019, 10, 16) - timedelta(days=3),
        date(2019, 10, 16),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        None,
    )
    prts = list(miner)
    assert len(prts) > 0
    for prt in prts:
        validate_pull_request_times(prt)
