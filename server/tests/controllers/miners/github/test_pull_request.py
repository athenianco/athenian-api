from collections import defaultdict
import dataclasses
from datetime import date, timedelta
from typing import List

import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal
import pytest

from athenian.api.controllers.miners.github.pull_request import PullRequestListMiner, \
    PullRequestMiner, PullRequestTimes, PullRequestTimesMiner
from athenian.api.controllers.miners.pull_request_list_item import ParticipationKind, \
    PullRequestListItem, Stage
from tests.conftest import has_memcached


async def test_pr_miner_iter(mdb):
    miner = await PullRequestMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        [],
        mdb,
        None,
    )
    with_data = defaultdict(int)
    for pr in miner:
        with_data["reviews"] += len(pr.reviews) > 0
        with_data["review_comments"] += len(pr.review_comments) > 0
        with_data["comments"] += len(pr.comments) > 0
        with_data["commits"] += len(pr.commits) > 0
    assert with_data["reviews"] > 0
    assert with_data["review_comments"] > 0
    assert with_data["comments"] > 0
    assert with_data["commits"] > 0


@pytest.mark.parametrize("with_memcached", [False] + ([True] if has_memcached else []))
async def test_pr_miner_iter_cache(mdb, cache, memcached, with_memcached):
    if with_memcached:
        cache = memcached
    miner = await PullRequestMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        [],
        mdb,
        cache,
    )
    if not with_memcached:
        assert len(cache.mem) == 1
    first_data = list(miner)
    miner = await PullRequestMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        [],
        None,
        cache,
    )
    second_data = list(miner)
    for first, second in zip(first_data, second_data):
        for fv, sv in zip(dataclasses.astuple(first), dataclasses.astuple(second)):
            if isinstance(fv, pd.Series):
                assert_series_equal(fv, sv)
            else:
                assert_frame_equal(fv.reset_index(), sv.reset_index())


def validate_pull_request_times(prt: PullRequestTimes):
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
        assert prt.last_commit_before_first_review.best <= prt.first_comment_on_first_review.best
        assert prt.first_review_request.best <= prt.first_comment_on_first_review.best
    if prt.approved:
        assert prt.first_comment_on_first_review.best <= prt.approved.best
        assert prt.first_review_request.best <= prt.approved.best
        if prt.merged:
            assert prt.approved.best <= prt.merged.best


async def test_pr_times_miner(mdb):
    miner = await PullRequestTimesMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        [],
        mdb,
        None,
    )
    for prt in miner:
        validate_pull_request_times(prt)


async def test_pr_times_miner_empty_review_comments(mdb):
    miner = await PullRequestTimesMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        [],
        mdb,
        None,
    )
    miner._review_comments = miner._review_comments.iloc[0:0]
    for prt in miner:
        validate_pull_request_times(prt)


async def test_pr_times_miner_empty_commits(mdb):
    miner = await PullRequestTimesMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        [],
        mdb,
        None,
    )
    miner._commits = miner._commits.iloc[0:0]
    for prt in miner:
        validate_pull_request_times(prt)


async def test_pr_times_miner_bug_less_timestamp_float(mdb):
    miner = await PullRequestTimesMiner.mine(
        date(2019, 10, 16) - timedelta(days=3),
        date(2019, 10, 16),
        ["src-d/go-git"],
        [],
        mdb,
        None,
    )
    prts = list(miner)
    assert len(prts) > 0
    for prt in prts:
        validate_pull_request_times(prt)


async def test_pr_list_miner_none(mdb):
    miner = await PullRequestListMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        [],
        mdb,
        None,
    )
    prs = list(miner)
    assert not prs


async def test_pr_list_miner_match_participants(mdb):
    miner = await PullRequestListMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        [],
        mdb,
        None,
    )
    miner.stages = set(Stage)
    miner.participants = {ParticipationKind.AUTHOR: ["github.com/mcuadros", "github.com/smola"],
                          ParticipationKind.COMMENTER: ["github.com/mcuadros"]}
    prs = list(miner)  # type: List[PullRequestListItem]
    assert prs
    for pr in prs:
        mcuadros_is_author = "github.com/mcuadros" in pr.participants[ParticipationKind.AUTHOR]
        smola_is_author = "github.com/smola" in pr.participants[ParticipationKind.AUTHOR]
        mcuadros_is_only_commenter = (
            ("github.com/mcuadros" in pr.participants[ParticipationKind.COMMENTER])
            and  # noqa
            (not mcuadros_is_author)
            and  # noqa
            (not smola_is_author)
        )
        assert mcuadros_is_author or smola_is_author or mcuadros_is_only_commenter


async def test_pr_list_miner_no_participants(mdb):
    miner = await PullRequestListMiner.mine(
        date.today() - timedelta(days=10 * 365),
        date.today(),
        ["src-d/go-git"],
        [],
        mdb,
        None,
    )
    miner.stages = set(Stage)
    prs = list(miner)
    assert prs
