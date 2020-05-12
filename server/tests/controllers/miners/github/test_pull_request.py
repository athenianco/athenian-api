from collections import defaultdict
import dataclasses
from datetime import date, datetime, timedelta, timezone
from typing import Any, Dict

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from athenian.api.controllers.miners.github.pull_request import PullRequestMiner, \
    PullRequestTimes, PullRequestTimesMiner
from athenian.api.controllers.settings import Match, ReleaseMatchSetting
from athenian.api.models.metadata.github import PullRequest
from tests.conftest import has_memcached


async def test_pr_miner_iter(mdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
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


@pytest.mark.parametrize("with_memcached", [False, True])
async def test_pr_miner_iter_cache(mdb, cache, memcached, release_match_setting_tag,
                                   with_memcached):
    if with_memcached:
        if not has_memcached:
            raise pytest.skip("no memcached")
        cache = memcached
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
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
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
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
    # we still use the cache here
    await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        ["src-d/go-git"],
        release_match_setting_tag,
        ["mcuadros"],
        None,
        cache,
    )
    if not with_memcached:
        cache_size = len(cache.mem)
        # check that the cache has not changed if we add some filters
        prs = list(await PullRequestMiner.mine(
            date_from,
            date_to,
            datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
            ["src-d/go-git"],
            release_match_setting_tag,
            ["mcuadros"],
            None,
            cache,
        ))
        assert len(cache.mem) == cache_size
        for pr in prs:
            text = ""
            for df in dataclasses.astuple(pr):
                try:
                    text += df.to_csv()
                except AttributeError:
                    text += str(df)
            assert "mcuadros" in text


async def test_pr_miner_iter_cache_incompatible(mdb, cache, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        cache,
    )
    with pytest.raises(AttributeError):
        await PullRequestMiner.mine(
            date_from,
            date_to,
            datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
            ["src-d/gitbase"],
            release_match_setting_tag,
            [],
            None,
            cache,
        )


def validate_pull_request_times(prmeta: Dict[str, Any], prt: PullRequestTimes):
    assert prmeta[PullRequest.node_id.key]
    assert prmeta[PullRequest.repository_full_name.key] == "src-d/go-git"
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


async def test_pr_times_miner_smoke(mdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        None,
    )
    times_miner = PullRequestTimesMiner()
    prts = [(pr.pr, times_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_times(*prt)


async def test_pr_times_miner_empty_review_comments(mdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        None,
    )
    miner._review_comments = miner._review_comments.iloc[0:0]
    times_miner = PullRequestTimesMiner()
    prts = [(pr.pr, times_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_times(*prt)


async def test_pr_times_miner_empty_commits(mdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        None,
    )
    miner._commits = miner._commits.iloc[0:0]
    times_miner = PullRequestTimesMiner()
    prts = [(pr.pr, times_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_times(*prt)


async def test_pr_times_miner_bug_less_timestamp_float(mdb, release_match_setting_tag):
    date_from = date(2019, 10, 16) - timedelta(days=3)
    date_to = date(2019, 10, 16)
    miner = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        ["src-d/go-git"],
        release_match_setting_tag,
        [],
        mdb,
        None,
    )
    times_miner = PullRequestTimesMiner()
    prts = [(pr.pr, times_miner(pr)) for pr in miner]
    assert len(prts) > 0
    for prt in prts:
        validate_pull_request_times(*prt)


async def test_pr_times_miner_empty_releases(mdb):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2018, month=1, day=1)
    miner = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        ["src-d/go-git"],
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="unknown", tags="", match=Match.branch)},
        [],
        mdb,
        None,
    )
    times_miner = PullRequestTimesMiner()
    prts = [(pr.pr, times_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_times(*prt)


async def test_pr_mine_by_ids(mdb, cache):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2018, month=1, day=1)
    time_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    time_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
    release_settings = {
        "github.com/src-d/go-git": ReleaseMatchSetting(
            branches="unknown", tags="", match=Match.branch),
    }
    miner = await PullRequestMiner.mine(
        date_from,
        date_to,
        time_from,
        time_to,
        ["src-d/go-git"],
        release_settings,
        [],
        mdb,
        None,
    )
    mined_prs = list(miner)
    prs = pd.DataFrame([pd.Series(pr.pr) for pr in mined_prs])
    prs.set_index(PullRequest.node_id.key, inplace=True)
    dfs1 = await PullRequestMiner.mine_by_ids(
        prs,
        time_from,
        time_to,
        release_settings,
        mdb,
        cache,
    )
    dfs2 = await PullRequestMiner.mine_by_ids(
        prs,
        time_from,
        time_to,
        release_settings,
        mdb,
        cache,
    )
    for df1, df2 in zip(dfs1, dfs2):
        assert (df1.fillna(0) == df2.fillna(0)).all().all()
    for i in range(len(dfs1) - 1):
        df = pd.concat([dataclasses.astuple(pr)[i + 1] for pr in mined_prs])
        df.sort_index(inplace=True)
        dfs1[i].index = dfs1[i].index.droplevel(0)
        dfs1[i].sort_index(inplace=True)
        assert (df.fillna(0) == dfs1[i].fillna(0)).all().all()
