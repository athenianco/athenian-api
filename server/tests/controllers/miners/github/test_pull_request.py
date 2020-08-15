from collections import defaultdict
import dataclasses
from datetime import date, datetime, timedelta, timezone
from itertools import chain
from typing import Any, Dict

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from athenian.api.controllers.miners.github.bots import bots
from athenian.api.controllers.miners.github.precomputed_prs import \
    discover_unreleased_prs, store_merged_unreleased_pull_request_facts, \
    store_open_pull_request_facts
from athenian.api.controllers.miners.github.pull_request import PullRequestFactsMiner, \
    PullRequestMiner
from athenian.api.controllers.miners.github.release import load_releases
from athenian.api.controllers.miners.types import Fallback, ParticipationKind, PullRequestFacts
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
import athenian.api.db
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.metadata.github import PullRequest
from tests.conftest import has_memcached


@with_defer
async def test_pr_miner_iter_smoke(
        branches, default_branches, mdb, pdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        mdb,
        pdb,
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


@with_defer
async def test_pr_miner_blacklist(branches, default_branches, mdb, pdb, release_match_setting_tag):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2017, month=1, day=12)
    miner, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        mdb,
        pdb,
        None,
    )
    node_ids = {pr.pr[PullRequest.node_id.key] for pr in miner}
    assert node_ids == {
        "MDExOlB1bGxSZXF1ZXN0MTAwMTI4Nzg0", "MDExOlB1bGxSZXF1ZXN0MTAwMjc5MDU4",
        "MDExOlB1bGxSZXF1ZXN0MTAwNjQ5MDk4", "MDExOlB1bGxSZXF1ZXN0MTAwNjU5OTI4",
        "MDExOlB1bGxSZXF1ZXN0OTI3NzM4NzY=", "MDExOlB1bGxSZXF1ZXN0OTUyMzA0Njg=",
        "MDExOlB1bGxSZXF1ZXN0OTg1NTIxMTc=",
    }
    miner, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        mdb,
        pdb,
        None,
        pr_blacklist=node_ids,
    )
    node_ids = [pr.pr[PullRequest.node_id.key] for pr in miner]
    assert len(node_ids) == 0


@pytest.mark.parametrize("with_memcached", [False, True])
@with_defer
async def test_pr_miner_iter_cache(branches, default_branches, mdb, pdb, cache, memcached,
                                   release_match_setting_tag, with_memcached):
    if with_memcached:
        if not has_memcached:
            raise pytest.skip("no memcached")
        cache = memcached
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        mdb,
        pdb,
        cache,
    )
    await wait_deferred()
    if not with_memcached:
        assert len(cache.mem) > 0
    first_data = list(miner)
    miner, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        None,
        None,
        cache,
    )
    await wait_deferred()
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
        {"src-d/go-git"},
        {ParticipationKind.AUTHOR: {"mcuadros"}, ParticipationKind.MERGER: {"mcuadros"}},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        None,
        None,
        cache,
    )
    if not with_memcached:
        cache_size = len(cache.mem)
        # check that the cache has not changed if we add some filters
        prs = list((await PullRequestMiner.mine(
            date_from,
            date_to,
            datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
            {"src-d/go-git"},
            {ParticipationKind.AUTHOR: {"mcuadros"}, ParticipationKind.MERGER: {"mcuadros"}},
            set(),
            branches, default_branches,
            False,
            release_match_setting_tag,
            None,
            None,
            cache,
        ))[0])
        await wait_deferred()
        assert len(cache.mem) == cache_size
        for pr in prs:
            text = ""
            for df in dataclasses.astuple(pr):
                try:
                    text += df.to_csv()
                except AttributeError:
                    text += str(df)
            assert "mcuadros" in text


@with_defer
async def test_pr_miner_iter_cache_incompatible(
        branches, default_branches, mdb, pdb, cache, release_match_setting_tag):
    date_from = date(year=2018, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        mdb,
        pdb,
        cache,
    )
    with pytest.raises(AttributeError):
        await PullRequestMiner.mine(
            date_from,
            date_to,
            datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
            {"src-d/gitbase"},
            {},
            set(),
            branches, default_branches,
            False,
            release_match_setting_tag,
            None,
            None,
            cache,
        )


@with_defer
async def test_pr_miner_cache_pr_blacklist(
        branches, default_branches, mdb, pdb, cache, release_match_setting_tag):
    """https://athenianco.atlassian.net/browse/DEV-206"""
    date_from = date(year=2018, month=1, day=11)
    date_to = date(year=2018, month=1, day=12)
    args = (
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        mdb,
        pdb,
        cache,
    )
    """
        There are 5 PRs:

        "MDExOlB1bGxSZXF1ZXN0MTM4MzExODAx",
        "MDExOlB1bGxSZXF1ZXN0MTU1NDg2ODkz",
        "MDExOlB1bGxSZXF1ZXN0MTYwODI1NTE4",
        "MDExOlB1bGxSZXF1ZXN0MTYyMDE1NTI4",
        "MDExOlB1bGxSZXF1ZXN0MTYyNDM2MzEx",
    """
    prs = list((await PullRequestMiner.mine(
        *args, pr_blacklist=["MDExOlB1bGxSZXF1ZXN0MTYyNDM2MzEx"]))[0])
    assert {pr.pr[PullRequest.node_id.key] for pr in prs} == {
        "MDExOlB1bGxSZXF1ZXN0MTM4MzExODAx", "MDExOlB1bGxSZXF1ZXN0MTU1NDg2ODkz",
        "MDExOlB1bGxSZXF1ZXN0MTYwODI1NTE4", "MDExOlB1bGxSZXF1ZXN0MTYyMDE1NTI4"}
    prs = list((await PullRequestMiner.mine(
        *args, pr_blacklist=["MDExOlB1bGxSZXF1ZXN0MTM4MzExODAx"]))[0])
    assert {pr.pr[PullRequest.node_id.key] for pr in prs} == {
        "MDExOlB1bGxSZXF1ZXN0MTYyNDM2MzEx", "MDExOlB1bGxSZXF1ZXN0MTU1NDg2ODkz",
        "MDExOlB1bGxSZXF1ZXN0MTYwODI1NTE4", "MDExOlB1bGxSZXF1ZXN0MTYyMDE1NTI4"}


@pytest.mark.parametrize("pk", [[v] for v in ParticipationKind] + [list(ParticipationKind)])
@with_defer
async def test_pr_miner_participant_filters(
        branches, default_branches, mdb, pdb, release_match_setting_tag, pk):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {v: {"mcuadros"} for v in pk},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        mdb,
        pdb,
        None,
    )
    count = 0
    for pr in miner:
        count += 1
        participants = pr.participants()
        if len(pk) == 1:
            assert "mcuadros" in participants[pk[0]], str(pr.pr)
        else:
            mentioned = False
            for v in pk:
                if "mcuadros" in participants[v]:
                    mentioned = True
                    break
            assert mentioned, str(pr.pr)
    assert count > 0


def validate_pull_request_facts(prmeta: Dict[str, Any], prt: PullRequestFacts):
    assert prmeta[PullRequest.node_id.key]
    assert prmeta[PullRequest.repository_full_name.key] == "src-d/go-git"
    for k, v in vars(prt).items():
        if not v or not isinstance(v, Fallback):
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


@with_defer
async def test_pr_facts_miner_smoke(
        branches, default_branches, mdb, pdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        mdb,
        pdb,
        None,
    )
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_empty_review_comments(
        branches, default_branches, mdb, pdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        mdb,
        pdb,
        None,
    )
    miner._review_comments = miner._review_comments.iloc[0:0]
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_empty_commits(
        branches, default_branches, mdb, pdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        mdb,
        pdb,
        None,
    )
    miner._commits = miner._commits.iloc[0:0]
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_bug_less_timestamp_float(
        branches, default_branches, mdb, pdb, release_match_setting_tag):
    date_from = date(2019, 10, 16) - timedelta(days=3)
    date_to = date(2019, 10, 16)
    miner, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        mdb,
        pdb,
        None,
    )
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    assert len(prts) > 0
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_empty_releases(branches, default_branches, mdb, pdb):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2018, month=1, day=1)
    miner, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        {"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="unknown", tags="", match=ReleaseMatch.branch)},
        mdb,
        pdb,
        None,
    )
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_mine_by_ids(branches, default_branches, dag, mdb, pdb, cache):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2018, month=1, day=1)
    time_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    time_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
    release_settings = {
        "github.com/src-d/go-git": ReleaseMatchSetting(
            branches="unknown", tags="", match=ReleaseMatch.branch),
    }
    miner, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        release_settings,
        mdb,
        pdb,
        None,
    )
    await wait_deferred()
    mined_prs = list(miner)
    prs = pd.DataFrame([pd.Series(pr.pr) for pr in mined_prs])
    prs.set_index(PullRequest.node_id.key, inplace=True)
    releases, matched_bys = await load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to, release_settings,
        mdb, pdb, cache)
    dfs1, _ = await PullRequestMiner.mine_by_ids(
        prs,
        [],
        time_to,
        releases,
        matched_bys,
        branches,
        default_branches,
        dag,
        release_settings,
        mdb,
        pdb,
        cache,
    )
    dfs2, _ = await PullRequestMiner.mine_by_ids(
        prs,
        [],
        time_to,
        releases,
        matched_bys,
        branches,
        default_branches,
        dag,
        release_settings,
        mdb,
        pdb,
        cache,
    )
    for df1, df2 in zip(dfs1, dfs2):
        assert (df1.fillna(0) == df2.fillna(0)).all().all()
    for i, df1 in enumerate(dfs1):
        if i == len(dfs1) - 2:
            # releases
            continue
        df = pd.concat([dataclasses.astuple(pr)[i + 1] for pr in mined_prs])
        df.sort_index(inplace=True)
        df1.index = df1.index.droplevel(0)
        df1.sort_index(inplace=True)
        assert (df.fillna(0) == df1.fillna(0)).all().all()


@with_defer
async def test_pr_miner_exclude_inactive(
        branches, default_branches, mdb, pdb, release_match_setting_tag):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2017, month=1, day=12)
    miner, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        True,
        release_match_setting_tag,
        mdb,
        pdb,
        None,
    )
    node_ids = {pr.pr[PullRequest.node_id.key] for pr in miner}
    assert node_ids == {
        "MDExOlB1bGxSZXF1ZXN0MTAwMTI4Nzg0", "MDExOlB1bGxSZXF1ZXN0MTAwMjc5MDU4",
        "MDExOlB1bGxSZXF1ZXN0MTAwNjQ5MDk4", "MDExOlB1bGxSZXF1ZXN0MTAwNjU5OTI4",
        "MDExOlB1bGxSZXF1ZXN0OTg1NTIxMTc=", "MDExOlB1bGxSZXF1ZXN0OTUyMzA0Njg=",
    }


@with_defer
async def test_pr_miner_unreleased_pdb(mdb, pdb, release_match_setting_tag):
    time_from = datetime(2018, 11, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    miner_incomplete, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, set(), pd.DataFrame(), {}, False,
        release_match_setting_tag, mdb, pdb, None)
    time_from_lookback = time_from - timedelta(days=60)
    await wait_deferred()
    # populate pdb
    await PullRequestMiner.mine(
        time_from_lookback.date(), time_to.date(), time_from_lookback, time_to,
        {"src-d/go-git"}, {}, set(), pd.DataFrame(), {}, False,
        release_match_setting_tag, mdb, pdb, None)
    await wait_deferred()
    miner_complete, facts, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, set(), pd.DataFrame(), {}, False,
        release_match_setting_tag, mdb, pdb, None)
    assert isinstance(facts, dict)
    assert len(facts) == 0
    await wait_deferred()
    assert len(miner_incomplete._prs) == 19
    assert len(miner_complete._prs) == 19 + 42
    miner_active, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, set(), pd.DataFrame(), {}, True,
        release_match_setting_tag, mdb, pdb, None)
    assert len(miner_active._prs) <= 19


@with_defer
async def test_pr_miner_labels(mdb, pdb, release_match_setting_tag, cache):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    miner, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, {"bug", "enhancement"}, pd.DataFrame(), {}, False,
        release_match_setting_tag, mdb, pdb, None)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.key] for pr in prs} == {887, 921, 958, 947, 950, 949}
    miner, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, {"bug", "enhancement"}, pd.DataFrame(), {}, False,
        release_match_setting_tag, mdb, pdb, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.key] for pr in prs} == {887, 921, 958, 947, 950, 949}
    miner, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, {"bug", "plumbing"}, pd.DataFrame(), {}, False,
        release_match_setting_tag, mdb, pdb, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.key] for pr in prs} == {921, 940, 946, 950, 958}
    await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, set(), pd.DataFrame(), {}, False,
        release_match_setting_tag, mdb, pdb, cache)
    miner, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, {"bug", "plumbing"}, pd.DataFrame(), {}, False,
        release_match_setting_tag, None, None, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.key] for pr in prs} == {921, 940, 946, 950, 958}
    miner, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, {"bug"}, pd.DataFrame(), {}, False,
        release_match_setting_tag, None, None, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.key] for pr in prs} == {921, 950, 958}


@with_defer
async def test_pr_miner_labels_unreleased(mdb, pdb, release_match_setting_tag):
    time_from = datetime(2018, 11, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    time_from_lookback = time_from - timedelta(days=60)
    # populate pdb
    await PullRequestMiner.mine(
        time_from_lookback.date(), time_to.date(), time_from_lookback, time_to,
        {"src-d/go-git"}, {}, set(), pd.DataFrame(), {}, False,
        release_match_setting_tag, mdb, pdb, None)
    await wait_deferred()
    miner_complete, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, {"bug"}, pd.DataFrame(), {}, False,
        release_match_setting_tag, mdb, pdb, None,
        pr_blacklist=["MDExOlB1bGxSZXF1ZXN0MjA5MjA0MDQz",
                      "MDExOlB1bGxSZXF1ZXN0MjE2MTA0NzY1",
                      "MDExOlB1bGxSZXF1ZXN0MjEzODQ1NDUx"])
    await wait_deferred()
    assert len(miner_complete._prs) == 3
    miner_complete, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, {"bug"}, pd.DataFrame(), {}, True,
        release_match_setting_tag, mdb, pdb, None)
    assert len(miner_complete._prs) == 0


@with_defer
async def test_pr_miner_unreleased_facts(
        branches, default_branches, mdb, pdb, release_match_setting_tag):
    date_from = date(year=2018, month=1, day=1)
    date_to = date(year=2020, month=4, day=1)
    time_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    time_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
    args = (
        date_from,
        date_to,
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        set(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        mdb,
        pdb,
        None,
    )
    athenian.api.db._testing = False
    try:
        miner, unreleased_facts, matched_bys = await PullRequestMiner.mine(*args)
    finally:
        athenian.api.db._testing = True
    await wait_deferred()
    assert unreleased_facts == {}
    open_prs_and_facts = []
    merged_unreleased_prs_and_facts = []
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    for pr in miner:
        facts = facts_miner(pr)
        if not facts.closed:
            open_prs_and_facts.append((pr.pr, facts))
        elif facts.merged and not facts.released:
            merged_unreleased_prs_and_facts.append((pr.pr, facts))
    assert len(open_prs_and_facts) == 21
    assert len(merged_unreleased_prs_and_facts) == 11
    discovered = await discover_unreleased_prs(
        miner._prs, time_to, matched_bys, default_branches, release_match_setting_tag, pdb)
    assert {pr[PullRequest.node_id.key] for pr, _ in merged_unreleased_prs_and_facts} == \
        set(discovered)
    await store_open_pull_request_facts(open_prs_and_facts, pdb)
    await store_merged_unreleased_pull_request_facts(
        merged_unreleased_prs_and_facts, matched_bys, default_branches,
        release_match_setting_tag, pdb)
    miner, unreleased_facts, _ = await PullRequestMiner.mine(*args)
    true_pr_node_set = {pr[PullRequest.node_id.key] for pr, _ in chain(
        open_prs_and_facts, merged_unreleased_prs_and_facts)}
    assert set(unreleased_facts) == true_pr_node_set
    assert len(miner) == 326
    dropped = miner.drop(unreleased_facts)
    assert set(dropped) == set(unreleased_facts)
    assert len(miner) == 294
