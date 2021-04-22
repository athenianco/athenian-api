from collections import defaultdict
import dataclasses
from datetime import date, datetime, timedelta, timezone
from itertools import chain
import pickle
from typing import Any, Dict

import pandas as pd
from pandas.core.dtypes.common import is_datetime64_any_dtype
from pandas.testing import assert_frame_equal
import pytest
from sqlalchemy import select, update

from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.bots import bots
from athenian.api.controllers.miners.github.branches import extract_branches
from athenian.api.controllers.miners.github.commit import _empty_dag
from athenian.api.controllers.miners.github.precomputed_prs import \
    load_merged_unreleased_pull_request_facts, store_merged_unreleased_pull_request_facts, \
    store_open_pull_request_facts
from athenian.api.controllers.miners.github.pull_request import PullRequestFactsMiner, \
    PullRequestMiner
from athenian.api.controllers.miners.github.release_load import load_releases
from athenian.api.controllers.miners.types import DAG, MinedPullRequest, PRParticipationKind, \
    PullRequestFacts
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting, ReleaseSettings
import athenian.api.db
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.metadata.github import Branch, PullRequest
from athenian.api.models.metadata.jira import Issue
from athenian.api.models.precomputed.models import GitHubCommitHistory, \
    GitHubMergedPullRequestFacts
from tests.conftest import has_memcached
from tests.controllers.conftest import FakeFacts, fetch_dag
from tests.controllers.test_filter_controller import force_push_dropped_go_git_pr_numbers


@with_defer
async def test_pr_miner_iter_smoke(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
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
async def test_pr_miner_blacklist(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2017, month=1, day=12)
    miner, _, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    node_ids = {pr.pr[PullRequest.node_id.key] for pr in miner}
    assert node_ids == {
        "MDExOlB1bGxSZXF1ZXN0MTAwMTI4Nzg0", "MDExOlB1bGxSZXF1ZXN0MTAwMjc5MDU4",
        "MDExOlB1bGxSZXF1ZXN0MTAwNjQ5MDk4", "MDExOlB1bGxSZXF1ZXN0MTAwNjU5OTI4",
        "MDExOlB1bGxSZXF1ZXN0OTI3NzM4NzY=", "MDExOlB1bGxSZXF1ZXN0OTUyMzA0Njg=",
        "MDExOlB1bGxSZXF1ZXN0OTg1NTIxMTc=",
    }
    miner, _, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        pr_blacklist=(node_ids, {}),
    )
    node_ids = [pr.pr[PullRequest.node_id.key] for pr in miner]
    assert len(node_ids) == 0


@pytest.mark.parametrize("with_memcached", [False, True])
@with_defer
async def test_pr_miner_iter_cache(branches, default_branches, mdb, pdb, rdb, cache, memcached,
                                   release_match_setting_tag, with_memcached):
    if with_memcached:
        if not has_memcached:
            raise pytest.skip("no memcached")
        cache = memcached
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    await wait_deferred()
    if not with_memcached:
        assert len(cache.mem) > 0
    first_data = list(miner)
    miner, _, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        None,
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
        {PRParticipationKind.AUTHOR: {"mcuadros"}, PRParticipationKind.MERGER: {"mcuadros"}},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        None,
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
            {PRParticipationKind.AUTHOR: {"mcuadros"}, PRParticipationKind.MERGER: {"mcuadros"}},
            LabelFilter.empty(),
            JIRAFilter.empty(),
            branches, default_branches,
            False,
            release_match_setting_tag,
            1,
            (6366825,),
            None,
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
        branches, default_branches, mdb, pdb, rdb, cache, release_match_setting_tag):
    date_from = date(year=2018, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    with pytest.raises(AssertionError):
        await PullRequestMiner.mine(
            date_from,
            date_to,
            datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
            datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
            {"src-d/gitbase"},
            {},
            LabelFilter.empty(),
            JIRAFilter.empty(),
            branches, default_branches,
            False,
            release_match_setting_tag,
            1,
            (6366825,),
            None,
            None,
            None,
            cache,
        )


@with_defer
async def test_pr_miner_cache_pr_blacklist(
        branches, default_branches, mdb, pdb, rdb, cache, release_match_setting_tag):
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
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
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
        *args, pr_blacklist=(["MDExOlB1bGxSZXF1ZXN0MTYyNDM2MzEx"], {})))[0])
    assert {pr.pr[PullRequest.node_id.key] for pr in prs} == {
        "MDExOlB1bGxSZXF1ZXN0MTM4MzExODAx", "MDExOlB1bGxSZXF1ZXN0MTU1NDg2ODkz",
        "MDExOlB1bGxSZXF1ZXN0MTYwODI1NTE4", "MDExOlB1bGxSZXF1ZXN0MTYyMDE1NTI4"}
    prs = list((await PullRequestMiner.mine(
        *args, pr_blacklist=(["MDExOlB1bGxSZXF1ZXN0MTM4MzExODAx"], {})))[0])
    assert {pr.pr[PullRequest.node_id.key] for pr in prs} == {
        "MDExOlB1bGxSZXF1ZXN0MTYyNDM2MzEx", "MDExOlB1bGxSZXF1ZXN0MTU1NDg2ODkz",
        "MDExOlB1bGxSZXF1ZXN0MTYwODI1NTE4", "MDExOlB1bGxSZXF1ZXN0MTYyMDE1NTI4"}


@pytest.mark.parametrize("pk", [[v] for v in PRParticipationKind] + [list(PRParticipationKind)])
@with_defer
async def test_pr_miner_participant_filters(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, pk):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {v: {"mcuadros"} for v in pk},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
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
    for k, v in prt.items():
        if not isinstance(v, pd.Timestamp) or not v:
            continue
        if k not in ("first_commit", "last_commit", "last_commit_before_first_review",
                     "work_began"):
            assert prt.created <= v, k
        assert prt.work_began <= v, k
        if prt.closed and k != "released":
            assert prt.closed >= v, k
        if prt.released:
            assert prt.released >= v
    for t in prt.reviews:
        assert t >= prt.created, "review before creation"
        assert t <= prt.last_review, "review after last review"
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
    if prt.first_review_request_exact:
        assert prt.first_review_request
    if prt.approved:
        if prt.last_commit_before_first_review:
            # force pushes can happen after the approval
            assert prt.last_commit_before_first_review <= prt.approved
        assert prt.first_comment_on_first_review <= prt.approved
        assert prt.first_review_request <= prt.approved
        if prt.last_review:
            assert prt.last_review >= prt.approved
        if prt.merged:
            assert prt.approved <= prt.merged
            assert prt.closed


@with_defer
async def test_pr_facts_miner_smoke(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_empty_review_comments(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    miner._dfs.review_comments = miner._dfs.review_comments.iloc[0:0]
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_empty_commits(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    miner._dfs.commits = miner._dfs.commits.iloc[0:0]
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_bug_less_timestamp_float(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag):
    date_from = date(2019, 10, 16) - timedelta(days=3)
    date_to = date(2019, 10, 16)
    miner, _, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    assert len(prts) > 0
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_empty_releases(branches, default_branches, mdb, pdb, rdb):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2018, month=1, day=1)
    miner, _, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="unknown", tags="", match=ReleaseMatch.branch)}),
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_mine_by_ids(branches, default_branches, dag, mdb, pdb, rdb, cache):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2018, month=1, day=1)
    time_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    time_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
    release_settings = ReleaseSettings({
        "github.com/src-d/go-git": ReleaseMatchSetting(
            branches="unknown", tags="", match=ReleaseMatch.branch),
    })
    miner, _, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_settings,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    await wait_deferred()
    mined_prs = list(miner)
    prs = pd.DataFrame([pd.Series(pr.pr) for pr in mined_prs])
    prs.set_index(PullRequest.node_id.key, inplace=True)
    releases, matched_bys = await load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to,
        release_settings, 1, (6366825,), mdb, pdb, rdb, cache)
    dfs1, _, _ = await PullRequestMiner.mine_by_ids(
        prs,
        [],
        time_to,
        releases,
        matched_bys,
        branches,
        default_branches,
        dag,
        release_settings,
        1,
        (6366825,),
        mdb,
        pdb,
        cache,
    )
    dfs2, _, _ = await PullRequestMiner.mine_by_ids(
        prs,
        [],
        time_to,
        releases,
        matched_bys,
        branches,
        default_branches,
        dag,
        release_settings,
        1,
        (6366825,),
        mdb,
        pdb,
        cache,
    )
    for df1, df2 in zip(dfs1, dfs2):
        assert (df1.fillna(0) == df2.fillna(0)).all().all()
    for field in dataclasses.fields(MinedPullRequest):
        field = field.name
        if field in ("release", "jira"):
            continue
        df1 = getattr(dfs1, field.rstrip("s") + "s")
        records = [getattr(pr, field) for pr in mined_prs]
        if field.endswith("s"):
            df = pd.concat(records)
        else:
            df = pd.DataFrame.from_records(records)
        if field != "pr":
            df1.index = df1.index.droplevel(0)
        else:
            df.set_index(PullRequest.node_id.key, inplace=True)
        df.sort_index(inplace=True)
        df1.sort_index(inplace=True)
        assert (df.fillna(0) == df1.fillna(0)).all().all()


@with_defer
async def test_pr_miner_exclude_inactive(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2017, month=1, day=12)
    miner, _, _, _ = await PullRequestMiner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        True,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    node_ids = {pr.pr[PullRequest.node_id.key] for pr in miner}
    assert node_ids == {
        "MDExOlB1bGxSZXF1ZXN0MTAwMTI4Nzg0", "MDExOlB1bGxSZXF1ZXN0MTAwMjc5MDU4",
        "MDExOlB1bGxSZXF1ZXN0MTAwNjQ5MDk4", "MDExOlB1bGxSZXF1ZXN0MTAwNjU5OTI4",
        "MDExOlB1bGxSZXF1ZXN0OTg1NTIxMTc=", "MDExOlB1bGxSZXF1ZXN0OTUyMzA0Njg=",
    }


@with_defer
async def test_pr_miner_unreleased_pdb(mdb, pdb, rdb, release_match_setting_tag):
    time_from = datetime(2018, 11, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    miner_incomplete, _, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None)
    time_from_lookback = time_from - timedelta(days=60)
    await wait_deferred()
    # populate pdb
    await PullRequestMiner.mine(
        time_from_lookback.date(), time_to.date(), time_from_lookback, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    miner_complete, facts, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None)
    assert isinstance(facts, dict)
    assert len(facts) == 0
    await wait_deferred()
    assert len(miner_incomplete._dfs.prs) == 19
    assert len(miner_complete._dfs.prs) == 19 + 42
    miner_active, _, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, True, release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None)
    assert len(miner_active._dfs.prs) <= 19


@with_defer
async def test_pr_miner_labels_torture(mdb, pdb, rdb, release_match_setting_tag, cache):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    miner, _, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug", "enhancement"}, set()), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.key] for pr in prs} == {887, 921, 958, 947, 950, 949}
    miner, _, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug", "enhancement"}, set()), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.key] for pr in prs} == {887, 921, 958, 947, 950, 949}
    miner, _, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug", "plumbing"}, set()), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.key] for pr in prs} == {921, 940, 946, 950, 958}
    await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, cache)
    miner, _, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug", "plumbing"}, set()), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), None, None, None, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.key] for pr in prs} == {921, 940, 946, 950, 958}
    miner, _, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug"}, set()), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), None, None, None, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.key] for pr in prs} == {921, 950, 958}
    miner, _, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug", "plumbing"}, {"plumbing"}), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), None, None, None, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.key] for pr in prs} == {921}
    miner, _, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug", "plumbing"}, {"plumbing"}), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.key] for pr in prs} == {921}
    miner, _, _, event = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug"}, {"plumbing"}), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), None, None, None, cache)
    assert event.is_set()
    prs = list(miner)
    assert {pr.pr[PullRequest.number.key] for pr in prs} == {921}


@with_defer
async def test_pr_miner_labels_unreleased(mdb, pdb, rdb, release_match_setting_tag):
    time_from = datetime(2018, 11, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    time_from_lookback = time_from - timedelta(days=60)
    # populate pdb
    await PullRequestMiner.mine(
        time_from_lookback.date(), time_to.date(), time_from_lookback, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    miner_complete, _, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug"}, set()), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None,
        pr_blacklist=(["MDExOlB1bGxSZXF1ZXN0MjA5MjA0MDQz",
                       "MDExOlB1bGxSZXF1ZXN0MjE2MTA0NzY1",
                       "MDExOlB1bGxSZXF1ZXN0MjEzODQ1NDUx"], {}))
    await wait_deferred()
    assert len(miner_complete._dfs.prs) == 3
    miner_complete, _, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug"}, {"ssh"}), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, False, release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None,
        pr_blacklist=(["MDExOlB1bGxSZXF1ZXN0MjA5MjA0MDQz",
                       "MDExOlB1bGxSZXF1ZXN0MjE2MTA0NzY1",
                       "MDExOlB1bGxSZXF1ZXN0MjEzODQ1NDUx"], {}))
    await wait_deferred()
    assert len(miner_complete._dfs.prs) == 2
    miner_complete, _, _, _ = await PullRequestMiner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug"}, set()), JIRAFilter.empty(),
        pd.DataFrame(columns=[Branch.commit_id.key, Branch.commit_sha.key,
                              Branch.repository_full_name.key]),
        {}, True, release_match_setting_tag, 1, (6366825,), mdb, pdb, rdb, None)
    assert len(miner_complete._dfs.prs) == 0


@with_defer
async def test_pr_miner_unreleased_facts(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag):
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
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    athenian.api.db._testing = False
    try:
        miner, unreleased_facts, matched_bys, event = await PullRequestMiner.mine(*args)
    finally:
        athenian.api.db._testing = True
    assert not event.is_set()
    await wait_deferred()
    await event.wait()
    assert unreleased_facts == {}
    open_prs_and_facts = []
    merged_unreleased_prs_and_facts = []
    force_push_dropped = []
    facts_miner = PullRequestFactsMiner(await bots(mdb))
    for pr in miner:
        facts = facts_miner(pr)
        if not facts.closed:
            open_prs_and_facts.append((pr, facts))
        elif facts.merged and not facts.released:
            if not facts.force_push_dropped:
                merged_unreleased_prs_and_facts.append((pr, facts))
            else:
                force_push_dropped.append((pr.pr, facts))
    assert len(open_prs_and_facts) == 21
    assert len(merged_unreleased_prs_and_facts) == 11
    assert len(force_push_dropped) == 0
    await pdb.execute(update(GitHubMergedPullRequestFacts).values({
        GitHubMergedPullRequestFacts.data: pickle.dumps(FakeFacts()),
        GitHubMergedPullRequestFacts.updated_at: datetime.now(timezone.utc),
    }))
    discovered = await load_merged_unreleased_pull_request_facts(
        miner._dfs.prs, time_to, LabelFilter.empty(), matched_bys, default_branches,
        release_match_setting_tag, 1, pdb)
    assert {pr.pr[PullRequest.node_id.key] for pr, _ in merged_unreleased_prs_and_facts} == \
        set(discovered)
    await store_open_pull_request_facts(open_prs_and_facts, 1, pdb)
    await store_merged_unreleased_pull_request_facts(
        merged_unreleased_prs_and_facts, matched_bys, default_branches,
        release_match_setting_tag, 1, pdb, event)
    miner, unreleased_facts, _, _ = await PullRequestMiner.mine(*args)
    true_pr_node_set = {pr.pr[PullRequest.node_id.key] for pr, _ in chain(
        open_prs_and_facts, merged_unreleased_prs_and_facts)}
    assert set(unreleased_facts) == true_pr_node_set
    assert len(miner) == 325
    dropped = miner.drop(unreleased_facts)
    assert set(dropped) == set(unreleased_facts)
    assert len(miner) == 293


@with_defer
async def test_pr_miner_jira_filter(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag):
    date_from = date(year=2018, month=1, day=1)
    date_to = date(year=2020, month=4, day=1)
    time_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    time_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
    args = [
        date_from,
        date_to,
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter(1, ["10003", "10009"], LabelFilter({"performance"}, set()),
                   set(), set(), False),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    ]
    miner, _, _, _ = await PullRequestMiner.mine(*args)
    numbers = {pr.pr[PullRequest.number.key] for pr in miner}
    assert {720, 721, 739, 740, 742,
            744, 751, 768, 771, 776,
            783, 784, 789, 797, 803,
            808, 815, 824, 825, 874} == numbers
    args[7] = JIRAFilter(1, ["10003", "10009"], LabelFilter.empty(), {"DEV-149"}, set(), False)
    miner, _, _, _ = await PullRequestMiner.mine(*args)
    numbers = {pr.pr[PullRequest.number.key] for pr in miner}
    assert {821, 833, 846, 861} == numbers
    args[7] = JIRAFilter(1, ["10003", "10009"], LabelFilter.empty(), set(), {"bug"}, False)
    miner, _, _, _ = await PullRequestMiner.mine(*args)
    numbers = {pr.pr[PullRequest.number.key] for pr in miner}
    assert {800, 769, 896, 762, 807, 778, 855, 816, 754, 724, 790, 759, 792, 794, 795} == numbers
    args[7] = JIRAFilter(1, ["10003", "10009"], LabelFilter({"api"}, set()),
                         set(), {"task"}, False)
    miner, _, _, _ = await PullRequestMiner.mine(*args)
    numbers = {pr.pr[PullRequest.number.key] for pr in miner}
    assert {710, 712, 716, 720, 721,
            739, 740, 742, 744, 751,
            766, 768, 771, 776, 782,
            783, 784, 786, 789, 797,
            803, 808, 810, 815, 824,
            825, 833, 846, 861} == numbers
    args[7] = JIRAFilter(1, ["10003", "10009"], LabelFilter.empty(), set(), set(), True)
    miner, _, _, _ = await PullRequestMiner.mine(*args)
    numbers = {pr.pr[PullRequest.number.key] for pr in miner}
    assert len(numbers) == 265


@with_defer
async def test_pr_miner_jira_fetch(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag):
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
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    miner, _, _, _ = await PullRequestMiner.mine(*args)
    labels = set()
    epics = set()
    types = set()
    for pr in miner:
        jira = pr.jiras
        if not (pr_labels := jira[Issue.labels.key]).empty:
            labels.update(pr_labels.iloc[0])
            assert is_datetime64_any_dtype(jira[Issue.created.key])
            assert is_datetime64_any_dtype(jira[Issue.updated.key])
        if not (pr_epic := jira["epic"]).empty:
            epics.add(pr_epic.iloc[0])
        if not (pr_type := jira[Issue.type.key]).empty:
            types.add(pr_type.iloc[0])
    assert labels == {"enhancement", "new-charts", "metrics", "usability", "security",
                      "api", "webapp", "data retrieval", "infrastructure", "reliability",
                      "code-quality", "accuracy", "bug", "performance", "functionality",
                      "sentry", "test"}
    assert epics == {"DEV-149", "DEV-776", "DEV-737", "DEV-667", "DEV-140", "DEV-818", None}
    assert types == {"task", "story", "epic", "bug"}


@with_defer
async def test_pr_miner_jira_cache(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, cache):
    date_from = date(year=2018, month=1, day=1)
    date_to = date(year=2020, month=4, day=1)
    time_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    time_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
    args = [
        date_from,
        date_to,
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        branches, default_branches,
        False,
        release_match_setting_tag,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    ]
    miner, _, _, _ = await PullRequestMiner.mine(*args)
    await wait_deferred()
    assert len(miner) == 325
    args[7] = JIRAFilter(1, ["10003", "10009"], LabelFilter({"enhancement"}, set()),
                         {"DEV-149"}, {"task"}, False)
    args[-4] = args[-3] = args[-2] = None
    miner, _, _, _ = await PullRequestMiner.mine(*args)
    assert len(miner) == 3
    for pr in miner:
        assert "enhancement" in pr.jiras["labels"].iloc[0]
        assert pr.jiras["epic"].iloc[0] == "DEV-149"
        assert pr.jiras[Issue.type.key].iloc[0] == "task"
    args[7] = JIRAFilter(1, ["10003", "10009"], LabelFilter({"enhancement,performance"}, set()),
                         {"DEV-149"}, {"task"}, False)
    miner, _, _, _ = await PullRequestMiner.mine(*args)
    assert len(miner) == 0
    args[-4] = mdb
    args[-3] = pdb
    args[-2] = rdb
    args[-1] = None
    miner, _, _, _ = await PullRequestMiner.mine(*args)
    assert len(miner) == 0


@with_defer
async def test_fetch_prs_no_branches(mdb, pdb, dag):
    branches, _ = await extract_branches(["src-d/go-git"], (6366825,), mdb, None)
    branches = branches[branches[Branch.branch_name.key] == "master"]
    branches[Branch.repository_full_name.key] = "xxx"
    branches[Branch.commit_date] = [datetime.now(timezone.utc)]
    dags = dag.copy()
    dags["xxx"] = _empty_dag()
    args = [
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 1, tzinfo=timezone.utc),
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        False, None, branches, dags, 1, (6366825,), mdb, pdb, None,
    ]
    prs, xdags = await PullRequestMiner.fetch_prs(*args)
    assert prs["dead"].sum() == 0
    assert xdags["src-d/go-git"]
    branches = branches.iloc[:0]
    args[-7] = branches
    prs, _ = await PullRequestMiner.fetch_prs(*args)
    assert prs["dead"].sum() == 0


@with_defer
async def test_fetch_prs_dead(mdb, pdb):
    branches, _ = await extract_branches(["src-d/go-git"], (6366825,), mdb, None)
    branches = branches[branches[Branch.branch_name.key] == "master"]
    branches[Branch.commit_date] = datetime.now(timezone.utc)
    args = [
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 1, tzinfo=timezone.utc),
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        False, None, branches, None, 1, (6366825,), mdb, pdb, None,
    ]
    prs, xdags = await PullRequestMiner.fetch_prs(*args)
    assert prs["dead"].sum() == len(force_push_dropped_go_git_pr_numbers)
    pdb_dag = DAG(await pdb.fetch_val(select([GitHubCommitHistory.dag])))
    dag = await fetch_dag(mdb, branches[Branch.commit_id.key].tolist())
    assert not (set(dag["src-d/go-git"][0]) - set(pdb_dag.hashes))
    for i in range(3):
        assert (xdags["src-d/go-git"][i] == pdb_dag[["hashes", "vertexes", "edges"][i]]).all()
