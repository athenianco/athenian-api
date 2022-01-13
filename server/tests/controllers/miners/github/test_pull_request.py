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
from sqlalchemy import delete, insert, select, update

from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.commit import _empty_dag
from athenian.api.controllers.miners.github.precomputed_prs import \
    store_merged_unreleased_pull_request_facts, store_open_pull_request_facts
from athenian.api.controllers.miners.github.pull_request import PullRequestFactsMiner
from athenian.api.controllers.miners.types import DAG, MinedPullRequest, PRParticipationKind, \
    PullRequestFacts
from athenian.api.controllers.settings import LogicalRepositorySettings, ReleaseMatch, \
    ReleaseMatchSetting, ReleaseSettings
import athenian.api.db
from athenian.api.defer import launch_defer, wait_deferred, with_defer, with_explicit_defer
from athenian.api.models.metadata.github import Branch, PullRequest
from athenian.api.models.metadata.jira import Issue
from athenian.api.models.persistentdata.models import ReleaseNotification
from athenian.api.models.precomputed.models import GitHubCommitHistory, \
    GitHubDonePullRequestFacts, GitHubMergedPullRequestFacts
from tests.conftest import has_memcached
from tests.controllers.conftest import FakeFacts, fetch_dag
from tests.controllers.test_filter_controller import force_push_dropped_go_git_pr_numbers


@with_defer
async def test_pr_miner_iter_smoke(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag,
        pr_miner, prefixer):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _, _ = await pr_miner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
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
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, pr_miner,
        prefixer):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2017, month=1, day=12)
    miner, _, _, _ = await pr_miner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    node_ids = {pr.pr[PullRequest.node_id.name] for pr in miner}
    assert node_ids == {
        162928, 162929,
        162930, 162931,
        163501, 163529,
        163568,
    }
    miner, _, _, _ = await pr_miner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
        pr_blacklist=(node_ids, {}),
    )
    node_ids = [pr.pr[PullRequest.node_id.name] for pr in miner]
    assert len(node_ids) == 0


@with_defer
async def test_pr_miner_iter_cache_compatible(
        branches, default_branches, mdb, pdb, rdb, cache, release_match_setting_tag,
        pr_miner, prefixer):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    args = [
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    ]
    miner, _, _, _ = await pr_miner.mine(*args)
    await wait_deferred()
    assert len(cache.mem) > 0
    first_data = list(miner)
    miner, _, _, _ = await pr_miner.mine(
        *args[:-4],
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
    cache_size = len(cache.mem)
    # we still use the cache here
    args[5] = {PRParticipationKind.AUTHOR: {"mcuadros"}, PRParticipationKind.MERGER: {"mcuadros"}}
    prs = list((await pr_miner.mine(
        *args[:-4],
        None,
        None,
        None,
        cache,
    ))[0])
    await wait_deferred()
    # check that the cache has not changed if we add some filters
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
        branches, default_branches, mdb, pdb, rdb, cache, release_match_setting_tag,
        pr_miner, prefixer):
    date_from = date(year=2018, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    args = [
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    ]
    await pr_miner.mine(*args)
    await wait_deferred()
    args[4] = {"src-d/gitbase"}
    with pytest.raises(AssertionError):
        await pr_miner.mine(
            *args[:-4],
            None,
            None,
            None,
            cache,
        )
    args[4] = {"src-d/go-git"}
    args[8] = True
    with pytest.raises(AssertionError):
        await pr_miner.mine(
            *args[:-4],
            None,
            None,
            None,
            cache,
        )


@pytest.mark.parametrize("with_memcached", [False, True])
@with_defer
async def test_pr_miner_cache_pr_blacklist(
        branches, default_branches, mdb, pdb, rdb, cache, release_match_setting_tag,
        pr_miner, prefixer, with_memcached, memcached):
    """https://athenianco.atlassian.net/browse/DEV-206"""
    if with_memcached:
        if not has_memcached:
            raise pytest.skip("no memcached")
        cache = memcached
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
        False,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    """
        There are 5 PRs:

        163047,
        163130,
        163167,
        163170,
        163172,
    """
    prs = list((await pr_miner.mine(
        *args, pr_blacklist=([163172], {})))[0])
    assert {pr.pr[PullRequest.node_id.name] for pr in prs} == {
        163047, 163130,
        163167, 163170}
    prs = list((await pr_miner.mine(
        *args, pr_blacklist=([163047], {})))[0])
    assert {pr.pr[PullRequest.node_id.name] for pr in prs} == {
        163172, 163130,
        163167, 163170}


@pytest.mark.parametrize("pk", [[v] for v in PRParticipationKind] + [list(PRParticipationKind)])
@with_defer
async def test_pr_miner_participant_filters(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, pk,
        pr_miner, prefixer):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _, _ = await pr_miner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {v: {"mcuadros"} for v in pk},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
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
        participants = pr.participant_logins()
        if len(pk) == 1:
            assert "mcuadros" in participants[pk[0]], (pk, str(pr.pr))
        else:
            mentioned = False
            for v in pk:
                if "mcuadros" in participants[v]:
                    mentioned = True
                    break
            assert mentioned, (pk, str(pr.pr))
    assert count > 0


def validate_pull_request_facts(prmeta: Dict[str, Any], prt: PullRequestFacts):
    assert prmeta[PullRequest.node_id.name]
    assert prmeta[PullRequest.repository_full_name.name] == "src-d/go-git"
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
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, pr_miner,
        prefixer, bots):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _, _ = await pr_miner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        True,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    facts_miner = PullRequestFactsMiner(bots)
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_empty_review_comments(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, pr_miner,
        prefixer, bots):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _, _ = await pr_miner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    miner._dfs.review_comments = miner._dfs.review_comments.iloc[0:0]
    facts_miner = PullRequestFactsMiner(bots)
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_empty_commits(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, pr_miner,
        prefixer, bots):
    date_from = date(year=2015, month=1, day=1)
    date_to = date(year=2020, month=1, day=1)
    miner, _, _, _ = await pr_miner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    miner._dfs.commits = miner._dfs.commits.iloc[0:0]
    facts_miner = PullRequestFactsMiner(bots)
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_bug_less_timestamp_float(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, pr_miner,
        prefixer, bots):
    date_from = date(2019, 10, 16) - timedelta(days=3)
    date_to = date(2019, 10, 16)
    miner, _, _, _ = await pr_miner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    facts_miner = PullRequestFactsMiner(bots)
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    assert len(prts) > 0
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_empty_releases(branches, default_branches, mdb, pdb, rdb,
                                             pr_miner, prefixer, bots):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2018, month=1, day=1)
    miner, _, _, _ = await pr_miner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        branches, default_branches,
        False,
        ReleaseSettings({"github.com/src-d/go-git": ReleaseMatchSetting(
            branches="unknown", tags="", events="", match=ReleaseMatch.branch)}),
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    facts_miner = PullRequestFactsMiner(bots)
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_mine_by_ids(branches, default_branches, dag, mdb, pdb, rdb, cache,
                              release_loader, pr_miner, prefixer):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2018, month=1, day=1)
    time_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    time_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
    release_settings = ReleaseSettings({
        "github.com/src-d/go-git": ReleaseMatchSetting(
            branches="unknown", tags="", events="", match=ReleaseMatch.branch),
    })
    miner, _, _, _ = await pr_miner.mine(
        date_from,
        date_to,
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        branches, default_branches,
        False,
        release_settings,
        LogicalRepositorySettings.empty(),
        prefixer,
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
    prs.set_index(PullRequest.node_id.name, inplace=True)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"], branches, default_branches, time_from, time_to,
        release_settings, LogicalRepositorySettings.empty(), prefixer,
        1, (6366825,), mdb, pdb, rdb, cache)
    dfs1, _, _ = await pr_miner.mine_by_ids(
        prs,
        [],
        {"src-d/go-git"},
        time_to,
        releases,
        matched_bys,
        branches,
        default_branches,
        dag,
        release_settings,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    await wait_deferred()
    dfs2, _, _ = await pr_miner.mine_by_ids(
        prs,
        [],
        {"src-d/go-git"},
        time_to,
        releases,
        matched_bys,
        branches,
        default_branches,
        dag,
        release_settings,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    for df1, df2 in zip(dfs1.values(), dfs2.values()):
        assert (df1.fillna(0) == df2.fillna(0)).all().all()
    for field in dataclasses.fields(MinedPullRequest):
        field = field.name
        if field == "release":
            continue
        df1 = getattr(dfs1, field.rstrip("s") + "s")
        records = [getattr(pr, field) for pr in mined_prs]
        if field.endswith("s"):
            df = pd.concat(records)
        else:
            df = pd.DataFrame.from_records(records)
        if field not in ("pr", "release"):
            df1.index = df1.index.droplevel(0)
        else:
            df.set_index(PullRequest.node_id.name, inplace=True)
        df.sort_index(inplace=True)
        df1.sort_index(inplace=True)
        df1 = df1[df.columns]
        assert (df.fillna(0) == df1.fillna(0)).all().all()


@with_defer
async def test_pr_miner_exclude_inactive(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, pr_miner,
        prefixer):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2017, month=1, day=12)
    miner, _, _, _ = await pr_miner.mine(
        date_from,
        date_to,
        datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc),
        datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        branches, default_branches,
        True,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    node_ids = {pr.pr[PullRequest.node_id.name] for pr in miner}
    assert node_ids == {
        162928, 162929,
        162930, 162931,
        163568, 163529,
    }


@with_defer
async def test_pr_miner_unreleased_pdb(mdb, pdb, rdb, release_match_setting_tag,
                                       pr_miner, prefixer, default_branches):
    time_from = datetime(2018, 11, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    miner_incomplete, _, _, _ = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    time_from_lookback = time_from - timedelta(days=60)
    await wait_deferred()
    # populate pdb
    await pr_miner.mine(
        time_from_lookback.date(), time_to.date(), time_from_lookback, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    miner_complete, facts, _, _ = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    assert isinstance(facts, dict)
    assert len(facts) == 0
    await wait_deferred()
    assert len(miner_incomplete._dfs.prs) == 19
    assert len(miner_complete._dfs.prs) == 19 + 42
    miner_active, _, _, _ = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, True, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    assert len(miner_active._dfs.prs) <= 19


@with_defer
async def test_pr_miner_labels_torture(mdb, pdb, rdb, release_match_setting_tag, cache,
                                       pr_miner, prefixer, default_branches):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug", "enhancement"}, set()), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {887, 921, 958, 947, 950, 949}
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug", "enhancement"}, set()), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {887, 921, 958, 947, 950, 949}
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug", "plumbing"}, set()), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {921, 940, 946, 950, 958}
    await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, cache)
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug", "plumbing"}, set()), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), None, None, None, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {921, 940, 946, 950, 958}
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug"}, set()), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), None, None, None, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {921, 950, 958}
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug", "plumbing"}, {"plumbing"}), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), None, None, None, cache)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {921}
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug", "plumbing"}, {"plumbing"}), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb,
        None)
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {921}
    miner, _, _, event = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug"}, {"plumbing"}), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), None, None, None, cache)
    assert event.is_set()
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {921}


@with_defer
async def test_pr_miner_labels_unreleased(mdb, pdb, rdb, release_match_setting_tag,
                                          pr_miner, prefixer, default_branches):
    time_from = datetime(2018, 11, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    time_from_lookback = time_from - timedelta(days=60)
    # populate pdb
    await pr_miner.mine(
        time_from_lookback.date(), time_to.date(), time_from_lookback, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    miner_complete, _, _, _ = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug"}, set()), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None,
        pr_blacklist=([163242,
                       163253,
                       163272], {}))
    await wait_deferred()
    assert len(miner_complete._dfs.prs) == 3
    miner_complete, _, _, _ = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug"}, {"ssh"}), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None,
        pr_blacklist=([163242,
                       163253,
                       163272], {}))
    await wait_deferred()
    assert len(miner_complete._dfs.prs) == 2
    miner_complete, _, _, _ = await pr_miner.mine(
        time_from.date(), time_to.date(), time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug"}, set()), JIRAFilter.empty(), False,
        pd.DataFrame(columns=[Branch.commit_id.name, Branch.commit_sha.name,
                              Branch.repository_full_name.name]),
        default_branches, True, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, 1, (6366825,), mdb, pdb, rdb, None)
    assert len(miner_complete._dfs.prs) == 0


@with_explicit_defer
async def test_pr_miner_unreleased_facts(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag,
        merged_prs_facts_loader, pr_miner, with_preloading_enabled, prefixer, bots):
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
        False,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    athenian.api.db._testing = False
    try:
        miner, unreleased_facts, matched_bys, event = await pr_miner.mine(*args)
    finally:
        athenian.api.db._testing = True
    assert not event.is_set()
    launch_defer(0, "enable_defer")
    await wait_deferred()
    if with_preloading_enabled:
        await pdb.cache.refresh()
    await event.wait()
    assert unreleased_facts == {}
    open_prs_and_facts = []
    merged_unreleased_prs_and_facts = []
    force_push_dropped = []
    facts_miner = PullRequestFactsMiner(bots)
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
    if with_preloading_enabled:
        await pdb.cache.refresh()

    discovered = await merged_prs_facts_loader.load_merged_unreleased_pull_request_facts(
        miner._dfs.prs, time_to, LabelFilter.empty(), matched_bys, default_branches,
        release_match_setting_tag, prefixer, 1, pdb)
    assert {pr.pr[PullRequest.node_id.name] for pr, _ in merged_unreleased_prs_and_facts} == \
        {node_id for node_id, _ in discovered}
    await store_open_pull_request_facts(open_prs_and_facts, 1, pdb)
    await store_merged_unreleased_pull_request_facts(
        merged_unreleased_prs_and_facts, matched_bys, default_branches,
        release_match_setting_tag, 1, pdb, event)
    if with_preloading_enabled:
        await pdb.cache.refresh()

    miner, unreleased_facts, _, _ = await pr_miner.mine(*args)
    true_pr_node_set = {pr.pr[PullRequest.node_id.name] for pr, _ in chain(
        open_prs_and_facts, merged_unreleased_prs_and_facts)}
    assert {node_id for node_id, _ in unreleased_facts} == true_pr_node_set
    assert len(miner) == 326
    unreleased_nodes = [node_id for node_id, _ in unreleased_facts]
    dropped = miner.drop(unreleased_nodes)
    assert set(dropped) == set(unreleased_nodes)
    assert len(miner) == 294


@with_defer
async def test_pr_miner_jira_filter(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, pr_miner,
        prefixer):
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
                   set(), set(), False, False),
        False,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    ]
    miner, _, _, _ = await pr_miner.mine(*args)
    assert miner.dfs.jiras.empty
    numbers = {pr.pr[PullRequest.number.name] for pr in miner}
    assert {720, 721, 739, 740, 742,
            744, 751, 768, 771, 776,
            783, 784, 789, 797, 803,
            808, 815, 824, 825, 874} == numbers
    args[7] = \
        JIRAFilter(1, ["10003", "10009"], LabelFilter.empty(), {"DEV-149"}, set(), False, False)
    miner, _, _, _ = await pr_miner.mine(*args)
    numbers = {pr.pr[PullRequest.number.name] for pr in miner}
    assert {821, 833, 846, 861} == numbers
    args[7] = JIRAFilter(1, ["10003", "10009"], LabelFilter.empty(), set(), {"bug"}, False, False)
    miner, _, _, _ = await pr_miner.mine(*args)
    numbers = {pr.pr[PullRequest.number.name] for pr in miner}
    assert {800, 769, 896, 762, 807, 778, 855, 816, 754, 724, 790, 759, 792, 794, 795} == numbers
    args[7] = JIRAFilter(1, ["10003", "10009"], LabelFilter({"api"}, set()),
                         set(), {"task"}, False, False)
    miner, _, _, _ = await pr_miner.mine(*args)
    numbers = {pr.pr[PullRequest.number.name] for pr in miner}
    assert {710, 712, 716, 720, 721,
            739, 740, 742, 744, 751,
            766, 768, 771, 776, 782,
            783, 784, 786, 789, 797,
            803, 808, 810, 815, 824,
            825, 833, 846, 861} == numbers
    args[7] = JIRAFilter(1, ["10003", "10009"], LabelFilter.empty(), set(), set(), False, True)
    miner, _, _, _ = await pr_miner.mine(*args)
    numbers = {pr.pr[PullRequest.number.name] for pr in miner}
    assert len(numbers) == 266


@with_defer
async def test_pr_miner_jira_fetch(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, pr_miner,
        prefixer):
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
        True,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    miner, _, _, _ = await pr_miner.mine(*args)
    labels = set()
    epics = set()
    types = set()
    for pr in miner:
        jira = pr.jiras
        if not (pr_labels := jira[Issue.labels.name]).empty:
            labels.update(pr_labels.iloc[0])
            assert is_datetime64_any_dtype(jira[Issue.created.name])
            assert is_datetime64_any_dtype(jira[Issue.updated.name])
        if not (pr_epic := jira["epic"]).empty:
            epics.add(pr_epic.iloc[0])
        if not (pr_type := jira[Issue.type.name]).empty:
            types.add(pr_type.iloc[0])
    assert labels == {"enhancement", "new-charts", "metrics", "usability", "security",
                      "api", "webapp", "data retrieval", "infrastructure", "reliability",
                      "code-quality", "accuracy", "bug", "performance", "functionality",
                      "sentry", "test"}
    assert epics == {"DEV-149", "DEV-776", "DEV-737", "DEV-667", "DEV-140", "DEV-818", None}
    assert types == {"task", "story", "epic", "bug"}
    # !!!!!!!!!!!!!!


@with_defer
async def test_pr_miner_jira_cache(
        branches, default_branches, mdb, pdb, rdb, release_match_setting_tag, cache,
        pr_miner, prefixer):
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
        True,
        branches, default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    ]
    miner, _, _, _ = await pr_miner.mine(*args)
    await wait_deferred()
    assert len(miner) == 326
    args[7] = JIRAFilter(1, ["10003", "10009"], LabelFilter({"enhancement"}, set()),
                         {"DEV-149"}, {"task"}, False, False)
    args[-4] = args[-3] = args[-2] = None
    miner, _, _, _ = await pr_miner.mine(*args)
    assert len(miner) == 3
    for pr in miner:
        assert len(pr.jiras) > 0
        assert "enhancement" in pr.jiras["labels"].iloc[0]
        assert pr.jiras["epic"].iloc[0] == "DEV-149"
        assert pr.jiras[Issue.type.name].iloc[0] == "task"
    args[7] = JIRAFilter(1, ["10003", "10009"], LabelFilter({"enhancement,performance"}, set()),
                         {"DEV-149"}, {"task"}, False, False)
    miner, _, _, _ = await pr_miner.mine(*args)
    assert len(miner) == 0
    args[-4] = mdb
    args[-3] = pdb
    args[-2] = rdb
    args[-1] = None
    miner, *_ = await pr_miner.mine(*args)
    assert len(miner) == 0


@with_defer
async def test_fetch_prs_no_branches(mdb, pdb, dag, branch_miner, pr_miner, prefixer):
    branches, _ = await branch_miner.extract_branches(
        ["src-d/go-git"], prefixer, (6366825,), mdb, None)
    branches = branches[branches[Branch.branch_name.name] == "master"]
    branches[Branch.repository_full_name.name] = "xxx"
    branches[Branch.commit_date] = [datetime.now(timezone.utc)]
    dags = dag.copy()
    dags["xxx"] = _empty_dag()
    args = [
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 1, tzinfo=timezone.utc),
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        False, None, None, branches, dags, 1, (6366825,), mdb, pdb, None,
    ]
    prs, xdags, _ = await pr_miner.fetch_prs(*args)
    assert prs["dead"].sum() == 0
    assert xdags["src-d/go-git"]
    branches = branches.iloc[:0]
    args[-7] = branches
    prs, *_ = await pr_miner.fetch_prs(*args)
    assert prs["dead"].sum() == 0


@with_defer
async def test_fetch_prs_dead(mdb, pdb, branch_miner, pr_miner, prefixer):
    branches, _ = await branch_miner.extract_branches(
        ["src-d/go-git"], prefixer, (6366825,), mdb, None)
    branches = branches[branches[Branch.branch_name.name] == "master"]
    branches[Branch.commit_date] = datetime.now(timezone.utc)
    args = [
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 1, tzinfo=timezone.utc),
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        False, None, None, branches, None, 1, (6366825,), mdb, pdb, None,
    ]
    prs, xdags, _ = await pr_miner.fetch_prs(*args)
    assert prs["dead"].sum() == len(force_push_dropped_go_git_pr_numbers)
    pdb_dag = DAG(await pdb.fetch_val(select([GitHubCommitHistory.dag])))
    dag = await fetch_dag(mdb, branches[Branch.commit_id.name].tolist())
    assert not (set(dag["src-d/go-git"][0]) - set(pdb_dag.hashes))
    for i in range(3):
        assert (xdags["src-d/go-git"][i] == pdb_dag[["hashes", "vertexes", "edges"][i]]).all()


@with_defer
async def test_mine_pull_requests_event_releases(
        metrics_calculator_factory, release_match_setting_event, mdb, pdb, rdb, prefixer, bots):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2019, 11, 19, tzinfo=timezone.utc)
    await rdb.execute(insert(ReleaseNotification).values(ReleaseNotification(
        account_id=1,
        repository_node_id=40550,
        commit_hash_prefix="1edb992",
        name="Pushed!",
        author_node_id=40020,
        url="www",
        published_at=datetime(2019, 9, 1, tzinfo=timezone.utc),
    ).create_defaults().explode(with_primary_keys=True)))
    args = (time_from, time_to, {"src-d/go-git"}, {},
            LabelFilter.empty(), JIRAFilter.empty(),
            False, bots, release_match_setting_event, LogicalRepositorySettings.empty(),
            prefixer, False, False)
    calc = metrics_calculator_factory(1, (6366825,))
    facts1 = await calc.calc_pull_request_facts_github(*args)
    facts1.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    await wait_deferred()
    await pdb.execute(
        delete(GitHubDonePullRequestFacts)
        .where(GitHubDonePullRequestFacts.pr_node_id == 163529))
    await pdb.execute(
        update(GitHubMergedPullRequestFacts)
        .where(GitHubMergedPullRequestFacts.pr_node_id == 163529)
        .values({
            GitHubMergedPullRequestFacts.checked_until: datetime.now(timezone.utc),
            GitHubMergedPullRequestFacts.updated_at: datetime.now(timezone.utc),
        }))
    facts2 = await calc.calc_pull_request_facts_github(*args)
    facts2.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    assert_frame_equal(facts1.iloc[1:], facts2.iloc[1:])
