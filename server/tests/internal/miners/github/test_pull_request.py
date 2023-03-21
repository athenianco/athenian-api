from collections import defaultdict
import contextlib
import dataclasses
from datetime import date, datetime, timedelta, timezone
from itertools import chain
import pickle
from typing import Any, Dict
from unittest import mock

import medvedi as md
from medvedi.testing import assert_frame_equal, assert_index_equal
import numpy as np
from numpy import typing as npt
from numpy.testing import assert_array_equal
import pytest
from sqlalchemy import delete, insert, select, update

from athenian.api.async_utils import read_sql_query
from athenian.api.controllers.search_controller.search_prs import _align_pr_numbers_to_ids
import athenian.api.db
from athenian.api.db import Database, DatabaseLike
from athenian.api.defer import launch_defer, wait_deferred, with_defer, with_explicit_defer
from athenian.api.internal.jira import get_jira_installation
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.commit import _empty_dag
from athenian.api.internal.miners.github.precomputed_prs import (
    store_merged_unreleased_pull_request_facts,
    store_open_pull_request_facts,
)
from athenian.api.internal.miners.github.pull_request import (
    PullRequestFactsMiner,
    PullRequestMiner,
    fetch_prs_numbers,
)
from athenian.api.internal.miners.github.released_pr import matched_by_column
from athenian.api.internal.miners.participation import PRParticipationKind
from athenian.api.internal.miners.types import (
    DAG,
    JIRAEntityToFetch,
    MinedPullRequest,
    PullRequestCheckRun,
    PullRequestFacts,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import (
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
    Settings,
)
from athenian.api.models.metadata.github import Branch, PullRequest, PullRequestCommit, Release
from athenian.api.models.metadata.jira import Issue
from athenian.api.models.persistentdata.models import DeploymentNotification, ReleaseNotification
from athenian.api.models.precomputed.models import (
    GitHubCommitHistory,
    GitHubDonePullRequestFacts,
    GitHubMergedPullRequestFacts,
)
from tests.conftest import build_fake_cache, fetch_dag, has_memcached
from tests.controllers.conftest import FakeFacts
from tests.controllers.test_filter_controller import force_push_dropped_go_git_pr_numbers
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_JIRA_ACCOUNT_ID, DEFAULT_MD_ACCOUNT_ID
from tests.testutils.time import dt


@with_defer
async def test_pr_miner_iter_smoke(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pr_miner,
    prefixer,
):
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
        branches,
        default_branches,
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
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pr_miner,
    prefixer,
):
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
        branches,
        default_branches,
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
        162928,
        162929,
        162930,
        162931,
        163501,
        163529,
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
        branches,
        default_branches,
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
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    cache,
    release_match_setting_tag,
    pr_miner,
    prefixer,
):
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
        branches,
        default_branches,
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
    prs = list(
        (
            await pr_miner.mine(
                *args[:-4],
                None,
                None,
                None,
                cache,
            )
        )[0],
    )
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
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    cache,
    release_match_setting_tag,
    pr_miner,
    prefixer,
):
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
        branches,
        default_branches,
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
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    cache,
    release_match_setting_tag,
    pr_miner,
    prefixer,
    with_memcached,
    memcached,
):
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
        branches,
        default_branches,
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
    prs = list((await pr_miner.mine(*args, pr_blacklist=([163172], {})))[0])
    assert {pr.pr[PullRequest.node_id.name] for pr in prs} == {163047, 163130, 163167, 163170}
    prs = list((await pr_miner.mine(*args, pr_blacklist=([163047], {})))[0])
    assert {pr.pr[PullRequest.node_id.name] for pr in prs} == {163172, 163130, 163167, 163170}


@pytest.mark.parametrize("pk", [[v] for v in PRParticipationKind] + [list(PRParticipationKind)])
@with_defer
async def test_pr_miner_participant_filters(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pk,
    pr_miner,
    prefixer,
):
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
        branches,
        default_branches,
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
        participants = pr.participant_nodes()
        if len(pk) == 1:
            assert 39789 in participants[pk[0]], (pk, str(pr.pr))
        else:
            mentioned = False
            for v in pk:
                if 39789 in participants[v]:
                    mentioned = True
                    break
            assert mentioned, (pk, str(pr.pr))
    assert count > 0


def validate_pull_request_facts(prmeta: Dict[str, Any], prt: PullRequestFacts):
    assert prmeta[PullRequest.node_id.name]
    assert prmeta[PullRequest.repository_full_name.name] == "src-d/go-git"
    for k, v in prt.items():
        if not isinstance(v, np.datetime64) or not v:
            continue
        if k not in (
            "first_commit",
            "last_commit",
            "last_commit_before_first_review",
            "work_began",
        ):
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
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pr_miner,
    prefixer,
    bots,
):
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
        branches,
        default_branches,
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
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pr_miner,
    prefixer,
    bots,
):
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
        branches,
        default_branches,
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
    miner._dfs.review_comments = miner._dfs.review_comments.iloc[:0]
    facts_miner = PullRequestFactsMiner(bots)
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_empty_commits(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pr_miner,
    prefixer,
    bots,
):
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
        branches,
        default_branches,
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
    miner._dfs.commits = miner._dfs.commits.iloc[:0]
    facts_miner = PullRequestFactsMiner(bots)
    prts = [(pr.pr, facts_miner(pr)) for pr in miner]
    for prt in prts:
        validate_pull_request_facts(*prt)


@with_defer
async def test_pr_facts_miner_bug_less_timestamp_float(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pr_miner,
    prefixer,
    bots,
):
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
        branches,
        default_branches,
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
async def test_pr_facts_miner_empty_releases(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    pr_miner,
    prefixer,
    bots,
):
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
        branches,
        default_branches,
        False,
        ReleaseSettings(
            {
                "github.com/src-d/go-git": ReleaseMatchSetting(
                    branches="unknown", tags="", events="", match=ReleaseMatch.branch,
                ),
            },
        ),
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
async def test_pr_mine_by_ids(
    branches,
    default_branches,
    dag,
    mdb,
    pdb,
    rdb,
    cache,
    release_loader,
    pr_miner,
    prefixer,
):
    date_from = date(year=2017, month=1, day=1)
    date_to = date(year=2018, month=1, day=1)
    time_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    time_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
    release_settings = ReleaseSettings(
        {
            "github.com/src-d/go-git": ReleaseMatchSetting(
                branches="unknown", tags="", events="", match=ReleaseMatch.branch,
            ),
        },
    )
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
        branches,
        default_branches,
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
    columns = defaultdict(list)
    for pr in mined_prs:
        for k, v in pr.pr.items():
            columns[k].append(v)
    prs = md.DataFrame(columns)
    prs.set_index(PullRequest.node_id.name, inplace=True)
    releases, matched_bys = await release_loader.load_releases(
        ["src-d/go-git"],
        branches,
        default_branches,
        time_from,
        time_to,
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
        assert_frame_equal(df1.fillna(0), df2.fillna(0))
    for field in dataclasses.fields(MinedPullRequest):
        field = field.name
        if field == "release":
            continue
        df1 = getattr(dfs1, field.rstrip("s") + "s")
        records = []
        pr_ids = []
        for pr in mined_prs:
            records.append(getattr(pr, field))
            pr_ids.append(pr.pr[PullRequest.node_id.name])
        if field == "check_run":
            ix = [i for i, r in enumerate(records) if r[PullRequestCheckRun.f.name] is not None]
            records = [records[i] for i in ix]
            pr_ids = [pr_ids[i] for i in ix]
        if field.endswith("s"):
            df = md.concat(*records)
        else:
            columns = defaultdict(list)
            for r in records:
                for k, v in r.items():
                    columns[k].append(v)
            df = md.DataFrame(columns)
        if field not in ("pr", "release", "check_run"):
            df1.set_index(df1.index.names[1:], inplace=True)
        else:
            df.set_index(PullRequest.node_id.name, inplace=True)
        df.sort_index(inplace=True)
        df1.sort_index(inplace=True)
        df1 = df1[df.columns]
        if field == "check_run":
            try:
                assert_index_equal(df.index, df1.index)
            except AssertionError as e:
                print(set(df.index.values).symmetric_difference(set(df1.index.values)))
                raise e from None
            for i, (name, name1) in enumerate(
                zip(df[PullRequestCheckRun.f.name], df1[PullRequestCheckRun.f.name]),
            ):
                assert name.tolist() == name1.tolist(), f"[{i}] {pr_ids[i]}"
        else:
            assert_frame_equal(df.astype(df1.dtype).fillna(0), df1.fillna(0))


@with_defer
async def test_pr_miner_exclude_inactive(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pr_miner,
    prefixer,
):
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
        branches,
        default_branches,
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
        162928,
        162929,
        162930,
        162931,
        163568,
        163529,
    }


@with_defer
async def test_pr_miner_unreleased_pdb(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pr_miner,
    prefixer,
    default_branches,
):
    time_from = datetime(2018, 11, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    miner_incomplete, _, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
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
    time_from_lookback = time_from - timedelta(days=60)
    await wait_deferred()
    # populate pdb
    await pr_miner.mine(
        time_from_lookback.date(),
        time_to.date(),
        time_from_lookback,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
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
    await wait_deferred()
    miner_complete, facts, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
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
    assert isinstance(facts, dict)
    assert len(facts) == 0
    await wait_deferred()
    assert len(miner_incomplete._dfs.prs) == 19
    assert len(miner_complete._dfs.prs) == 19 + 42
    miner_active, _, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
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
    assert len(miner_active._dfs.prs) <= 19


@with_defer
async def test_pr_miner_labels_torture(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    cache,
    pr_miner,
    prefixer,
    default_branches,
):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug", "enhancement"}, set()),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
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
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {887, 921, 958, 947, 950, 949}
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug", "enhancement"}, set()),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
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
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {887, 921, 958, 947, 950, 949}
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug", "plumbing"}, set()),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
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
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {921, 940, 946, 950, 958}
    await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
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
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug", "plumbing"}, set()),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        None,
        None,
        None,
        cache,
    )
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {921, 940, 946, 950, 958}
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug"}, set()),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        None,
        None,
        None,
        cache,
    )
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {921, 950, 958}
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug", "plumbing"}, {"plumbing"}),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        None,
        None,
        None,
        cache,
    )
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {921}
    miner, _, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug", "plumbing"}, {"plumbing"}),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
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
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {921}
    miner, _, _, event = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug"}, {"plumbing"}),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
        False,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        1,
        (6366825,),
        None,
        None,
        None,
        cache,
    )
    assert event.is_set()
    prs = list(miner)
    assert {pr.pr[PullRequest.number.name] for pr in prs} == {921}


@with_defer
async def test_pr_miner_labels_unreleased(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pr_miner,
    prefixer,
    default_branches,
):
    time_from = datetime(2018, 11, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    time_from_lookback = time_from - timedelta(days=60)
    # populate pdb
    await pr_miner.mine(
        time_from_lookback.date(),
        time_to.date(),
        time_from_lookback,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
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
    await wait_deferred()
    miner_complete, _, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug"}, set()),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
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
        pr_blacklist=([163242, 163253, 163272], {}),
    )
    await wait_deferred()
    assert len(miner_complete._dfs.prs) == 3
    miner_complete, _, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug"}, {"ssh"}),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
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
        pr_blacklist=([163242, 163253, 163272], {}),
    )
    await wait_deferred()
    assert len(miner_complete._dfs.prs) == 2
    miner_complete, _, _, _ = await pr_miner.mine(
        time_from.date(),
        time_to.date(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug"}, set()),
        JIRAFilter.empty(),
        False,
        md.DataFrame(
            {
                Branch.commit_id.name: [],
                Branch.commit_sha.name: [],
                Branch.repository_full_name.name: [],
            },
        ),
        default_branches,
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
    assert len(miner_complete._dfs.prs) == 0


@with_explicit_defer
async def test_pr_miner_unreleased_facts(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    merged_prs_facts_loader,
    pr_miner,
    prefixer,
    bots,
):
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
        branches,
        default_branches,
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
    athenian.api.is_testing = False
    try:
        miner, unreleased_facts, matched_bys, event = await pr_miner.mine(*args)
    finally:
        athenian.api.is_testing = True
    assert not event.is_set()
    launch_defer(0, "enable_defer")
    await wait_deferred()
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
    await pdb.execute(
        update(GitHubMergedPullRequestFacts).values(
            {
                GitHubMergedPullRequestFacts.data: pickle.dumps(FakeFacts()),
                GitHubMergedPullRequestFacts.updated_at: datetime.now(timezone.utc),
            },
        ),
    )

    discovered = await merged_prs_facts_loader.load_merged_unreleased_pull_request_facts(
        miner._dfs.prs,
        time_to,
        LabelFilter.empty(),
        matched_bys,
        default_branches,
        release_match_setting_tag,
        prefixer,
        1,
        pdb,
    )
    assert {pr.pr[PullRequest.node_id.name] for pr, _ in merged_unreleased_prs_and_facts} == {
        node_id for node_id, _ in discovered
    }
    await store_open_pull_request_facts(open_prs_and_facts, 1, pdb)
    await store_merged_unreleased_pull_request_facts(
        merged_unreleased_prs_and_facts,
        datetime(2050, 1, 1, tzinfo=timezone.utc),
        matched_bys,
        default_branches,
        release_match_setting_tag,
        1,
        pdb,
        event,
    )

    miner, unreleased_facts, _, _ = await pr_miner.mine(*args)
    true_pr_node_set = {
        pr.pr[PullRequest.node_id.name]
        for pr, _ in chain(open_prs_and_facts, merged_unreleased_prs_and_facts)
    }
    assert {node_id for node_id, _ in unreleased_facts} == true_pr_node_set
    assert len(miner) == 326
    dropped = miner.drop(unreleased_facts)
    assert set(zip(*dropped)) == set(unreleased_facts)
    assert len(miner) == 294


@with_defer
async def test_pr_miner_jira_filter(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pr_miner,
    prefixer,
):
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
        JIRAFilter(
            1,
            frozenset(("10003", "10009")),
            LabelFilter(frozenset(("performance",)), set()),
            custom_projects=False,
        ),
        False,
        branches,
        default_branches,
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
    assert {
        720,
        721,
        739,
        740,
        742,
        744,
        751,
        768,
        771,
        776,
        783,
        784,
        789,
        797,
        803,
        808,
        815,
        824,
        825,
        874,
    } == numbers
    args[7] = JIRAFilter(
        1, ("10003", "10009"), LabelFilter.empty(), {"DEV-149"}, set(), set(), False, False,
    )
    miner, _, _, _ = await pr_miner.mine(*args)
    numbers = {pr.pr[PullRequest.number.name] for pr in miner}
    assert {821, 833, 846, 861} == numbers
    args[7] = JIRAFilter(
        1, frozenset(("10003", "10009")), issue_types=frozenset(["bug"]), custom_projects=False,
    )
    miner, _, _, _ = await pr_miner.mine(*args)
    numbers = {pr.pr[PullRequest.number.name] for pr in miner}
    assert {800, 769, 896, 762, 807, 778, 855, 816, 754, 724, 790, 759, 792, 794, 795} == numbers
    args[7] = JIRAFilter(
        1,
        frozenset(("10003", "10009")),
        LabelFilter(frozenset(["api"]), frozenset()),
        issue_types=frozenset(["task"]),
        custom_projects=False,
    )
    miner, _, _, _ = await pr_miner.mine(*args)
    numbers = {pr.pr[PullRequest.number.name] for pr in miner}
    assert {
        710,
        712,
        716,
        720,
        721,
        739,
        740,
        742,
        744,
        751,
        766,
        768,
        771,
        776,
        782,
        783,
        784,
        786,
        789,
        797,
        803,
        808,
        810,
        815,
        824,
        825,
        833,
        846,
        861,
    } == numbers
    args[7] = JIRAFilter(1, ("10003", "10009"), custom_projects=False, unmapped=True)
    miner, _, _, _ = await pr_miner.mine(*args)
    numbers = {pr.pr[PullRequest.number.name] for pr in miner}
    assert len(numbers) == 266


@with_defer
async def test_pr_miner_jira_fetch(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    pr_miner,
    prefixer,
):
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
        JIRAEntityToFetch.EVERYTHING(),
        branches,
        default_branches,
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
    priorities = set()
    projects = set()
    for pr in miner:
        jira = pr.jiras
        assert jira[Issue.created.name].dtype.kind == "M"
        assert jira[Issue.updated.name].dtype.kind == "M"
        if len(pr_labels := jira[Issue.labels.name]):
            labels.update(pr_labels[0])
        if len(pr_epic := jira["epic"]):
            epics.add(pr_epic[0])
        if len(pr_type := jira[Issue.type.name]):
            types.add(pr_type[0])
        if len(pr_priority := jira[Issue.priority_id.name]):
            priorities.add(pr_priority[0])
        if len(pr_projects := jira[Issue.project_id.name]):
            projects.add(pr_projects[0])

    assert labels == {
        "enhancement",
        "new-charts",
        "metrics",
        "usability",
        "security",
        "api",
        "webapp",
        "data retrieval",
        "infrastructure",
        "reliability",
        "code-quality",
        "accuracy",
        "bug",
        "performance",
        "functionality",
        "sentry",
        "test",
    }
    assert epics == {"DEV-149", "DEV-776", "DEV-737", "DEV-667", "DEV-140", "DEV-818", None}
    assert types == {"task", "story", "epic", "bug"}
    assert priorities == {b"1", b"2", b"3", b"4", b"5", b"6"}
    assert projects == {b"10009"}
    # !!!!!!!!!!!!!!


@with_defer
async def test_pr_miner_jira_cache(
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    cache,
    pr_miner,
    prefixer,
):
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
        branches,
        default_branches,
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
    args[7] = JIRAFilter(
        1,
        ("10003", "10009"),
        LabelFilter(frozenset(["enhancement"]), frozenset()),
        epics=frozenset(["DEV-149"]),
        issue_types=frozenset(["task"]),
        custom_projects=False,
    )
    args[-4] = args[-3] = args[-2] = None
    miner, _, _, _ = await pr_miner.mine(*args)
    assert len(miner) == 3
    for pr in miner:
        assert len(pr.jiras) > 0
        assert "enhancement" in pr.jiras["labels"][0]
        assert pr.jiras["epic"][0] == "DEV-149"
        assert pr.jiras[Issue.type.name][0] == "task"
    args[7] = args[7].replace(
        labels=LabelFilter(frozenset(["enhancement,performance"]), frozenset()),
    )
    miner, _, _, _ = await pr_miner.mine(*args)
    assert len(miner) == 0
    args[-4] = mdb
    args[-3] = pdb
    args[-2] = rdb
    args[-1] = None
    miner, *_ = await pr_miner.mine(*args)
    assert len(miner) == 0


@pytest.mark.parametrize("extra_meta_ids", [(), (1,)])
@with_defer
async def test_fetch_prs_no_branches(
    mdb,
    pdb,
    dag,
    branch_miner,
    pr_miner,
    prefixer,
    meta_ids,
    extra_meta_ids,
):
    branches, _ = await branch_miner.load_branches(
        ["src-d/go-git"], prefixer, 1, meta_ids, mdb, None, None,
    )
    branches = branches.take(branches[Branch.branch_name.name] == "master").copy()
    branches[Branch.repository_full_name.name] = "xxx"
    branches[Branch.commit_date.name] = datetime.now(timezone.utc)
    dags = dag.copy()
    dags["xxx"] = True, _empty_dag()
    args = [
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 1, tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        None,
        None,
        branches,
        dags,
        1,
        meta_ids + extra_meta_ids,
        mdb,
        pdb,
        None,
    ]
    prs, xdags, _ = await pr_miner.fetch_prs(*args)
    assert prs[PullRequest.dead].sum() == 0
    assert xdags["src-d/go-git"]
    branches = branches.iloc[:0]
    args[-7] = branches
    prs, *_ = await pr_miner.fetch_prs(*args)
    assert prs[PullRequest.dead].sum() == 0


@with_defer
async def test_fetch_prs_dead(mdb, pdb, branch_miner, pr_miner, prefixer, meta_ids):
    branches, _ = await branch_miner.load_branches(
        ["src-d/go-git"], prefixer, 1, meta_ids, mdb, None, None,
    )
    branches = branches.take(branches[Branch.branch_name.name] == "master").copy()
    branches[Branch.commit_date.name] = datetime.now(timezone.utc)
    args = [
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2021, 1, 1, tzinfo=timezone.utc),
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        None,
        None,
        branches,
        None,
        1,
        meta_ids,
        mdb,
        pdb,
        None,
    ]
    prs, xdags, _ = await pr_miner.fetch_prs(*args)
    assert prs[PullRequest.dead].sum() == len(force_push_dropped_go_git_pr_numbers)
    pdb_dag = DAG(await pdb.fetch_val(select(GitHubCommitHistory.dag)))
    dag = await fetch_dag(meta_ids, mdb, branches[Branch.commit_id.name].tolist())
    assert not (set(dag["src-d/go-git"][1][0]) - set(pdb_dag.hashes))
    assert np.in1d(xdags["src-d/go-git"][1][0], pdb_dag["hashes"]).all()
    vertexes = np.searchsorted(pdb_dag["hashes"], xdags["src-d/go-git"][1][0])
    assert_array_equal(
        np.diff(xdags["src-d/go-git"][1][1]),
        pdb_dag["vertexes"][vertexes + 1] - pdb_dag["vertexes"][vertexes],
    )


@with_defer
async def test_mine_pull_requests_event_releases(
    pr_facts_calculator_factory,
    release_match_setting_event,
    mdb,
    pdb,
    rdb,
    prefixer,
    bots,
):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2019, 11, 19, tzinfo=timezone.utc)
    await rdb.execute(
        insert(ReleaseNotification).values(
            ReleaseNotification(
                account_id=1,
                repository_node_id=40550,
                commit_hash_prefix="1edb992",
                name="Pushed!",
                author_node_id=40020,
                url="www",
                published_at=datetime(2019, 9, 1, tzinfo=timezone.utc),
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    args = (
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        bots,
        release_match_setting_event,
        LogicalRepositorySettings.empty(),
        prefixer,
        False,
        0,
    )
    calc = pr_facts_calculator_factory(1, (6366825,))
    facts1 = await calc(*args)
    facts1.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    await wait_deferred()
    await pdb.execute(
        delete(GitHubDonePullRequestFacts).where(GitHubDonePullRequestFacts.pr_node_id == 163529),
    )
    await pdb.execute(
        update(GitHubMergedPullRequestFacts)
        .where(GitHubMergedPullRequestFacts.pr_node_id == 163529)
        .values(
            {
                GitHubMergedPullRequestFacts.checked_until: datetime.now(timezone.utc),
                GitHubMergedPullRequestFacts.updated_at: datetime.now(timezone.utc),
            },
        ),
    )
    facts2 = await calc(*args)
    facts2.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    assert_frame_equal(facts1.iloc[1:], facts2.iloc[1:])


def test_pull_request_review_request_without_review_open():
    miner = PullRequestFactsMiner(set())
    pr = MinedPullRequest(
        pr={
            PullRequest.created_at.name: np.datetime64("2023-01-01 00:00:00", "us"),
            PullRequest.repository_full_name.name: "athenianco/athenian-api",
            PullRequest.user_login.name: "vmarkovtsev",
            PullRequest.user_node_id.name: 40020,
            PullRequest.merged_by_login.name: "mcuadros",
            PullRequest.merged_by_id.name: 39789,
            PullRequest.merged_at.name: None,
            PullRequest.closed_at.name: None,
            PullRequest.number.name: 7777,
            PullRequest.node_id.name: 100500,
            PullRequest.additions.name: 10,
            PullRequest.deletions.name: 0,
        },
        release={
            Release.published_at.name: None,
            matched_by_column: None,
            Release.author.name: None,
        },
        comments=md.DataFrame(
            {
                "user_login": ["vmarkovtsev"],
                "user_node_id": [40020],
                "created_at": [np.datetime64("2023-01-01 00:05:00", "us")],
                "submitted_at": [np.datetime64("2023-01-01 00:05:00", "us")],
            },
        ),
        commits=md.DataFrame(
            {
                PullRequestCommit.committer_login.name: ["vmarkovtsev", "vmarkovtsev"],
                PullRequestCommit.author_login.name: ["vmarkovtsev", "vmarkovtsev"],
                PullRequestCommit.committer_user_id.name: [40020, 40020],
                PullRequestCommit.author_user_id.name: [40020, 40020],
                PullRequestCommit.committed_date.name: [
                    np.datetime64("2022-12-31 23:59:59", "us"),
                    np.datetime64("2023-01-01 00:10:00", "us"),
                ],
                PullRequestCommit.authored_date.name: [
                    np.datetime64("2022-12-31 23:59:59", "us"),
                    np.datetime64("2023-01-01 00:10:00", "us"),
                ],
            },
        ),
        reviews=md.DataFrame(
            {"user_login": [], "user_node_id": [], "created_at": [], "submitted_at": []},
        ),
        review_comments=md.DataFrame(
            {"user_login": [], "user_node_id": [], "created_at": [], "submitted_at": []},
        ),
        review_requests=md.DataFrame(
            {
                "user_login": ["vmarkovtsev"],
                "user_node_id": [40020],
                "created_at": [np.datetime64("2023-01-01 00:00:01", "us")],
                "submitted_at": [np.datetime64("2023-01-01 00:05:00", "us")],
            },
        ),
        labels=md.DataFrame({"name": []}),
        jiras=md.DataFrame(),
        deployments=md.DataFrame(
            {
                DeploymentNotification.environment.name: [],
                DeploymentNotification.conclusion.name: [],
                DeploymentNotification.finished_at.name: [],
                "node_id": [],
                "name": [],
            },
            index=("node_id", "name"),
        ),
        check_run={PullRequestCheckRun.f.name: None},
    )
    facts = miner(pr)
    assert facts.first_review_request == facts.last_commit


class TestPullRequestMinerMine:
    @with_defer
    async def test_jira_priorities_caching(self, sdb, pdb, rdb, mdb) -> None:
        cache = build_fake_cache()
        meta_ids = (DEFAULT_MD_ACCOUNT_ID,)
        prefixer = await Prefixer.load(meta_ids, mdb, None)
        settings = Settings.from_account(1, prefixer, sdb, mdb, None, None)
        release_settings = await settings.list_release_matches()
        repos = release_settings.native.keys()
        branches, default_branches = await BranchMiner.load_branches(
            repos, prefixer, 1, meta_ids, mdb, None, None,
        )
        jira_config = await get_jira_installation(DEFAULT_JIRA_ACCOUNT_ID, sdb, mdb, None)
        jira_all = JIRAFilter.from_jira_config(jira_config).replace(custom_projects=False)

        kwargs = {
            "date_from": date(2019, 1, 1),
            "date_to": date(2021, 1, 1),
            "time_from": dt(2019, 1, 1),
            "time_to": dt(2021, 1, 1),
            "repositories": {"src-d/go-git"},
            "participants": {},
            "labels": LabelFilter.empty(),
            "with_jira": JIRAEntityToFetch.EVERYTHING(),
            "branches": branches,
            "default_branches": default_branches,
            "exclude_inactive": True,
            "release_settings": release_settings,
            "logical_settings": LogicalRepositorySettings.empty(),
            "prefixer": prefixer,
            "account": 1,
            "meta_ids": meta_ids,
            "mdb": mdb,
            "pdb": pdb,
            "rdb": rdb,
            "cache": cache,
        }

        miner, *_ = await PullRequestMiner.mine(**{**kwargs, "cache": None}, jira=jira_all)
        assert len(list(miner)) == 118

        with mock.patch.object(
            PullRequestMiner, "fetch_prs", wraps=PullRequestMiner.fetch_prs,
        ) as fetch_prs_wrapper:
            print("all")
            miner, *_ = await PullRequestMiner.mine(**kwargs, jira=jira_all)
            assert len(list(miner)) == 118
            prs_w_jiras = [pr for pr in miner if not pr.jiras.empty]
            assert len(prs_w_jiras) == 2
            # cache miss, fetch_prs() is called
            fetch_prs_wrapper.assert_called_once()

            jira_high = jira_all.replace(priorities=["high"])
            miner, *_ = await PullRequestMiner.mine(**kwargs, jira=jira_high)
            prs_w_jiras = [pr for pr in miner if not pr.jiras.empty]
            assert len(prs_w_jiras) == 1
            assert list(prs_w_jiras[0].jiras[Issue.priority_id.name]) == [b"5"]
            # cache hit, fetch_prs() is not called again
            fetch_prs_wrapper.assert_called_once()

            jira_medium = jira_all.replace(priorities=["medium"])
            miner, *_ = await PullRequestMiner.mine(**kwargs, jira=jira_medium)
            prs_w_jiras = [pr for pr in miner if not pr.jiras.empty]
            assert len(prs_w_jiras) == 1
            assert list(prs_w_jiras[0].jiras[Issue.priority_id.name]) == [b"4"]
            # cache hit, fetch_prs() is not called again
            fetch_prs_wrapper.assert_called_once()


async def fetch_prs_numbers_e2e(
    node_ids: npt.NDArray[int],
    meta_ids: tuple[int, ...],
    mdb: DatabaseLike,
) -> npt.NDArray[int]:
    return _align_pr_numbers_to_ids(await fetch_prs_numbers(node_ids, meta_ids, mdb), node_ids)


class TestFetchPRsNumbers:
    async def test_single_meta_acc_id(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestFactory(acc_id=3, node_id=30, number=100),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=31, number=101),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=32, number=102),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prs_numbers = await fetch_prs_numbers_e2e(np.array([30, 31, 32, 33]), (3,), mdb_rw)
            assert_array_equal(prs_numbers, np.array([100, 101, 102, 0]))

            prs_numbers = await fetch_prs_numbers_e2e(np.array([30, 33, 31]), (3,), mdb_rw)
            assert_array_equal(prs_numbers, np.array([100, 0, 101]))

            prs_numbers = await fetch_prs_numbers_e2e(np.array([33, 35, 30]), (3,), mdb_rw)
            assert_array_equal(prs_numbers, np.array([0, 0, 100]))

    async def test_multiple_meta_acc_ids(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestFactory(acc_id=3, node_id=30, number=1),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=31, number=2),
                md_factory.NodePullRequestFactory(acc_id=4, node_id=32, number=1),
                md_factory.NodePullRequestFactory(acc_id=4, node_id=33, number=3),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prs_numbers = await fetch_prs_numbers_e2e(np.array([30, 31, 32, 33]), (3, 4), mdb_rw)
            assert_array_equal(prs_numbers, np.array([1, 2, 1, 3]))

            prs_numbers = await fetch_prs_numbers_e2e(np.array([30, 31, 32, 35]), (3, 4), mdb_rw)
            assert_array_equal(prs_numbers, np.array([1, 2, 1, 0]))

    async def test_a_whole_lot_of_node_ids(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                *[
                    md_factory.NodePullRequestFactory(acc_id=3, node_id=n, number=100 + n)
                    for n in range(1, 106)
                ],
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prs_numbers = await fetch_prs_numbers_e2e(
                np.array(list(range(1, 111))), (3, 4), mdb_rw,
            )
            assert_array_equal(prs_numbers[:105], np.arange(1, 106) + 100)
            assert_array_equal(prs_numbers[105:], np.zeros(5))

    async def test_unsorted_pr_from_db(self, mdb_rw: Database) -> None:
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.NodePullRequestFactory(acc_id=3, node_id=20, number=3),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=21, number=5),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=22, number=1),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=23, number=4),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=24, number=2),
                md_factory.NodePullRequestFactory(acc_id=3, node_id=25, number=6),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            # DB will often return rows ordered by node_id anyway, mock is needed to have
            # a true chaotic order
            with self._shuffle_read_sql_query_result():
                prs_numbers = await fetch_prs_numbers_e2e(
                    np.array([23, 25, 19, 21, 22]), (3,), mdb_rw,
                )
            assert_array_equal(prs_numbers, np.array([4, 6, 0, 5, 1]))

    async def test_no_pr_found_pr_order_noise_meta_acc_id(self, mdb_rw: Database) -> None:
        prs_numbers = await fetch_prs_numbers_e2e(np.array([22, 23]), (3,), mdb_rw)
        assert_array_equal(prs_numbers, np.array([0, 0]))

    @contextlib.contextmanager
    def _shuffle_read_sql_query_result(self):
        mock_path = f"{fetch_prs_numbers.__module__}.read_sql_query"

        async def _read_sql_query(*args, **kwargs):
            res = await read_sql_query(*args, **kwargs)
            return res.sample(frac=1)

        with mock.patch(mock_path, new=_read_sql_query):
            yield
