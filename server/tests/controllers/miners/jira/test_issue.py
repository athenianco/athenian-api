from datetime import datetime, timezone
from typing import Sequence

import pandas as pd
from pandas._testing import assert_frame_equal
from sqlalchemy import insert

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.jira import JIRAConfig
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.precomputed_prs import store_precomputed_done_facts
from athenian.api.internal.miners.github.released_pr import matched_by_column
from athenian.api.internal.miners.jira.issue import (
    ISSUE_PRS_BEGAN,
    ISSUE_PRS_RELEASED,
    _fetch_released_prs,
    fetch_jira_issues,
)
from athenian.api.internal.miners.types import (
    MinedPullRequest,
    PullRequestCheckRun,
    PullRequestFacts,
)
from athenian.api.internal.settings import (
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
)
from athenian.api.models.metadata.github import PullRequest, PullRequestCommit, Release
from athenian.precomputer.db.models import GitHubDonePullRequestFacts


@with_defer
async def test_fetch_jira_issues_releases(
    metrics_calculator_factory,
    mdb,
    pdb,
    rdb,
    default_branches,
    release_match_setting_tag,
    prefixer,
    bots,
    cache,
):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2016, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2021, 1, 1, tzinfo=timezone.utc)
    await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        False,
        0,
    )
    await wait_deferred()
    args = [
        JIRAConfig(1, ["10003", "10009"], {}),
        time_from,
        time_to,
        False,
        LabelFilter.empty(),
        [],
        [],
        [],
        [],
        [],
        [],
        False,
        default_branches,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        1,
        (6366825,),
        mdb,
        pdb,
        cache,
    ]
    issues = await fetch_jira_issues(*args)

    assert issues[ISSUE_PRS_BEGAN].notnull().sum() == 55  # 56 without cleaning
    assert issues[ISSUE_PRS_RELEASED].notnull().sum() == 54  # 55 without cleaning
    assert (
        issues[ISSUE_PRS_RELEASED][issues[ISSUE_PRS_RELEASED].notnull()]
        > issues[ISSUE_PRS_BEGAN][issues[ISSUE_PRS_RELEASED].notnull()]
    ).all()

    await wait_deferred()
    args[-3] = args[-2] = None
    cached_issues = await fetch_jira_issues(*args)
    assert_frame_equal(issues, cached_issues)
    args[-7] = ReleaseSettings({})
    args[-3] = mdb
    args[-2] = pdb
    ghdprf = GitHubDonePullRequestFacts
    await pdb.execute(
        insert(ghdprf).values(
            {
                ghdprf.acc_id: 1,
                ghdprf.pr_node_id: 163250,
                ghdprf.repository_full_name: "src-d/go-git",
                ghdprf.release_match: "branch|master",
                ghdprf.pr_done_at: datetime(2018, 7, 17, tzinfo=timezone.utc),
                ghdprf.pr_created_at: datetime(2018, 5, 17, tzinfo=timezone.utc),
                ghdprf.number: 1,
                ghdprf.updated_at: datetime.now(timezone.utc),
                ghdprf.format_version: ghdprf.__table__.columns[
                    ghdprf.format_version.key
                ].default.arg,
                ghdprf.data: b"test",
            },
        ),
    )
    issues = await fetch_jira_issues(*args)
    assert issues[ISSUE_PRS_BEGAN].notnull().sum() == 55
    assert issues[ISSUE_PRS_RELEASED].notnull().sum() == 55


@with_defer
async def test_fetch_jira_issues_no_times(
    mdb,
    pdb,
    default_branches,
    release_match_setting_tag,
    cache,
):
    args = [
        JIRAConfig(1, ["10003", "10009"], {}),
        None,
        None,
        False,
        LabelFilter.empty(),
        [],
        [],
        [],
        [],
        [],
        [],
        False,
        default_branches,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        1,
        (6366825,),
        mdb,
        pdb,
        cache,
    ]
    issues = await fetch_jira_issues(*args)
    await wait_deferred()
    cached_issues = await fetch_jira_issues(*args)
    assert_frame_equal(issues, cached_issues)


@with_defer
async def test_fetch_jira_issues_none_assignee(
    mdb,
    pdb,
    default_branches,
    release_match_setting_tag,
    cache,
):
    args = [
        JIRAConfig(1, ["10003", "10009"], {}),
        None,
        None,
        False,
        LabelFilter.empty(),
        [],
        [],
        [],
        [],
        ["vadim markovtsev", None],
        [],
        False,
        default_branches,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        1,
        (6366825,),
        mdb,
        pdb,
        cache,
    ]
    issues = await fetch_jira_issues(*args)
    assert len(issues) == 716  # 730 without cleaning
    await wait_deferred()
    cached_issues = await fetch_jira_issues(*args)
    assert_frame_equal(issues, cached_issues)


def gen_dummy_df(dt: datetime) -> pd.DataFrame:
    return pd.DataFrame.from_records(
        [["vmarkovtsev", 40020, dt, dt]],
        columns=["user_login", "user_node_id", "created_at", "submitted_at"],
    )


async def test__fetch_released_prs_release_settings_events(
    pr_samples,
    pdb,
    done_prs_facts_loader,
    prefixer,
):
    samples = pr_samples(12)  # type: Sequence[PullRequestFacts]
    names = ["one", "two", "three"]
    settings = ReleaseSettings(
        {
            "github.com/"
            + k: ReleaseMatchSetting("{{default}}", ".*", ".*", ReleaseMatch.tag_or_branch)
            for k in names
        },
    )
    default_branches = {k: "master" for k in names}
    prs = [
        MinedPullRequest(
            pr={
                PullRequest.created_at.name: s.created,
                PullRequest.repository_full_name.name: names[i % len(names)],
                PullRequest.user_login.name: ["vmarkovtsev", "marnovo"][i % 2],
                PullRequest.user_node_id.name: [40020, 39792][i % 2],
                PullRequest.merged_by_login.name: "mcuadros",
                PullRequest.merged_by_id.name: 39789,
                PullRequest.number.name: i + 1,
                PullRequest.node_id.name: i + 100500,
            },
            release={
                matched_by_column: match,
                Release.author.name: ["marnovo", "mcarmonaa"][i % 2],
                Release.author_node_id.name: [39792, 39818][i % 2],
                Release.url.name: "https://release",
                Release.node_id.name: i,
            },
            comments=gen_dummy_df(s.first_comment_on_first_review),
            commits=pd.DataFrame.from_records(
                [["mcuadros", "mcuadros", 39789, 39789, s.first_commit]],
                columns=[
                    PullRequestCommit.committer_login.name,
                    PullRequestCommit.author_login.name,
                    PullRequestCommit.committer_user_id.name,
                    PullRequestCommit.author_user_id.name,
                    PullRequestCommit.committed_date.name,
                ],
            ),
            reviews=gen_dummy_df(s.first_comment_on_first_review),
            review_comments=gen_dummy_df(s.first_comment_on_first_review),
            review_requests=gen_dummy_df(s.first_review_request),
            labels=pd.DataFrame.from_records(([["bug"]], [["feature"]])[i % 2], columns=["name"]),
            jiras=pd.DataFrame(),
            deployments=None,
            check_run={PullRequestCheckRun.f.name: None},
        )
        for match in (ReleaseMatch.tag, ReleaseMatch.event)
        for i, s in enumerate(samples)
    ]

    def with_mutables(s, i):
        s.repository_full_name = names[i % len(names)]
        s.author = ["vmarkovtsev", "marnovo"][i % 2]
        s.merger = "mcuadros"
        s.releaser = ["marnovo", "mcarmonaa"][i % 2]
        return s

    await store_precomputed_done_facts(
        prs,
        [with_mutables(s, i) for i, s in enumerate(samples)] * 2,
        datetime(2050, 1, 1, tzinfo=timezone.utc),
        default_branches,
        settings,
        1,
        pdb,
    )

    new_prs = await _fetch_released_prs(
        [i + 100500 for i in range(len(samples))],
        default_branches,
        settings,
        1,
        pdb,
    )

    assert len(new_prs) == len(samples)
