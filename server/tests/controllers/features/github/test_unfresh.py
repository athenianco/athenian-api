from datetime import datetime, timezone

import numpy as np
from numpy.testing import assert_array_equal
import pandas as pd
from pandas.testing import assert_frame_equal
import pytest
from sqlalchemy import delete, select, update

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.features import entries
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.types import JIRAEntityToFetch, PullRequestFacts
from athenian.api.internal.settings import LogicalRepositorySettings
from athenian.api.models.precomputed.models import (
    GitHubDonePullRequestFacts,
    GitHubMergedPullRequestFacts,
)


@with_defer
async def test_fetch_pull_request_facts_unfresh_smoke(
    metrics_calculator_factory,
    release_match_setting_tag,
    mdb,
    pdb,
    rdb,
    prefixer,
    bots,
    precomputed_deployments,
):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2017, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    facts_fresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
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
        JIRAEntityToFetch.NOTHING,
    )
    facts_fresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    await wait_deferred()
    assert len(facts_fresh) == 230
    assert (facts_fresh["repository_full_name"] == "src-d/go-git").all()
    facts_fresh = facts_fresh.take(
        np.flatnonzero(
            ~(facts_fresh[PullRequestFacts.f.closed].values > pd.Timestamp(time_to).to_numpy()),
        ),
    )
    facts_fresh.reset_index(inplace=True, drop=True)
    orig_threshold = entries.unfresh_prs_threshold
    entries.unfresh_prs_threshold = 1
    try:
        facts_unfresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
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
            JIRAEntityToFetch.NOTHING,
        )
        assert len(facts_unfresh) == 222
        facts_unfresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
        assert_frame_equal(facts_fresh, facts_unfresh)
    finally:
        entries.unfresh_prs_threshold = orig_threshold


@with_defer
async def test_fetch_pull_request_facts_unfresh_labels(
    metrics_calculator_factory,
    release_match_setting_tag,
    mdb,
    pdb,
    rdb,
    prefixer,
    bots,
):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2017, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    label_filter = LabelFilter({"enhancement", "bug"}, set())
    facts_fresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        label_filter,
        JIRAFilter.empty(),
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        False,
        JIRAEntityToFetch.NOTHING,
    )
    facts_fresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    await wait_deferred()
    assert len(facts_fresh) == 6
    orig_threshold = entries.unfresh_prs_threshold
    entries.unfresh_prs_threshold = 1
    try:
        facts_unfresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
            time_from,
            time_to,
            {"src-d/go-git"},
            {},
            label_filter,
            JIRAFilter.empty(),
            False,
            bots,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            prefixer,
            False,
            JIRAEntityToFetch.NOTHING,
        )
        assert len(facts_unfresh) == 6
        facts_unfresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
        assert_frame_equal(facts_fresh, facts_unfresh)
    finally:
        entries.unfresh_prs_threshold = orig_threshold


@with_defer
async def test_fetch_pull_request_facts_unfresh_jira(
    metrics_calculator_factory,
    release_match_setting_tag,
    mdb,
    pdb,
    rdb,
    prefixer,
    bots,
):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2017, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    jira_filter = JIRAFilter(
        1, ["10003", "10009"], LabelFilter.empty(), set(), {"task"}, False, False,
    )
    facts_fresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        jira_filter,
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        False,
        JIRAEntityToFetch.EVERYTHING(),
    )
    await wait_deferred()
    assert len(facts_fresh) == 36
    for node_id in (163160, 163206):
        data = await pdb.fetch_val(
            select([GitHubDonePullRequestFacts.data]).where(
                GitHubDonePullRequestFacts.pr_node_id == node_id,
            ),
        )
        await pdb.execute(
            update(GitHubMergedPullRequestFacts)
            .where(GitHubMergedPullRequestFacts.pr_node_id == node_id)
            .values(
                {
                    GitHubMergedPullRequestFacts.checked_until: datetime.now(timezone.utc),
                    GitHubMergedPullRequestFacts.data: data,
                    GitHubMergedPullRequestFacts.updated_at: datetime.now(timezone.utc),
                },
            ),
        )
        await pdb.execute(
            delete(GitHubDonePullRequestFacts).where(
                GitHubDonePullRequestFacts.pr_node_id == node_id,
            ),
        )
    facts_fresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    facts_fresh = facts_fresh.take(
        np.flatnonzero(
            ~(facts_fresh[PullRequestFacts.f.closed].values > pd.Timestamp(time_to).to_numpy()),
        ),
    )
    facts_fresh.reset_index(inplace=True, drop=True)
    orig_threshold = entries.unfresh_prs_threshold
    entries.unfresh_prs_threshold = 1
    try:
        facts_unfresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
            time_from,
            time_to,
            {"src-d/go-git"},
            {},
            LabelFilter.empty(),
            jira_filter,
            False,
            bots,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            prefixer,
            False,
            JIRAEntityToFetch.EVERYTHING(),
        )
        assert len(facts_unfresh) == 35
        facts_unfresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
        for i in (5, 23):
            facts_unfresh.loc[i, PullRequestFacts.f.releaser] = "mcuadros"
        assert_frame_equal(facts_fresh, facts_unfresh)
    finally:
        entries.unfresh_prs_threshold = orig_threshold

    pr_facts = facts_unfresh[facts_unfresh.node_id == 163168]

    assert_array_equal(pr_facts.jira_ids.iloc[0], np.array(["DEV-261"], dtype="U"))
    assert_array_equal(pr_facts.jira_projects.values[0], np.array([b"10009"], dtype="S"))
    assert_array_equal(pr_facts.jira_types.values[0], np.array([b"10016"], dtype="S"))
    assert_array_equal(pr_facts.jira_priorities.values[0], np.array([b"3"], dtype="S"))


@pytest.mark.parametrize(
    "repos, count",
    [
        ({"src-d/go-git/alpha"}, 67),
        ({"src-d/go-git/alpha", "src-d/go-git/beta"}, 119),
        ({"src-d/go-git", "src-d/go-git/alpha"}, 179),
    ],
)
@pytest.mark.parametrize("exclude_inactive", [False, True])
# there are no precomputed PRs before `time_from`so count-s stay the same
@with_defer
async def test_fetch_pull_request_facts_unfresh_logical_title(
    metrics_calculator_factory,
    release_match_setting_tag_logical,
    mdb,
    pdb,
    rdb,
    prefixer,
    bots,
    logical_settings,
    repos,
    exclude_inactive,
    count,
):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2017, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    facts_fresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from,
        time_to,
        repos,
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        exclude_inactive,
        bots,
        release_match_setting_tag_logical,
        logical_settings,
        prefixer,
        False,
        JIRAEntityToFetch.NOTHING,
    )
    facts_fresh.sort_values(
        [PullRequestFacts.f.created, PullRequestFacts.f.repository_full_name],
        inplace=True,
        ignore_index=True,
    )
    facts_fresh = facts_fresh.take(
        np.flatnonzero(
            ~(facts_fresh[PullRequestFacts.f.closed].values > pd.Timestamp(time_to).to_numpy()),
        ),
    )
    facts_fresh.reset_index(inplace=True, drop=True)
    await wait_deferred()
    assert len(facts_fresh) == count
    assert facts_fresh["repository_full_name"].isin(repos).all()
    orig_threshold = entries.unfresh_prs_threshold
    entries.unfresh_prs_threshold = 1
    try:
        facts_unfresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
            time_from,
            time_to,
            repos,
            {},
            LabelFilter.empty(),
            JIRAFilter.empty(),
            exclude_inactive,
            bots,
            release_match_setting_tag_logical,
            logical_settings,
            prefixer,
            False,
            JIRAEntityToFetch.NOTHING,
        )
        assert len(facts_unfresh) == count
        facts_unfresh.sort_values(
            [PullRequestFacts.f.created, PullRequestFacts.f.repository_full_name],
            inplace=True,
            ignore_index=True,
        )
        assert_frame_equal(facts_fresh, facts_unfresh)
    finally:
        entries.unfresh_prs_threshold = orig_threshold
