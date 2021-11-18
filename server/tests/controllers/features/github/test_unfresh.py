from datetime import datetime, timezone

from pandas.testing import assert_frame_equal
from sqlalchemy import delete, select, update

from athenian.api.controllers.features import entries
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.types import PullRequestFacts
from athenian.api.controllers.settings import LogicalRepositorySettings
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts, \
    GitHubMergedPullRequestFacts


@with_defer
async def test_fetch_pull_request_facts_unfresh_smoke(
        metrics_calculator_factory, release_match_setting_tag, mdb, pdb, rdb, prefixer,
        precomputed_deployments):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2017, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    facts_fresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, False, False,
    )
    facts_fresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    await wait_deferred()
    assert len(facts_fresh) == 230
    assert (facts_fresh["repository_full_name"] == "src-d/go-git").all()
    orig_threshold = entries.unfresh_prs_threshold
    entries.unfresh_prs_threshold = 1
    try:
        facts_unfresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
            time_from, time_to,
            {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
            False, release_match_setting_tag, LogicalRepositorySettings.empty(),
            prefixer, False, False,
        )
        assert len(facts_unfresh) == 230
        facts_unfresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
        assert_frame_equal(facts_fresh, facts_unfresh)
    finally:
        entries.unfresh_prs_threshold = orig_threshold


@with_defer
async def test_fetch_pull_request_facts_unfresh_labels(
        metrics_calculator_factory, release_match_setting_tag, mdb, pdb, rdb, prefixer):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2017, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    label_filter = LabelFilter({"enhancement", "bug"}, set())
    facts_fresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from, time_to,
        {"src-d/go-git"}, {}, label_filter, JIRAFilter.empty(),
        False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, False, False,
    )
    facts_fresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    await wait_deferred()
    assert len(facts_fresh) == 6
    orig_threshold = entries.unfresh_prs_threshold
    entries.unfresh_prs_threshold = 1
    try:
        facts_unfresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
            time_from, time_to,
            {"src-d/go-git"}, {}, label_filter, JIRAFilter.empty(),
            False, release_match_setting_tag, LogicalRepositorySettings.empty(),
            prefixer, False, False,
        )
        assert len(facts_unfresh) == 6
        facts_unfresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
        assert_frame_equal(facts_fresh, facts_unfresh)
    finally:
        entries.unfresh_prs_threshold = orig_threshold


@with_defer
async def test_fetch_pull_request_facts_unfresh_jira(
        metrics_calculator_factory, release_match_setting_tag, mdb, pdb, rdb, prefixer):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2017, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    jira_filter = JIRAFilter(1, ["10003", "10009"], LabelFilter.empty(), set(), {"task"}, False)
    facts_fresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), jira_filter,
        False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, False, True,
    )
    await wait_deferred()
    assert len(facts_fresh) == 36
    for node_id in (163160, 163206):
        data = await pdb.fetch_val(select([GitHubDonePullRequestFacts.data])
                                   .where(GitHubDonePullRequestFacts.pr_node_id == node_id))
        await pdb.execute(
            update(GitHubMergedPullRequestFacts)
            .where(GitHubMergedPullRequestFacts.pr_node_id == node_id)
            .values({GitHubMergedPullRequestFacts.checked_until: datetime.now(timezone.utc),
                     GitHubMergedPullRequestFacts.data: data,
                     GitHubMergedPullRequestFacts.updated_at: datetime.now(timezone.utc)}))
        await pdb.execute(delete(GitHubDonePullRequestFacts)
                          .where(GitHubDonePullRequestFacts.pr_node_id == node_id))
    facts_fresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    orig_threshold = entries.unfresh_prs_threshold
    entries.unfresh_prs_threshold = 1
    try:
        facts_unfresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
            time_from, time_to,
            {"src-d/go-git"}, {}, LabelFilter.empty(), jira_filter,
            False, release_match_setting_tag, LogicalRepositorySettings.empty(),
            prefixer, False, True,
        )
        assert len(facts_unfresh) == 36
        facts_unfresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
        for i in (5, 23):
            facts_unfresh.loc[i, PullRequestFacts.f.releaser] = "mcuadros"
        assert_frame_equal(facts_fresh, facts_unfresh)
    finally:
        entries.unfresh_prs_threshold = orig_threshold
