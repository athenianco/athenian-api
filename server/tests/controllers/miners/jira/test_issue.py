from datetime import datetime, timezone

from pandas._testing import assert_frame_equal

from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.jira.issue import fetch_jira_issues, ISSUE_PRS_BEGAN, \
    ISSUE_PRS_RELEASED
from athenian.api.controllers.settings import LogicalRepositorySettings
from athenian.api.defer import wait_deferred, with_defer


@with_defer
async def test_fetch_jira_issues_releases(
        metrics_calculator_factory, mdb, pdb, rdb, default_branches, release_match_setting_tag,
        prefixer, cache):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2016, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2021, 1, 1, tzinfo=timezone.utc)
    await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter.empty(), JIRAFilter.empty(),
        False, release_match_setting_tag, LogicalRepositorySettings.empty(),
        prefixer, False, False)
    await wait_deferred()
    args = [(1, ["10003", "10009"]), time_from, time_to, False,
            LabelFilter.empty(), [], [], [], [], [], [], False,
            default_branches, release_match_setting_tag, LogicalRepositorySettings.empty(),
            1, (6366825,), mdb, pdb, cache]
    issues = await fetch_jira_issues(*args)
    assert issues[ISSUE_PRS_BEGAN].notnull().sum() == 55  # 56 without cleaning
    assert issues[ISSUE_PRS_RELEASED].notnull().sum() == 54  # 55 without cleaning
    assert (issues[ISSUE_PRS_RELEASED][issues[ISSUE_PRS_RELEASED].notnull()] >
            issues[ISSUE_PRS_BEGAN][issues[ISSUE_PRS_RELEASED].notnull()]).all()
    await wait_deferred()
    args[-3] = args[-2] = None
    cached_issues = await fetch_jira_issues(*args)
    assert_frame_equal(issues, cached_issues)


@with_defer
async def test_fetch_jira_issues_no_times(
        mdb, pdb, default_branches, release_match_setting_tag, cache):
    args = [(1, ["10003", "10009"]), None, None, False,
            LabelFilter.empty(), [], [], [], [], [], [], False,
            default_branches, release_match_setting_tag, LogicalRepositorySettings.empty(),
            1, (6366825,), mdb, pdb, cache]
    issues = await fetch_jira_issues(*args)
    await wait_deferred()
    cached_issues = await fetch_jira_issues(*args)
    assert_frame_equal(issues, cached_issues)


@with_defer
async def test_fetch_jira_issues_none_assignee(
        mdb, pdb, default_branches, release_match_setting_tag, cache):
    args = [(1, ["10003", "10009"]), None, None, False,
            LabelFilter.empty(), [], [], [], [], ["vadim markovtsev", None], [], False,
            default_branches, release_match_setting_tag, LogicalRepositorySettings.empty(),
            1, (6366825,), mdb, pdb, cache]
    issues = await fetch_jira_issues(*args)
    assert len(issues) == 716  # 730 without cleaning
    await wait_deferred()
    cached_issues = await fetch_jira_issues(*args)
    assert_frame_equal(issues, cached_issues)
