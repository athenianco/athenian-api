from datetime import datetime, timezone

from athenian.api.controllers.features.entries import calc_pull_request_facts_github
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.jira.issue import fetch_jira_issues, ISSUE_PRS_BEGAN, \
    ISSUE_PRS_RELEASED
from athenian.api.defer import wait_deferred, with_defer


@with_defer
async def test_fetch_jira_issues_releases(mdb, pdb, default_branches, release_match_setting_tag):
    time_from = datetime(2016, 1, 1, tzinfo=timezone.utc)
    time_to = datetime(2021, 1, 1, tzinfo=timezone.utc)
    await calc_pull_request_facts_github(
        time_from, time_to, {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        False, release_match_setting_tag, False, mdb, pdb, None)
    await wait_deferred()
    issues = await fetch_jira_issues(1,
                                     time_from, time_to, False,
                                     LabelFilter.empty(), [], [], [], [], [], False,
                                     default_branches, release_match_setting_tag,
                                     mdb, pdb, None)
    assert issues[ISSUE_PRS_BEGAN].notnull().sum() == 56
    assert issues[ISSUE_PRS_RELEASED].notnull().sum() == 55
    assert (issues[ISSUE_PRS_RELEASED][issues[ISSUE_PRS_RELEASED].notnull()] >
            issues[ISSUE_PRS_BEGAN][issues[ISSUE_PRS_RELEASED].notnull()]).all()
