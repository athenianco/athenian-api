from datetime import datetime, timezone

from pandas.testing import assert_frame_equal

from athenian.api.controllers.features import entries
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.types import PullRequestFacts
from athenian.api.defer import wait_deferred, with_defer


@with_defer
async def test_fetch_pull_request_facts_unfresh_smoke(
        metrics_calculator_factory, release_match_setting_tag, mdb, pdb, rdb, prefixer_promise):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2017, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    facts_fresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        False, release_match_setting_tag, prefixer_promise, False, False,
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
            False, release_match_setting_tag, prefixer_promise, False, False,
        )
        assert len(facts_unfresh) == 230
        facts_unfresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
        assert_frame_equal(facts_fresh, facts_unfresh)
    finally:
        entries.unfresh_prs_threshold = orig_threshold


@with_defer
async def test_fetch_pull_request_facts_unfresh_labels(
        metrics_calculator_factory, release_match_setting_tag, mdb, pdb, rdb, prefixer_promise):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2017, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    label_filter = LabelFilter({"enhancement", "bug"}, set())
    facts_fresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from, time_to,
        {"src-d/go-git"}, {}, label_filter, JIRAFilter.empty(),
        False, release_match_setting_tag, prefixer_promise, False, False,
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
            False, release_match_setting_tag, prefixer_promise, False, False,
        )
        assert len(facts_unfresh) == 6
        facts_unfresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
        assert_frame_equal(facts_fresh, facts_unfresh)
    finally:
        entries.unfresh_prs_threshold = orig_threshold


@with_defer
async def test_fetch_pull_request_facts_unfresh_jira(
        metrics_calculator_factory, release_match_setting_tag, mdb, pdb, rdb, prefixer_promise):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2017, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    jira_filter = JIRAFilter(1, ["10003", "10009"], LabelFilter.empty(), set(), {"task"}, False)
    facts_fresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), jira_filter,
        False, release_match_setting_tag, prefixer_promise, False, False,
    )
    await wait_deferred()
    assert len(facts_fresh) == 36
    facts_fresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    orig_threshold = entries.unfresh_prs_threshold
    entries.unfresh_prs_threshold = 1
    try:
        facts_unfresh = await metrics_calculator_no_cache.calc_pull_request_facts_github(
            time_from, time_to,
            {"src-d/go-git"}, {}, LabelFilter.empty(), jira_filter,
            False, release_match_setting_tag, prefixer_promise, False, False,
        )
        assert len(facts_unfresh) == 36
        facts_unfresh.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
        assert_frame_equal(facts_fresh, facts_unfresh)
    finally:
        entries.unfresh_prs_threshold = orig_threshold
