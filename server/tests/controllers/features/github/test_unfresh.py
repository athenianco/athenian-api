from datetime import datetime, timezone

from athenian.api.controllers.features import entries
from athenian.api.controllers.features.entries import calc_pull_request_facts_github
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.defer import wait_deferred, with_defer


@with_defer
async def test_fetch_pull_request_facts_unfresh_smoke(release_match_setting_tag, mdb, pdb, rdb):
    time_from = datetime(2017, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    facts_fresh = await calc_pull_request_facts_github(
        time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        False, release_match_setting_tag, False, False,
        1, (6366825,), mdb, pdb, rdb, None,
    )
    await wait_deferred()
    assert len(facts_fresh) == 230
    for f in facts_fresh:
        assert f.repository_full_name == "src-d/go-git"
    orig_threshold = entries.unfresh_prs_threshold
    entries.unfresh_prs_threshold = 1
    try:
        facts_unfresh = await calc_pull_request_facts_github(
            time_from, time_to,
            {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
            False, release_match_setting_tag, False, False,
            1, (6366825,), mdb, pdb, rdb, None,
        )
        assert len(facts_unfresh) == 230
        for i, (fresh, unfresh) in enumerate(zip(sorted(facts_fresh), sorted(facts_unfresh))):
            assert fresh == unfresh, i
    finally:
        entries.unfresh_prs_threshold = orig_threshold


@with_defer
async def test_fetch_pull_request_facts_unfresh_labels(release_match_setting_tag, mdb, pdb, rdb):
    time_from = datetime(2017, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    label_filter = LabelFilter({"enhancement", "bug"}, set())
    facts_fresh = await calc_pull_request_facts_github(
        time_from, time_to,
        {"src-d/go-git"}, {}, label_filter, JIRAFilter.empty(),
        False, release_match_setting_tag, False, False,
        1, (6366825,), mdb, pdb, rdb, None,
    )
    await wait_deferred()
    assert len(facts_fresh) == 6
    orig_threshold = entries.unfresh_prs_threshold
    entries.unfresh_prs_threshold = 1
    try:
        facts_unfresh = await calc_pull_request_facts_github(
            time_from, time_to,
            {"src-d/go-git"}, {}, label_filter, JIRAFilter.empty(),
            False, release_match_setting_tag, False, False,
            1, (6366825,), mdb, pdb, rdb, None,
        )
        assert len(facts_unfresh) == 6
        for i, (fresh, unfresh) in enumerate(zip(sorted(facts_fresh), sorted(facts_unfresh))):
            assert fresh == unfresh, i
    finally:
        entries.unfresh_prs_threshold = orig_threshold


@with_defer
async def test_fetch_pull_request_facts_unfresh_jira(release_match_setting_tag, mdb, pdb, rdb):
    time_from = datetime(2017, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    jira_filter = JIRAFilter(1, ["10003", "10009"], LabelFilter.empty(), set(), {"task"}, False)
    facts_fresh = await calc_pull_request_facts_github(
        time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter.empty(), jira_filter,
        False, release_match_setting_tag, False, False,
        1, (6366825,), mdb, pdb, rdb, None,
    )
    await wait_deferred()
    assert len(facts_fresh) == 36
    orig_threshold = entries.unfresh_prs_threshold
    entries.unfresh_prs_threshold = 1
    try:
        facts_unfresh = await calc_pull_request_facts_github(
            time_from, time_to,
            {"src-d/go-git"}, {}, LabelFilter.empty(), jira_filter,
            False, release_match_setting_tag, False, False,
            1, (6366825,), mdb, pdb, rdb, None,
        )
        assert len(facts_unfresh) == 36
        for i, (fresh, unfresh) in enumerate(zip(sorted(facts_fresh), sorted(facts_unfresh))):
            assert fresh == unfresh, i
    finally:
        entries.unfresh_prs_threshold = orig_threshold
