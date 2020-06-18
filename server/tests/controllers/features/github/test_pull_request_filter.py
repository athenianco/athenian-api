from datetime import date, datetime, timezone

import pytest
from sqlalchemy import delete

from athenian.api.controllers.features.entries import calc_pull_request_metrics_line_github
from athenian.api.controllers.features.github.pull_request_filter import filter_pull_requests
from athenian.api.controllers.miners.types import ParticipationKind, Property
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
from athenian.api.models.precomputed.models import GitHubMergedPullRequest
from athenian.api.models.web import PullRequestMetricID


@pytest.fixture(scope="module")
def time_from_to():
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=1, day=1, tzinfo=timezone.utc)
    return time_from, time_to


async def test_pr_list_miner_none(mdb, pdb, release_match_setting_tag, time_from_to):
    prs = list(await filter_pull_requests(set(), *time_from_to, {"src-d/go-git"}, {}, set(), False,
                                          release_match_setting_tag, mdb, pdb, None))
    assert not prs


async def test_pr_list_miner_match_participants(mdb, pdb, release_match_setting_tag, time_from_to):
    participants = {ParticipationKind.AUTHOR: {"mcuadros", "smola"},
                    ParticipationKind.COMMENTER: {"mcuadros"}}
    prs = list(await filter_pull_requests(
        set(Property), *time_from_to, {"src-d/go-git"}, participants, set(), False,
        release_match_setting_tag, mdb, pdb, None))
    assert len(prs) == 320
    for pr in prs:
        mcuadros_is_author = "mcuadros" in pr.participants[ParticipationKind.AUTHOR]
        smola_is_author = "smola" in pr.participants[ParticipationKind.AUTHOR]
        mcuadros_is_only_commenter = (
            ("mcuadros" in pr.participants[ParticipationKind.COMMENTER])
            and  # noqa
            (not mcuadros_is_author)
            and  # noqa
            (not smola_is_author)
        )
        assert mcuadros_is_author or smola_is_author or mcuadros_is_only_commenter, str(pr)


@pytest.mark.parametrize("date_from, date_to", [(date(year=2018, month=1, day=1),
                                                 date(year=2019, month=1, day=1)),
                                                (date(year=2016, month=12, day=1),
                                                 date(year=2016, month=12, day=15)),
                                                (date(year=2016, month=11, day=17),
                                                 date(year=2016, month=12, day=1))])
async def test_pr_list_miner_match_metrics_all_count(
        mdb, pdb, release_match_setting_tag, date_from, date_to):
    time_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    time_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
    prs = list(await filter_pull_requests(
        set(Property), time_from, time_to, {"src-d/go-git"}, {}, set(), False,
        release_match_setting_tag, mdb, pdb, None))
    assert prs
    await pdb.execute(delete(GitHubMergedPullRequest))  # ignore inactive unreleased
    metric = (await calc_pull_request_metrics_line_github(
        [PullRequestMetricID.PR_ALL_COUNT], [[time_from, time_to]],
        {"src-d/go-git"}, {}, set(), False, release_match_setting_tag, mdb, pdb, None,
    ))[0][0][0]
    assert len(prs) == metric.value
    if date_from.year == 2018:
        # check labels to save some time
        true_labels = {"bug", "enhancement", "plumbing", "ssh", "performance"}
        labels = set()
        colors = set()
        for pr in prs:
            if pr.labels:
                labels.update(label.name for label in pr.labels)
                colors.update(label.color for label in pr.labels)
        assert labels == true_labels
        assert colors == {"84b6eb", "b0f794", "fc2929", "4faccc", "fbca04"}


async def test_pr_list_miner_release_settings(mdb, pdb, release_match_setting_tag, cache):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2019, month=1, day=1, tzinfo=timezone.utc)
    prs1 = list(await filter_pull_requests(
        {Property.RELEASING}, time_from, time_to, {"src-d/go-git"}, {}, set(), False,
        release_match_setting_tag, mdb, pdb, cache))
    assert prs1
    release_match_setting_tag = {
        "github.com/src-d/go-git": ReleaseMatchSetting("master", "unknown", ReleaseMatch.branch),
    }
    prs2 = list(await filter_pull_requests(
        {Property.RELEASING}, time_from, time_to, {"src-d/go-git"}, {}, set(), False,
        release_match_setting_tag, mdb, pdb, cache))
    assert prs2
    assert prs1 != prs2


async def test_pr_list_miner_release_cache_participants(
        mdb, pdb, release_match_setting_tag, cache):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2019, month=1, day=1, tzinfo=timezone.utc)
    participants = {ParticipationKind.AUTHOR: {"mcuadros", "smola"},
                    ParticipationKind.COMMENTER: {"mcuadros"},
                    ParticipationKind.REVIEWER: {"mcuadros", "alcortes"}}
    prs1 = list(await filter_pull_requests(
        {Property.RELEASING}, time_from, time_to, {"src-d/go-git"}, participants, set(), False,
        release_match_setting_tag, mdb, pdb, cache))
    assert prs1
    # reorder
    participants = {ParticipationKind.REVIEWER: {"alcortes", "mcuadros"},
                    ParticipationKind.COMMENTER: {"mcuadros"},
                    ParticipationKind.AUTHOR: {"smola", "mcuadros"}}
    prs2 = list(await filter_pull_requests(
        {Property.RELEASING}, time_from, time_to, {"src-d/go-git"}, participants, set(), False,
        release_match_setting_tag, None, None, cache))
    assert len(prs1) == len(prs2)


async def test_pr_list_miner_exclude_inactive(mdb, pdb, release_match_setting_tag, cache):
    time_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=1, day=11, tzinfo=timezone.utc)
    prs1 = list(await filter_pull_requests(
        set(Property), time_from, time_to, {"src-d/go-git"}, {}, set(), False,
        release_match_setting_tag, mdb, pdb, cache))
    assert len(prs1) == 7
    prs1 = list(await filter_pull_requests(
        set(Property), time_from, time_to, {"src-d/go-git"}, {}, set(), True,
        release_match_setting_tag, mdb, pdb, cache))
    assert len(prs1) == 6


async def test_pr_list_miner_filter_labels_cache(mdb, pdb, release_match_setting_tag, cache):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    prs1 = list(await filter_pull_requests(
        set(Property), time_from, time_to, {"src-d/go-git"}, {}, {"bug", "enhancement"}, False,
        release_match_setting_tag, mdb, pdb, cache))
    assert len(prs1) == 6
    for pr in prs1:
        labels = {label.name for label in pr.labels}
        assert labels.intersection({"bug", "enhancement"})
    prs2 = list(await filter_pull_requests(
        set(Property), time_from, time_to, {"src-d/go-git"}, {}, {"bug"}, False,
        release_match_setting_tag, mdb, pdb, cache))
    assert len(prs2) == 3
    for pr in prs2:
        labels = {label.name for label in pr.labels}
        assert "bug" in labels
    prs3 = list(await filter_pull_requests(
        set(Property), time_from, time_to, {"src-d/go-git"}, {}, {"bug", "plumbing"}, False,
        release_match_setting_tag, mdb, pdb, cache))
    assert len(prs3) == 5
    for pr in prs3:
        labels = {label.name for label in pr.labels}
        assert labels.intersection({"bug", "plumbing"})


async def test_pr_list_miner_filter_labels_pdb(mdb, pdb, release_match_setting_tag):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    await calc_pull_request_metrics_line_github(
        [PullRequestMetricID.PR_ALL_COUNT], [[time_from, time_to]],
        {"src-d/go-git"}, {}, {"bug", "enhancement"}, False,
        release_match_setting_tag, mdb, pdb, None,
    )
    prs = list(await filter_pull_requests(
        set(Property), time_from, time_to, {"src-d/go-git"}, {}, {"bug", "enhancement"}, False,
        release_match_setting_tag, mdb, pdb, None))
    assert len(prs) == 6
    for pr in prs:
        labels = {label.name for label in pr.labels}
        assert labels.intersection({"bug", "enhancement"})
    prs = list(await filter_pull_requests(
        set(Property), time_from, time_to, {"src-d/go-git"}, {}, {"bug"}, False,
        release_match_setting_tag, mdb, pdb, None))
    assert len(prs) == 3
    for pr in prs:
        assert "bug" in {label.name for label in (pr.labels or [])}
