from datetime import date, datetime, timezone
from typing import Dict, List

import pytest
from sqlalchemy import delete, select

from athenian.api.controllers.features.github.pull_request_filter import fetch_pull_requests, \
    filter_pull_requests
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.deployment import mine_deployments
from athenian.api.controllers.miners.types import DeployedComponent, Deployment, \
    PRParticipationKind, PullRequestEvent, PullRequestListItem, PullRequestStage
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.precomputed.models import GitHubDonePullRequestFacts, \
    GitHubMergedPullRequestFacts, GitHubOpenPullRequestFacts
from athenian.api.models.web import PullRequestMetricID


@pytest.fixture(scope="module")
def time_from_to():
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=1, day=1, tzinfo=timezone.utc)
    return time_from, time_to


@pytest.mark.parametrize("stage, n", [
    (PullRequestStage.WIP, 6),
    (PullRequestStage.REVIEWING, 7),
    (PullRequestStage.MERGING, 4),
    (PullRequestStage.RELEASING, 130),
    (PullRequestStage.DONE, 529),
    (PullRequestStage.FORCE_PUSH_DROPPED, 10),
])
@with_defer
async def test_pr_list_miner_stages(
        mdb, pdb, rdb, release_match_setting_tag, time_from_to, stage, n, prefixer_promise):
    prs, _ = await filter_pull_requests(
        set(), {stage}, *time_from_to, {"src-d/go-git"}, {},
        LabelFilter.empty(), JIRAFilter.empty(),
        False, release_match_setting_tag, None, None,
        prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    assert len(prs) == n


@with_defer
async def test_pr_list_miner_match_participants(
        mdb, pdb, rdb, release_match_setting_tag, time_from_to, prefixer_promise):
    participants = {PRParticipationKind.AUTHOR: {"mcuadros", "smola"},
                    PRParticipationKind.COMMENTER: {"mcuadros"}}
    prs, _ = await filter_pull_requests(
        set(), set(), *time_from_to, {"src-d/go-git"},
        participants, LabelFilter.empty(), JIRAFilter.empty(), False, release_match_setting_tag,
        None, None, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    assert isinstance(prs, list)
    assert len(prs) == 320
    for pr in prs:
        mcuadros_is_author = "mcuadros" in pr.participants[PRParticipationKind.AUTHOR]
        smola_is_author = "smola" in pr.participants[PRParticipationKind.AUTHOR]
        mcuadros_is_only_commenter = (
            ("mcuadros" in pr.participants[PRParticipationKind.COMMENTER])
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
@with_defer
async def test_pr_list_miner_match_metrics_all_count(
        metrics_calculator_factory, mdb, pdb, rdb, release_match_setting_tag,
        prefixer_promise, date_from, date_to):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    time_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
    prs, _ = await filter_pull_requests(
        set(), set(),
        time_from, time_to, {"src-d/go-git"}, {}, LabelFilter.empty(),
        JIRAFilter.empty(), False, release_match_setting_tag, None, None,
        prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    assert prs
    await pdb.execute(delete(GitHubMergedPullRequestFacts))  # ignore inactive unreleased
    metric = (await metrics_calculator_no_cache.calc_pull_request_metrics_line_github(
        [PullRequestMetricID.PR_ALL_COUNT], [[time_from, time_to]], [0, 1], [],
        [{"src-d/go-git"}], [{}], LabelFilter.empty(), JIRAFilter.empty(), False,
        release_match_setting_tag, prefixer_promise, False,
    ))[0][0][0][0][0][0]
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


@with_defer
async def test_pr_list_miner_release_settings(
        mdb, pdb, rdb, release_match_setting_tag, release_match_setting_branch, cache,
        prefixer_promise):
    time_from = datetime(year=2016, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    prs1, _ = await filter_pull_requests(
        set(), {PullRequestStage.RELEASING}, time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter.empty(), JIRAFilter.empty(), False, release_match_setting_tag,
        None, None, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    assert prs1
    await wait_deferred()
    prs2, _ = await filter_pull_requests(
        set(), {PullRequestStage.RELEASING}, time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter.empty(), JIRAFilter.empty(), False, release_match_setting_branch,
        None, None, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    assert len(prs2) == 96  # force-push-dropped PRs still accessible from the artificial branches


@with_defer
async def test_pr_list_miner_release_cache_participants(
        mdb, pdb, rdb, release_match_setting_tag, cache, prefixer_promise):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2019, month=1, day=1, tzinfo=timezone.utc)
    participants = {PRParticipationKind.AUTHOR: {"mcuadros", "smola"},
                    PRParticipationKind.COMMENTER: {"mcuadros"},
                    PRParticipationKind.REVIEWER: {"mcuadros", "alcortes"}}
    prs1, _ = await filter_pull_requests(
        set(), {PullRequestStage.RELEASING}, time_from, time_to, {"src-d/go-git"},
        participants, LabelFilter.empty(), JIRAFilter.empty(), False, release_match_setting_tag,
        None, None, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    assert prs1
    # reorder
    participants = {PRParticipationKind.REVIEWER: {"alcortes", "mcuadros"},
                    PRParticipationKind.COMMENTER: {"mcuadros"},
                    PRParticipationKind.AUTHOR: {"smola", "mcuadros"}}
    prs2, _ = await filter_pull_requests(
        set(), {PullRequestStage.RELEASING}, time_from, time_to, {"src-d/go-git"},
        participants, LabelFilter.empty(), JIRAFilter.empty(), False, release_match_setting_tag,
        None, None, prefixer_promise, 1, (6366825,), None, None, None, cache)
    assert len(prs1) == len(prs2)


@with_defer
async def test_pr_list_miner_exclude_inactive(
        mdb, pdb, rdb, release_match_setting_tag, cache, prefixer_promise):
    time_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=1, day=11, tzinfo=timezone.utc)
    prs1, _ = await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter.empty(), JIRAFilter.empty(), False, release_match_setting_tag,
        None, None, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    assert len(prs1) == 7
    prs1, _ = await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter.empty(), JIRAFilter.empty(), True, release_match_setting_tag,
        None, None, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    assert len(prs1) == 6


@with_defer
async def test_pr_list_miner_filter_labels_cache(
        mdb, pdb, rdb, release_match_setting_tag, cache, prefixer_promise):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    prs1, _ = await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug", "enhancement"}, set()), JIRAFilter.empty(), False,
        release_match_setting_tag, None, None,
        prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    assert len(prs1) == 6
    for pr in prs1:
        labels = {label.name for label in pr.labels}
        assert labels.intersection({"bug", "enhancement"})
    prs2, _ = await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug"}, set()), JIRAFilter.empty(), False,
        release_match_setting_tag, None, None,
        prefixer_promise, 1, (6366825,), None, None, None, cache)
    await wait_deferred()
    assert len(prs2) == 3
    for pr in prs2:
        labels = {label.name for label in pr.labels}
        assert "bug" in labels
    prs3, _ = await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug", "plumbing"}, set()), JIRAFilter.empty(), False,
        release_match_setting_tag, None, None,
        prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    assert len(prs3) == 5
    for pr in prs3:
        labels = {label.name for label in pr.labels}
        assert labels.intersection({"bug", "plumbing"})


@with_defer
async def test_pr_list_miner_filter_labels_cache_postprocess(
        mdb, pdb, rdb, release_match_setting_tag, cache, prefixer_promise):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter.empty(), JIRAFilter.empty(), False, release_match_setting_tag,
        None, None, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    prs1, _ = await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug"}, set()), JIRAFilter.empty(), False,
        release_match_setting_tag, None, None,
        prefixer_promise, 1, (6366825,), None, None, None, cache)
    await wait_deferred()
    assert len(prs1) == 3
    prs2, _ = await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug"}, set()), JIRAFilter.empty(), False,
        release_match_setting_tag, None, None,
        prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    assert prs1 == prs2
    cache.mem = {}
    await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug"}, set()), JIRAFilter.empty(), False,
        release_match_setting_tag, None, None,
        prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    with pytest.raises(Exception):
        await filter_pull_requests(
            set(), set(), time_from, time_to, {"src-d/go-git"}, {},
            LabelFilter.empty(), JIRAFilter.empty(), False, release_match_setting_tag,
            None, None, prefixer_promise, 1, (6366825,), None, None, None, cache)


@with_defer
async def test_pr_list_miner_filter_labels_cache_exclude(
        mdb, pdb, rdb, release_match_setting_tag, cache, prefixer_promise):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    prs1, _ = await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter(set(), {"bug"}), JIRAFilter.empty(), False,
        release_match_setting_tag, None, None,
        prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    assert len(prs1) == 71

    prs2, _ = await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter.empty(), JIRAFilter.empty(), False, release_match_setting_tag,
        None, None, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    assert len(prs2) == 74

    prs3, _ = await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter(set(), {"bug"}), JIRAFilter.empty(), False,
        release_match_setting_tag, None, None,
        prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    assert len(prs3) == 71


@with_defer
async def test_pr_list_miner_filter_labels_pdb(
        metrics_calculator_factory, mdb, pdb, rdb, release_match_setting_tag, prefixer_promise):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug", "enhancement"}, set()), JIRAFilter.empty(),
        False, release_match_setting_tag, prefixer_promise,
        False, False,
    )
    await wait_deferred()
    prs, _ = await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug", "enhancement"}, set()), JIRAFilter.empty(), False,
        release_match_setting_tag, None, None,
        prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    assert len(prs) == 6
    for pr in prs:
        labels = {label.name for label in pr.labels}
        assert labels.intersection({"bug", "enhancement"})
    prs, _ = await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter({"bug"}, set()), JIRAFilter.empty(), False, release_match_setting_tag,
        None, None, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    assert len(prs) == 3
    for pr in prs:
        assert "bug" in {label.name for label in (pr.labels or [])}


@with_defer
async def test_fetch_pull_requests_smoke(
        metrics_calculator_factory, mdb, pdb, rdb, release_match_setting_tag,
        prefixer_promise, cache):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    await metrics_calculator_no_cache.calc_pull_request_facts_github(
        time_from, time_to,
        {"src-d/go-git"}, {}, LabelFilter({"bug", "enhancement"}, set()), JIRAFilter.empty(),
        False, release_match_setting_tag, prefixer_promise,
        False, False,
    )
    await wait_deferred()
    prs1, deps = await filter_pull_requests(
        set(), set(), time_from, time_to, {"src-d/go-git"}, {},
        LabelFilter.empty(), JIRAFilter.empty(), False, release_match_setting_tag,
        None, None, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    assert deps == {}
    prs1 = {pr.number: pr for pr in prs1}
    # 921 is needed to check done_times
    prs2, deps2 = await fetch_pull_requests(
        {"src-d/go-git": set(list(range(1000, 1011)) + [921])},
        release_match_setting_tag, prefixer_promise, 1, (6366825,), mdb, pdb, rdb, cache)
    await wait_deferred()
    prs3, deps3 = await fetch_pull_requests(
        {"src-d/go-git": set(list(range(1000, 1011)) + [921])},
        release_match_setting_tag, prefixer_promise, 1, (6366825,), None, None, None, cache)
    assert prs2 == prs3
    assert deps2 == deps3 == {}
    del prs3
    assert len(prs2) == 12
    for pr2 in prs2:
        assert 1000 <= pr2.number <= 1010 or pr2.number == 921
        pr1 = prs1[pr2.number]
        object.__setattr__(pr1, "stages_time_machine", None)
        object.__setattr__(pr1, "events_time_machine", None)
        assert pr1 == pr2, pr1.number
    with pytest.raises(Exception):
        await fetch_pull_requests(
            {"src-d/go-git": set(list(range(1000, 1011)) + [922])},
            release_match_setting_tag, prefixer_promise, 1, (6366825,), None, None, None, cache)


@with_defer
async def test_fetch_pull_requests_no_merged(
        mdb, pdb, rdb, release_match_setting_tag, prefixer_promise, cache):
    prs, _ = await fetch_pull_requests(
        {"src-d/go-git": {1069}}, release_match_setting_tag, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, cache)
    assert len(prs) == 1
    assert prs[0].number == 1069
    assert PullRequestStage.WIP in prs[0].stages_now
    assert PullRequestEvent.CREATED in prs[0].events_now
    assert PullRequestEvent.COMMITTED in prs[0].events_now
    assert prs[0].stages_time_machine is None
    assert prs[0].events_time_machine is None


@with_defer
async def test_fetch_pull_requests_empty(
        mdb, pdb, rdb, release_match_setting_tag, prefixer_promise, cache):
    prs, deployments = await fetch_pull_requests(
        {"src-d/go-git": {0}}, release_match_setting_tag, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, cache)
    assert len(prs) == 0
    assert len(deployments) == 0


@with_defer
async def test_pr_list_miner_filter_open_precomputed(
        mdb, pdb, rdb, release_match_setting_tag, prefixer_promise):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    args = [set(),
            {PullRequestStage.WIP, PullRequestStage.REVIEWING, PullRequestStage.MERGING},
            time_from, time_to, {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
            False, release_match_setting_tag, None, None,
            prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None]
    prs1, _ = await filter_pull_requests(*args)
    await wait_deferred()
    assert len(prs1) == 21
    open_facts = await pdb.fetch_all(select([GitHubOpenPullRequestFacts]))
    assert len(open_facts) == 21

    # the following is offtopic but saves the precious execution time
    done_facts = await pdb.fetch_all(select([GitHubDonePullRequestFacts]))
    assert len(done_facts) == 293
    merged_facts = await pdb.fetch_all(select([GitHubMergedPullRequestFacts]))
    assert len(merged_facts) == 245
    # offtopic ends

    prs2, _ = await filter_pull_requests(*args)
    assert {pr.number for pr in prs1} == {pr.number for pr in prs2}
    assert {tuple(sorted(pr.stage_timings)) for pr in prs1} == \
           {tuple(sorted(pr.stage_timings)) for pr in prs2}


@with_defer
async def test_pr_list_miner_filter_stages_events_aggregation(
        mdb, pdb, rdb, release_match_setting_tag, prefixer_promise):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    args = [{PullRequestEvent.REVIEWED},
            {PullRequestStage.WIP, PullRequestStage.REVIEWING, PullRequestStage.MERGING},
            time_from, time_to, {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
            False, release_match_setting_tag, None, None,
            prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None]
    prs, _ = await filter_pull_requests(*args)
    assert len(prs) == 132


@with_defer
async def test_pr_list_miner_filter_updated_min_max(
        mdb, pdb, rdb, release_match_setting_tag, prefixer_promise):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    args = [{PullRequestEvent.REVIEWED},
            {PullRequestStage.WIP, PullRequestStage.REVIEWING, PullRequestStage.MERGING},
            time_from, time_to, {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
            False, release_match_setting_tag,
            datetime(2018, 2, 1, tzinfo=timezone.utc),
            datetime(2019, 2, 1, tzinfo=timezone.utc),
            prefixer_promise, 1, (6366825,), mdb, pdb, rdb, None]
    prs, _ = await filter_pull_requests(*args)
    assert len(prs) == 72


@with_defer
async def test_filter_pull_requests_deployments(
        mdb, pdb, rdb, release_match_setting_tag, prefixer_promise, branches, default_branches):
    time_from = datetime(year=2019, month=6, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2019, month=12, day=1, tzinfo=timezone.utc)
    await mine_deployments(
        [40550], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag,
        branches, default_branches, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    args = [{PullRequestEvent.REVIEWED},
            {PullRequestStage.WIP, PullRequestStage.REVIEWING, PullRequestStage.MERGING},
            time_from, time_to, {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
            False, release_match_setting_tag, None, None, prefixer_promise,
            1, (6366825,), mdb, pdb, rdb, None]
    prs, deps = await filter_pull_requests(*args)
    check_pr_deployments(prs, deps)


def check_pr_deployments(prs: List[PullRequestListItem], deps: Dict[str, Deployment]) -> None:
    prs_have_deps = False
    for pr in prs:
        if pr.deployments is not None:
            assert len(pr.deployments) == 1 and pr.deployments[0] == "Dummy deployment", pr
            prs_have_deps = True
    assert prs_have_deps
    assert deps == {
        "Dummy deployment": Deployment(
            name="Dummy deployment",
            conclusion="SUCCESS",
            environment="production",
            url=None,
            started_at=datetime(2019, 11, 1, 12, 0, tzinfo=timezone.utc),
            finished_at=datetime(2019, 11, 1, 12, 15, tzinfo=timezone.utc),
            components=[
                DeployedComponent(
                    repository_id=40550,
                    reference="v4.13.1",
                    sha="0d1a009cbb604db18be960db5f1525b99a55d727",
                ),
            ],
            labels=None),
    }


@with_defer
async def test_fetch_pull_requests_deployments(
        mdb, pdb, rdb, release_match_setting_tag, prefixer_promise, branches, default_branches):
    time_from = datetime(year=2019, month=6, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2019, month=12, day=1, tzinfo=timezone.utc)
    await mine_deployments(
        [40550], {},
        time_from, time_to,
        ["production", "staging"],
        [], {}, {}, LabelFilter.empty(), JIRAFilter.empty(),
        release_match_setting_tag,
        branches, default_branches, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, None)
    await wait_deferred()
    prs, deps = await fetch_pull_requests(
        {"src-d/go-git": {1160, 1179}}, release_match_setting_tag, prefixer_promise,
        1, (6366825,), mdb, pdb, rdb, None)
    check_pr_deployments(prs, deps)
