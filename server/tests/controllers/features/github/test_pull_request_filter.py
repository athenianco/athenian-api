from datetime import date, datetime, timedelta, timezone
from typing import Dict, List

import pandas as pd
import pytest
from sqlalchemy import delete, select

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.features.github.pull_request_filter import (
    fetch_pull_requests,
    filter_pull_requests,
)
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.deployment import mine_deployments
from athenian.api.internal.miners.types import (
    DeployedComponent,
    Deployment,
    DeploymentConclusion,
    PRParticipationKind,
    PullRequestEvent,
    PullRequestListItem,
    PullRequestStage,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, Settings
from athenian.api.models.precomputed.models import (
    GitHubDonePullRequestFacts,
    GitHubMergedPullRequestFacts,
    GitHubOpenPullRequestFacts,
)
from athenian.api.models.web import PullRequestMetricID
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_ACCOUNT_ID, DEFAULT_MD_ACCOUNT_ID
from tests.testutils.factory.wizards import insert_repo, pr_models


@pytest.fixture(scope="module")
def time_from_to():
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=1, day=1, tzinfo=timezone.utc)
    return time_from, time_to


@pytest.mark.parametrize(
    "stage, n",
    [
        (PullRequestStage.WIP, 6),
        (PullRequestStage.REVIEWING, 7),
        (PullRequestStage.MERGING, 4),
        (PullRequestStage.RELEASING, 130),
        (PullRequestStage.DONE, 529),
        (PullRequestStage.FORCE_PUSH_DROPPED, 5),
    ],
)
@with_defer
async def test_pr_list_miner_stages(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    time_from_to,
    stage,
    n,
    prefixer,
    bots,
):
    prs, _ = await filter_pull_requests(
        set(),
        {stage},
        *time_from_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert len(prs) == n


@with_defer
async def test_pr_list_miner_match_participants(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    time_from_to,
    prefixer,
    bots,
):
    participants = {
        PRParticipationKind.AUTHOR: {"mcuadros", "smola"},
        PRParticipationKind.COMMENTER: {"mcuadros"},
    }
    prs, _ = await filter_pull_requests(
        set(),
        set(),
        *time_from_to,
        {"src-d/go-git"},
        participants,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert isinstance(prs, list)
    assert len(prs) == 320
    for pr in prs:
        mcuadros_is_author = 39789 in pr.participants[PRParticipationKind.AUTHOR]
        smola_is_author = 40070 in pr.participants[PRParticipationKind.AUTHOR]
        mcuadros_is_only_commenter = (
            (39789 in pr.participants[PRParticipationKind.COMMENTER])
            and (not mcuadros_is_author)  # noqa
            and (not smola_is_author)  # noqa
        )
        assert mcuadros_is_author or smola_is_author or mcuadros_is_only_commenter, str(pr)


@pytest.mark.parametrize(
    "date_from, date_to",
    [
        (date(year=2018, month=1, day=1), date(year=2019, month=1, day=1)),
        (date(year=2016, month=12, day=1), date(year=2016, month=12, day=15)),
        (date(year=2016, month=11, day=17), date(year=2016, month=12, day=1)),
    ],
)
@with_defer
async def test_pr_list_miner_match_metrics_all_count(
    metrics_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
    date_from,
    date_to,
    branches,
    default_branches,
):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime.combine(date_from, datetime.min.time(), tzinfo=timezone.utc)
    time_to = datetime.combine(date_to, datetime.min.time(), tzinfo=timezone.utc)
    prs, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    await wait_deferred()
    assert prs
    await pdb.execute(delete(GitHubMergedPullRequestFacts))  # ignore inactive unreleased
    metric = (
        await metrics_calculator_no_cache.calc_pull_request_metrics_line_github(
            [PullRequestMetricID.PR_ALL_COUNT],
            [[time_from, time_to]],
            [0, 1],
            [],
            [],
            [{"src-d/go-git"}],
            [{}],
            LabelFilter.empty(),
            JIRAFilter.empty(),
            False,
            bots,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            prefixer,
            branches,
            default_branches,
            False,
        )
    )[0][0][0][0][0][0]
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
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    release_match_setting_branch,
    cache,
    prefixer,
    bots,
):
    time_from = datetime(year=2016, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    prs1, _ = await filter_pull_requests(
        set(),
        {PullRequestStage.RELEASING},
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    assert prs1
    await wait_deferred()
    prs2, _ = await filter_pull_requests(
        set(),
        {PullRequestStage.RELEASING},
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_branch,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    assert len(prs2) == 96  # force-push-dropped PRs still accessible from the artificial branches


@with_defer
async def test_pr_list_miner_release_cache_participants(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    cache,
    prefixer,
    bots,
):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2019, month=1, day=1, tzinfo=timezone.utc)
    participants = {
        PRParticipationKind.AUTHOR: {"mcuadros", "smola"},
        PRParticipationKind.COMMENTER: {"mcuadros"},
        PRParticipationKind.REVIEWER: {"mcuadros", "alcortes"},
    }
    prs1, _ = await filter_pull_requests(
        set(),
        {PullRequestStage.RELEASING},
        time_from,
        time_to,
        {"src-d/go-git"},
        participants,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    await wait_deferred()
    assert prs1
    # reorder
    participants = {
        PRParticipationKind.REVIEWER: {"alcortes", "mcuadros"},
        PRParticipationKind.COMMENTER: {"mcuadros"},
        PRParticipationKind.AUTHOR: {"smola", "mcuadros"},
    }
    prs2, _ = await filter_pull_requests(
        set(),
        {PullRequestStage.RELEASING},
        time_from,
        time_to,
        {"src-d/go-git"},
        participants,
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        None,
        None,
        None,
        cache,
    )
    assert len(prs1) == len(prs2)


@with_defer
async def test_pr_list_miner_exclude_inactive(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    cache,
    prefixer,
    bots,
):
    time_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2017, month=1, day=11, tzinfo=timezone.utc)
    prs1, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    assert len(prs1) == 7
    prs1, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        True,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    assert len(prs1) == 6


@with_defer
async def test_pr_list_miner_filter_labels_cache(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    cache,
    prefixer,
    bots,
):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    prs1, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug", "enhancement"}, set()),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    await wait_deferred()
    assert len(prs1) == 6
    for pr in prs1:
        labels = {label.name for label in pr.labels}
        assert labels.intersection({"bug", "enhancement"})
    prs2, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug"}, set()),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        None,
        None,
        None,
        cache,
    )
    await wait_deferred()
    assert len(prs2) == 3
    for pr in prs2:
        labels = {label.name for label in pr.labels}
        assert "bug" in labels
    prs3, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug", "plumbing"}, set()),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    assert len(prs3) == 5
    for pr in prs3:
        labels = {label.name for label in pr.labels}
        assert labels.intersection({"bug", "plumbing"})


@with_defer
async def test_pr_list_miner_filter_labels_cache_postprocess(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    cache,
    prefixer,
    bots,
):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    await wait_deferred()
    prs1, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug"}, set()),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        None,
        None,
        None,
        cache,
    )
    await wait_deferred()
    assert len(prs1) == 3
    prs2, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug"}, set()),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    await wait_deferred()
    assert prs1 == prs2
    cache.mem = {}
    await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug"}, set()),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    await wait_deferred()
    with pytest.raises(Exception):
        await filter_pull_requests(
            set(),
            set(),
            time_from,
            time_to,
            {"src-d/go-git"},
            {},
            LabelFilter.empty(),
            JIRAFilter.empty(),
            ["production"],
            False,
            bots,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            None,
            None,
            prefixer,
            1,
            (6366825,),
            None,
            None,
            None,
            cache,
        )


@with_defer
async def test_pr_list_miner_filter_labels_cache_exclude(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    cache,
    prefixer,
    bots,
):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    prs1, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter(set(), {"bug"}),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    await wait_deferred()
    assert len(prs1) == 71

    prs2, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    await wait_deferred()
    assert len(prs2) == 74

    prs3, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter(set(), {"bug"}),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    await wait_deferred()
    assert len(prs3) == 71


@with_defer
async def test_pr_list_miner_filter_labels_pdb(
    pr_facts_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
):
    pr_facts_calculator_no_cache = pr_facts_calculator_factory(1, (6366825,))
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    await pr_facts_calculator_no_cache(
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug", "enhancement"}, set()),
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
    prs, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug", "enhancement"}, set()),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    await wait_deferred()
    assert len(prs) == 6
    for pr in prs:
        labels = {label.name for label in pr.labels}
        assert labels.intersection({"bug", "enhancement"})
    prs, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug"}, set()),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert len(prs) == 3
    for pr in prs:
        assert "bug" in {label.name for label in (pr.labels or [])}


@pytest.mark.parametrize("with_precomputed", [True, False])
@with_defer
async def test_pr_list_miner_deployments_production(
    pr_facts_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
    precomputed_deployments,
    with_precomputed,
    detect_deployments,
):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2019, 12, 19, tzinfo=timezone.utc)
    if with_precomputed:
        pr_facts_calculator_no_cache = pr_facts_calculator_factory(1, (6366825,))
        await pr_facts_calculator_no_cache(
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
    prs, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert len(prs) == 581
    deps = 0
    merged_undeployed = 0
    deployed_margin = datetime(2019, 11, 1, 12, 15) - datetime(2015, 5, 2)
    undeployed_margin = (datetime.now(timezone.utc) - time_to) - timedelta(seconds=60)
    for pr in prs:
        if pr.deployments:
            deps += 1
            assert pr.stage_timings["deploy"]["production"] < deployed_margin
            assert pr.stage_timings["deploy"]["production"] > timedelta(0)
            assert PullRequestEvent.DEPLOYED in pr.events_now
            assert PullRequestEvent.DEPLOYED in pr.events_time_machine
            assert PullRequestStage.DEPLOYED in pr.stages_now
            assert PullRequestStage.DEPLOYED in pr.stages_time_machine
        else:
            if PullRequestEvent.MERGED in pr.events_now:
                merged_undeployed += 1
                assert pr.stage_timings["deploy"]["production"] > undeployed_margin
            else:
                assert "deploy" not in pr.stage_timings
            assert PullRequestEvent.DEPLOYED not in pr.events_now
            assert PullRequestEvent.DEPLOYED not in pr.events_time_machine
            assert PullRequestStage.DEPLOYED not in pr.stages_now
            assert PullRequestStage.DEPLOYED not in pr.stages_time_machine
    assert deps == 513
    assert merged_undeployed == 11


@with_defer
async def test_pr_list_miner_deployments_early(
    metrics_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
    precomputed_deployments,
    detect_deployments,
):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 12, 19, tzinfo=timezone.utc)
    prs, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        None,
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert len(prs) == 79
    deps = 0
    deployed_margin = datetime(2019, 11, 1, 12, 15) - datetime(2015, 5, 2)
    for pr in prs:
        if pr.deployments:
            deps += 1
            assert pr.stage_timings["deploy"]["production"] < deployed_margin
            assert pr.stage_timings["deploy"]["production"] > timedelta(0)
            assert PullRequestEvent.DEPLOYED in pr.events_now
            assert PullRequestEvent.DEPLOYED not in pr.events_time_machine
            assert PullRequestStage.DEPLOYED in pr.stages_now
            assert PullRequestStage.DEPLOYED not in pr.stages_time_machine
    assert deps == 54


@with_defer
async def test_pr_list_miner_deployments_staging(
    metrics_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
    precomputed_deployments,
    detect_deployments,
):
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2019, 12, 19, tzinfo=timezone.utc)
    prs, _ = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["staging"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    assert len(prs) == 581
    deps = 0
    for pr in prs:
        if pr.deployments:
            deps += 1
            assert "deploy" not in pr.stage_timings
        assert PullRequestEvent.DEPLOYED not in pr.events_now
        assert PullRequestEvent.DEPLOYED not in pr.events_time_machine
        assert PullRequestStage.DEPLOYED not in pr.stages_now
        assert PullRequestStage.DEPLOYED not in pr.stages_time_machine
    assert deps == 513


@pytest.mark.parametrize(
    "stage, n",
    [
        (
            PullRequestStage.WIP,
            {
                "": 3,
                "/alpha": 1,
                "/beta": 2,
            },
        ),
        (
            PullRequestStage.REVIEWING,
            {
                "": 4,
                "/alpha": 2,
                "/beta": 1,
            },
        ),
        (
            PullRequestStage.MERGING,
            {
                "": 2,
                "/alpha": 2,
                "/beta": 0,
            },
        ),
        (
            PullRequestStage.RELEASING,
            {
                "": 70,
                "/alpha": 37,
                "/beta": 27,
            },
        ),
        (
            PullRequestStage.DONE,
            {
                "": 290,
                "/alpha": 146,
                "/beta": 108,
            },
        ),
        (
            PullRequestStage.FORCE_PUSH_DROPPED,
            {
                "": 4,
                "/alpha": 0,
                "/beta": 1,
            },
        ),
    ],
)
@with_defer
async def test_pr_list_miner_logical(
    mdb,
    pdb,
    rdb,
    logical_settings,
    release_match_setting_tag_logical,
    time_from_to,
    stage,
    n,
    prefixer,
    bots,
):
    prs, _ = await filter_pull_requests(
        set(),
        {stage},
        *time_from_to,
        {"src-d/go-git", "src-d/go-git/alpha", "src-d/go-git/beta"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag_logical,
        logical_settings,
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    stats = {
        "": 0,
        "/alpha": 0,
        "/beta": 0,
    }
    for pr in prs:
        stats[pr.repository[len("src-d/go-git") :]] += 1
    assert stats == n


@with_defer
async def test_fetch_pull_requests_smoke(
    pr_facts_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
    cache,
):
    pr_facts_calculator_no_cache = pr_facts_calculator_factory(1, (6366825,))
    time_from = datetime(2018, 9, 1, tzinfo=timezone.utc)
    time_to = datetime(2018, 11, 19, tzinfo=timezone.utc)
    await pr_facts_calculator_no_cache(
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter({"bug", "enhancement"}, set()),
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
    prs1, deps = await filter_pull_requests(
        set(),
        set(),
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    await wait_deferred()
    assert deps == {}
    prs1 = {pr.number: pr for pr in prs1}
    # 921 is needed to check done_times
    prs2, deps2 = await fetch_pull_requests(
        {"src-d/go-git": set(list(range(1000, 1011)) + [921])},
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        cache,
    )
    await wait_deferred()
    prs3, deps3 = await fetch_pull_requests(
        {"src-d/go-git": set(list(range(1000, 1011)) + [921])},
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        prefixer,
        1,
        (6366825,),
        None,
        None,
        None,
        cache,
    )
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
            bots,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            None,
            prefixer,
            1,
            (6366825,),
            None,
            None,
            None,
            cache,
        )


class TestFetchPullRequests:
    @with_defer
    async def test_no_merged(
        self,
        mdb,
        pdb,
        rdb,
        release_match_setting_tag,
        prefixer,
        bots,
    ):
        prs, _ = await self._fetch(
            {"src-d/go-git": {1069}},
            bots,
            release_match_setting_tag,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
        )
        assert len(prs) == 1
        assert prs[0].number == 1069
        assert PullRequestStage.WIP in prs[0].stages_now
        assert PullRequestEvent.CREATED in prs[0].events_now
        assert PullRequestEvent.COMMITTED in prs[0].events_now
        assert prs[0].stages_time_machine is None
        assert prs[0].events_time_machine is None

    @with_defer
    async def test_empty(
        self,
        mdb,
        pdb,
        rdb,
        release_match_setting_tag,
        prefixer,
        bots,
    ):
        prs, deployments = await self._fetch(
            {"src-d/go-git": {0}},
            bots,
            release_match_setting_tag,
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
        )
        assert len(prs) == 0
        assert len(deployments) == 0

    @with_defer
    async def test_deployments(
        self,
        mdb,
        pdb,
        rdb,
        release_match_setting_tag,
        prefixer,
        bots,
        branches,
        default_branches,
    ):
        time_from = datetime(year=2019, month=6, day=1, tzinfo=timezone.utc)
        time_to = datetime(year=2019, month=12, day=1, tzinfo=timezone.utc)
        await mine_deployments(
            ["src-d/go-git"],
            {},
            time_from,
            time_to,
            ["production", "staging"],
            [],
            {},
            {},
            LabelFilter.empty(),
            JIRAFilter.empty(),
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            branches,
            default_branches,
            prefixer,
            1,
            None,
            (6366825,),
            mdb,
            pdb,
            rdb,
            None,
        )
        await wait_deferred()
        prs, deps = await self._fetch(
            {"src-d/go-git": {1160, 1168}},
            bots,
            release_match_setting_tag,
            environments=["production"],
            prefixer=prefixer,
            mdb=mdb,
            pdb=pdb,
            rdb=rdb,
        )
        check_pr_deployments(prs, deps, 1)

    async def test_many_repositories_batching(
        self,
        mdb_rw,
        sdb,
        pdb,
        rdb,
        bots,
    ):
        N_PRS = 23
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = []
            for i in range(1, N_PRS + 1):
                repo = md_factory.RepositoryFactory(node_id=100 + i, full_name=f"org0/{i}")
                await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
                models.extend(pr_models(100 + i, 100 + i, i, repository_full_name=f"org0/{i}"))
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            prefixer = await Prefixer.load((DEFAULT_MD_ACCOUNT_ID,), mdb_rw, None)
            settings = Settings.from_account(1, prefixer, sdb, mdb_rw, None, None)
            release_settings = await settings.list_release_matches()
            prs, _ = await self._fetch(
                {f"org0/{i}": {i} for i in range(1, N_PRS + 1)},
                bots,
                release_settings,
                prefixer=prefixer,
                mdb=mdb_rw,
                pdb=pdb,
                rdb=rdb,
            )
            assert sorted((pr.number, pr.repository) for pr in prs) == [
                (i, f"org0/{i}") for i in range(1, N_PRS + 1)
            ]

    async def _fetch(self, *args, **kwargs):
        kwargs.setdefault("account", DEFAULT_ACCOUNT_ID)
        kwargs.setdefault("meta_ids", (DEFAULT_MD_ACCOUNT_ID,))
        kwargs.setdefault("logical_settings", LogicalRepositorySettings.empty())
        kwargs.setdefault("cache", None)
        kwargs.setdefault("environments", None)
        return await fetch_pull_requests(*args, **kwargs)


@with_defer
async def test_pr_list_miner_filter_open_precomputed(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    args = [
        set(),
        {PullRequestStage.WIP, PullRequestStage.REVIEWING, PullRequestStage.MERGING},
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    ]
    prs1, _ = await filter_pull_requests(*args)
    await wait_deferred()
    assert len(prs1) == 21
    open_facts = await pdb.fetch_all(select([GitHubOpenPullRequestFacts]))
    assert len(open_facts) == 21

    # the following is offtopic but saves the precious execution time
    done_facts = await pdb.fetch_all(select([GitHubDonePullRequestFacts]))
    assert len(done_facts) == 294
    merged_facts = await pdb.fetch_all(select([GitHubMergedPullRequestFacts]))
    assert len(merged_facts) == 246
    # offtopic ends

    prs2, _ = await filter_pull_requests(*args)
    assert {pr.number for pr in prs1} == {pr.number for pr in prs2}
    assert {tuple(sorted(pr.stage_timings)) for pr in prs1} == {
        tuple(sorted(pr.stage_timings)) for pr in prs2
    }


@with_defer
async def test_pr_list_miner_filter_stages_events_aggregation(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    args = [
        {PullRequestEvent.REVIEWED},
        {PullRequestStage.WIP, PullRequestStage.REVIEWING, PullRequestStage.MERGING},
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    ]
    prs, _ = await filter_pull_requests(*args)
    assert len(prs) == 132


@with_defer
async def test_pr_list_miner_filter_updated_min_max(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    args = [
        {PullRequestEvent.REVIEWED},
        {PullRequestStage.WIP, PullRequestStage.REVIEWING, PullRequestStage.MERGING},
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        datetime(2018, 2, 1, tzinfo=timezone.utc),
        datetime(2019, 2, 1, tzinfo=timezone.utc),
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    ]
    prs, _ = await filter_pull_requests(*args)
    assert len(prs) == 72


@pytest.mark.parametrize("exclude_inactive", [False, True])
@with_defer
async def test_filter_pull_requests_deployments(
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
    branches,
    default_branches,
    exclude_inactive,
):
    time_from = datetime(year=2019, month=6, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2019, month=12, day=1, tzinfo=timezone.utc)
    args = [
        set(),
        {PullRequestStage.WIP, PullRequestStage.REVIEWING, PullRequestStage.MERGING},
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        ["production"],
        exclude_inactive,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        None,
        None,
        prefixer,
        1,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    ]
    # we have set `environment` but nothing has been deployed yet, will we break?
    prs, deps = await filter_pull_requests(*args)
    assert len(prs) == 8 if exclude_inactive else 15
    for pr in prs:
        assert pr.deployments is None
    await mine_deployments(
        ["src-d/go-git"],
        {},
        time_from,
        time_to,
        ["production", "staging"],
        [],
        {},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        branches,
        default_branches,
        prefixer,
        1,
        None,
        (6366825,),
        mdb,
        pdb,
        rdb,
        None,
    )
    await wait_deferred()
    prs, deps = await filter_pull_requests(*args)
    await wait_deferred()
    assert len(prs) == 8 if exclude_inactive else 15
    check_pr_deployments(prs, deps, 0)  # unmerged PR cannot be deployed!
    args[1] = {PullRequestStage.DONE}
    prs, deps = await filter_pull_requests(*args)
    assert len(prs) == 404
    check_pr_deployments(prs, deps, 394)


def check_pr_deployments(
    prs: List[PullRequestListItem],
    deps: Dict[str, Deployment],
    prs_must_have_deps: int,
) -> None:
    prs_have_deps = 0
    for pr in prs:
        if pr.deployments is not None:
            assert len(pr.deployments) == 1 and pr.deployments[0] == "Dummy deployment", pr
            prs_have_deps += 1
    assert prs_have_deps == prs_must_have_deps
    assert deps == {
        "Dummy deployment": Deployment(
            name="Dummy deployment",
            conclusion=DeploymentConclusion.SUCCESS,
            environment="production",
            url=None,
            started_at=pd.Timestamp(datetime(2019, 11, 1, 12, 0, tzinfo=timezone.utc)),
            finished_at=pd.Timestamp(datetime(2019, 11, 1, 12, 15, tzinfo=timezone.utc)),
            components=[
                DeployedComponent(
                    repository_full_name="src-d/go-git",
                    reference="v4.13.1",
                    sha="0d1a009cbb604db18be960db5f1525b99a55d727",
                ),
            ],
            labels=None,
        ),
    }
