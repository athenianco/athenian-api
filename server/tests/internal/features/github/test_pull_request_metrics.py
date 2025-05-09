from copy import deepcopy
from datetime import datetime, timedelta, timezone
from functools import partial
import itertools
from typing import Optional, Sequence, Tuple

import medvedi as md
from medvedi.testing import assert_frame_equal
import numpy as np
from numpy.testing import assert_array_equal
import pytest
from sqlalchemy import insert, select

from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.features.github.pull_request_metrics import (
    AllCounter,
    AverageReviewCommentsCalculator,
    ClosedCalculator,
    CycleCounter,
    CycleCounterWithQuantiles,
    CycleTimeBelowThresholdRatio,
    CycleTimeCalculator,
    DoneCalculator,
    FlowRatioCalculator,
    LiveCycleCounter,
    LiveCycleCounterWithQuantiles,
    LiveCycleTimeCalculator,
    MergingCounter,
    MergingCounterWithQuantiles,
    MergingTimeBelowThresholdRatio,
    MergingTimeCalculator,
    NotReviewedCalculator,
    OpenCounter,
    OpenCounterWithQuantiles,
    OpenedCalculator,
    OpenTimeBelowThresholdRatio,
    OpenTimeCalculator,
    PullRequestBinnedMetricCalculator,
    PullRequestMetricCalculatorEnsemble,
    ReleaseCounter,
    ReleaseCounterWithQuantiles,
    ReleaseTimeCalculator,
    ReviewCommentsAboveThresholdRatio,
    ReviewCounter,
    ReviewCounterWithQuantiles,
    ReviewedCalculator,
    ReviewedRatioCalculator,
    ReviewTimeBelowThresholdRatio,
    ReviewTimeCalculator,
    SizeBelowThresholdRatio,
    SizeCalculator,
    WaitFirstReviewTimeBelowThresholdRatio,
    WaitFirstReviewTimeCalculator,
    WorkInProgressCounter,
    WorkInProgressCounterWithQuantiles,
    WorkInProgressTimeBelowThresholdRatio,
    WorkInProgressTimeCalculator,
    _ReviewedPlusNotReviewedCalculator,
    group_prs_by_participants,
    histogram_calculators,
    register_metric,
)
from athenian.api.internal.features.histogram import Scale
from athenian.api.internal.features.metric import MetricInt, MetricTimeDelta
from athenian.api.internal.features.metric_calculator import (
    MetricCalculator,
    MetricCalculatorEnsemble,
)
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.deployment import mine_deployments
from athenian.api.internal.miners.participation import PRParticipationKind
from athenian.api.internal.miners.types import JIRAEntityToFetch, PullRequestFacts
from athenian.api.internal.settings import (
    LogicalDeploymentSettings,
    LogicalRepositorySettings,
    ReleaseMatch,
    ReleaseMatchSetting,
    ReleaseSettings,
)
from athenian.api.models.persistentdata.models import ReleaseNotification
from athenian.api.models.precomputed.models import (
    GitHubMergedPullRequestFacts,
    GitHubOpenPullRequestFacts,
)
from athenian.api.models.web import Granularity, PullRequestMetricID
from athenian.api.typing_utils import df_from_structs
from tests.conftest import generate_pr_samples, has_memcached
from tests.testutils.factory.miners import PullRequestFactsFactory
from tests.testutils.time import dt, dt64arr_s, dt64arr_us


def random_dropout(pr, prob):
    fields = sorted(PullRequestFacts.dtype.fields.items())
    fields = [f for f, dt in fields if np.issubdtype(dt, np.datetime64)]
    killed = np.random.choice(fields, int(len(fields) * prob), replace=False).tolist()
    if "closed" in killed and "merged" not in killed:
        killed.append("merged")
    if "created" in killed:
        # "created" must always exist
        killed.remove("created")
    if "work_began" in killed:
        # dependent property
        killed.remove("work_began")
    if "first_review_request" in killed and "first_review_request_exact" not in killed:
        killed.append("first_review_request_exact")
        killed.append("last_review")
    kwargs = dict(pr)
    for k in killed:
        kwargs[k] = None
    if "first_commit" in killed:
        kwargs["work_began"] = kwargs["created"]
    if "released" in killed or "closed" in killed:
        kwargs["done"] = (
            kwargs["released"]
            or pr.force_push_dropped
            or (kwargs["closed"] and not kwargs["merged"])
        )
    return PullRequestFacts.from_fields(**kwargs)


@pytest.mark.parametrize(
    "cls",
    [
        WorkInProgressTimeCalculator,
        ReviewTimeCalculator,
        MergingTimeCalculator,
        ReleaseTimeCalculator,
        CycleTimeCalculator,
        WaitFirstReviewTimeCalculator,
    ],
)
def test_pull_request_metrics_2d(pr_samples, cls):  # noqa: F811
    calc = cls(quantiles=(0, 1))
    time_froms = np.array(
        [datetime.utcnow() - timedelta(days=i * 200) for i in range(1, 3)], dtype="datetime64[us]",
    )
    time_tos = np.array(
        [datetime.utcnow(), datetime.utcnow() - timedelta(days=100)], dtype="datetime64[us]",
    )
    prs = df_from_structs(random_dropout(pr, 0.5) for pr in pr_samples(1000))
    r = calc._analyze(prs, time_froms, time_tos)
    assert (r[0, r[0] == r[0]] >= np.array(0, dtype=r.dtype)).any()
    assert (r[1, r[1] == r[1]] >= np.array(0, dtype=r.dtype)).any()


@pytest.mark.parametrize(
    "cls",
    [
        WorkInProgressTimeCalculator,
        ReviewTimeCalculator,
        MergingTimeCalculator,
        ReleaseTimeCalculator,
        CycleTimeCalculator,
        WaitFirstReviewTimeCalculator,
    ],
)
def test_pull_request_metrics_timedelta_stability(pr_samples, cls):  # noqa: F811
    calc = cls(quantiles=(0, 1))
    time_from = datetime.utcnow() - timedelta(days=10000)
    time_to = datetime.utcnow()
    prs = df_from_structs(random_dropout(pr, 0.5) for pr in pr_samples(1000))
    r = calc._analyze(prs, dt64arr_us(time_from), dt64arr_us(time_to))
    assert (r[~np.isnat(r)] >= np.array(0, dtype=r.dtype)).all()


def test_pull_request_metrics_empty_input(pr_samples):
    calc = WorkInProgressTimeCalculator(quantiles=(0, 1))
    df = df_from_structs(pr_samples(1)).iloc[:0]
    time_to = datetime.utcnow()
    time_from = time_to - timedelta(days=180)
    calc(df, dt64arr_us(time_from), dt64arr_us(time_to), None, np.full((1, len(df)), True))
    assert len(calc.values) == 1
    assert len(calc.values[0]) == 1
    assert not calc.values[0][0].exists


@pytest.mark.parametrize("fill_val", [False, True])
def test_pull_request_metrics_empty_group(pr_samples, fill_val):
    calc = WorkInProgressTimeCalculator(quantiles=(0, 0.9))
    df = df_from_structs(pr_samples(100))
    time_to = datetime.utcnow()
    time_from = time_to - timedelta(days=180)
    time_from = np.concatenate([dt64arr_us(time_from)] * 2)
    time_to = np.concatenate([dt64arr_us(time_to)] * 2)
    calc(df, time_from, time_to, 1, np.full((1, len(df)), fill_val))
    assert len(calc.values) == 1
    assert len(calc.values[0]) == 1
    assert calc.values[0][0].exists == fill_val


@pytest.mark.parametrize(
    "cls, peak_attr",
    [
        (WorkInProgressTimeCalculator, "first_review_request"),
        (ReviewTimeCalculator, "approved,last_review"),
        (MergingTimeCalculator, "closed"),
        (ReleaseTimeCalculator, "released"),
        (CycleTimeCalculator, "released"),
        (WaitFirstReviewTimeCalculator, "first_comment_on_first_review"),
    ],
)
def test_pull_request_metrics_out_of_bounds(pr_samples, cls, peak_attr):  # noqa: F811
    calc = cls(quantiles=(0, 1))
    for pr in pr_samples(100):
        time_from = datetime.utcnow() - timedelta(days=10000)
        for attr in peak_attr.split(","):
            time_from = max(getattr(pr, attr), dt64arr_s(time_from)).item()
        time_from += timedelta(days=1)
        time_to = time_from + timedelta(days=7)
        assert calc._analyze(
            df_from_structs([pr]), dt64arr_us(time_from), dt64arr_us(time_to),
        ) == np.array([None])

        time_from = datetime.utcnow()
        for attr in peak_attr.split(","):
            time_from = min(getattr(pr, attr), dt64arr_s(time_from)).item()
        time_from -= timedelta(days=7)
        time_to = time_from + timedelta(days=1)
        assert calc._analyze(
            df_from_structs([pr]), dt64arr_us(time_from), dt64arr_us(time_to),
        ) == np.array([None])


@pytest.mark.parametrize(
    "metric",
    [
        PullRequestMetricID.PR_OPENED,
        PullRequestMetricID.PR_MERGED,
        PullRequestMetricID.PR_REJECTED,
        PullRequestMetricID.PR_CLOSED,
    ],
)
def test_pull_request_metrics_float_binned(pr_samples, metric):  # noqa: F811
    time_from = (datetime.now(tz=timezone.utc) - timedelta(days=365 * 3 // 2)).date()
    time_to = (datetime.now(tz=timezone.utc) - timedelta(days=365 // 2)).date()
    time_intervals = [
        [
            datetime.combine(i, datetime.min.time(), tzinfo=timezone.utc)
            for i in Granularity.split("month", time_from, time_to)
        ],
    ]
    binned = PullRequestBinnedMetricCalculator([metric], quantiles=(0, 1), quantile_stride=0)
    samples = pr_samples(1000)
    if metric == PullRequestMetricID.PR_REJECTED:
        for i, s in enumerate(samples):
            data = dict(s)
            data["merged"] = None
            samples[i] = PullRequestFacts.from_fields(**data)
    result = binned(df_from_structs(samples), time_intervals, np.array([np.arange(len(samples))]))
    # the last interval is null and that's intended
    for i, m in enumerate(result[0][0][:-1]):
        assert m[0].exists, str(i)
        assert m[0].value > 1, str(i)
        assert m[0].confidence_min is None, str(i)
        assert m[0].confidence_max is None, str(i)


def test_pull_request_opened_no(pr_samples):  # noqa: F811
    calc = OpenedCalculator(quantiles=(0, 1))
    time_to = datetime.utcnow()
    time_from = time_to - timedelta(days=180)
    prs = df_from_structs(pr for pr in pr_samples(100) if pr.closed and pr.closed < time_to)
    calc(prs, dt64arr_us(time_from), dt64arr_us(time_to), None, np.full((1, len(prs)), True))
    assert len(prs) > 0
    m = calc.values[0][0]
    assert m.exists
    assert m.value == 0


def test_pull_request_closed_no(pr_samples):  # noqa: F811
    calc_closed = ClosedCalculator(quantiles=(0, 1))
    calc_released = DoneCalculator(quantiles=(0, 1))
    time_from = datetime.utcnow() - timedelta(days=365 * 3)
    time_to = time_from + timedelta(days=7)
    prs = df_from_structs(pr_samples(100))
    args = prs, dt64arr_us(time_from), dt64arr_us(time_to), None, np.full((1, len(prs)), True)
    calc_closed(*args)
    calc_released(*args)
    assert calc_closed.values[0][0].exists
    assert calc_closed.values[0][0].value == 0
    assert calc_released.values[0][0].exists
    assert calc_released.values[0][0].value == 0


def test_pull_request_flow_ratio(pr_samples):  # noqa: F811
    calc = FlowRatioCalculator(
        OpenedCalculator(quantiles=(0, 1)), ClosedCalculator(quantiles=(0, 1)), quantiles=(0, 1),
    )
    open_calc = OpenedCalculator(quantiles=(0, 1))
    closed_calc = ClosedCalculator(quantiles=(0, 1))
    time_from = datetime.utcnow() - timedelta(days=365)
    time_to = datetime.utcnow()
    prs = df_from_structs(pr_samples(1000))
    args = prs, dt64arr_us(time_from), dt64arr_us(time_to), None, np.full((1, len(prs)), True)
    for dep in calc._calcs:
        dep(*args)
    calc(*args)
    open_calc(*args)
    closed_calc(*args)
    m = calc.values[0][0]
    assert m.exists
    assert 0 < m.value < 1
    assert m.confidence_min is None
    assert m.confidence_max is None
    assert m.value == np.float32(
        (open_calc.values[0][0].value + 1) / (closed_calc.values[0][0].value + 1),
    )


def test_pull_request_flow_ratio_zeros(pr_samples):
    calc = FlowRatioCalculator(
        OpenedCalculator(quantiles=(0, 1)), ClosedCalculator(quantiles=(0, 1)), quantiles=(0, 1),
    )
    calc._representative_time_interval_indexes = calc._calcs[
        0
    ]._representative_time_interval_indexes = calc._calcs[
        1
    ]._representative_time_interval_indexes = [0]
    assert len(calc.values) == 0


def test_pull_request_flow_ratio_no_opened(pr_samples):  # noqa: F811
    calc = FlowRatioCalculator(
        OpenedCalculator(quantiles=(0, 1)), ClosedCalculator(quantiles=(0, 1)), quantiles=(0, 1),
    )
    time_to = datetime.utcnow()
    time_from = time_to - timedelta(days=180)
    for pr in pr_samples(100):
        if pr.closed and time_from <= pr.closed < time_to:
            df = df_from_structs([pr])
            args = (
                df,
                dt64arr_us(time_from),
                dt64arr_us(time_to),
                None,
                np.full((1, len(df)), True),
            )
            for dep in calc._calcs:
                dep(*args)
            calc(*args)
            break
    m = calc.values[0][0]
    assert m.exists
    assert m.value == 0.5


def test_pull_request_flow_ratio_no_closed(pr_samples):  # noqa: F811
    calc = FlowRatioCalculator(
        OpenedCalculator(quantiles=(0, 1)), ClosedCalculator(quantiles=(0, 1)), quantiles=(0, 1),
    )
    time_to = datetime.utcnow() - timedelta(days=180)
    time_from = time_to - timedelta(days=180)
    for pr in pr_samples(100):
        if pr.closed and pr.closed > time_to > pr.created >= time_from:
            args = (
                df_from_structs([pr]),
                dt64arr_us(time_from),
                dt64arr_us(time_to),
                None,
                np.array([[True]]),
            )
            for dep in calc._calcs:
                dep(*args)
            calc(*args)
            break
    m = calc.values[0][0]
    assert m.exists
    assert m.value == 2


@pytest.mark.parametrize(
    "cls",
    [
        WorkInProgressCounter,
        ReviewCounter,
        MergingCounter,
        ReleaseCounter,
        OpenCounter,
        CycleCounter,
        LiveCycleCounter,
        AllCounter,
    ],
)
def test_pull_request_metrics_counts_nq(pr_samples, cls):  # noqa: F811
    calc = cls(
        *(
            dep1(*(dep2(quantiles=(0, 1)) for dep2 in dep1.deps), quantiles=(0, 1))
            for dep1 in cls.deps
        ),
        quantiles=(0, 1),
    )
    prs = df_from_structs(pr_samples(1000))
    time_tos = np.full(2, datetime.utcnow(), "datetime64[us]")
    time_froms = time_tos - np.timedelta64(timedelta(days=10000))
    args = prs, time_froms, time_tos, None, np.full((1, len(prs)), True)
    for dep1 in calc._calcs:
        for dep2 in dep1._calcs:
            dep2(*args)
        dep1(*args)
    calc(*args)
    delta = calc.peek
    assert isinstance(delta, np.ndarray)
    assert delta.shape == (2, 1000)
    assert (delta[0][delta[0] == delta[0]] == delta[1][delta[1] == delta[1]]).all()
    if cls != AllCounter:
        peek = calc._calcs[0].peek
    else:
        peek = calc.peek
    assert (peek[0][peek[0] == peek[0]] == peek[1][peek[1] == peek[1]]).all()
    if cls != AllCounter:
        nonones = (peek == peek).sum()
    else:
        nonones = (peek != 0).sum()
    nones = (peek.shape[0] * peek.shape[1]) - nonones
    if cls not in (WorkInProgressCounter, LiveCycleCounter, AllCounter):
        assert nones > 0, cls
    assert nonones > 0, cls


@pytest.mark.flaky(reruns=3)
@pytest.mark.parametrize(
    "cls_q, cls",
    [
        (WorkInProgressCounterWithQuantiles, WorkInProgressCounter),
        (ReviewCounterWithQuantiles, ReviewCounter),
        (MergingCounterWithQuantiles, MergingCounter),
        (ReleaseCounterWithQuantiles, ReleaseCounter),
        (OpenCounterWithQuantiles, OpenCounter),
        (CycleCounterWithQuantiles, CycleCounter),
        (LiveCycleCounterWithQuantiles, LiveCycleCounter),
    ],
)
def test_pull_request_metrics_counts_q(pr_samples, cls_q, cls):  # noqa: F811
    calc_q = cls_q(
        *(
            dep1(*(dep2(quantiles=(0, 0.95)) for dep2 in dep1.deps), quantiles=(0, 0.95))
            for dep1 in cls_q.deps
        ),
        quantiles=(0, 0.95),
    )
    calc = cls(*calc_q._calcs, quantiles=(0, 0.95))
    prs = df_from_structs(pr_samples(1000))
    time_to = np.concatenate([dt64arr_us(datetime.utcnow())] * 2)
    time_from = time_to - np.array([timedelta(days=10000)], dtype="timedelta64")
    args = prs, time_from, time_to, 1, np.full((1, len(prs)), True)
    for dep1 in calc._calcs:
        for dep2 in dep1._calcs:
            dep2(*args)
        dep1(*args)
    calc_q(*args)
    calc(*args)
    assert 0 < calc_q.values[0][0].value < calc.values[0][0].value


@pytest.mark.parametrize(
    "with_memcached, with_mine_cache_wipe",
    itertools.product(*([[False, True]] * 2)),
)
@with_defer
async def test_calc_pull_request_metrics_line_github_cache_reset(
    metrics_calculator_factory,
    branches,
    default_branches,
    mdb,
    pdb,
    rdb,
    cache,
    memcached,
    with_memcached,
    metrics_calculator_factory_memcached,
    release_match_setting_tag,
    with_mine_cache_wipe,
    pr_miner,
    prefixer,
    bots,
):
    if with_memcached:
        if not has_memcached:
            raise pytest.skip("no memcached")
        cache = memcached

    if with_memcached:
        factory = metrics_calculator_factory_memcached
    else:
        factory = metrics_calculator_factory
    pr_facts_calculator = factory(1, (6366825,), with_cache=True)
    date_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2019, month=10, day=1, tzinfo=timezone.utc)
    args = (
        [PullRequestMetricID.PR_CYCLE_TIME],
        [[date_from, date_to]],
        [0, 1],
        [],
        [],
        [{"src-d/go-git"}],
        [{}],
        LabelFilter.empty(),
        [JIRAFilter.empty()],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        branches,
        default_branches,
        False,
    )
    metrics1 = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][
        0
    ][0][0][0]
    await wait_deferred()
    assert await pr_facts_calculator.calc_pull_request_metrics_line_github.reset_cache(*args)
    if with_mine_cache_wipe:
        assert await pr_miner._mine.reset_cache(
            None,
            date_from,
            date_to,
            {"src-d/go-git"},
            {},
            LabelFilter.empty(),
            [JIRAFilter.empty()],
            False,
            branches,
            default_branches,
            False,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            None,
            None,
            None,
            True,
            prefixer,
            1,
            (6366825,),
            mdb,
            pdb,
            rdb,
            cache,
        )
    metrics2 = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][
        0
    ][0][0][0]
    assert metrics1.exists and metrics2.exists
    assert metrics1.value == metrics2.value
    assert metrics1.confidence_score() == metrics2.confidence_score()
    assert metrics1.confidence_min < metrics1.value < metrics1.confidence_max


@with_defer
async def test_calc_pull_request_metrics_line_github_cache_lines(
    metrics_calculator_factory,
    release_match_setting_tag,
    prefixer,
    bots,
    branches,
    default_branches,
):
    pr_facts_calculator = metrics_calculator_factory(1, (6366825,), with_cache=True)
    date_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2019, month=10, day=1, tzinfo=timezone.utc)
    args = [
        [PullRequestMetricID.PR_CYCLE_TIME],
        [[date_from, date_to]],
        [0, 1],
        [0, 1000],
        [],
        [{"src-d/go-git"}],
        [{}],
        LabelFilter.empty(),
        [JIRAFilter.empty()],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        branches,
        default_branches,
        False,
    ]
    metrics1 = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][
        0
    ][0][0]
    await wait_deferred()
    args[3] = []
    metrics2 = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][
        0
    ][0][0]
    assert metrics1 != metrics2


@with_defer
async def test_calc_pull_request_metrics_line_github_changed_releases(
    metrics_calculator_factory,
    mdb,
    pdb,
    rdb,
    cache,
    release_match_setting_tag,
    prefixer,
    bots,
    branches,
    default_branches,
):
    pr_facts_calculator = metrics_calculator_factory(1, (6366825,), with_cache=True)
    date_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2017, month=10, day=1, tzinfo=timezone.utc)
    args = [
        [PullRequestMetricID.PR_CYCLE_TIME],
        [[date_from, date_to]],
        [0, 1],
        [],
        [],
        [{"src-d/go-git"}],
        [{}],
        LabelFilter.empty(),
        [JIRAFilter.empty()],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        branches,
        default_branches,
        False,
    ]
    metrics1 = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][
        0
    ][0][0]
    await wait_deferred()
    release_match_setting_tag = ReleaseSettings(
        {
            "github.com/src-d/go-git": ReleaseMatchSetting(
                "master", ".*", ".*", ReleaseMatch.branch,
            ),
        },
    )
    args[-6] = release_match_setting_tag
    metrics2 = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][
        0
    ][0][0]
    assert metrics1 != metrics2


@with_defer
async def test_pr_list_miner_match_metrics_all_count_david_bug(
    metrics_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
    branches,
    default_branches,
):
    metrics_calculator_no_cache = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(year=2016, month=11, day=17, tzinfo=timezone.utc)
    time_middle = time_from + timedelta(days=14)
    time_to = datetime(year=2016, month=12, day=15, tzinfo=timezone.utc)
    metric1 = (
        await metrics_calculator_no_cache.calc_pull_request_metrics_line_github(
            [PullRequestMetricID.PR_ALL_COUNT],
            [[time_from, time_middle]],
            [0, 1],
            [],
            [],
            [{"src-d/go-git"}],
            [{}],
            LabelFilter.empty(),
            [JIRAFilter.empty()],
            False,
            bots,
            release_match_setting_tag,
            LogicalRepositorySettings.empty(),
            prefixer,
            branches,
            default_branches,
            False,
        )
    )[0][0][0][0][0][0][0].value
    await wait_deferred()
    metric2 = (
        await metrics_calculator_no_cache.calc_pull_request_metrics_line_github(
            [PullRequestMetricID.PR_ALL_COUNT],
            [[time_middle, time_to]],
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
    )[0][0][0][0][0][0][0].value
    await wait_deferred()
    metric1_ext, metric2_ext = (
        m[0].value
        for m in (
            await metrics_calculator_no_cache.calc_pull_request_metrics_line_github(
                [PullRequestMetricID.PR_ALL_COUNT],
                [[time_from, time_middle, time_to]],
                [0, 1],
                [],
                [],
                [{"src-d/go-git"}],
                [{}],
                LabelFilter.empty(),
                [JIRAFilter.empty()],
                False,
                bots,
                release_match_setting_tag,
                LogicalRepositorySettings.empty(),
                prefixer,
                branches,
                default_branches,
                False,
            )
        )[0][0][0][0][0]
    )
    assert metric1 == metric1_ext
    assert metric2 == metric2_ext


@with_defer
async def test_calc_pull_request_metrics_line_github_exclude_inactive(
    metrics_calculator_factory,
    release_match_setting_tag,
    prefixer,
    bots,
    branches,
    default_branches,
):
    pr_facts_calculator = metrics_calculator_factory(1, (6366825,), with_cache=True)
    date_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2017, month=1, day=12, tzinfo=timezone.utc)
    args = [
        [PullRequestMetricID.PR_ALL_COUNT],
        [[date_from, date_to]],
        [0, 1],
        [],
        [],
        [{"src-d/go-git"}],
        [{}],
        LabelFilter.empty(),
        [JIRAFilter.empty()],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        branches,
        default_branches,
        False,
    ]
    metrics = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][0][
        0
    ][0][0]
    await wait_deferred()
    assert metrics.value == 7
    args[9] = True
    metrics = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][0][
        0
    ][0][0]
    await wait_deferred()
    assert metrics.value == 6
    date_from = datetime(year=2017, month=5, day=23, tzinfo=timezone.utc)
    date_to = datetime(year=2017, month=5, day=25, tzinfo=timezone.utc)
    args[0] = [PullRequestMetricID.PR_RELEASE_COUNT]
    args[1] = [[date_from, date_to]]
    args[9] = False
    metrics = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][0][
        0
    ][0][0]
    await wait_deferred()
    assert metrics.value == 71
    metrics = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][0][
        0
    ][0][0]
    await wait_deferred()
    assert metrics.value == 71
    args[9] = True
    metrics = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][0][
        0
    ][0][0]
    assert metrics.value == 71


@with_defer
async def test_calc_pull_request_metrics_line_github_quantiles(
    metrics_calculator_factory,
    release_match_setting_tag,
    prefixer,
    bots,
    branches,
    default_branches,
):
    pr_facts_calculator = metrics_calculator_factory(1, (6366825,), with_cache=True)
    date_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2017, month=1, day=12, tzinfo=timezone.utc)
    args = [
        [PullRequestMetricID.PR_ALL_COUNT],
        [[date_from, date_to]],
        [0, 0.95],
        [],
        [],
        [{"src-d/go-git"}],
        [{}],
        LabelFilter.empty(),
        [JIRAFilter.empty()],
        False,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        branches,
        default_branches,
        False,
    ]
    metrics = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][0][
        0
    ][0][0]
    await wait_deferred()
    assert metrics.value == 25
    args[2] = [0, 1]
    metrics = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][0][
        0
    ][0][0]
    await wait_deferred()
    assert metrics.value == 25
    # yes, see _fetch_inactive_merged_unreleased_prs


@with_defer
async def test_calc_pull_request_metrics_line_github_tag_after_branch(
    metrics_calculator_factory,
    mdb,
    pdb,
    rdb,
    cache,
    prefixer,
    bots,
    release_match_setting_branch,
    release_match_setting_tag_or_branch,
    branches,
    default_branches,
):
    pr_facts_calculator = metrics_calculator_factory(1, (6366825,), with_cache=True)
    date_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2018, month=1, day=12, tzinfo=timezone.utc)
    args = [
        [PullRequestMetricID.PR_RELEASE_TIME],
        [[date_from, date_to]],
        [0, 1],
        [],
        [],
        [{"src-d/go-git"}],
        [{}],
        LabelFilter.empty(),
        [JIRAFilter.empty()],
        False,
        bots,
        release_match_setting_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        branches,
        default_branches,
        False,
    ]
    res = await pr_facts_calculator.calc_pull_request_metrics_line_github(*args)
    metrics = res[0][0][0][0][0][0][0]
    await wait_deferred()
    assert metrics.value == timedelta(seconds=0)
    args[-6] = release_match_setting_tag_or_branch
    res = await pr_facts_calculator.calc_pull_request_metrics_line_github(*args)
    metrics = res[0][0][0][0][0][0][0]
    assert metrics.value == timedelta(days=40, seconds=72739)


@with_defer
async def test_calc_pull_request_metrics_line_github_deployment_hazard(
    metrics_calculator_factory,
    mdb,
    pdb,
    rdb,
    cache,
    prefixer,
    bots,
    release_match_setting_branch,
    precomputed_deployments,
    branches,
    default_branches,
):
    pr_facts_calculator = metrics_calculator_factory(1, (6366825,), with_cache=True)
    date_from = datetime(year=2019, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2020, month=1, day=12, tzinfo=timezone.utc)
    args = [
        [PullRequestMetricID.PR_RELEASE_TIME],
        [[date_from, date_to]],
        [0, 1],
        [],
        [],
        [{"src-d/go-git"}],
        [{}],
        LabelFilter.empty(),
        [JIRAFilter.empty()],
        False,
        bots,
        release_match_setting_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        branches,
        default_branches,
        False,
    ]
    metrics = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][0][
        0
    ][0][0]
    await wait_deferred()
    assert metrics.value == timedelta(seconds=0)  # 396 days without loading deployed releases


@with_defer
async def test_calc_pull_request_metrics_line_jira_map(
    metrics_calculator_factory,
    mdb,
    pdb,
    rdb,
    cache,
    release_match_setting_tag_or_branch,
    prefixer,
    bots,
    branches,
    default_branches,
):
    pr_facts_calculator = metrics_calculator_factory(1, (6366825,), with_cache=True)
    date_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2018, month=1, day=12, tzinfo=timezone.utc)
    metrics = [
        PullRequestMetricID.PR_OPENED_MAPPED_TO_JIRA,
        PullRequestMetricID.PR_DONE_MAPPED_TO_JIRA,
        PullRequestMetricID.PR_ALL_MAPPED_TO_JIRA,
    ]
    args = [
        metrics,
        [[date_from, date_to]],
        [0, 1],
        [],
        [],
        [{"src-d/go-git"}],
        [{}],
        LabelFilter.empty(),
        [JIRAFilter.empty()],
        False,
        bots,
        release_match_setting_tag_or_branch,
        LogicalRepositorySettings.empty(),
        prefixer,
        branches,
        default_branches,
        False,
    ]
    metrics = (await pr_facts_calculator.calc_pull_request_metrics_line_github(*args))[0][0][0][0][
        0
    ][0]
    await wait_deferred()
    assert metrics[0].value == 0.021739130839705467
    assert metrics[1].value == 0.00800000037997961
    assert metrics[2].value == 0.02150537632405758


@with_defer
async def test_calc_pull_request_metrics_deep_filters(
    metrics_calculator_factory,
    mdb,
    pdb,
    rdb,
    cache,
    release_match_setting_tag_or_branch,
    prefixer,
    bots,
    branches,
    default_branches,
):
    pr_facts_calculator = metrics_calculator_factory(1, (6366825,), with_cache=True)
    release_settings = release_match_setting_tag_or_branch.copy()
    for r in ("gitbase", "hercules"):
        release_settings.native["src-d/" + r] = release_settings.prefixed[
            "github.com/src-d/" + r
        ] = release_settings.native["src-d/go-git"]
    date_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2018, month=1, day=12, tzinfo=timezone.utc)
    metrics = [
        PullRequestMetricID.PR_OPENED,
        PullRequestMetricID.PR_CLOSED,
        PullRequestMetricID.PR_MERGED,
    ]
    args = [
        metrics,
        [[date_from, date_to], [date_from, date_from + (date_to - date_from) / 2, date_to]],
        [0, 1],
        [0, 50, 10000],
        [],
        [{"src-d/go-git"}, {"src-d/gitbase"}, {"src-d/hercules"}],
        {},
        LabelFilter.empty(),
        [JIRAFilter.empty()],
        False,
        bots,
        release_settings,
        LogicalRepositorySettings.empty(),
        prefixer,
        branches,
        default_branches,
        False,
    ]
    # 1. jira: 1 group
    # 2. line: 2 groups
    # 3. repository: 3 groups
    # 4. participants: 1 group
    # 5. time series primary: 2 groups
    # 6. time series secondary: 1 and 2 groups
    # 7. metrics: 3 groups
    metrics = await pr_facts_calculator.calc_pull_request_metrics_line_github(*args)
    metric = partial(MetricInt.from_fields, exists=True, confidence_min=None, confidence_max=None)

    ground_truth = np.array(
        [
            [  # jira group 1
                [  # line group 1
                    [  # repository group 1
                        [  # participants group 1
                            [  # time series primary 1
                                [metric(value=134), metric(value=131), metric(value=110)],
                            ],
                            [  # time series primary 2
                                [metric(value=65), metric(value=62), metric(value=54)],
                                [metric(value=69), metric(value=69), metric(value=56)],
                            ],
                        ],
                    ],
                    # repository group 2 and 3
                    *[
                        [  # repository group
                            [  # participants group 1
                                [  # time series primary 1
                                    [metric(value=0)] * 3,
                                ],
                                [  # time series primary 2
                                    [metric(value=0)] * 3,
                                ]
                                * 2,
                            ],
                        ],
                    ]
                    * 2,
                ],
                [  # line group 2
                    [  # repository group 1
                        [  # participants group 1
                            [  # time series primary 1
                                [metric(value=142), metric(value=142), metric(value=130)],
                            ],
                            [  # time series primary 2
                                [metric(value=69), metric(value=70), metric(value=64)],
                                [metric(value=73), metric(value=72), metric(value=66)],
                            ],
                        ],
                    ],
                    # repository group 2 and 3
                    *[
                        [  # repository group
                            [  # participants group 1
                                [  # time series primary 1
                                    [metric(value=0)] * 3,
                                ],
                                [  # time series primary 2
                                    [metric(value=0)] * 3,
                                ]
                                * 2,
                            ],
                        ],
                    ]
                    * 2,
                ],
            ],
        ],
        dtype=object,
    )
    np.testing.assert_array_equal(np.array(metrics.tolist(), dtype=object), ground_truth)


def test_pull_request_metric_calculator_ensemble_accuracy(pr_samples):
    qargs = {"quantiles": (0, 1)}
    ensemble = PullRequestMetricCalculatorEnsemble(
        PullRequestMetricID.PR_LIVE_CYCLE_TIME,
        PullRequestMetricID.PR_WIP_COUNT,
        PullRequestMetricID.PR_RELEASE_TIME,
        PullRequestMetricID.PR_CLOSED,
        quantile_stride=0,
        **qargs,
    )
    release_time = ReleaseTimeCalculator(**qargs)
    wip_count = WorkInProgressCounter(WorkInProgressTimeCalculator(**qargs), **qargs)
    cycle_time = LiveCycleTimeCalculator(
        WorkInProgressTimeCalculator(**qargs),
        ReviewTimeCalculator(**qargs),
        MergingTimeCalculator(**qargs),
        ReleaseTimeCalculator(**qargs),
        **qargs,
    )
    closed = ClosedCalculator(**qargs)
    time_from = datetime.utcnow() - timedelta(days=365)
    time_to = datetime.utcnow()
    for _ in range(2):
        prs = df_from_structs(pr_samples(100))
        args = [
            prs,
            dt64arr_us(time_from),
            dt64arr_us(time_to),
            None,
            np.full((1, len(prs)), True),
        ]
        ensemble(*args[:-2], args[-1])
        release_time(*args)
        wip_count._calcs[0](*args)
        wip_count(*args)
        for c in cycle_time._calcs:
            c(*args)
        cycle_time(*args)
        closed(*args)
        ensemble_metrics = ensemble.values()
        assert ensemble_metrics[PullRequestMetricID.PR_LIVE_CYCLE_TIME] == cycle_time.values
        assert ensemble_metrics[PullRequestMetricID.PR_RELEASE_TIME] == release_time.values
        assert ensemble_metrics[PullRequestMetricID.PR_WIP_COUNT] == wip_count.values
        assert ensemble_metrics[PullRequestMetricID.PR_CLOSED] == closed.values


def test_pull_request_metric_calculator_empty_facts(pr_samples):
    binned = PullRequestBinnedMetricCalculator(
        [PullRequestMetricID.PR_WIP_COUNT], quantiles=(0, 0.9), quantile_stride=210,
    )
    prs = df_from_structs(pr_samples(1)).iloc[:0]
    time_to = datetime.now(timezone.utc)
    time_from = time_to - timedelta(days=365)
    groups = np.full((1, 1, 1), None, dtype=object)
    groups.fill(np.empty(0, int))
    metrics = binned(prs, [[time_from, time_to]], groups)
    assert metrics[0][0][0][0][0][0].value == 0


def test_pull_request_metric_calculator_ensemble_empty(pr_samples):
    ensemble = PullRequestMetricCalculatorEnsemble(quantiles=(0, 1), quantile_stride=73)
    time_from = datetime.utcnow() - timedelta(days=365)
    time_to = datetime.utcnow()
    ensemble(
        df_from_structs(pr_samples(1)), dt64arr_us(time_from), dt64arr_us(time_to), [np.arange(1)],
    )
    assert ensemble.values() == {}


@with_defer
async def test_pr_facts_calculator_open_precomputed(
    pr_facts_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
):
    pr_facts_calculator_no_cache = pr_facts_calculator_factory(1, (6366825,))
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    args = (
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
    facts1 = await pr_facts_calculator_no_cache(*args)
    facts1.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    await wait_deferred()
    open_facts = await pdb.fetch_all(select(GitHubOpenPullRequestFacts))
    assert len(open_facts) == 21
    facts2 = await pr_facts_calculator_no_cache(*args)
    facts2.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    assert_frame_equal(facts1, facts2)


@with_defer
async def test_pr_facts_calculator_unreleased_precomputed(
    pr_facts_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
):
    pr_facts_calculator_no_cache = pr_facts_calculator_factory(1, (6366825,))
    time_from = datetime(year=2019, month=10, day=30, tzinfo=timezone.utc)
    time_to = datetime(year=2019, month=11, day=2, tzinfo=timezone.utc)
    args = (
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
    facts1 = await pr_facts_calculator_no_cache(*args)
    facts1.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    await wait_deferred()
    unreleased_facts = await pdb.fetch_all(select(GitHubMergedPullRequestFacts))
    assert len(unreleased_facts) == 2
    for row in unreleased_facts:
        assert row[GitHubMergedPullRequestFacts.data.name] is not None, row[
            GitHubMergedPullRequestFacts.pr_node_id.name
        ]
    facts2 = await pr_facts_calculator_no_cache(*args)
    facts2.sort_values(PullRequestFacts.f.created, inplace=True, ignore_index=True)
    assert_frame_equal(facts1, facts2)


@with_defer
async def test_pr_facts_calculator_jira(
    pr_facts_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
    cache,
):
    pr_facts_calculator = pr_facts_calculator_factory(1, (6366825,), with_cache=True)
    pr_facts_calculator_cache_only = pr_facts_calculator_factory(1, (6366825,), cache_only=True)
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    args = [
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
    ]
    facts = await pr_facts_calculator(*args)
    await wait_deferred()
    assert facts.notnull(PullRequestFacts.f.released).sum() == 235
    args[5] = JIRAFilter(
        account=1,
        projects=frozenset(("10003", "10009")),
        labels=LabelFilter({"performance", "task"}, frozenset()),
        custom_projects=False,
    )
    facts = await pr_facts_calculator(*args)
    assert facts.notnull(PullRequestFacts.f.released).sum() == 16

    args[5] = JIRAFilter.empty()
    args[-1] = JIRAEntityToFetch.ISSUES
    facts = await pr_facts_calculator(*args)

    assert facts[PullRequestFacts.INDIRECT_FIELDS.JIRA_IDS].astype(bool).sum() == 60
    await wait_deferred()
    facts = await pr_facts_calculator_cache_only(*args)
    assert facts[PullRequestFacts.INDIRECT_FIELDS.JIRA_IDS].astype(bool).sum() == 60


@with_defer
async def test_pr_facts_calculator_jira_everything(
    pr_facts_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
    cache,
):
    pr_facts_calculator = pr_facts_calculator_factory(1, (6366825,), with_cache=True)
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    args = [
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
        JIRAEntityToFetch.EVERYTHING(),
    ]
    facts = await pr_facts_calculator(*args)
    pr_fact = facts.take(facts[PullRequestFacts.f.node_id] == 163209)
    assert facts[PullRequestFacts.INDIRECT_FIELDS.JIRA_IDS].astype(bool).sum() == 60
    assert_array_equal(
        pr_fact[PullRequestFacts.INDIRECT_FIELDS.JIRA_IDS][0], np.array(["DEV-678"]),
    )
    assert_array_equal(
        pr_fact[PullRequestFacts.INDIRECT_FIELDS.JIRA_PROJECTS][0], np.array([b"10009"]),
    )
    assert_array_equal(
        pr_fact[PullRequestFacts.INDIRECT_FIELDS.JIRA_TYPES][0], np.array([b"10024"]),
    )
    assert_array_equal(
        pr_fact[PullRequestFacts.INDIRECT_FIELDS.JIRA_PRIORITIES][0], np.array([b"5"]),
    )


@with_defer
async def test_pr_facts_calculator_event_releases(
    pr_facts_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_event,
    prefixer,
    bots,
    cache,
):
    await rdb.execute(
        insert(ReleaseNotification).values(
            ReleaseNotification(
                account_id=1,
                repository_node_id=40550,
                commit_hash_prefix="1edb992",
                name="Pushed!",
                author_node_id=40020,
                url="www",
                published_at=datetime(2019, 9, 1, tzinfo=timezone.utc),
            )
            .create_defaults()
            .explode(with_primary_keys=True),
        ),
    )
    pr_facts_calculator = pr_facts_calculator_factory(1, (6366825,))
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    args = [
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        False,
        bots,
        release_match_setting_event,
        LogicalRepositorySettings.empty(),
        prefixer,
        False,
        JIRAEntityToFetch.NOTHING,
    ]
    facts = await pr_facts_calculator(*args)
    await wait_deferred()
    assert facts.notnull(PullRequestFacts.f.released).sum() == 381
    facts = await pr_facts_calculator(*args)
    assert facts.notnull(PullRequestFacts.f.released).sum() == 381


@with_defer
async def test_calc_pull_request_facts_empty(
    pr_facts_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag,
    prefixer,
    bots,
    cache,
):
    pr_facts_calculator = pr_facts_calculator_factory(1, (6366825,), with_cache=True)
    time_from = datetime(year=2022, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2023, month=4, day=1, tzinfo=timezone.utc)
    args = [
        time_from,
        time_to,
        {"src-d/go-git"},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        True,
        bots,
        release_match_setting_tag,
        LogicalRepositorySettings.empty(),
        prefixer,
        False,
        JIRAEntityToFetch.NOTHING,
    ]
    facts = await pr_facts_calculator(*args)
    assert facts.empty
    assert len(facts.columns) == len(PullRequestFacts.f)
    assert PullRequestFacts.f.done in facts.columns


def test_size_calculator_shift_log():
    calc = histogram_calculators[PullRequestMetricID.PR_SIZE](quantiles=(0, 1))
    calc._samples = [[np.array([0, 10, 0, 20, 150, 0])]]
    h = calc.histogram(Scale.LOG, 3, None)[0][0]
    assert h.ticks[0] == 1
    for f in h.frequencies:
        assert f == f


@register_metric("test")
class QuantileTestingMetric(MetricCalculator):
    metric = MetricTimeDelta

    def _analyze(
        self,
        facts: md.DataFrame,
        min_times: np.ndarray,
        max_times: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        """Calculate the actual state update."""
        return np.repeat((facts["released"] - facts["created"])[None, :], len(min_times), axis=0)

    def _value(self, samples: Sequence[timedelta]) -> Tuple[timedelta, int]:
        """Calculate the actual current metric value."""
        return np.asarray(samples).sum(), len(samples)


def test_quantiles(pr_samples):
    time_from = datetime.utcnow() - timedelta(days=365)
    time_to = datetime.utcnow()
    samples = df_from_structs(pr_samples(200))
    min_times = dt64arr_us(time_from)
    max_times = dt64arr_us(time_to)
    groups = [np.arange(len(samples))]
    ensemble = PullRequestMetricCalculatorEnsemble("test", quantiles=(0, 1), quantile_stride=0)
    ensemble(samples, min_times, max_times, groups)
    m1, c1 = ensemble.values()["test"][0][0]
    ensemble = PullRequestMetricCalculatorEnsemble("test", quantiles=(0, 0.9), quantile_stride=73)
    ensemble(samples, min_times, max_times, groups)
    m2, c2 = ensemble.values()["test"][0][0]
    ensemble = PullRequestMetricCalculatorEnsemble(
        "test", quantiles=(0.1, 0.9), quantile_stride=73,
    )
    ensemble(samples, min_times, max_times, groups)
    m3, c3 = ensemble.values()["test"][0][0]
    assert m1 > m2 > m3
    assert c1 > c2 > c3


def test_counter_quantiles(pr_samples):
    time_from = datetime.utcnow() - timedelta(days=365)
    time_to = datetime.utcnow()
    samples = df_from_structs(pr_samples(100))
    quantiles = [0.25, 0.75]
    c_base = WorkInProgressTimeCalculator(quantiles=quantiles)
    c_with = WorkInProgressCounterWithQuantiles(c_base, quantiles=quantiles)
    c_without = WorkInProgressCounter(c_base, quantiles=quantiles)
    min_times = dt64arr_us(time_from)
    max_times = dt64arr_us(time_to)
    qmins, qmaxs = MetricCalculatorEnsemble.compose_quantile_time_intervals(
        min_times[0], max_times[0], 73,
    )
    min_times = np.concatenate([min_times, qmins])
    max_times = np.concatenate([max_times, qmaxs])
    groups = np.full((1, len(samples)), True)
    c_base(samples, min_times, max_times, 1, groups)
    c_with(samples, min_times, max_times, 1, groups)
    c_without(samples, min_times, max_times, 1, groups)
    v_with = c_with.values[0][0].value
    v_without = c_without.values[0][0].value
    assert v_without > v_with


@pytest.fixture(scope="function")
@with_defer
async def real_pr_samples(
    release_match_setting_tag,
    pr_facts_calculator_factory,
    prefixer,
    bots,
) -> Tuple[datetime, datetime, md.DataFrame]:
    pr_facts_calculator_no_cache = pr_facts_calculator_factory(1, (6366825,))
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    args = (
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
    samples = await pr_facts_calculator_no_cache(*args)
    return time_from, time_to, samples


@with_defer
async def test_pull_request_count_logical_alpha_beta(
    logical_settings,
    metrics_calculator_factory,
    mdb,
    pdb,
    rdb,
    release_match_setting_tag_logical,
    prefixer,
    bots,
    branches,
    default_branches,
    dag,
):
    logical_settings = deepcopy(logical_settings)
    logical_settings._deployments["src-d/go-git"] = LogicalDeploymentSettings(
        {
            "src-d/go-git/alpha": {"title": ".*"},
        },
        "src-d/go-git",
    )
    await mine_deployments(
        ["src-d/go-git/alpha"],
        {},
        datetime(2015, 1, 1, tzinfo=timezone.utc),
        datetime(2020, 1, 1, tzinfo=timezone.utc),
        ["production", "staging"],
        [],
        {},
        {},
        LabelFilter.empty(),
        JIRAFilter.empty(),
        release_match_setting_tag_logical,
        logical_settings,
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
    pr_facts_calculator = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    metrics = [
        PullRequestMetricID.PR_MERGED,
        PullRequestMetricID.PR_REJECTED,
        PullRequestMetricID.PR_REVIEW_COUNT,
        PullRequestMetricID.PR_DEPLOYMENT_COUNT,
        PullRequestMetricID.PR_RELEASE_COUNT,
    ]
    for _ in range(2):  # test the second run with filled pdb
        args = [
            metrics,
            [[time_from, time_to]],
            (0, 1),
            [],
            ["production"],
            [["src-d/go-git/alpha"], ["src-d/go-git/beta"]],
            [],
            LabelFilter.empty(),
            [JIRAFilter.empty()],
            False,
            bots,
            release_match_setting_tag_logical,
            logical_settings,
            prefixer,
            branches,
            default_branches,
            False,
        ]
        values = await pr_facts_calculator.calc_pull_request_metrics_line_github(*args)
        await wait_deferred()
        assert values.shape == (1, 1, 2, 1, 1)

        def check_metrics():
            assert values[0, 0, 0, 0, 0][0][0].value == 159
            assert values[0, 0, 0, 0, 0][0][1].value == 24
            assert values[0, 0, 0, 0, 0][0][2].value == 105
            assert values[0, 0, 0, 0, 0][0][3].value == 148
            assert values[0, 0, 0, 0, 0][0][4].value == 122

            assert values[0, 0, 1, 0, 0][0][0].value == 107
            assert values[0, 0, 1, 0, 0][0][1].value == 29
            assert values[0, 0, 1, 0, 0][0][2].value == 99
            assert (
                values[0, 0, 1, 0, 0][0][3].value == 0
            )  # TODO(vmarkovtsev): set to 79 when ready
            assert values[0, 0, 1, 0, 0][0][4].value == 79

        check_metrics()


@with_defer
async def test_pull_request_count_logical_root(
    logical_settings,
    precomputed_deployments,
    metrics_calculator_factory,
    release_match_setting_tag_logical,
    prefixer,
    bots,
    branches,
    default_branches,
):
    pr_facts_calculator = metrics_calculator_factory(1, (6366825,))
    time_from = datetime(year=2015, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    metrics = [
        PullRequestMetricID.PR_MERGED,
        PullRequestMetricID.PR_REJECTED,
        PullRequestMetricID.PR_REVIEW_COUNT,
        PullRequestMetricID.PR_DEPLOYMENT_COUNT,
        PullRequestMetricID.PR_RELEASE_COUNT,
    ]
    for _ in range(2):  # test the second run with filled pdb
        args = [
            metrics,
            [[time_from, time_to]],
            (0, 1),
            [],
            ["production"],
            [["src-d/go-git/alpha"], ["src-d/go-git/beta"]],
            [],
            LabelFilter.empty(),
            [JIRAFilter.empty()],
            False,
            bots,
            release_match_setting_tag_logical,
            logical_settings,
            prefixer,
            branches,
            default_branches,
            False,
        ]
        args[5].append(["src-d/go-git"])
        values = await pr_facts_calculator.calc_pull_request_metrics_line_github(*args)
        await wait_deferred()
        assert values.shape == (1, 1, 3, 1, 1)
        assert values[0, 0, 2, 0, 0][0][0].value == 304
        assert values[0, 0, 2, 0, 0][0][1].value == 57
        assert values[0, 0, 2, 0, 0][0][2].value == 267
        assert values[0, 0, 2, 0, 0][0][3].value == 283
        assert values[0, 0, 2, 0, 0][0][4].value == 230
        args[5] = [["src-d/go-git"]]
        values = await pr_facts_calculator.calc_pull_request_metrics_line_github(*args)
        await wait_deferred()
        assert values.shape == (1, 1, 1, 1, 1)
        assert values[0, 0, 0, 0, 0][0][0].value == 554
        assert values[0, 0, 0, 0, 0][0][1].value == 107
        assert values[0, 0, 0, 0, 0][0][2].value == 461
        assert values[0, 0, 0, 0, 0][0][3].value == 513
        assert values[0, 0, 0, 0, 0][0][4].value == 419


async def test_pull_request_stage_times(precomputed_deployments, real_pr_samples):
    ensemble = PullRequestMetricCalculatorEnsemble(
        PullRequestMetricID.PR_WIP_TIME,
        PullRequestMetricID.PR_REVIEW_TIME,
        PullRequestMetricID.PR_MERGING_TIME,
        PullRequestMetricID.PR_OPEN_TIME,
        PullRequestMetricID.PR_RELEASE_TIME,
        PullRequestMetricID.PR_DEPLOYMENT_TIME,
        PullRequestMetricID.PR_LEAD_DEPLOYMENT_TIME,
        quantile_stride=0,
        quantiles=(0, 1),
        environments=["staging", "mirror", "production"],
    )
    time_from, time_to, samples = real_pr_samples
    ensemble(samples, dt64arr_us(time_from), dt64arr_us(time_to), [np.arange(len(samples))])
    values = ensemble.values()
    for metric, td in [
        (PullRequestMetricID.PR_WIP_TIME, timedelta(days=3, seconds=58592)),
        (PullRequestMetricID.PR_REVIEW_TIME, timedelta(days=4, seconds=85421)),
        (PullRequestMetricID.PR_MERGING_TIME, timedelta(days=2, seconds=30368)),
        (PullRequestMetricID.PR_OPEN_TIME, timedelta(days=9, seconds=20554)),
        (PullRequestMetricID.PR_RELEASE_TIME, timedelta(days=29, seconds=25591)),
        (PullRequestMetricID.PR_DEPLOYMENT_TIME, [None, None, timedelta(days=723, seconds=65837)]),
        (
            PullRequestMetricID.PR_LEAD_DEPLOYMENT_TIME,
            [None, None, timedelta(days=754, seconds=12887)],
        ),
    ]:
        assert values[metric][0][0].value == td, metric


async def test_pull_request_deployment_stage_counts(precomputed_deployments, real_pr_samples):
    ensemble = PullRequestMetricCalculatorEnsemble(
        PullRequestMetricID.PR_DEPLOYMENT_COUNT,
        PullRequestMetricID.PR_DEPLOYMENT_COUNT_Q,
        PullRequestMetricID.PR_LEAD_DEPLOYMENT_COUNT,
        PullRequestMetricID.PR_LEAD_DEPLOYMENT_COUNT_Q,
        quantile_stride=180,
        quantiles=(0, 0.5),
        environments=["staging", "mirror", "production"],
    )
    time_from, time_to, samples = real_pr_samples
    ensemble(samples, dt64arr_us(time_from), dt64arr_us(time_to), [np.arange(len(samples))])
    values = ensemble.values()
    for metric, td in [
        (PullRequestMetricID.PR_DEPLOYMENT_COUNT, [0, 0, 513]),
        (PullRequestMetricID.PR_DEPLOYMENT_COUNT_Q, [0, 0, 262]),
        (PullRequestMetricID.PR_LEAD_DEPLOYMENT_COUNT, [0, 0, 513]),
        (PullRequestMetricID.PR_LEAD_DEPLOYMENT_COUNT_Q, [0, 0, 257]),
    ]:
        assert values[metric][0][0].value == td, metric


@pytest.mark.parametrize("with_origin", [False, True])
async def test_pull_request_cycle_deployment_time(
    precomputed_deployments,
    real_pr_samples,
    with_origin,
):
    ensemble = PullRequestMetricCalculatorEnsemble(
        PullRequestMetricID.PR_CYCLE_DEPLOYMENT_TIME,
        PullRequestMetricID.PR_CYCLE_DEPLOYMENT_COUNT,
        PullRequestMetricID.PR_CYCLE_DEPLOYMENT_COUNT_Q,
        *((PullRequestMetricID.PR_CYCLE_TIME,) if with_origin else ()),
        quantile_stride=180,
        quantiles=(0, 0.95),
        environments=["staging", "mirror", "production"],
    )
    time_from, time_to, samples = real_pr_samples
    ensemble(samples, dt64arr_us(time_from), dt64arr_us(time_to), [np.arange(len(samples))])
    values = ensemble.values()
    for metric, td in [
        (
            PullRequestMetricID.PR_CYCLE_DEPLOYMENT_TIME,
            [None, None, timedelta(days=729, seconds=85842)],
        ),
        (PullRequestMetricID.PR_CYCLE_DEPLOYMENT_COUNT, [0, 0, 513]),
        (PullRequestMetricID.PR_CYCLE_DEPLOYMENT_COUNT_Q, [0, 0, 460]),
    ]:
        assert values[metric][0][0].value == td, metric


async def test_pull_request_deployment_time_with_failed(
    precomputed_sample_deployments,
    real_pr_samples,
):
    ensemble = PullRequestMetricCalculatorEnsemble(
        PullRequestMetricID.PR_DEPLOYMENT_TIME,
        quantile_stride=0,
        quantiles=(0, 1),
        environments=["staging", "production"],
    )
    time_from, time_to, samples = real_pr_samples
    ensemble(samples, dt64arr_us(time_from), dt64arr_us(time_to), [np.arange(len(samples))])
    values = ensemble.values()
    assert (
        values[PullRequestMetricID.PR_DEPLOYMENT_TIME][0][0].value
        == [timedelta(days=187, seconds=62469)] * 2
    )


class TestGroupPRsByParticipants:
    def test_multiple_groups(self) -> None:
        items = md.DataFrame({"author": [20, 30], "values": [100, 200]})

        participants = [
            {PRParticipationKind.AUTHOR: {10, 30}},
            {PRParticipationKind.AUTHOR: {20}},
        ]

        res = group_prs_by_participants(participants, False, items)
        # group {10, 30} has row 1, group {20} has row 0
        assert len(res) == 2
        assert np.array_equal(res[0], [1])
        assert np.array_equal(res[1], [0])

    def test_single_participants_group_items_only_specified_participants(self) -> None:
        items = md.DataFrame({"a": [1, 2], "b": [3, 4]})
        participants = [{PRParticipationKind.AUTHOR: {2}}]

        res = group_prs_by_participants(participants, True, items)
        # all rows are selected with a single group
        assert len(res) == 1
        assert np.array_equal(res[0], [0, 1])

    def test_single_participants_group(self) -> None:
        items = md.DataFrame({"author": [1, 2, 2], "b": [3, 3, 3]})
        participants = [{PRParticipationKind.AUTHOR: {2}}]

        res = group_prs_by_participants(participants, False, items)
        assert len(res) == 1
        assert np.array_equal(res[0], [1, 2])

    def test_no_participant_groups(self) -> None:
        items = md.DataFrame({"a": [1, 2], "b": [3, 4]})
        res = group_prs_by_participants([], False, items)
        # all rows are selected with no groups
        assert len(res) == 1
        assert np.array_equal(res[0], [0, 1])

    def test_empty_items_multiple_participant_groups(self) -> None:
        items = md.DataFrame()
        participants = [
            {PRParticipationKind.AUTHOR: {1, 3}},
            {PRParticipationKind.AUTHOR: {2}},
        ]

        res = group_prs_by_participants(participants, False, items)
        assert len(res) == 2
        assert np.array_equal(res[0], [])
        assert np.array_equal(res[1], [])


class TestWorkInProgressTimeBelowThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 2, 1))

        wip_time_calc = WorkInProgressTimeCalculator(quantiles=quantiles)
        calc = WorkInProgressTimeBelowThresholdRatio(
            wip_time_calc, quantiles=quantiles, threshold=timedelta(hours=3),
        )
        prs = [
            self._mk_pr(dt(2022, 1, 1, 5), dt(2022, 1, 1, 6)),
            self._mk_pr(dt(2022, 1, 1, 2), dt(2022, 1, 1, 3)),
            self._mk_pr(dt(2022, 1, 1, 3), dt(2022, 1, 1, 7)),
            self._mk_pr(dt(2022, 1, 1, 4), None, closed=None, last_commit=None),
        ]
        facts = df_from_structs(prs)

        groups_mask = np.full((1, len(prs)), True, bool)

        wip_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(2 / 3)

    @classmethod
    def _mk_pr(
        cls,
        work_began: datetime,
        first_review_request: Optional[datetime],
        **kwargs,
    ) -> PullRequestFacts:
        return PullRequestFactsFactory(
            work_began=np.datetime64(work_began.replace(tzinfo=None), "s"),
            first_review_request=np.datetime64(first_review_request.replace(tzinfo=None), "s")
            if first_review_request is not None
            else None,
            **kwargs,
        )


class TestReviewedCalculator:
    def test_base(self) -> None:
        prs = df_from_structs(generate_pr_samples(100))

        calc = ReviewedCalculator(quantiles=[0, 1])

        min_times = dt64arr_us(dt(2001, 1, 1))
        max_times = dt64arr_us(dt(2135, 1, 1))

        calc(prs, min_times, max_times, None, np.full((1, len(prs)), True))

        assert calc.values[0][0].value is not None
        assert 0 <= calc.values[0][0].value <= 100


class TestReviewedRatioCalculator:
    def test_base(self) -> None:
        prs = df_from_structs(generate_pr_samples(30))

        reviewed_calc = ReviewedCalculator(quantiles=(0, 1))
        closed_calc = ClosedCalculator(quantiles=(0, 1))
        not_reviewed_calc = NotReviewedCalculator(reviewed_calc, closed_calc, quantiles=(0, 1))
        rev_non_rev_calc = _ReviewedPlusNotReviewedCalculator(
            reviewed_calc, not_reviewed_calc, quantiles=(0, 1),
        )
        reviewed_ratio_calc = ReviewedRatioCalculator(
            reviewed_calc, rev_non_rev_calc, quantiles=(0, 1),
        )

        min_times = dt64arr_us(dt(2001, 1, 1))
        max_times = dt64arr_us(dt(2135, 1, 1))
        calc_args = (prs, min_times, max_times, None, np.full((1, len(prs)), True))

        reviewed_calc(*calc_args)
        closed_calc(*calc_args)
        not_reviewed_calc(*calc_args)
        rev_non_rev_calc(*calc_args)
        reviewed_ratio_calc(*calc_args)

        reviewed = reviewed_calc.values[0][0].value
        not_reviewed = not_reviewed_calc.values[0][0].value
        reviewed_ratio = reviewed_ratio_calc.values[0][0].value

        assert reviewed is not None
        assert not_reviewed is not None
        assert reviewed_ratio == pytest.approx(reviewed / (reviewed + not_reviewed), rel=0.001)


class TestReviewTimeCalculator:
    def test_base(self) -> None:
        calc = ReviewTimeCalculator(quantiles=(0, 1))
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 3, 1))

        prs = [
            PullRequestFactsFactory(
                first_review_request_exact=np.datetime64(dt(2022, 1, 1), "s"),
                approved=np.datetime64(dt(2022, 1, 4), "s"),
            ),
            PullRequestFactsFactory(
                first_review_request_exact=np.datetime64(dt(2022, 1, 1), "s"),
                approved=np.datetime64(dt(2022, 1, 2), "s"),
            ),
            PullRequestFactsFactory(
                first_review_request_exact=None, approved=np.datetime64(dt(2022, 1, 2), "s"),
            ),  # review not requested, doesn't affect metric
            PullRequestFactsFactory(
                first_review_request_exact=np.datetime64(dt(2022, 1, 1), "s"), approved=None,
            ),  # not reviewed, doesn't affect metric
        ]
        facts = df_from_structs(prs)

        calc(facts, min_times, max_times, None, np.full((1, len(prs)), True, bool))
        assert calc.values[0][0].value == timedelta(days=2)


class TestReviewTimeBelowThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        threshold = timedelta(hours=5)
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 4, 1))

        review_time_calc = ReviewTimeCalculator(quantiles=quantiles)
        calc = ReviewTimeBelowThresholdRatio(
            review_time_calc, quantiles=quantiles, threshold=threshold,
        )

        prs = [
            self._mk_pr(dt(2022, 1, 1, 5), dt(2022, 1, 1, 10)),
            self._mk_pr(dt(2022, 1, 1, 2), dt(2022, 1, 1, 8)),
            self._mk_pr(dt(2022, 1, 1, 12), dt(2022, 1, 1, 14)),
            self._mk_pr(dt(2022, 1, 1, 12), None),
        ]
        facts = df_from_structs(prs)

        groups_mask = np.full((1, len(prs)), True, bool)
        review_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(2 / 3)

    def test_complex_groups_mask(self) -> None:
        quantiles = (0, 1)
        threshold = timedelta(hours=3)
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 4, 1))

        review_time_calc = ReviewTimeCalculator(quantiles=quantiles)
        calc = ReviewTimeBelowThresholdRatio(
            review_time_calc, quantiles=quantiles, threshold=threshold,
        )

        prs = [
            self._mk_pr(dt(2022, 1, 1, 5), dt(2022, 1, 1, 10)),
            self._mk_pr(dt(2022, 1, 1, 2), dt(2022, 1, 1, 5)),
            self._mk_pr(dt(2022, 1, 1, 12), dt(2022, 1, 1, 14)),
            self._mk_pr(dt(2022, 1, 1, 3), dt(2022, 1, 1, 4)),
        ]
        facts = df_from_structs(prs)

        groups_mask = np.array(
            [[True, True, False, True], [True, False, False, True], [True, False, False, False]],
            dtype=bool,
        )

        review_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 3
        assert all(len(v) == 1 for v in calc.values)

        assert calc.values[0][0].value == pytest.approx(2 / 3)
        assert calc.values[1][0].value == pytest.approx(1 / 2)
        assert calc.values[2][0].value == pytest.approx(0)

    def test_empty_groups_in_the_middle(self) -> None:
        quantiles = (0, 1)
        threshold = timedelta(hours=3)
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 4, 1))

        review_time_calc = ReviewTimeCalculator(quantiles=quantiles)
        calc = ReviewTimeBelowThresholdRatio(
            review_time_calc, quantiles=quantiles, threshold=threshold,
        )

        prs = [
            self._mk_pr(dt(2022, 1, 1, 5), dt(2022, 1, 1, 10)),
            self._mk_pr(dt(2022, 1, 1, 2), dt(2022, 1, 1, 5)),
            self._mk_pr(dt(2022, 1, 1, 3), dt(2022, 1, 1, 4)),
        ]
        facts = df_from_structs(prs)

        groups_mask = np.array(
            [
                [True, True, False],
                [True, False, False],
                [False, False, False],
                [False, True, True],
            ],
            dtype=bool,
        )

        review_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 4
        assert all(len(v) == 1 for v in calc.values)

        assert calc.values[0][0].value == pytest.approx(1 / 2)
        assert calc.values[1][0].value == pytest.approx(0)
        assert calc.values[2][0].value == pytest.approx(0)
        assert calc.values[3][0].value == pytest.approx(1)

    @classmethod
    def _mk_pr(cls, review_request: datetime, approved: Optional[datetime]) -> PullRequestFacts:
        return PullRequestFactsFactory(
            first_review_request_exact=np.datetime64(review_request.replace(tzinfo=None), "s"),
            approved=np.datetime64(approved.replace(tzinfo=None), "s") if approved else None,
        )


class TestWaitFirstReviewTimeBelowThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 2, 1))

        wait_review_calc = WaitFirstReviewTimeCalculator(quantiles=quantiles)
        calc = WaitFirstReviewTimeBelowThresholdRatio(
            wait_review_calc, quantiles=quantiles, threshold=timedelta(hours=24),
        )

        prs = [
            self._mk_pr(dt(2022, 1, 1, 2), dt(2022, 1, 1, 4)),
            self._mk_pr(dt(2022, 1, 2), dt(2022, 1, 3)),
            self._mk_pr(dt(2022, 1, 2, 2), dt(2022, 1, 3, 1)),
            self._mk_pr(dt(2022, 1, 3, 1), dt(2022, 1, 4, 5)),
            self._mk_pr(dt(2022, 1, 3, 1), None),
            self._mk_pr(None, dt(2022, 1, 3, 1)),
        ]
        facts = df_from_structs(prs)
        groups_mask = np.full((1, len(prs)), True, bool)

        wait_review_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(3 / 4)

        calc = WaitFirstReviewTimeBelowThresholdRatio(
            wait_review_calc, quantiles=quantiles, threshold=timedelta(hours=48),
        )
        calc(facts, min_times, max_times, None, groups_mask)
        assert calc.values[0][0].value == 1

        calc = WaitFirstReviewTimeBelowThresholdRatio(
            wait_review_calc, quantiles=quantiles, threshold=timedelta(hours=12),
        )
        calc(facts, min_times, max_times, None, groups_mask)
        assert calc.values[0][0].value == pytest.approx(1 / 4)

    @classmethod
    def _mk_pr(
        cls,
        review_request: Optional[datetime],
        first_comment: Optional[datetime],
    ) -> PullRequestFacts:
        return PullRequestFactsFactory(
            first_review_request_exact=np.datetime64(review_request.replace(tzinfo=None), "s")
            if review_request is not None
            else None,
            first_comment_on_first_review=np.datetime64(first_comment.replace(tzinfo=None), "s")
            if first_comment is not None
            else None,
        )


class TestSizeBelowThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 7, 1))
        prs = [
            self._mk_pr(50),
            self._mk_pr(100),
            self._mk_pr(120),
            self._mk_pr(101),
            self._mk_pr(1, created=dt(2023, 1, 1)),  # out of interval
        ]
        facts = df_from_structs(prs)

        groups_mask = np.full((1, len(prs)), True, bool)

        size_calc = SizeCalculator(quantiles=quantiles)
        calc = SizeBelowThresholdRatio(size_calc, quantiles=quantiles)  # default threshold is 100

        size_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(2 / 4)

    _DEFAULT_DT = dt(2022, 1, 15)

    @classmethod
    def _mk_pr(cls, size: int, created: datetime = _DEFAULT_DT) -> PullRequestFacts:
        return PullRequestFactsFactory(
            size=size, created=np.datetime64(created.replace(tzinfo=None), "s"),
        )


class TestReviewCommentsAboveThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 7, 1))
        prs = [
            self._mk_pr(0),  # ignored, not reviewed
            self._mk_pr(1),
            self._mk_pr(2),
            self._mk_pr(3),
            self._mk_pr(4),
            self._mk_pr(1, created=dt(2022, 8, 1)),  # ignored, out of time
            self._mk_pr(5, created=dt(2022, 8, 1)),  # ignored, out of time
        ]
        groups_mask = np.full((1, len(prs)), True, bool)
        facts = df_from_structs(prs)

        all_calc = AllCounter(quantiles=quantiles)
        review_comments_calc = AverageReviewCommentsCalculator(all_calc, quantiles=quantiles)
        calc = ReviewCommentsAboveThresholdRatio(review_comments_calc, quantiles=quantiles)

        all_calc(facts, min_times, max_times, None, groups_mask)
        review_comments_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(2 / 4)

        calc = ReviewCommentsAboveThresholdRatio(
            review_comments_calc, quantiles=quantiles, threshold=4,
        )
        calc(facts, min_times, max_times, None, groups_mask)
        assert calc.values[0][0].value == pytest.approx(1 / 4)

    _DEFAULT_CREATED = dt(2022, 2, 1)

    @classmethod
    def _mk_pr(cls, review_comments: int, created=_DEFAULT_CREATED) -> PullRequestFacts:
        return PullRequestFactsFactory(
            review_comments=review_comments,
            created=np.datetime64(created.replace(tzinfo=None), "s"),
        )


class TestMergingTimeBelowThresholdRatio:
    def test_with_groups(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_us(dt(2022, 4, 1))
        max_times = dt64arr_us(dt(2022, 5, 1))
        prs = [
            self._mk_pr(dt(2022, 4, 1, 1), dt(2022, 4, 1, 5)),
            self._mk_pr(dt(2022, 4, 1, 2), dt(2022, 4, 1, 3)),
            self._mk_pr(dt(2022, 4, 1, 4), dt(2022, 4, 1, 10)),
            self._mk_pr(dt(2022, 4, 1, 12), dt(2022, 4, 1, 18)),
        ]

        groups_mask = np.array(
            [[True, True, True, True], [False, False, False, False], [True, False, True, True]],
            dtype=bool,
        )
        facts = df_from_structs(prs)

        time_calc = MergingTimeCalculator(quantiles=quantiles)
        calc = MergingTimeBelowThresholdRatio(
            time_calc, quantiles=quantiles, threshold=timedelta(hours=4),
        )

        time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 3
        assert calc.values[0][0].value == pytest.approx(2 / 4)
        assert calc.values[1][0].value == 0
        assert calc.values[2][0].value == pytest.approx(1 / 3)

    @classmethod
    def _mk_pr(cls, approved: datetime, merged: datetime) -> PullRequestFacts:
        return PullRequestFactsFactory(
            approved=np.datetime64(approved.replace(tzinfo=None), "s"),
            merged=np.datetime64(merged.replace(tzinfo=None), "s"),
        )


class TestOpenTimeBelowThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 7, 1))
        prs = [
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 5)),
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 8)),
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 5)),
            self._mk_pr(dt(2022, 1, 3), None),
        ]
        groups_mask = np.full((1, len(prs)), True, bool)
        facts = df_from_structs(prs)

        open_time_calc = OpenTimeCalculator(quantiles=quantiles)
        calc = OpenTimeBelowThresholdRatio(open_time_calc, quantiles=quantiles)

        open_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 1
        assert len(calc.values[0]) == 1
        assert calc.values[0][0].value == pytest.approx(2 / 3)

        calc = OpenTimeBelowThresholdRatio(
            open_time_calc, quantiles=quantiles, threshold=timedelta(days=10),
        )
        calc(facts, min_times, max_times, None, groups_mask)
        assert calc.values[0][0].value == 1

    def test_more_groups(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 7, 1))
        prs = [
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 5)),
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 8)),
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 5)),
            self._mk_pr(dt(2022, 1, 3), None),
        ]
        groups_mask = np.array(
            [[True, True, True, False], [True, False, False, True], [False, True, True, True]],
            dtype=bool,
        )

        facts = df_from_structs(prs)

        open_time_calc = OpenTimeCalculator(quantiles=quantiles)
        calc = OpenTimeBelowThresholdRatio(open_time_calc, quantiles=quantiles)

        open_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 3
        assert calc.values[0][0].value == pytest.approx(2 / 3)
        assert calc.values[1][0].value == pytest.approx(1 / 1)
        assert calc.values[2][0].value == pytest.approx(1 / 2)

    def test_other_groups(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 7, 1))
        prs = [
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 5)),
            self._mk_pr(dt(2022, 1, 3), dt(2022, 1, 4)),
        ]
        groups_mask = np.array(
            [[True, True], [False, False], [False, True], [True, False]],
            dtype=bool,
        )

        facts = df_from_structs(prs)

        open_time_calc = OpenTimeCalculator(quantiles=quantiles)
        calc = OpenTimeBelowThresholdRatio(
            open_time_calc, quantiles=quantiles, threshold=timedelta(hours=25),
        )

        open_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)

        assert len(calc.values) == 4
        assert calc.values[0][0].value == 1 / 2
        assert calc.values[1][0].value == 0
        assert calc.values[2][0].value == 1
        assert calc.values[3][0].value == 0

    @classmethod
    def _mk_pr(cls, created: datetime, closed: Optional[datetime]) -> PullRequestFacts:
        return PullRequestFactsFactory(
            created=np.datetime64(created.replace(tzinfo=None), "s"),
            closed=None if closed is None else np.datetime64(closed.replace(tzinfo=None), "s"),
        )


class TestCycleTimeBelowThresholdRatio:
    def test_base(self) -> None:
        quantiles = (0, 1)
        min_times = dt64arr_us(dt(2022, 1, 1))
        max_times = dt64arr_us(dt(2022, 1, 20))
        prs = [
            self._mk_pr(dt(2022, 1, 1), dt(2022, 1, 4)),
            self._mk_pr(dt(2022, 1, 2), dt(2022, 1, 4)),
            self._mk_pr(dt(2022, 1, 1), dt(2022, 1, 10)),
            self._mk_pr(dt(2022, 2, 1), dt(2022, 2, 10)),
        ]
        groups_mask = np.full((1, len(prs)), True, bool)

        facts = df_from_structs(prs)

        lead_time_calc = CycleTimeCalculator(quantiles=quantiles)
        calc = CycleTimeBelowThresholdRatio(lead_time_calc, quantiles=quantiles)

        lead_time_calc(facts, min_times, max_times, None, groups_mask)
        calc(facts, min_times, max_times, None, groups_mask)
        assert len(calc.values) == 1
        assert calc.values[0][0].value == pytest.approx(2 / 3)

    @classmethod
    def _mk_pr(cls, work_began: datetime, released: datetime) -> PullRequestFacts:
        return PullRequestFactsFactory(
            released=np.datetime64(released.replace(tzinfo=None), "s"),
            work_began=np.datetime64(work_began.replace(tzinfo=None), "s"),
        )
