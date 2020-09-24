from datetime import datetime, timedelta, timezone
import itertools
from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import select

from athenian.api.controllers.features.entries import calc_pull_request_facts_github, \
    calc_pull_request_metrics_line_github
from athenian.api.controllers.features.github.pull_request_metrics import AllCounter, \
    ClosedCalculator, CycleCounter, CycleCounterWithQuantiles, CycleTimeCalculator, \
    FlowRatioCalculator, histogram_calculators, LeadCounter, LeadCounterWithQuantiles, \
    LeadTimeCalculator, MergingCounter, MergingCounterWithQuantiles, MergingTimeCalculator, \
    OpenedCalculator, PullRequestBinnedMetricCalculator, \
    PullRequestMetricCalculatorEnsemble, register_metric, ReleaseCounter, \
    ReleaseCounterWithQuantiles, ReleasedCalculator, ReleaseTimeCalculator, ReviewCounter, \
    ReviewCounterWithQuantiles, ReviewTimeCalculator, WaitFirstReviewTimeCalculator, \
    WorkInProgressCounter, WorkInProgressCounterWithQuantiles, WorkInProgressTimeCalculator
from athenian.api.controllers.features.histogram import Scale
from athenian.api.controllers.features.metric_calculator import df_from_dataclasses, \
    MetricCalculator
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.pull_request import PullRequestMiner
from athenian.api.controllers.miners.types import PullRequestFacts
from athenian.api.controllers.settings import ReleaseMatch, ReleaseMatchSetting
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.precomputed.models import GitHubMergedPullRequestFacts, \
    GitHubOpenPullRequestFacts
from athenian.api.models.web import Granularity, PullRequestMetricID
from tests.conftest import has_memcached


def random_dropout(pr, prob):
    fields = sorted(PullRequestFacts.__dataclass_fields__)
    fields = [f for f in fields if isinstance(getattr(pr, f), pd.Timestamp)]
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
    kwargs = pr.__dict__.copy()
    for k in killed:
        kwargs[k] = None
    if "first_commit" in killed:
        kwargs["work_began"] = kwargs["created"]
    if "released" in killed or "closed" in killed:
        kwargs["done"] = kwargs["released"] or pr.force_push_dropped or (
            kwargs["closed"] and not kwargs["merged"])
    return PullRequestFacts(**kwargs)


@pytest.mark.parametrize("cls", [
    WorkInProgressTimeCalculator, ReviewTimeCalculator, MergingTimeCalculator,
    ReleaseTimeCalculator, LeadTimeCalculator, WaitFirstReviewTimeCalculator,
])
def test_pull_request_metrics_timedelta_stability(pr_samples, cls):  # noqa: F811
    calc = cls(quantiles=(0, 1))
    time_from = datetime.utcnow() - timedelta(days=10000)
    time_to = datetime.utcnow()
    prs = df_from_dataclasses(random_dropout(pr, 0.5) for pr in pr_samples(1000))
    r = calc._analyze(prs, time_from, time_to)
    assert (r[r != np.array(None)] >= 0).all()


@pytest.mark.parametrize("cls, peak_attr",
                         [(WorkInProgressTimeCalculator, "first_review_request"),
                          (ReviewTimeCalculator, "approved,last_review"),
                          (MergingTimeCalculator, "closed"),
                          (ReleaseTimeCalculator, "released"),
                          (LeadTimeCalculator, "released"),
                          (WaitFirstReviewTimeCalculator, "first_comment_on_first_review"),
                          ])
def test_pull_request_metrics_out_of_bounds(pr_samples, cls, peak_attr):  # noqa: F811
    calc = cls(quantiles=(0, 1))
    for pr in pr_samples(100):
        time_from = datetime.utcnow() - timedelta(days=10000)
        for attr in peak_attr.split(","):
            time_from = max(
                getattr(pr, attr),
                time_from.replace(tzinfo=timezone.utc),
            ).replace(tzinfo=None)
        time_from += timedelta(days=1)
        time_to = time_from + timedelta(days=7)
        assert calc._analyze(df_from_dataclasses([pr]), time_from, time_to)[0] is None

        time_from = datetime.utcnow()
        for attr in peak_attr.split(","):
            time_from = min(
                getattr(pr, attr),
                time_from.replace(tzinfo=timezone.utc),
            ).replace(tzinfo=None)
        time_from -= timedelta(days=7)
        time_to = time_from + timedelta(days=1)
        assert calc._analyze(df_from_dataclasses([pr]), time_from, time_to)[0] is None


@pytest.mark.parametrize("metric", [PullRequestMetricID.PR_OPENED,
                                    PullRequestMetricID.PR_MERGED,
                                    PullRequestMetricID.PR_REJECTED,
                                    PullRequestMetricID.PR_CLOSED])
def test_pull_request_metrics_float_binned(pr_samples, metric):  # noqa: F811
    time_from = (datetime.now(tz=timezone.utc) - timedelta(days=365 * 3 // 2)).date()
    time_to = (datetime.now(tz=timezone.utc) - timedelta(days=365 // 2)).date()
    time_intervals = [datetime.combine(i, datetime.min.time(), tzinfo=timezone.utc)
                      for i in Granularity.split("month", time_from, time_to)]
    binned = PullRequestBinnedMetricCalculator([metric], time_intervals, quantiles=(0, 1))
    samples = pr_samples(1000)
    if metric == PullRequestMetricID.PR_REJECTED:
        for i, s in enumerate(samples):
            data = vars(s)
            data["merged"] = None
            samples[i] = PullRequestFacts(**data)
    result = binned(samples)
    # the last interval is null and that's intended
    for i, m in enumerate(result[:-1]):
        assert m[0].exists, str(i)
        assert m[0].value > 1, str(i)
        assert m[0].confidence_min is None, str(i)
        assert m[0].confidence_max is None, str(i)


def test_pull_request_opened_no(pr_samples):  # noqa: F811
    calc = OpenedCalculator(quantiles=(0, 1))
    time_to = datetime.utcnow()
    time_from = time_to - timedelta(days=180)
    prs = df_from_dataclasses(pr for pr in pr_samples(100)
                              if pr.closed and pr.closed < time_to.replace(tzinfo=timezone.utc))
    calc(prs, time_from, time_to)
    assert len(prs) > 0
    m = calc.value
    assert not m.exists


def test_pull_request_closed_no(pr_samples):  # noqa: F811
    calc_closed = ClosedCalculator(quantiles=(0, 1))
    calc_released = ReleasedCalculator(quantiles=(0, 1))
    time_from = datetime.utcnow() - timedelta(days=365 * 3)
    time_to = time_from + timedelta(days=7)
    prs = df_from_dataclasses(pr_samples(100))
    calc_closed(prs, time_from, time_to)
    calc_released(prs, time_from, time_to)
    assert not calc_closed.value.exists
    assert not calc_released.value.exists


def test_pull_request_flow_ratio(pr_samples):  # noqa: F811
    calc = FlowRatioCalculator(OpenedCalculator(quantiles=(0, 1)),
                               ClosedCalculator(quantiles=(0, 1)),
                               quantiles=(0, 1))
    open_calc = OpenedCalculator(quantiles=(0, 1))
    closed_calc = ClosedCalculator(quantiles=(0, 1))
    time_from = datetime.utcnow() - timedelta(days=365)
    time_to = datetime.utcnow()
    prs = df_from_dataclasses(pr_samples(1000))
    for dep in calc._calcs:
        dep(prs, time_from, time_to)
    calc(prs, time_from, time_to)
    open_calc(prs, time_from, time_to)
    closed_calc(prs, time_from, time_to)
    m = calc.value
    assert m.exists
    assert 0 < m.value < 1
    assert m.confidence_min is None
    assert m.confidence_max is None
    assert m.value == (open_calc.value.value + 1) / (closed_calc.value.value + 1)


def test_pull_request_flow_ratio_zeros():
    calc = FlowRatioCalculator(OpenedCalculator(quantiles=(0, 1)),
                               ClosedCalculator(quantiles=(0, 1)),
                               quantiles=(0, 1))
    m = calc.value
    assert not m.exists


def test_pull_request_flow_ratio_no_opened(pr_samples):  # noqa: F811
    calc = FlowRatioCalculator(OpenedCalculator(quantiles=(0, 1)),
                               ClosedCalculator(quantiles=(0, 1)),
                               quantiles=(0, 1))
    time_to = datetime.utcnow()
    time_from = time_to - timedelta(days=180)
    for pr in pr_samples(100):
        if pr.closed and time_from.replace(tzinfo=timezone.utc) <= pr.closed < \
                time_to.replace(tzinfo=timezone.utc):
            df = df_from_dataclasses([pr])
            for dep in calc._calcs:
                dep(df, time_from, time_to)
            calc(df, time_from, time_to)
            break
    m = calc.value
    assert m.exists
    assert m.value == 0.5


def test_pull_request_flow_ratio_no_closed(pr_samples):  # noqa: F811
    calc = FlowRatioCalculator(OpenedCalculator(quantiles=(0, 1)),
                               ClosedCalculator(quantiles=(0, 1)),
                               quantiles=(0, 1))
    time_to = datetime.utcnow() - timedelta(days=180)
    time_from = time_to - timedelta(days=180)
    for pr in pr_samples(100):
        if pr.closed and pr.closed > time_to.replace(tzinfo=timezone.utc) > \
                pr.created >= time_from.replace(tzinfo=timezone.utc):
            df = df_from_dataclasses([pr])
            for dep in calc._calcs:
                dep(df, time_from, time_to)
            calc(df, time_from, time_to)
            break
    m = calc.value
    assert m.exists
    assert m.value == 2


@pytest.mark.parametrize("cls",
                         [WorkInProgressCounter,
                          ReviewCounter,
                          MergingCounter,
                          ReleaseCounter,
                          LeadCounter,
                          CycleCounter,
                          AllCounter])
def test_pull_request_metrics_counts(pr_samples, cls):  # noqa: F811
    calc = cls(*(dep1(*(dep2(quantiles=(0, 1)) for dep2 in dep1.deps),
                      quantiles=(0, 1)) for dep1 in cls.deps),
               quantiles=(0, 1))
    prs = df_from_dataclasses(pr_samples(1000))
    time_to = datetime.utcnow()
    time_from = time_to - timedelta(days=10000)
    for dep1 in calc._calcs:
        for dep2 in dep1._calcs:
            dep2(prs, time_from, time_to)
        dep1(prs, time_from, time_to)
    calc(prs, time_from, time_to)
    delta = calc.peek
    assert isinstance(delta, np.ndarray)
    assert len(delta) == 1000
    if cls != AllCounter:
        peek = calc._calcs[0].peek
    else:
        peek = calc.peek
    nonones = (peek != np.array(None)).sum()
    nones = len(peek) - nonones
    if cls not in (WorkInProgressCounter, CycleCounter, AllCounter):
        assert nones > 0, cls
    assert nonones > 0


@pytest.mark.parametrize("cls_q, cls",
                         [(WorkInProgressCounterWithQuantiles, WorkInProgressCounter),
                          (ReviewCounterWithQuantiles, ReviewCounter),
                          (MergingCounterWithQuantiles, MergingCounter),
                          (ReleaseCounterWithQuantiles, ReleaseCounter),
                          (LeadCounterWithQuantiles, LeadCounter),
                          (CycleCounterWithQuantiles, CycleCounter)])
def test_pull_request_metrics_counts_q(pr_samples, cls_q, cls):  # noqa: F811
    calc_q = cls_q(*(dep1(*(dep2(quantiles=(0, 0.95)) for dep2 in dep1.deps),
                          quantiles=(0, 0.95)) for dep1 in cls_q.deps),
                   quantiles=(0, 0.95))
    calc = cls(*calc_q._calcs, quantiles=(0, 0.95))
    prs = df_from_dataclasses(pr_samples(1000))
    time_to = datetime.utcnow()
    time_from = time_to - timedelta(days=10000)
    for dep1 in calc._calcs:
        for dep2 in dep1._calcs:
            dep2(prs, time_from, time_to)
        dep1(prs, time_from, time_to)
    calc_q(prs, time_from, time_to)
    calc(prs, time_from, time_to)
    assert 0 < calc_q.value.value < calc.value.value


@pytest.mark.parametrize("with_memcached, with_mine_cache_wipe",
                         itertools.product(*([[False, True]] * 2)))
@with_defer
async def test_calc_pull_request_metrics_line_github_cache(
        branches, default_branches, mdb, pdb, cache, memcached, with_memcached,
        release_match_setting_tag, with_mine_cache_wipe):
    if with_memcached:
        if not has_memcached:
            raise pytest.skip("no memcached")
        cache = memcached
    date_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2019, month=10, day=1, tzinfo=timezone.utc)
    args = ([PullRequestMetricID.PR_CYCLE_TIME], [[date_from, date_to]], [0, 1],
            [{"src-d/go-git"}], {}, LabelFilter.empty(), JIRAFilter.empty(), False,
            release_match_setting_tag, False, mdb, pdb, cache)
    metrics1 = (await calc_pull_request_metrics_line_github(*args))[0][0][0][0]
    await wait_deferred()
    assert await calc_pull_request_metrics_line_github.reset_cache(*args)
    if with_mine_cache_wipe:
        assert await PullRequestMiner._mine.reset_cache(
            None, date_from, date_to, {"src-d/go-git"}, {}, LabelFilter.empty(),
            JIRAFilter.empty(), branches, default_branches,
            False, release_match_setting_tag, 0, mdb, pdb, cache,
            pr_blacklist=None, truncate=True)
    metrics2 = (await calc_pull_request_metrics_line_github(*args))[0][0][0][0]
    assert metrics1.exists and metrics2.exists
    assert metrics1.value == metrics2.value
    assert metrics1.confidence_score() == metrics2.confidence_score()
    assert metrics1.confidence_min < metrics1.value < metrics1.confidence_max


@with_defer
async def test_calc_pull_request_metrics_line_github_changed_releases(
        mdb, pdb, cache, release_match_setting_tag):
    date_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2017, month=10, day=1, tzinfo=timezone.utc)
    args = [[PullRequestMetricID.PR_CYCLE_TIME], [[date_from, date_to]], [0, 1],
            [{"src-d/go-git"}], {}, LabelFilter.empty(), JIRAFilter.empty(), False,
            release_match_setting_tag, False, mdb, pdb, cache]
    metrics1 = (await calc_pull_request_metrics_line_github(*args))[0][0][0]
    release_match_setting_tag = {
        "github.com/src-d/go-git": ReleaseMatchSetting("master", ".*", ReleaseMatch.branch),
    }
    args[-5] = release_match_setting_tag
    metrics2 = (await calc_pull_request_metrics_line_github(*args))[0][0][0]
    assert metrics1 != metrics2


@with_defer
async def test_pr_list_miner_match_metrics_all_count_david_bug(
        mdb, pdb, release_match_setting_tag):
    time_from = datetime(year=2016, month=11, day=17, tzinfo=timezone.utc)
    time_middle = time_from + timedelta(days=14)
    time_to = datetime(year=2016, month=12, day=15, tzinfo=timezone.utc)
    metric1 = (await calc_pull_request_metrics_line_github(
        [PullRequestMetricID.PR_ALL_COUNT], [[time_from, time_middle]], [0, 1],
        [{"src-d/go-git"}], {}, LabelFilter.empty(), JIRAFilter.empty(), False,
        release_match_setting_tag, False, mdb, pdb, None,
    ))[0][0][0][0].value
    metric2 = (await calc_pull_request_metrics_line_github(
        [PullRequestMetricID.PR_ALL_COUNT], [[time_middle, time_to]], [0, 1],
        [{"src-d/go-git"}], {}, LabelFilter.empty(), JIRAFilter.empty(), False,
        release_match_setting_tag, False, mdb, pdb, None,
    ))[0][0][0][0].value
    metric1_ext, metric2_ext = (m[0].value for m in (
        await calc_pull_request_metrics_line_github(
            [PullRequestMetricID.PR_ALL_COUNT], [[time_from, time_middle, time_to]], [0, 1],
            [{"src-d/go-git"}], {}, LabelFilter.empty(), JIRAFilter.empty(), False,
            release_match_setting_tag, False, mdb, pdb, None,
        )
    )[0][0])
    assert metric1 == metric1_ext
    assert metric2 == metric2_ext


@with_defer
async def test_calc_pull_request_metrics_line_github_exclude_inactive(
        mdb, pdb, cache, release_match_setting_tag):
    date_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2017, month=1, day=12, tzinfo=timezone.utc)
    args = [[PullRequestMetricID.PR_ALL_COUNT], [[date_from, date_to]], [0, 1],
            [{"src-d/go-git"}], {}, LabelFilter.empty(), JIRAFilter.empty(),
            False, release_match_setting_tag, False, mdb, pdb, cache]
    metrics = (await calc_pull_request_metrics_line_github(*args))[0][0][0][0]
    await wait_deferred()
    assert metrics.value == 7
    args[7] = True
    metrics = (await calc_pull_request_metrics_line_github(*args))[0][0][0][0]
    await wait_deferred()
    assert metrics.value == 6
    date_from = datetime(year=2017, month=5, day=23, tzinfo=timezone.utc)
    date_to = datetime(year=2017, month=5, day=25, tzinfo=timezone.utc)
    args[0] = [PullRequestMetricID.PR_RELEASE_COUNT]
    args[1] = [[date_from, date_to]]
    args[7] = False
    metrics = (await calc_pull_request_metrics_line_github(*args))[0][0][0][0]
    await wait_deferred()
    assert metrics.value == 70
    metrics = (await calc_pull_request_metrics_line_github(*args))[0][0][0][0]
    await wait_deferred()
    assert metrics.value == 70
    args[7] = True
    metrics = (await calc_pull_request_metrics_line_github(*args))[0][0][0][0]
    assert metrics.value == 71


def test_pull_request_metric_calculator_ensemble_accuracy(pr_samples):
    ensemble = PullRequestMetricCalculatorEnsemble(PullRequestMetricID.PR_CYCLE_TIME,
                                                   PullRequestMetricID.PR_WIP_COUNT,
                                                   PullRequestMetricID.PR_RELEASE_TIME,
                                                   PullRequestMetricID.PR_CLOSED,
                                                   quantiles=(0, 1))
    release_time = ReleaseTimeCalculator(quantiles=(0, 1))
    wip_count = WorkInProgressCounter(WorkInProgressTimeCalculator(quantiles=(0, 1)),
                                      quantiles=(0, 1))
    cycle_time = CycleTimeCalculator(WorkInProgressTimeCalculator(quantiles=(0, 1)),
                                     ReviewTimeCalculator(quantiles=(0, 1)),
                                     MergingTimeCalculator(quantiles=(0, 1)),
                                     ReleaseTimeCalculator(quantiles=(0, 1)),
                                     quantiles=(0, 1))
    closed = ClosedCalculator(quantiles=(0, 1))
    time_from = datetime.utcnow() - timedelta(days=365)
    time_to = datetime.utcnow()
    for _ in range(2):
        prs = df_from_dataclasses(pr_samples(100))
        ensemble(prs, time_from, time_to)
        release_time(prs, time_from, time_to)
        wip_count._calcs[0](prs, time_from, time_to)
        wip_count(prs, time_from, time_to)
        for c in cycle_time._calcs:
            c(prs, time_from, time_to)
        cycle_time(prs, time_from, time_to)
        closed(prs, time_from, time_to)
        ensemble_metrics = ensemble.values()
        assert ensemble_metrics[PullRequestMetricID.PR_CYCLE_TIME] == cycle_time.value
        assert ensemble_metrics[PullRequestMetricID.PR_RELEASE_TIME] == release_time.value
        assert ensemble_metrics[PullRequestMetricID.PR_WIP_COUNT] == wip_count.value
        assert ensemble_metrics[PullRequestMetricID.PR_CLOSED] == closed.value
        for c in (ensemble, release_time, wip_count, cycle_time, closed):
            c.reset()


def test_pull_request_metric_calculator_ensemble_empty(pr_samples):
    ensemble = PullRequestMetricCalculatorEnsemble(quantiles=(0, 1))
    time_from = datetime.utcnow() - timedelta(days=365)
    time_to = datetime.utcnow()
    ensemble(df_from_dataclasses(pr_samples(1)), time_from, time_to)
    assert ensemble.values() == {}


@with_defer
async def test_calc_pull_request_facts_github_open_precomputed(
        mdb, pdb, release_match_setting_tag):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    args = (time_from, time_to, {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
            False, release_match_setting_tag, False, mdb, pdb, None)
    facts1 = await calc_pull_request_facts_github(*args)
    await wait_deferred()
    open_facts = await pdb.fetch_all(select([GitHubOpenPullRequestFacts]))
    assert len(open_facts) == 21
    facts2 = await calc_pull_request_facts_github(*args)
    assert set(facts1["src-d/go-git"]) == set(facts2["src-d/go-git"])


@with_defer
async def test_calc_pull_request_facts_github_unreleased_precomputed(
        mdb, pdb, release_match_setting_tag):
    time_from = datetime(year=2019, month=10, day=30, tzinfo=timezone.utc)
    time_to = datetime(year=2019, month=11, day=2, tzinfo=timezone.utc)
    args = (time_from, time_to, {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
            False, release_match_setting_tag, False, mdb, pdb, None)
    facts1 = await calc_pull_request_facts_github(*args)
    await wait_deferred()
    unreleased_facts = await pdb.fetch_all(select([GitHubMergedPullRequestFacts]))
    assert len(unreleased_facts) == 2
    for row in unreleased_facts:
        assert row[GitHubMergedPullRequestFacts.data.key] is not None, \
            row[GitHubMergedPullRequestFacts.pr_node_id.key]
    facts2 = await calc_pull_request_facts_github(*args)
    assert set(facts1["src-d/go-git"]) == set(facts2["src-d/go-git"])


@with_defer
async def test_calc_pull_request_facts_github_jira(
        mdb, pdb, release_match_setting_tag):
    time_from = datetime(year=2018, month=1, day=1, tzinfo=timezone.utc)
    time_to = datetime(year=2020, month=4, day=1, tzinfo=timezone.utc)
    args = [time_from, time_to, {"src-d/go-git"}, {}, LabelFilter.empty(), JIRAFilter.empty(),
            False, release_match_setting_tag, False, mdb, pdb, None]
    facts = (await calc_pull_request_facts_github(*args))["src-d/go-git"]
    await wait_deferred()
    assert sum(1 for f in facts if f.released) == 233
    args[5] = JIRAFilter(1, LabelFilter({"performance", "task"}, set()), set(), set())
    facts = (await calc_pull_request_facts_github(*args))["src-d/go-git"]
    assert sum(1 for f in facts if f.released) == 16


def test_size_calculator_shift_log():
    calc = histogram_calculators[PullRequestMetricID.PR_SIZE](quantiles=(0, 1))
    calc._samples = np.array([0, 10, 0, 20, 150, 0])
    h = calc.histogram(Scale.LOG, 3, None)
    assert h.ticks[0] == 1
    for f in h.frequencies:
        assert f == f


@register_metric("test")
class QuantileTestingMetric(MetricCalculator):
    dtype = "timedelta64[s]"

    def _analyze(self, facts: pd.DataFrame, min_time: datetime, max_time: datetime,
                 **kwargs) -> np.ndarray:
        """Calculate the actual state update."""
        return (facts["released"] - facts["created"]).values

    def _value(self, samples: Sequence[timedelta]) -> Tuple[timedelta, int]:
        """Calculate the actual current metric value."""
        return np.asarray(samples).sum(), len(samples)


def test_quantiles(pr_samples):
    time_from = datetime.utcnow() - timedelta(days=365)
    time_to = datetime.utcnow()
    samples = df_from_dataclasses(pr_samples(200))
    ensemble = PullRequestMetricCalculatorEnsemble("test", quantiles=(0, 1))
    ensemble(samples, time_from, time_to)
    m1, c1 = ensemble.values()["test"]
    ensemble = PullRequestMetricCalculatorEnsemble("test", quantiles=(0, 0.9))
    ensemble(samples, time_from, time_to)
    m2, c2 = ensemble.values()["test"]
    ensemble = PullRequestMetricCalculatorEnsemble("test", quantiles=(0.1, 0.9))
    ensemble(samples, time_from, time_to)
    m3, c3 = ensemble.values()["test"]
    assert m1 > m2 > m3
    assert c1 > c2 > c3


def test_counter_quantiles(pr_samples):
    time_from = datetime.utcnow() - timedelta(days=365)
    time_to = datetime.utcnow()
    samples = df_from_dataclasses(pr_samples(100))
    c_base = WorkInProgressTimeCalculator(quantiles=[0.25, 0.75])
    c_with = WorkInProgressCounterWithQuantiles(c_base, quantiles=[0.25, 0.75])
    c_without = WorkInProgressCounter(c_base, quantiles=[0.25, 0.75])
    c_base(samples, time_from, time_to)
    c_with(samples, time_from, time_to)
    c_without(samples, time_from, time_to)
    v_with = c_with.value.value
    v_without = c_without.value.value
    assert v_without > v_with
