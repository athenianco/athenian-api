from datetime import datetime, timedelta, timezone
import itertools

import numpy as np
import pandas as pd
import pytest

from athenian.api.controllers.features.entries import calc_pull_request_metrics_line_github
from athenian.api.controllers.features.github.pull_request import BinnedPullRequestMetricCalculator
from athenian.api.controllers.features.github.pull_request_metrics import AllCounter, \
    ClosedCalculator, CycleCounter, FlowRatioCalculator, LeadCounter, LeadTimeCalculator, \
    MergedCalculator, MergingCounter, MergingTimeCalculator, OpenedCalculator, ReleaseCounter, \
    ReleaseTimeCalculator, ReviewCounter, ReviewTimeCalculator, WaitFirstReviewTimeCalculator, \
    WorkInProgressCounter, WorkInProgressTimeCalculator
from athenian.api.controllers.miners.github.pull_request import Fallback, PullRequestMiner, \
    PullRequestTimes
from athenian.api.controllers.settings import Match, ReleaseMatchSetting
from athenian.api.models.web import Granularity, PullRequestMetricID
from tests.conftest import has_memcached
from tests.controllers.features.github.test_pull_request import ensure_dtype


def random_dropout(pr, prob):
    fields = sorted(PullRequestTimes.__dataclass_fields__)
    killed = np.random.choice(fields, int(len(fields) * prob), replace=False)
    kwargs = {f: getattr(pr, f) for f in fields}
    for k in killed:
        # "created" must always exist
        if k != "created":
            kwargs[k] = Fallback(None, None)
    return PullRequestTimes(**kwargs)


@pytest.mark.parametrize("cls, dtypes", itertools.product(
    [WorkInProgressTimeCalculator, ReviewTimeCalculator, MergingTimeCalculator,
     ReleaseTimeCalculator, LeadTimeCalculator, WaitFirstReviewTimeCalculator,
     ], ((datetime, timedelta), (pd.Timestamp, pd.Timedelta))))
def test_pull_request_metrics_timedelta_stability(pr_samples, cls, dtypes):  # noqa: F811
    calc = cls()
    time_from = datetime.now(tz=timezone.utc) - timedelta(days=10000)
    time_to = datetime.now(tz=timezone.utc)
    for pr in pr_samples(1000):
        pr = random_dropout(ensure_dtype(pr, dtypes[0]), 0.5)
        r = calc.analyze(pr, time_from, time_to)
        assert (r is None) or ((isinstance(r, dtypes[1])) and r >= dtypes[1](0)), str(pr)


@pytest.mark.parametrize("cls, peak_attr",
                         [(WorkInProgressTimeCalculator, "first_review_request"),
                          (ReviewTimeCalculator, "approved,last_review"),
                          (MergingTimeCalculator, "closed"),
                          (ReleaseTimeCalculator, "released"),
                          (LeadTimeCalculator, "released"),
                          (WaitFirstReviewTimeCalculator, "first_comment_on_first_review"),
                          ])
def test_pull_request_metrics_out_of_bounds(pr_samples, cls, peak_attr):  # noqa: F811
    calc = cls()
    for pr in pr_samples(100):
        time_from = datetime.now(tz=timezone.utc) - timedelta(days=10000)
        for attr in peak_attr.split(","):
            time_from = max(getattr(pr, attr).best, time_from)
        time_from += timedelta(days=1)
        time_to = time_from + timedelta(days=7)
        assert calc.analyze(pr, time_from, time_to) is None

        time_from = datetime.now(tz=timezone.utc)
        for attr in peak_attr.split(","):
            time_from = min(getattr(pr, attr).best, time_from)
        time_from -= timedelta(days=7)
        time_to = time_from + timedelta(days=1)
        assert calc.analyze(pr, time_from, time_to) is None


@pytest.mark.parametrize("cls", [OpenedCalculator, MergedCalculator, ClosedCalculator])
def test_pull_request_metrics_float_binned(pr_samples, cls):  # noqa: F811
    time_from = (datetime.now(tz=timezone.utc) - timedelta(days=365 * 3 // 2)).date()
    time_to = (datetime.now(tz=timezone.utc) - timedelta(days=365 // 2)).date()
    time_intervals = [datetime.combine(i, datetime.min.time(), tzinfo=timezone.utc)
                      for i in Granularity.split("month", time_from, time_to)]
    binned = BinnedPullRequestMetricCalculator([cls()], time_intervals)
    result = binned(pr_samples(1000))
    # the last interval is null and that's intended
    for i, m in enumerate(result[:-1]):
        assert m[0].exists, str(i)
        assert m[0].value > 1, str(i)
        assert m[0].confidence_min is None, str(i)
        assert m[0].confidence_max is None, str(i)


def test_pull_request_opened_no(pr_samples):  # noqa: F811
    calc = OpenedCalculator()
    time_to = datetime.now(tz=timezone.utc)
    time_from = time_to - timedelta(days=180)
    n = 0
    for pr in pr_samples(100):
        if pr.closed and pr.closed.best < time_to:
            n += 1
            calc(pr, time_from, time_to)
    assert n > 0
    m = calc.value()
    assert not m.exists


def test_pull_request_closed_no(pr_samples):  # noqa: F811
    calc = ClosedCalculator()
    time_from = datetime.now(tz=timezone.utc) - timedelta(days=365 * 3)
    time_to = time_from + timedelta(days=7)
    for pr in pr_samples(100):
        calc(pr, time_from, time_to)
    m = calc.value()
    assert not m.exists


def test_pull_request_flow_ratio(pr_samples):  # noqa: F811
    calc = FlowRatioCalculator()
    open_calc = OpenedCalculator()
    closed_calc = ClosedCalculator()
    time_from = datetime.now(tz=timezone.utc) - timedelta(days=365)
    time_to = datetime.now(tz=timezone.utc)
    for pr in pr_samples(1000):
        calc(pr, time_from, time_to)
        open_calc(pr, time_from, time_to)
        closed_calc(pr, time_from, time_to)
    m = calc.value()
    assert m.exists
    assert 0 < m.value < 1
    assert m.confidence_min is None
    assert m.confidence_max is None
    assert m.value == open_calc.value().value / closed_calc.value().value


def test_pull_request_flow_ratio_no_closed():
    calc = FlowRatioCalculator()
    m = calc.value()
    assert not m.exists


def test_pull_request_flow_ratio_no_opened(pr_samples):  # noqa: F811
    calc = FlowRatioCalculator()
    time_to = datetime.now(tz=timezone.utc)
    time_from = time_to - timedelta(days=180)
    for pr in pr_samples(100):
        if pr.closed and pr.closed.best < time_to:
            calc(pr, time_from, time_to)
    m = calc.value()
    assert m.exists
    assert m.value == 0


@pytest.mark.parametrize("cls",
                         [WorkInProgressCounter,
                          ReviewCounter,
                          MergingCounter,
                          ReleaseCounter,
                          LeadCounter,
                          CycleCounter,
                          AllCounter,
                          ])
def test_pull_request_metrics_counts(pr_samples, cls):  # noqa: F811
    calc = cls()
    if isinstance(calc, AllCounter):
        calc.calc = calc
    nones = nonones = 0
    for pr in pr_samples(1000):
        time_to = datetime.now(tz=timezone.utc)
        time_from = time_to - timedelta(days=10000)
        delta = calc.analyze(pr, time_from, time_to)
        assert isinstance(delta, int)
        if calc.calc.analyze(pr, time_from, time_to) is not None:
            assert delta == 1
            nonones += 1
        else:
            assert delta == 0
            nones += 1
    if cls not in (WorkInProgressCounter, CycleCounter, AllCounter):
        assert nones > 0
    assert nonones > 0


@pytest.mark.parametrize("with_memcached, with_mine_cache_wipe",
                         itertools.product(*([[False, True]] * 2)))
async def test_calc_pull_request_metrics_line_github_cache(
        mdb, cache, memcached, with_memcached, release_match_setting_tag, with_mine_cache_wipe):
    if with_memcached:
        if not has_memcached:
            raise pytest.skip("no memcached")
        cache = memcached
    date_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2019, month=10, day=1, tzinfo=timezone.utc)
    args = ([PullRequestMetricID.PR_CYCLE_TIME], [[date_from, date_to]],
            ["src-d/go-git"], release_match_setting_tag, [], mdb, cache)
    metrics1 = (await calc_pull_request_metrics_line_github(*args))[0][0][0]
    assert await calc_pull_request_metrics_line_github.reset_cache(*args)
    if with_mine_cache_wipe:
        assert await PullRequestMiner._mine.reset_cache(
            None, date_from, date_to, ["src-d/go-git"], release_match_setting_tag, [], mdb, cache)
    metrics2 = (await calc_pull_request_metrics_line_github(*args))[0][0][0]
    assert metrics1.exists and metrics2.exists
    assert metrics1.value == metrics2.value
    assert metrics1.confidence_score() == metrics2.confidence_score()
    assert metrics1.confidence_min < metrics1.value < metrics1.confidence_max


async def test_calc_pull_request_metrics_line_github_changed_releases(
        mdb, cache, release_match_setting_tag):
    date_from = datetime(year=2017, month=1, day=1, tzinfo=timezone.utc)
    date_to = datetime(year=2017, month=10, day=1, tzinfo=timezone.utc)
    args = [[PullRequestMetricID.PR_CYCLE_TIME], [[date_from, date_to]],
            ["src-d/go-git"], release_match_setting_tag, [], mdb, cache]
    metrics1 = (await calc_pull_request_metrics_line_github(*args))[0][0][0]
    release_match_setting_tag = {
        "github.com/src-d/go-git": ReleaseMatchSetting("master", ".*", Match.branch),
    }
    args[-4] = release_match_setting_tag
    metrics2 = (await calc_pull_request_metrics_line_github(*args))[0][0][0]
    assert metrics1 != metrics2


async def test_pr_list_miner_match_metrics_all_count_david_bug(mdb, release_match_setting_tag):
    time_from = datetime(year=2016, month=11, day=17, tzinfo=timezone.utc)
    time_middle = time_from + timedelta(days=14)
    time_to = datetime(year=2016, month=12, day=15, tzinfo=timezone.utc)
    metric1 = (await calc_pull_request_metrics_line_github(
        [PullRequestMetricID.PR_ALL_COUNT], [[time_from, time_middle]],
        ["src-d/go-git"], release_match_setting_tag, [], mdb, None,
    ))[0][0][0].value
    metric2 = (await calc_pull_request_metrics_line_github(
        [PullRequestMetricID.PR_ALL_COUNT], [[time_middle, time_to]],
        ["src-d/go-git"], release_match_setting_tag, [], mdb, None,
    ))[0][0][0].value
    metric1_ext, metric2_ext = (m[0].value for m in (
        await calc_pull_request_metrics_line_github(
            [PullRequestMetricID.PR_ALL_COUNT], [[time_from, time_middle, time_to]],
            ["src-d/go-git"], release_match_setting_tag, [], mdb, None,
        )
    )[0])
    assert metric1 == metric1_ext
    assert metric2 == metric2_ext
