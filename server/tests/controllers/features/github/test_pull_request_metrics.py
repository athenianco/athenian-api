from datetime import datetime, timedelta, timezone
import itertools

import numpy as np
import pandas as pd
import pytest

from athenian.api.controllers.features.github.pull_request import BinnedPullRequestMetricCalculator
from athenian.api.controllers.features.github.pull_request_metrics import ClosedCalculator, \
    FlowRatioCalculator, LeadCounter, LeadTimeCalculator, MergedCalculator, MergingCounter, \
    MergingTimeCalculator, OpenedCalculator, ReleaseCounter, ReleaseTimeCalculator, \
    ReviewCounter, ReviewTimeCalculator, WaitFirstReviewTimeCalculator, WorkInProgressCounter, \
    WorkInProgressTimeCalculator
from athenian.api.controllers.miners.github.pull_request import Fallback, PullRequestTimes
from athenian.api.models.web import Granularity
from tests.controllers.features.github.test_pull_request import ensure_dtype, pr_samples  # noqa


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
    time_from = datetime.now(tz=timezone.utc) - timedelta(days=365 * 3 // 2)
    time_to = datetime.now(tz=timezone.utc) - timedelta(days=365 // 2)
    time_intervals = Granularity.split("month", time_from, time_to)
    binned = BinnedPullRequestMetricCalculator([cls()], time_intervals)
    result = binned(pr_samples(1000))
    for m in result:
        assert m[0].exists
        assert m[0].value > 1
        assert m[0].confidence_min is None
        assert m[0].confidence_max is None


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
    time_from = datetime.now(tz=timezone.utc) - timedelta(days=365)
    time_to = datetime.now(tz=timezone.utc)
    for pr in pr_samples(1000):
        calc(pr, time_from, time_to)
    m = calc.value()
    assert m.exists
    assert 0 < m.value < 1
    assert m.confidence_min is None
    assert m.confidence_max is None


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
                          ])
def test_pull_request_metrics_counts(pr_samples, cls):  # noqa: F811
    calc = cls()
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
    if cls is not WorkInProgressCounter:
        assert nones > 0
    assert nonones > 0
