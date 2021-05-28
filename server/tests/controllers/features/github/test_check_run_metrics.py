from datetime import datetime, timedelta, timezone
from typing import List

import pytest

from athenian.api.controllers.features.entries import MetricEntriesCalculator
from athenian.api.controllers.miners.filters import JIRAFilter
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.web import CodeCheckMetricID


@pytest.fixture(scope="function")
def metrics_calculator(mdb, pdb, rdb, cache):
    return MetricEntriesCalculator(1, (6366825,), mdb, pdb, rdb, cache)


@pytest.fixture(scope="function")
def metrics_calculator_force_cache(cache):
    return MetricEntriesCalculator(1, (6366825,), None, None, None, cache)


@pytest.mark.parametrize("split_by_check_runs, suite_freqs, suite_sizes, metrics", [
    (True, [[[983, 399, 495, 302, 7, 12]]], [1, 2, 3, 4, 5, 6],
     [[983, 648, 319, 0],
      [399, 55, 343, 0],
      [495, 345, 150, 0],
      [302, 133, 169, 0],
      [7, 1, 6, 0],
      [12, 1, 11, 0]]),
    (False, [[[0]]], [], [[2198, 1183, 998, 0]]),
])
@with_defer
async def test_check_run_metrics_suite_counts(
        metrics_calculator: MetricEntriesCalculator,
        metrics_calculator_force_cache: MetricEntriesCalculator,
        split_by_check_runs: bool,
        suite_freqs: List[int],
        suite_sizes: List[int],
        metrics):
    args = [
        [CodeCheckMetricID.SUITES_COUNT, CodeCheckMetricID.SUCCESSFUL_SUITES_COUNT,
         CodeCheckMetricID.FAILED_SUITES_COUNT, CodeCheckMetricID.CANCELLED_SUITES_COUNT],
        [[datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc)]],
        [0, 1], [["src-d/go-git"]], [], split_by_check_runs, JIRAFilter.empty(),
    ]

    def check(result):
        assert result[1].tolist() == suite_freqs
        assert result[2].tolist() == suite_sizes
        assert result[0].shape == (1, 1, max(len(suite_sizes), 1), 1)
        mm = [[m.value for m in result[0][0, 0, i, 0][0]] for i in range(result[0].shape[2])]
        assert mm == metrics

    check(await metrics_calculator.calc_check_run_metrics_line_github(*args))
    await wait_deferred()
    check(await metrics_calculator_force_cache.calc_check_run_metrics_line_github(*args))


@pytest.mark.parametrize("metric, value", [
    (CodeCheckMetricID.SUITE_TIME, timedelta(0)),
    (CodeCheckMetricID.SUITE_TIME_PER_PR, timedelta(0)),
    (CodeCheckMetricID.SUITES_PER_PR, 1.9697428139183055),
    (CodeCheckMetricID.PRS_WITH_CHECKS_COUNT, 661),
    (CodeCheckMetricID.FLAKY_COMMIT_CHECKS_COUNT, 0),
    (CodeCheckMetricID.PRS_MERGED_WITH_FAILED_CHECKS_COUNT, 238),
    (CodeCheckMetricID.PRS_MERGED_WITH_FAILED_CHECKS_RATIO, 0.43933823529411764),
    (CodeCheckMetricID.ROBUST_SUITE_TIME, timedelta(0)),
    (CodeCheckMetricID.CONCURRENCY, 1.0),
    (CodeCheckMetricID.CONCURRENCY_MAX, 1),
])
@with_defer
async def test_check_run_metrics_blitz(metrics_calculator: MetricEntriesCalculator,
                                       metric: str,
                                       value):
    metrics, _, _ = await metrics_calculator.calc_check_run_metrics_line_github(
        [metric],
        [[datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc)]],
        [0, 1], [["src-d/go-git"]], [], False, JIRAFilter.empty())
    assert metrics[0, 0, 0, 0][0][0].value == value


@with_defer
async def test_check_run_metrics_robust_empty(metrics_calculator: MetricEntriesCalculator):
    metrics, _, _ = await metrics_calculator.calc_check_run_metrics_line_github(
        [CodeCheckMetricID.ROBUST_SUITE_TIME],
        [[datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2017, 6, 1, tzinfo=timezone.utc)]],
        [0.8, 1], [["src-d/go-git"]], [], False, JIRAFilter.empty())
    assert metrics[0, 0, 0, 0][0][0].value is None


@with_defer
async def test_check_run_metrics_robust_quantiles(metrics_calculator: MetricEntriesCalculator):
    metrics, _, _ = await metrics_calculator.calc_check_run_metrics_line_github(
        [CodeCheckMetricID.ROBUST_SUITE_TIME],
        [[datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2017, 6, 1, tzinfo=timezone.utc),
         datetime(2020, 1, 1, tzinfo=timezone.utc)]],
        [0.8, 1], [["src-d/go-git"]], [], False, JIRAFilter.empty())
    assert metrics[0, 0, 0, 0][0][0].value == timedelta(seconds=2)
    assert metrics[0, 0, 0, 0][1][0].value == timedelta(seconds=2)
