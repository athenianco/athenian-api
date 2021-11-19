from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest

from athenian.api.controllers.features.entries import MetricEntriesCalculator
from athenian.api.controllers.features.github.check_run_metrics import \
    CheckRunHistogramCalculatorEnsemble, CheckRunMetricCalculatorEnsemble
from athenian.api.controllers.features.github.check_run_metrics_accelerated import \
    calculate_interval_intersections
from athenian.api.controllers.features.histogram import Histogram, Scale
from athenian.api.controllers.miners.filters import JIRAFilter, LabelFilter
from athenian.api.controllers.miners.github.check_run import _postprocess_check_runs, \
    _split_duplicate_check_runs
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.models.metadata.github import CheckRun
from athenian.api.models.web import CodeCheckMetricID


@pytest.fixture(scope="function")
def metrics_calculator(mdb, pdb, rdb, cache):
    return MetricEntriesCalculator(1, (6366825,), 28, mdb, pdb, rdb, cache)


@pytest.fixture(scope="function")
def metrics_calculator_force_cache(cache):
    return MetricEntriesCalculator(1, (6366825,), 28, None, None, None, cache)


@pytest.mark.parametrize("split_by_check_runs, suite_freqs, suite_sizes, metrics", [
    (True, [[[983, 399, 495, 302, 7, 12]]], [1, 2, 3, 4, 5, 6],
     [[982, 649, 333, 0, 591],
      [399, 56, 343, 0, 188],
      [495, 345, 150, 0, 314],
      [302, 133, 169, 0, 194],
      [7, 1, 6, 0, 4],
      [12, 1, 11, 0, 10]]),
    (False, [[[0]]], [], [[2197, 1185, 1012, 0, 1301]]),
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
         CodeCheckMetricID.FAILED_SUITES_COUNT, CodeCheckMetricID.CANCELLED_SUITES_COUNT,
         CodeCheckMetricID.SUITES_IN_PRS_COUNT],
        [[datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc)]],
        [0, 1], [["src-d/go-git"]], [], split_by_check_runs,
        LabelFilter.empty(), JIRAFilter.empty(),
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
    (CodeCheckMetricID.SUITE_TIME, None),
    (CodeCheckMetricID.SUITE_TIME_PER_PR, None),
    (CodeCheckMetricID.SUITES_PER_PR, 1.9682300090789795),
    (CodeCheckMetricID.SUCCESS_RATIO, 0.5393718481063843),
    (CodeCheckMetricID.PRS_WITH_CHECKS_COUNT, 661),
    (CodeCheckMetricID.FLAKY_COMMIT_CHECKS_COUNT, 0),
    (CodeCheckMetricID.PRS_MERGED_WITH_FAILED_CHECKS_COUNT, 238),
    (CodeCheckMetricID.PRS_MERGED_WITH_FAILED_CHECKS_RATIO, 0.43830570578575134),
    (CodeCheckMetricID.ROBUST_SUITE_TIME, None),
    (CodeCheckMetricID.CONCURRENCY, 1),
    (CodeCheckMetricID.CONCURRENCY_MAX, 1),
    (CodeCheckMetricID.ELAPSED_TIME_PER_CONCURRENCY, None),
    (CodeCheckMetricID.SUITE_OCCUPANCY, None),
    (CodeCheckMetricID.SUITE_CRITICAL_OCCUPANCY, None),
    (CodeCheckMetricID.SUITE_IMBALANCE, None),
])
@with_defer
async def test_check_run_metrics_blitz(metrics_calculator: MetricEntriesCalculator,
                                       metric: str,
                                       value):
    metrics, _, _ = await metrics_calculator.calc_check_run_metrics_line_github(
        [metric],
        [[datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2020, 1, 1, tzinfo=timezone.utc)]],
        [0, 1], [["src-d/go-git"]], [], False, LabelFilter.empty(), JIRAFilter.empty())
    assert metrics[0, 0, 0, 0][0][0].value == value


@with_defer
async def test_check_run_metrics_ratio_0_0(metrics_calculator: MetricEntriesCalculator):
    metrics, _, _ = await metrics_calculator.calc_check_run_metrics_line_github(
        [CodeCheckMetricID.SUCCESS_RATIO, CodeCheckMetricID.PRS_MERGED_WITH_FAILED_CHECKS_RATIO],
        [[datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2015, 2, 1, tzinfo=timezone.utc)]],
        [0, 1], [["src-d/go-git"]], [], False, LabelFilter.empty(), JIRAFilter.empty())
    assert metrics[0, 0, 0, 0][0][0].value is None


@with_defer
async def test_check_run_metrics_robust_empty(metrics_calculator: MetricEntriesCalculator):
    metrics, _, _ = await metrics_calculator.calc_check_run_metrics_line_github(
        [CodeCheckMetricID.ROBUST_SUITE_TIME],
        [[datetime(2015, 1, 1, tzinfo=timezone.utc), datetime(2017, 6, 1, tzinfo=timezone.utc)]],
        [0.8, 1], [["src-d/go-git"]], [], False, LabelFilter.empty(), JIRAFilter.empty())
    assert metrics[0, 0, 0, 0][0][0].value is None


@with_defer
async def test_check_run_metrics_robust_quantiles(alternative_facts):
    cls = CheckRunMetricCalculatorEnsemble(CodeCheckMetricID.ROBUST_SUITE_TIME,
                                           quantiles=(0.8, 1), quantile_stride=30)
    cls(alternative_facts,
        np.array([datetime(2021, 1, 1), datetime(2021, 6, 1)], dtype="datetime64[ns]"),
        np.array([datetime(2021, 6, 1), datetime(2021, 11, 1)], dtype="datetime64[ns]"),
        [np.arange(len(alternative_facts))],
        )
    values = cls.values()[CodeCheckMetricID.ROBUST_SUITE_TIME]
    assert values[0][0].value == timedelta(seconds=1919)
    assert values[1][0].value == timedelta(seconds=18152)


ii_starts = np.array([
    100,
    200,
    150,
    175,
    125,
    50,
    200,
    300,
    50,
    0,
    150,
    175,
    200,
], dtype=np.uint64)


ii_finishes = np.array([
    200,
    250,
    300,
    225,
    250,
    150,
    300,
    400,
    250,
    300,
    350,
    200,
    450,
], dtype=np.uint64)


def test_calculate_interval_intersections_smoke():
    # 100 -> 200 ?

    # 50 -> 150
    # 150 -> 300
    # 175 -> 225
    # 125 -> 250

    # 100..125: 2 * 25
    # 125..150: 3 * 25
    # 150..175: 3 * 25
    # 175..200: 4 * 25

    # (2 + 3 + 3 + 4) * 25 / 100 = 12 * 25 / 100 = 12 / 4 = 3

    borders = np.array([6, len(ii_starts)])
    result = calculate_interval_intersections(ii_starts, ii_finishes, borders)
    assert result.tolist() == [
        3.,  # <<< checked this
        3.5,
        2.6666666666666665,
        4.,
        3.4,
        1.75,
        4.5,
        2.5,
        3.125,
        2.9166666666666665,
        3.875,
        4.,
        3.,
    ]


def test_calculate_interval_intersections_empty():
    empty = np.array([], dtype=np.uint64)
    result = calculate_interval_intersections(empty, empty, np.array([6, len(ii_starts)]))
    assert len(result) == 0
    assert result.dtype == float


def test_elapsed_time_per_concurrency_histogram():
    calc = CheckRunHistogramCalculatorEnsemble(CodeCheckMetricID.ELAPSED_TIME_PER_CONCURRENCY,
                                               quantiles=[0, 0.95])
    facts = pd.DataFrame({
        CheckRun.started_at.name: ii_starts.view("datetime64[s]").astype("datetime64[ns]"),
        CheckRun.completed_at.name: ii_finishes.view("datetime64[s]").astype("datetime64[ns]"),
        CheckRun.repository_full_name.name: "athenianco/athenian-api",
        CheckRun.name.name: ["name1"] * 6 + ["name2"] * 7,
    })
    calc(facts,
         np.array([0], dtype="datetime64[s]").astype("datetime64[ns]"),
         np.array([500], dtype="datetime64[s]").astype("datetime64[ns]"),
         [np.arange(len(facts))])
    hists = calc.histograms(None, None, None)
    assert hists == {
        CodeCheckMetricID.ELAPSED_TIME_PER_CONCURRENCY: [[
            Histogram(
                scale=Scale.LINEAR,
                bins=4,
                ticks=[1, 2, 3, 4, 5],
                frequencies=[
                    timedelta(seconds=100),
                    timedelta(seconds=250),
                    timedelta(seconds=925),
                    timedelta(seconds=175),
                ],
                interquartile=(2.9166666865348816, 3.90625)),
        ]],
    }


@pytest.fixture(scope="module")
def alternative_facts() -> pd.DataFrame:
    df = pd.read_csv(Path(__file__).parent / "check_runs.csv.gz")
    for col in (CheckRun.started_at,
                CheckRun.completed_at,
                CheckRun.pull_request_created_at,
                CheckRun.pull_request_closed_at,
                CheckRun.committed_date):
        df[col.name] = df[col.name].astype(np.datetime64)
    df = _split_duplicate_check_runs(df)
    _postprocess_check_runs(df)
    return df


@pytest.mark.parametrize("metric, value", [
    (CodeCheckMetricID.SUITE_TIME, timedelta(seconds=822)),
    (CodeCheckMetricID.SUITE_TIME_PER_PR, timedelta(seconds=822)),
    (CodeCheckMetricID.ROBUST_SUITE_TIME, timedelta(seconds=9770)),
    (CodeCheckMetricID.CONCURRENCY, 1.497145652770996),
    (CodeCheckMetricID.CONCURRENCY_MAX, 10),
    (CodeCheckMetricID.ELAPSED_TIME_PER_CONCURRENCY, None),
    (CodeCheckMetricID.SUITE_OCCUPANCY, 0.4226307272911072),
    (CodeCheckMetricID.SUITE_CRITICAL_OCCUPANCY, 0.3870483636856079),
    (CodeCheckMetricID.SUITE_IMBALANCE, timedelta(seconds=10)),
])
def test_check_run_metrics_alternative_blitz(alternative_facts, metric, value):
    cls = CheckRunMetricCalculatorEnsemble(metric, quantiles=(0, 1), quantile_stride=0)
    cls(alternative_facts,
        np.array([datetime(2021, 1, 1)], dtype="datetime64[ns]"),
        np.array([datetime(2021, 11, 1)], dtype="datetime64[ns]"),
        [np.arange(len(alternative_facts))],
        )
    assert cls.values()[metric][0][0].value == value
