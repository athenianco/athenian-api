from datetime import datetime, timedelta, timezone
import lzma
from pathlib import Path
import pickle
import warnings

import pytest

from athenian.api.controllers.features.github.pull_request import \
    BinnedPullRequestMetricCalculator, metric_calculators
import athenian.api.controllers.features.github.pull_request_metrics  # noqa
from athenian.api.models.web import PullRequestMetricID


@pytest.fixture(scope="function")
def no_warnings():
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)


def test_binned_calc_lab(benchmark, pr_samples, no_warnings):
    metric_calcs = [metric_calculators[m]() for m in PullRequestMetricID]
    ts = [datetime.now(timezone.utc) - timedelta(days=365 * 3), datetime.now(timezone.utc)]
    bin_calc = BinnedPullRequestMetricCalculator(metric_calcs, ts)
    prs = pr_samples(2000)
    benchmark(bin_calc, prs)


def test_binned_calc_es(benchmark, no_warnings):
    with lzma.open(Path(__file__).parent / "es.pickle.xz", "rb") as fin:
        mined_facts, time_intervals = pickle.load(fin)
    metrics = list(PullRequestMetricID)

    def calc():
        return [BinnedPullRequestMetricCalculator(metrics, ts)(mined_facts)
                for ts in time_intervals]

    benchmark(calc)
