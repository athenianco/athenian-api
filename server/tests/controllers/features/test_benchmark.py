from datetime import datetime, timedelta, timezone
import lzma
from pathlib import Path
import pickle

from athenian.api.internal.features.github.pull_request_metrics import metric_calculators
from athenian.api.internal.features.github.pull_request_metrics import \
    PullRequestBinnedMetricCalculator
from athenian.api.models.web import PullRequestMetricID


def test_binned_calc_lab(benchmark, pr_samples, no_deprecation_warnings):
    metric_calcs = [metric_calculators[m](quantiles=[0, 0.95]) for m in PullRequestMetricID]
    ts = [datetime.now(timezone.utc) - timedelta(days=365 * 3), datetime.now(timezone.utc)]
    bin_calc = PullRequestBinnedMetricCalculator(metric_calcs, ts)
    prs = pr_samples(2000)
    benchmark(bin_calc, prs)


def test_binned_calc_client_data(benchmark, no_deprecation_warnings):
    with lzma.open(Path(__file__).parent / "client_data.pickle.xz", "rb") as fin:
        calc, df, time_intervals, groups = pickle.load(fin)

    benchmark(calc, df, time_intervals, groups)
