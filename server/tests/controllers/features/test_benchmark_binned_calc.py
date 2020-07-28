from datetime import datetime, timedelta, timezone
import warnings

from athenian.api.controllers.features.github.pull_request import \
    BinnedPullRequestMetricCalculator, metric_calculators
import athenian.api.controllers.features.github.pull_request_metrics  # noqa
from athenian.api.models.web import PullRequestMetricID


def calculate_binned_metrics(calc, prs):
    calc(prs)


def test_binned_calc(benchmark, pr_samples):
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
    metric_calcs = [metric_calculators[m]() for m in PullRequestMetricID]
    ts = [datetime.now(timezone.utc) - timedelta(days=365 * 3), datetime.now(timezone.utc)]
    bin_calc = BinnedPullRequestMetricCalculator(metric_calcs, ts)
    prs = pr_samples(2000)
    benchmark(bin_calc, prs)
