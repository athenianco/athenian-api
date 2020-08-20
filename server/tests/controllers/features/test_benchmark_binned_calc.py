from datetime import datetime, timedelta, timezone
import lzma
from pathlib import Path
import pickle

from athenian.api.controllers.features.github.pull_request import \
    metric_calculators
from athenian.api.controllers.features.metric_calculator import BinnedMetricCalculator
import athenian.api.controllers.features.github.pull_request_metrics  # noqa
from athenian.api.models.web import PullRequestMetricID


def test_binned_calc_lab(benchmark, pr_samples, no_deprecation_warnings):
    metric_calcs = [metric_calculators[m]() for m in PullRequestMetricID]
    ts = [datetime.now(timezone.utc) - timedelta(days=365 * 3), datetime.now(timezone.utc)]
    bin_calc = BinnedMetricCalculator(metric_calcs, ts)
    prs = pr_samples(2000)
    benchmark(bin_calc, prs)


def test_binned_calc_es(benchmark, no_deprecation_warnings):
    with lzma.open(Path(__file__).parent / "es_bins.pickle.xz", "rb") as fin:
        mined_facts, time_intervals = pickle.load(fin)
    metrics = list(PullRequestMetricID)

    def calc():
        return [BinnedMetricCalculator(metrics, ts, quantiles=(0, 0.95))(mined_facts)
                for ts in time_intervals]

    benchmark(calc)


def test_pr_list_miner_iter(benchmark):
    with lzma.open(Path(__file__).parent / "es_list.pickle.xz", "rb") as fin:
        miner = pickle.load(fin)

    def list_prs():
        return list(miner)

    benchmark(list_prs)
