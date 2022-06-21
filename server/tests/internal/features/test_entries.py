from datetime import timedelta
from functools import partial
from itertools import chain
from typing import Any
from unittest import mock

from athenian.api.db import Database
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.features.entries import (
    MetricEntriesCalculator,
    PullRequestMetricsLineRequest,
)
from athenian.api.internal.features.metric_calculator import DEFAULT_QUANTILE_STRIDE
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, Settings
from athenian.api.models.web import PullRequestMetricID
from tests.conftest import build_fake_cache
from tests.testutils.time import dt


class TestBatchCalcPullRequestMetrics:
    @with_defer
    async def test_compare_with_separate_calc(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
    ) -> None:
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        shared_kwargs = await _calc_shared_kwargs(meta_ids, mdb, sdb)

        base_kwargs = {
            "quantiles": [0, 1],
            "exclude_inactive": False,
            "bots": set(),
            "fresh": False,
            **shared_kwargs,
        }

        # the requests for the batch call
        requests = [
            PullRequestMetricsLineRequest(
                [PullRequestMetricID.PR_REVIEW_TIME],
                [[dt(2018, 1, 1), dt(2019, 9, 1)]],
                [],
                [],
                [{"src-d/go-git"}],
                [{}],
            ),
            PullRequestMetricsLineRequest(
                [PullRequestMetricID.PR_REVIEW_COUNT],
                [[dt(2017, 1, 1), dt(2017, 10, 1)]],
                [],
                [],
                [{"src-d/go-git"}],
                [{}],
            ),
            PullRequestMetricsLineRequest(
                [PullRequestMetricID.PR_SIZE],
                [[dt(2017, 8, 10), dt(2017, 8, 12)]],
                [],
                [],
                [{"src-d/go-git"}],
                [{}],
            ),
        ]

        # simulate the same call using the normal calc method, which computes
        # all the combinations of metrics and intervals
        all_metrics = list(chain.from_iterable(req.metrics for req in requests))
        all_intervals = list(chain.from_iterable(req.time_intervals for req in requests))
        calculator = MetricEntriesCalculator(
            1, meta_ids, DEFAULT_QUANTILE_STRIDE, mdb, pdb, rdb, None,
        )
        global_calc_res = await calculator.calc_pull_request_metrics_line_github(
            metrics=all_metrics,
            time_intervals=all_intervals,
            lines=[],
            environments=[],
            repositories=[{"src-d/go-git"}],
            participants=[{}],
            **base_kwargs,
        )

        await wait_deferred()
        batch_calc_res = await calculator.batch_calc_pull_request_metrics_line_github(
            requests, **base_kwargs,
        )
        batched_res_values = [req_res[0][0][0][0][0][0].value for req_res in batch_calc_res]
        for i in range(len(requests)):
            # for each batch request the same value in global result must be
            # found in interval index i and metric index i
            global_value = global_calc_res[0][0][0][i][0][i].value

            assert batched_res_values[i] == global_value

        assert batched_res_values[0] == timedelta(days=8, seconds=34252)
        assert batched_res_values[1] == 174
        assert batched_res_values[2] == 21

    @with_defer
    async def test_with_cache(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
    ) -> None:
        cache = build_fake_cache()
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        shared_kwargs = await _calc_shared_kwargs(meta_ids, mdb, sdb)
        requests = [
            PullRequestMetricsLineRequest(
                [PullRequestMetricID.PR_ALL_COUNT],
                [[dt(2019, 8, 25), dt(2019, 9, 1)]],
                [],
                [],
                [{"src-d/go-git"}],
                [{}],
            ),
        ]
        calculator = MetricEntriesCalculator(
            1, meta_ids, DEFAULT_QUANTILE_STRIDE, mdb, pdb, rdb, cache,
        )
        calc = partial(
            calculator.batch_calc_pull_request_metrics_line_github,
            requests,
            quantiles=(0, 1),
            exclude_inactive=False,
            bots=set(),
            fresh=False,
            **shared_kwargs,
        )

        calc_res = await calc()
        pr_all_count = calc_res[0][0][0][0][0][0][0].value
        await wait_deferred()

        # the second time the cache is used and the underlying calc function must not be called
        with mock.patch.object(
            calculator,
            "calc_pull_request_facts_github",
            wraps=calculator.calc_pull_request_facts_github,
        ) as calc_mock:
            second_calc_res = await calc()

        calc_mock.assert_not_called()
        assert second_calc_res[0][0][0][0][0][0][0].value == pr_all_count


async def _calc_shared_kwargs(
    meta_ids: tuple[int, ...],
    mdb: Database,
    sdb: Database,
) -> dict[str, Any]:
    prefixer = await Prefixer.load(meta_ids, mdb, None)
    settings = Settings.from_account(1, sdb, mdb, None, None)
    release_settings = await settings.list_release_matches()
    repos = release_settings.native.keys()
    branches, default_branches = await BranchMiner.extract_branches(
        repos, prefixer, meta_ids, mdb, None,
    )
    return {
        "release_settings": release_settings,
        "logical_settings": LogicalRepositorySettings.empty(),
        "prefixer": prefixer,
        "branches": branches,
        "default_branches": default_branches,
        "labels": LabelFilter.empty(),
        "jira": JIRAFilter.empty(),
    }
