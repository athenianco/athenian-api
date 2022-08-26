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
    MetricsLineRequest,
    TeamSpecificFilters,
    make_calculator,
)
from athenian.api.internal.features.metric_calculator import DEFAULT_QUANTILE_STRIDE
from athenian.api.internal.jira import JIRAConfig, get_jira_installation
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.release_mine import mine_releases
from athenian.api.internal.miners.jira.issue import fetch_jira_issues
from athenian.api.internal.miners.types import (
    JIRAParticipationKind,
    PRParticipationKind,
    ReleaseParticipationKind,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
from athenian.api.models.web import JIRAMetricID, PullRequestMetricID, ReleaseMetricID
from tests.conftest import build_fake_cache
from tests.testutils.time import dt


class TestMakeCalculator:
    async def test_get_calculator(self, mdb, pdb, rdb, cache):
        calc = make_calculator(1, (1,), mdb, pdb, rdb, cache)
        assert isinstance(calc, MetricEntriesCalculator)


class TestCalcPullRequestFactsGithub:
    @with_defer
    async def test_gaetano_bug(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
        release_match_setting_tag: ReleaseSettings,
    ) -> None:
        meta_ids = (6366825,)
        calculator = MetricEntriesCalculator(
            1, meta_ids, DEFAULT_QUANTILE_STRIDE, mdb, pdb, rdb, None,
        )
        prefixer = await Prefixer.load(meta_ids, mdb, None)
        repos = release_match_setting_tag.native.keys()
        branches, default_branches = await BranchMiner.extract_branches(
            repos, prefixer, meta_ids, mdb, None,
        )
        base_kwargs = dict(
            repositories={"src-d/go-git"},
            participants={},
            labels=LabelFilter.empty(),
            jira=JIRAFilter.empty(),
            exclude_inactive=False,
            bots=set(),
            release_settings=release_match_setting_tag,
            logical_settings=LogicalRepositorySettings.empty(),
            prefixer=prefixer,
            fresh=False,
            with_jira_map=False,
            branches=branches,
            default_branches=default_branches,
        )
        facts = await calculator.calc_pull_request_facts_github(
            time_from=dt(2017, 8, 10), time_to=dt(2017, 9, 1), **base_kwargs,
        )
        last_review = facts[facts.node_id == 163078].last_review.values[0]

        await calculator.calc_pull_request_facts_github(
            time_from=dt(2017, 8, 10), time_to=dt(2017, 8, 20), **base_kwargs,
        )
        facts = await calculator.calc_pull_request_facts_github(
            time_from=dt(2017, 8, 10), time_to=dt(2017, 9, 1), **base_kwargs,
        )
        assert facts[facts.node_id == 163078].last_review.values[0] == last_review

        facts = await calculator.calc_pull_request_facts_github(
            time_from=dt(2017, 8, 10), time_to=dt(2017, 9, 1), **base_kwargs,
        )
        assert facts[facts.node_id == 163078].last_review.values[0] == last_review


class TestCalcPullRequestMetricsLineGithub:
    _default_kwargs = {
        "quantiles": [0, 1],
        "exclude_inactive": False,
        "bots": set(),
        "fresh": False,
        "lines": [],
        "environments": [],
        "repositories": [{"src-d/go-git"}],
        "participants": [{}],
    }

    @with_defer
    async def test_cache_shared_with_different_metrics_order(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
    ) -> None:
        cache = build_fake_cache()
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        shared_kwargs = await _calc_shared_kwargs(meta_ids, mdb, sdb)
        time_intervals = [[dt(2017, 8, 10), dt(2017, 8, 12)], [dt(2017, 8, 13), dt(2017, 9, 1)]]
        kwargs = {
            "time_intervals": time_intervals,
            **self._default_kwargs,
            **shared_kwargs,
            "labels": LabelFilter.empty(),
            "jira": JIRAFilter.empty(),
        }
        calculator = MetricEntriesCalculator(
            1, meta_ids, DEFAULT_QUANTILE_STRIDE, mdb, pdb, rdb, cache,
        )

        metrics = [PullRequestMetricID.PR_REVIEW_TIME, PullRequestMetricID.PR_REVIEW_COUNT]

        with mock.patch.object(
            calculator,
            "calc_pull_request_facts_github",
            wraps=calculator.calc_pull_request_facts_github,
        ) as calc_mock:
            res0 = await calculator.calc_pull_request_metrics_line_github(
                metrics=metrics, **kwargs,
            )
            # wait cache to be written
            await wait_deferred()
            res1 = await calculator.calc_pull_request_metrics_line_github(
                metrics=list(reversed(metrics)), **kwargs,
            )

        # result must be cached, so calc_pull_request_facts_github only called once
        calc_mock.assert_called_once()

        assert res1[0][0][0][0][0][0].value == res0[0][0][0][0][0][1].value
        assert res1[0][0][0][0][0][1].value == res0[0][0][0][0][0][0].value
        assert res1[0][0][0][1][0][0].value == res0[0][0][0][1][0][1].value
        assert res1[0][0][0][1][0][1].value == res0[0][0][0][1][0][0].value

    @with_defer
    async def test_cache_different_metrics_order_multiple_intervals(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
    ) -> None:
        cache = build_fake_cache()
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        shared_kwargs = await _calc_shared_kwargs(meta_ids, mdb, sdb)
        time_intervals = [[dt(2017, 8, 10), dt(2017, 8, 12), dt(2017, 9, 1)]]
        kwargs = {
            "time_intervals": time_intervals,
            **self._default_kwargs,
            **shared_kwargs,
            "labels": LabelFilter.empty(),
            "jira": JIRAFilter.empty(),
        }
        calculator = MetricEntriesCalculator(
            1, meta_ids, DEFAULT_QUANTILE_STRIDE, mdb, pdb, rdb, cache,
        )

        metrics0 = [
            PullRequestMetricID.PR_REVIEW_TIME,
            PullRequestMetricID.PR_CLOSED,
            PullRequestMetricID.PR_REVIEW_COUNT,
        ]
        metrics1 = [
            PullRequestMetricID.PR_REVIEW_COUNT,
            PullRequestMetricID.PR_REVIEW_TIME,
            PullRequestMetricID.PR_CLOSED,
        ]

        with mock.patch.object(
            calculator,
            "calc_pull_request_facts_github",
            wraps=calculator.calc_pull_request_facts_github,
        ) as calc_mock:
            res0 = await calculator.calc_pull_request_metrics_line_github(
                metrics=metrics0, **kwargs,
            )
            await wait_deferred()
            res1 = await calculator.calc_pull_request_metrics_line_github(
                metrics=metrics1, **kwargs,
            )

        calc_mock.assert_called_once()

        for intvl_idx in range(1):
            assert res0[0][0][0][0][intvl_idx][0].value == res1[0][0][0][0][intvl_idx][1].value
            assert res0[0][0][0][0][intvl_idx][1].value == res1[0][0][0][0][intvl_idx][2].value
            assert res0[0][0][0][0][intvl_idx][2].value == res1[0][0][0][0][intvl_idx][0].value


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
            MetricsLineRequest(
                [PullRequestMetricID.PR_REVIEW_TIME],
                [[dt(2018, 1, 1), dt(2019, 9, 1)]],
                [
                    TeamSpecificFilters(
                        1, ["src-d/go-git"], {PRParticipationKind.AUTHOR: {"mcuadros"}},
                    ),
                ],
            ),
            MetricsLineRequest(
                [PullRequestMetricID.PR_REVIEW_COUNT],
                [[dt(2017, 1, 1), dt(2017, 10, 1)]],
                [
                    TeamSpecificFilters(
                        1, ["src-d/go-git"], {PRParticipationKind.AUTHOR: {"mcuadros"}},
                    ),
                ],
            ),
            MetricsLineRequest(
                [PullRequestMetricID.PR_SIZE],
                [[dt(2017, 8, 10), dt(2017, 8, 12)]],
                [
                    TeamSpecificFilters(
                        1, ["src-d/go-git"], {PRParticipationKind.AUTHOR: {"mcuadros"}},
                    ),
                ],
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
            labels=LabelFilter.empty(),
            jira=JIRAFilter.empty(),
            environments=[],
            repositories=[{"src-d/go-git"}],
            participants=[{PRParticipationKind.AUTHOR: {"mcuadros"}}],
            **base_kwargs,
        )

        await wait_deferred()
        batch_calc_res = await calculator.batch_calc_pull_request_metrics_line_github(
            requests, **base_kwargs,
        )
        batched_res_values = [req_res[0][0][0][0].value for req_res in batch_calc_res]
        for i in range(len(requests)):
            # for each batch request the same value in global result must be
            # found in interval index i and metric index i
            global_value = global_calc_res[0][0][0][i][0][i].value

            assert batched_res_values[i] == global_value

        assert batched_res_values[0] == timedelta(days=6, seconds=42664)
        assert batched_res_values[1] == 44
        assert batched_res_values[2] is None

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
            MetricsLineRequest(
                [PullRequestMetricID.PR_ALL_COUNT],
                [[dt(2019, 8, 25), dt(2019, 9, 1)]],
                [
                    TeamSpecificFilters(
                        1, ["src-d/go-git"], {PRParticipationKind.AUTHOR: {"mcuadros"}},
                    ),
                ],
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
        pr_all_count = calc_res[0][0][0][0][0].value
        await wait_deferred()

        # the second time the cache is used and the underlying calc function must not be called
        with mock.patch.object(
            calculator,
            "calc_pull_request_facts_github",
            wraps=calculator.calc_pull_request_facts_github,
        ) as calc_mock:
            second_calc_res = await calc()

        calc_mock.assert_not_called()
        assert second_calc_res[0][0][0][0][0].value == pr_all_count


class TestCalcReleaseMetricsLineGithub:
    @with_defer
    async def test_cache_shared_with_different_metrics_order(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
    ) -> None:
        cache = build_fake_cache()
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        calculator = MetricEntriesCalculator(1, meta_ids, 28, mdb, pdb, rdb, cache)
        shared_kwargs = await _calc_shared_kwargs(meta_ids, mdb, sdb)
        time_intervals = [[dt(2018, 1, 1), dt(2018, 6, 1), dt(2018, 8, 1)]]

        kwargs = {
            "time_intervals": time_intervals,
            "repositories": [["src-d/go-git"]],
            "participants": [],
            "quantiles": [0, 1],
            "labels": LabelFilter.empty(),
            "jira": JIRAFilter.empty(),
            **shared_kwargs,
        }

        metrics0 = [
            ReleaseMetricID.RELEASE_PRS,
            ReleaseMetricID.RELEASE_COUNT,
            ReleaseMetricID.RELEASE_LINES,
        ]
        metrics1 = [
            ReleaseMetricID.RELEASE_LINES,
            ReleaseMetricID.RELEASE_PRS,
            ReleaseMetricID.RELEASE_COUNT,
        ]

        with mock.patch(
            f"{MetricEntriesCalculator.__module__}.mine_releases", wraps=mine_releases,
        ) as mine_releases_mock:
            res0 = await calculator.calc_release_metrics_line_github(metrics=metrics0, **kwargs)
            await wait_deferred()

            res1 = await calculator.calc_release_metrics_line_github(metrics=metrics1, **kwargs)

        mine_releases_mock.assert_called_once()

        for intvl_idx in range(1):
            assert res0[0][0][0][0][intvl_idx][0].value == res1[0][0][0][0][intvl_idx][1].value
            assert res0[0][0][0][0][intvl_idx][1].value == res1[0][0][0][0][intvl_idx][2].value
            assert res0[0][0][0][0][intvl_idx][2].value == res1[0][0][0][0][intvl_idx][0].value


class TestBatchCalcReleaseMetrics:
    @with_defer
    async def test_compare_with_unbatched_calc(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
    ):
        meta_ids = await get_metadata_account_ids(1, sdb, None)

        requests = [
            MetricsLineRequest(
                metrics=[ReleaseMetricID.RELEASE_PRS, ReleaseMetricID.RELEASE_COUNT],
                time_intervals=[[dt(2018, 6, 12), dt(2020, 11, 11)]],
                teams=[
                    TeamSpecificFilters(
                        1, ["src-d/go-git"], {ReleaseParticipationKind.COMMIT_AUTHOR: [39789]},
                    ),
                ],
            ),
            MetricsLineRequest(
                metrics=[ReleaseMetricID.RELEASE_AGE],
                time_intervals=[[dt(2018, 1, 1), dt(2018, 6, 1)]],
                teams=[
                    TeamSpecificFilters(
                        1, ["src-d/go-git"], {ReleaseParticipationKind.COMMIT_AUTHOR: [39789]},
                    ),
                ],
            ),
        ]

        all_metrics = list(chain.from_iterable(req.metrics for req in requests))
        all_intervals = list(chain.from_iterable(req.time_intervals for req in requests))
        calculator = MetricEntriesCalculator(1, meta_ids, 28, mdb, pdb, rdb, None)

        shared_kwargs = await _calc_shared_kwargs(meta_ids, mdb, sdb)

        global_calc_res, _ = await calculator.calc_release_metrics_line_github(
            all_metrics,
            all_intervals,
            repositories=[["src-d/go-git"]],
            participants=[{ReleaseParticipationKind.COMMIT_AUTHOR: [39789]}],
            labels=LabelFilter.empty(),
            jira=JIRAFilter.empty(),
            quantiles=[0, 1],
            **shared_kwargs,
        )
        await wait_deferred()

        batched_calc_res = await calculator.batch_calc_release_metrics_line_github(
            requests, quantiles=[0, 1], **shared_kwargs,
        )

        release_prs = global_calc_res[0][0][0][0][0]
        assert release_prs.value == 131
        assert batched_calc_res[0][0][0][0][0] == release_prs

        release_count = global_calc_res[0][0][0][0][1]
        assert release_count.value == 13
        assert batched_calc_res[0][0][0][0][1] == release_count

        release_age = global_calc_res[0][0][1][0][2]
        assert release_age.value == timedelta(days=31, seconds=61494)
        assert batched_calc_res[1][0][0][0][0] == release_age

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
            MetricsLineRequest(
                metrics=[ReleaseMetricID.RELEASE_PRS, ReleaseMetricID.RELEASE_COUNT],
                time_intervals=[[dt(2019, 1, 12), dt(2019, 3, 11)]],
                teams=[
                    TeamSpecificFilters(
                        1, ["src-d/go-git"], {ReleaseParticipationKind.COMMIT_AUTHOR: [39789]},
                    ),
                ],
            ),
        ]

        calculator = MetricEntriesCalculator(1, meta_ids, 28, mdb, pdb, rdb, cache)
        first_res = await calculator.batch_calc_release_metrics_line_github(
            requests, quantiles=[0, 1], **shared_kwargs,
        )
        await wait_deferred()

        with mock.patch(
            f"{MetricEntriesCalculator.__module__}.mine_releases", wraps=mine_releases,
        ) as mine_mock:
            second_res = await calculator.batch_calc_release_metrics_line_github(
                requests, quantiles=[0, 1], **shared_kwargs,
            )

        mine_mock.assert_not_called()
        assert second_res[0][0][0][0][0].value == first_res[0][0][0][0][0].value
        assert second_res[0][0][0][0][1].value == first_res[0][0][0][0][1].value


class BaseCalcJIRAMetricsTest:
    @classmethod
    async def _base_kwargs(
        cls,
        meta_ids: tuple[int, ...],
        sdb: Database,
        mdb: Database,
    ) -> dict[str, Any]:
        jira_ids = await get_jira_installation(1, sdb, mdb, None)
        jira_config = JIRAConfig(jira_ids.acc_id, jira_ids.projects, jira_ids.epics)

        shared_kwargs = await _calc_shared_kwargs(meta_ids, mdb, sdb)
        for f in ("prefixer", "branches"):
            shared_kwargs.pop(f)

        return {
            "quantiles": [0, 1],
            "exclude_inactive": False,
            "jira_ids": jira_config,
            **shared_kwargs,
        }


class TestCalcJIRAMetricsLineGithub(BaseCalcJIRAMetricsTest):
    @with_defer
    async def test_cache_shared_with_different_metrics_order(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
    ) -> None:
        cache = build_fake_cache()
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        calculator = MetricEntriesCalculator(1, meta_ids, 28, mdb, pdb, rdb, cache)
        time_intervals = [[dt(2020, 5, 1), dt(2020, 5, 5)]]

        metrics0 = [JIRAMetricID.JIRA_RAISED, JIRAMetricID.JIRA_OPEN, JIRAMetricID.JIRA_LEAD_TIME]
        metrics1 = [JIRAMetricID.JIRA_LEAD_TIME, JIRAMetricID.JIRA_OPEN, JIRAMetricID.JIRA_RAISED]

        base_kwargs = await self._base_kwargs(meta_ids, sdb, mdb)
        base_kwargs.update(
            {
                "label_filter": LabelFilter.empty(),
                "split_by_label": False,
                "priorities": [],
                "types": [],
                "epics": [],
            },
        )
        kwargs = {"time_intervals": time_intervals, "participants": [], **base_kwargs}

        with mock.patch(
            f"{MetricEntriesCalculator.__module__}.fetch_jira_issues", wraps=fetch_jira_issues,
        ) as fetch_issues_mock:
            res0 = await calculator.calc_jira_metrics_line_github(metrics=metrics0, **kwargs)
            await wait_deferred()
            res1 = await calculator.calc_jira_metrics_line_github(metrics=metrics1, **kwargs)

        fetch_issues_mock.assert_called_once()

        assert res0[0][0][0][0][0][0].value == res1[0][0][0][0][0][2].value
        assert res0[0][0][0][0][0][1].value == res1[0][0][0][0][0][1].value
        assert res0[0][0][0][0][0][2].value == res1[0][0][0][0][0][0].value


class TestBatchCalcJIRAMetrics(BaseCalcJIRAMetricsTest):
    @with_defer
    async def test_compare_with_unbatched_calc(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
    ) -> None:
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        requests = [
            MetricsLineRequest(
                [JIRAMetricID.JIRA_OPEN],
                [[dt(2019, 1, 1), dt(2020, 1, 1)]],
                [
                    TeamSpecificFilters(
                        1,
                        ["src-d/go-git"],
                        {JIRAParticipationKind.REPORTER: ["vadim markovtsev"]},
                    ),
                ],
            ),
            MetricsLineRequest(
                [JIRAMetricID.JIRA_RESOLVED, JIRAMetricID.JIRA_LEAD_TIME],
                [[dt(2019, 1, 1), dt(2020, 1, 1)]],
                [
                    TeamSpecificFilters(
                        1,
                        ["src-d/go-git"],
                        {JIRAParticipationKind.REPORTER: ["vadim markovtsev"]},
                    ),
                ],
            ),
        ]

        base_kwargs = await self._base_kwargs(meta_ids, sdb, mdb)

        calculator = MetricEntriesCalculator(1, meta_ids, 28, mdb, pdb, rdb, None)

        all_metrics = list(chain.from_iterable(req.metrics for req in requests))
        all_intervals = list(chain.from_iterable(req.time_intervals for req in requests))
        global_calc_res, _ = await calculator.calc_jira_metrics_line_github(
            metrics=all_metrics,
            time_intervals=all_intervals,
            label_filter=LabelFilter.empty(),
            participants=[{JIRAParticipationKind.REPORTER: ["vadim markovtsev"]}],
            split_by_label=False,
            priorities=[],
            types=[],
            epics=[],
            **base_kwargs,
        )
        await wait_deferred()

        batch_res = await calculator.batch_calc_jira_metrics_line_github(requests, **base_kwargs)

        jira_open = global_calc_res[0][0][0][0][0]
        assert jira_open.value == 7
        assert batch_res[0][0][0][0][0] == jira_open

        jira_resolved = global_calc_res[0][0][1][0][1]
        assert jira_resolved.value == 2
        assert batch_res[1][0][0][0][0] == jira_resolved

        jira_lead_time = global_calc_res[0][0][1][0][2]
        assert jira_lead_time.value == timedelta(days=10, seconds=83141)
        assert batch_res[1][0][0][0][1] == jira_lead_time

    @with_defer
    async def test_with_cache(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
    ):
        cache = build_fake_cache()
        meta_ids = await get_metadata_account_ids(1, sdb, None)

        requests = [
            MetricsLineRequest(
                [JIRAMetricID.JIRA_RAISED],
                [[dt(2019, 9, 1), dt(2020, 1, 1)]],
                [
                    TeamSpecificFilters(
                        1,
                        ["src-d/go-git"],
                        {JIRAParticipationKind.REPORTER: ["vadim markovtsev"]},
                    ),
                ],
            ),
        ]
        base_kwargs = await self._base_kwargs(meta_ids, sdb, mdb)

        calculator = MetricEntriesCalculator(1, meta_ids, 28, mdb, pdb, rdb, cache)

        first_res = await calculator.batch_calc_jira_metrics_line_github(requests, **base_kwargs)
        await wait_deferred()

        with mock.patch(
            f"{MetricEntriesCalculator.__module__}.fetch_jira_issues", wraps=fetch_jira_issues,
        ) as fetch_issues_mock:
            second_res = await calculator.batch_calc_jira_metrics_line_github(
                requests, **base_kwargs,
            )

        fetch_issues_mock.assert_not_called()

        assert first_res[0][0][0][0][0].value == second_res[0][0][0][0][0].value


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
    }
