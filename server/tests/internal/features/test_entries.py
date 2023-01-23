from datetime import timedelta
from functools import partial
from itertools import chain
from typing import Any
from unittest import mock

import numpy as np
from numpy.testing import assert_array_equal

from athenian.api.db import Database
from athenian.api.defer import wait_deferred, with_defer
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.features.entries import (
    MetricEntriesCalculator,
    MetricsLineRequest,
    MinePullRequestMetrics,
    PRFactsCalculator,
    TeamSpecificFilters,
    make_calculator,
)
from athenian.api.internal.features.github.unfresh_pull_request_metrics import (
    UnfreshPullRequestFactsFetcher,
)
from athenian.api.internal.features.metric_calculator import DEFAULT_QUANTILE_STRIDE
from athenian.api.internal.jira import JIRAConfig, get_jira_installation
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.github.precomputed_prs import DonePRFactsLoader
from athenian.api.internal.miners.github.pull_request import PullRequestMiner
from athenian.api.internal.miners.github.release_mine import mine_releases
from athenian.api.internal.miners.jira.issue import PullRequestJiraMapper, fetch_jira_issues
from athenian.api.internal.miners.participation import (
    JIRAParticipationKind,
    PRParticipationKind,
    ReleaseParticipationKind,
)
from athenian.api.internal.miners.types import JIRAEntityToFetch
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
from athenian.api.models.web import JIRAMetricID, PullRequestMetricID, ReleaseMetricID
from tests.conftest import build_fake_cache
from tests.testutils.db import DBCleaner, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.common import DEFAULT_JIRA_ACCOUNT_ID, DEFAULT_MD_ACCOUNT_ID
from tests.testutils.factory.wizards import insert_repo, pr_models
from tests.testutils.time import dt


class TestMakeCalculator:
    async def test_get_calculator(self, mdb, pdb, rdb, cache):
        calc = make_calculator(1, (1,), mdb, pdb, rdb, cache)
        assert isinstance(calc, MetricEntriesCalculator)


class TestPRFactsCalculator:
    @with_defer
    async def test_gaetano_bug(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
        release_match_setting_tag: ReleaseSettings,
    ) -> None:
        meta_ids = (DEFAULT_MD_ACCOUNT_ID,)
        calculator = PRFactsCalculator(
            1, meta_ids, mdb, pdb, rdb, **self._init_kwargs(), cache=None,
        )
        metrics = MinePullRequestMetrics.empty()

        base_kwargs = await self._kwargs(meta_ids, mdb, sdb)
        base_kwargs["metrics"] = metrics
        facts = await calculator(time_from=dt(2017, 8, 10), time_to=dt(2017, 9, 1), **base_kwargs)
        assert metrics == MinePullRequestMetrics(
            count=33, done_count=29, merged_count=3, open_count=1, undead_count=1,
        )
        last_review = facts[facts.node_id == 163078].last_review.values[0]

        await calculator(time_from=dt(2017, 8, 10), time_to=dt(2017, 8, 20), **base_kwargs)
        facts = await calculator(time_from=dt(2017, 8, 10), time_to=dt(2017, 9, 1), **base_kwargs)
        assert facts[facts.node_id == 163078].last_review.values[0] == last_review

        facts = await calculator(time_from=dt(2017, 8, 10), time_to=dt(2017, 9, 1), **base_kwargs)
        assert facts[facts.node_id == 163078].last_review.values[0] == last_review

    @with_defer
    async def test_with_jira_map(
        self,
        mdb_rw: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
        release_match_setting_tag: ReleaseSettings,
    ) -> None:
        calculator = PRFactsCalculator(
            1, (DEFAULT_MD_ACCOUNT_ID,), mdb_rw, pdb, rdb, **self._init_kwargs(), cache=None,
        )

        kwargs = await self._kwargs((DEFAULT_MD_ACCOUNT_ID,), mdb_rw, sdb)
        kwargs.update(
            time_from=dt(2017, 8, 10),
            time_to=dt(2017, 9, 1),
            with_jira=JIRAEntityToFetch.EVERYTHING(),
        )
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.JIRAIssueFactory(
                    id="20", key="I1", project_id="P1", type_id="T1", priority_id=None,
                ),
                md_factory.JIRAIssueFactory(
                    id="21", key="I2", project_id="P1", type_id="T2", priority_id=None,
                ),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162990, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162990, jira_id="21"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            facts = await calculator(**kwargs)

        pr_facts = facts[facts.node_id == 162990]
        assert sorted(pr_facts.jira_ids.iloc[0]) == ["I1", "I2"]
        assert_array_equal(pr_facts.jira_projects.values[0], np.array([b"P1", b"P1"], dtype="S"))
        assert_array_equal(pr_facts.jira_types.values[0], np.array(["T1", "T2"], dtype="S"))
        assert_array_equal(pr_facts.jira_priorities.values[0], np.array(["", ""], dtype="S"))

    @with_defer
    async def test_cache_with_jira_issues(
        self,
        mdb_rw: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
        release_match_setting_tag: ReleaseSettings,
    ) -> None:
        meta_ids = (DEFAULT_MD_ACCOUNT_ID,)
        cache = build_fake_cache()
        calculator = PRFactsCalculator(
            1, meta_ids, mdb_rw, pdb, rdb, **self._init_kwargs(), cache=cache,
        )

        base_kw = await self._kwargs(meta_ids, mdb_rw, sdb)
        base_kw.update(time_from=dt(2017, 8, 10), time_to=dt(2017, 9, 1))
        base_kw.pop("with_jira")

        async with DBCleaner(mdb_rw) as mdb_cleaner:
            models = [
                md_factory.JIRAIssueFactory(id="20", key="1"),
                md_factory.JIRAIssueFactory(id="21", key="2"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162990, jira_id="20"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=162990, jira_id="21"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=163027, jira_id="20"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            # spy called load_precomputed_done_facts_filters() to verify cache hits/misses
            with mock.patch.object(
                calculator._done_prs_facts_loader,
                "load_precomputed_done_facts_filters",
                wraps=calculator._done_prs_facts_loader.load_precomputed_done_facts_filters,
            ) as load_pdb_mock:
                r_jira0 = await calculator(**base_kw, with_jira=JIRAEntityToFetch.ISSUES)
                await wait_deferred()
                assert load_pdb_mock.call_count == 1

                assert sorted(r_jira0[r_jira0.node_id == 162990].jira_ids.values[0]) == ["1", "2"]
                assert r_jira0[r_jira0.node_id == 163027].jira_ids.values[0] == ["1"]

                r_jira1 = await calculator(**base_kw, with_jira=JIRAEntityToFetch.ISSUES)
                await wait_deferred()
                assert load_pdb_mock.call_count == 1
                assert sorted(r_jira1[r_jira1.node_id == 162990].jira_ids.values[0]) == [
                    "1",
                    "2",
                ]
                assert r_jira1[r_jira1.node_id == 163027].jira_ids.values[0] == ["1"]

                await calculator(**base_kw, with_jira=JIRAEntityToFetch.NOTHING)
                assert load_pdb_mock.call_count == 1

    async def _kwargs(self, meta_ids: tuple[int, ...], mdb: Database, sdb: Database) -> dict:
        shared_kwargs = await _calc_shared_kwargs(meta_ids, mdb, sdb)
        return {
            "repositories": {"src-d/go-git"},
            "participants": {},
            "labels": LabelFilter.empty(),
            "jira": JIRAFilter.empty(),
            "exclude_inactive": False,
            "bots": set(),
            "fresh": False,
            "with_jira": JIRAEntityToFetch.NOTHING,
            **shared_kwargs,
        }

    @classmethod
    def _init_kwargs(cls, **extra) -> dict:
        return {
            "pr_miner": PullRequestMiner,
            "branch_miner": BranchMiner,
            "done_prs_facts_loader": DonePRFactsLoader,
            "unfresh_pr_facts_fetcher": UnfreshPullRequestFactsFetcher,
            "pr_jira_mapper": PullRequestJiraMapper,
            **extra,
        }


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
            "jiragroups": [],
        }
        calculator = MetricEntriesCalculator(
            1, meta_ids, DEFAULT_QUANTILE_STRIDE, mdb, pdb, rdb, cache,
        )

        metrics = [PullRequestMetricID.PR_REVIEW_TIME, PullRequestMetricID.PR_REVIEW_COUNT]

        with mock.patch.object(
            PRFactsCalculator,
            "__call__",
            side_effect=PRFactsCalculator.__call__,
            autospec=True,
        ) as calc_mock:
            res0 = await calculator.calc_pull_request_metrics_line_github(
                metrics=metrics, **kwargs,
            )
            # wait cache to be written
            await wait_deferred()
            res1 = await calculator.calc_pull_request_metrics_line_github(
                metrics=list(reversed(metrics)), **kwargs,
            )

        # result must be cached, so PRFactsCalculator.__call__ only called once
        calc_mock.assert_called_once()

        assert res1[0][0][0][0][0][0][0].value == res0[0][0][0][0][0][0][1].value
        assert res1[0][0][0][0][0][0][1].value == res0[0][0][0][0][0][0][0].value
        assert res1[0][0][0][0][1][0][0].value == res0[0][0][0][0][1][0][1].value
        assert res1[0][0][0][0][1][0][1].value == res0[0][0][0][0][1][0][0].value

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
            "jiragroups": [],
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
            PRFactsCalculator, "__call__", side_effect=PRFactsCalculator.__call__, autospec=True,
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
            assert (
                res0[0][0][0][0][0][intvl_idx][0].value == res1[0][0][0][0][0][intvl_idx][1].value
            )
            assert (
                res0[0][0][0][0][0][intvl_idx][1].value == res1[0][0][0][0][0][intvl_idx][2].value
            )
            assert (
                res0[0][0][0][0][0][intvl_idx][2].value == res1[0][0][0][0][0][intvl_idx][0].value
            )

    @with_defer
    async def test_lead_time_alias_for_pr_cycle_time(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
    ) -> None:
        meta_ids = await get_metadata_account_ids(1, sdb, None)
        shared_kwargs = await _calc_shared_kwargs(meta_ids, mdb, sdb)
        calculator = MetricEntriesCalculator(
            1, meta_ids, DEFAULT_QUANTILE_STRIDE, mdb, pdb, rdb, None,
        )
        res = await calculator.calc_pull_request_metrics_line_github(
            metrics=[PullRequestMetricID.PR_CYCLE_TIME, PullRequestMetricID.PR_LEAD_TIME],
            time_intervals=[[dt(2017, 8, 10), dt(2017, 9, 30)]],
            **self._default_kwargs,
            **shared_kwargs,
            labels=LabelFilter.empty(),
            jiragroups=[JIRAFilter.empty()],
        )
        # result for the two metrics must be the same
        assert res[0][0][0][0][0][0][0] == res[0][0][0][0][0][0][1]
        await wait_deferred()

    @with_defer
    async def test_multiple_jira_groups_projects(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
    ) -> None:
        shared_kwargs = await _calc_shared_kwargs((DEFAULT_MD_ACCOUNT_ID,), mdb, sdb)
        calculator = MetricEntriesCalculator(
            1, (DEFAULT_MD_ACCOUNT_ID,), DEFAULT_QUANTILE_STRIDE, mdb, pdb, rdb, None,
        )
        jira_filter = JIRAFilter.with_defaults(
            account=DEFAULT_JIRA_ACCOUNT_ID, projects=frozenset(["10009"]),
        )
        res = await calculator.calc_pull_request_metrics_line_github(
            metrics=[PullRequestMetricID.PR_CYCLE_TIME, PullRequestMetricID.PR_CYCLE_COUNT],
            time_intervals=[[dt(2018, 1, 1), dt(2018, 10, 1)]],
            **self._default_kwargs,
            **shared_kwargs,
            labels=LabelFilter.empty(),
            jiragroups=[JIRAFilter.empty(), jira_filter],
        )
        count_all = res[0][0][0][0][0][0][1].value
        count_proj_10009 = res[1][0][0][0][0][0][1].value
        assert count_proj_10009 < count_all

        cycle_time_all = res[0][0][0][0][0][0][0].value
        cycle_time_proj_10009 = res[1][0][0][0][0][0][0].value
        assert cycle_time_all != cycle_time_proj_10009

    @with_defer
    async def test_multiple_jira_groups_priorities(
        self,
        mdb_rw: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
    ) -> None:
        calculator = MetricEntriesCalculator(
            1, (DEFAULT_MD_ACCOUNT_ID,), DEFAULT_QUANTILE_STRIDE, mdb_rw, pdb, rdb, None,
        )
        jira_filter0 = JIRAFilter.with_defaults(
            account=DEFAULT_JIRA_ACCOUNT_ID, priorities=frozenset(["p0"]),
        )
        jira_filter1 = JIRAFilter.with_defaults(
            account=DEFAULT_JIRA_ACCOUNT_ID, priorities=frozenset(["p1"]),
        )
        async with DBCleaner(mdb_rw) as mdb_cleaner:
            repo = md_factory.RepositoryFactory(node_id=99, full_name="o/r")
            await insert_repo(repo, mdb_cleaner, mdb_rw, sdb)
            models = [
                *pr_models(99, 11, 1, repository_full_name="o/r", created_at=dt(2015, 1, 2)),
                *pr_models(99, 12, 2, repository_full_name="o/r", created_at=dt(2015, 1, 2)),
                *pr_models(99, 13, 3, repository_full_name="o/r", created_at=dt(2015, 1, 2)),
                *pr_models(99, 14, 4, repository_full_name="o/r", created_at=dt(2015, 1, 2)),
                md_factory.JIRAPriorityFactory(id="p0", name="P0"),
                md_factory.JIRAPriorityFactory(id="p1", name="P1"),
                md_factory.JIRAIssueFactory(id="20", priority_id="p0"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=11, jira_id="20"),
                md_factory.JIRAIssueFactory(id="30", priority_id="p1"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=12, jira_id="30"),
                md_factory.NodePullRequestJiraIssuesFactory(node_id=13, jira_id="30"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb_rw, *models)

            shared_kwargs = await _calc_shared_kwargs((DEFAULT_MD_ACCOUNT_ID,), mdb_rw, sdb)
            kwargs = {
                **shared_kwargs,
                **self._default_kwargs,
                "metrics": [PullRequestMetricID.PR_ALL_COUNT],
                "time_intervals": [[dt(2015, 1, 1), dt(2015, 2, 1)]],
                "labels": LabelFilter.empty(),
                "jiragroups": [jira_filter0, jira_filter1, JIRAFilter.empty()],
                "repositories": [["o/r"]],
            }

            res = await calculator.calc_pull_request_metrics_line_github(**kwargs)

        assert res[0][0][0][0][0][0][0].value == 1
        assert res[1][0][0][0][0][0][0].value == 2
        assert res[2][0][0][0][0][0][0].value == 4


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
                        1,
                        ["src-d/go-git"],
                        {PRParticipationKind.AUTHOR: {"mcuadros"}},
                        JIRAFilter.empty(),
                    ),
                ],
            ),
            MetricsLineRequest(
                [PullRequestMetricID.PR_REVIEW_COUNT],
                [[dt(2017, 1, 1), dt(2017, 10, 1)]],
                [
                    TeamSpecificFilters(
                        1,
                        ["src-d/go-git"],
                        {PRParticipationKind.AUTHOR: {"mcuadros"}},
                        JIRAFilter.empty(),
                    ),
                ],
            ),
            MetricsLineRequest(
                [PullRequestMetricID.PR_SIZE],
                [[dt(2017, 8, 10), dt(2017, 8, 12)]],
                [
                    TeamSpecificFilters(
                        1,
                        ["src-d/go-git"],
                        {PRParticipationKind.AUTHOR: {"mcuadros"}},
                        JIRAFilter.empty(),
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
            jiragroups=[JIRAFilter.empty()],
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
            global_value = global_calc_res[0][0][0][0][i][0][i].value

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
                        1,
                        ["src-d/go-git"],
                        {PRParticipationKind.AUTHOR: {"mcuadros"}},
                        JIRAFilter.empty(),
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
            PRFactsCalculator,
            "__call__",
            side_effect=PRFactsCalculator.__call__,
            autospec=True,
        ) as calc_mock:
            second_calc_res = await calc()

        calc_mock.assert_not_called()
        assert second_calc_res[0][0][0][0][0].value == pr_all_count

    @with_defer
    async def test_with_cache_threshold_comparison_metric(
        self,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        sdb: Database,
    ) -> None:
        cache = build_fake_cache()
        shared_kwargs = await _calc_shared_kwargs((DEFAULT_MD_ACCOUNT_ID,), mdb, sdb)
        requests = [
            MetricsLineRequest(
                [PullRequestMetricID.PR_REVIEW_TIME_BELOW_THRESHOLD_RATIO],
                [[dt(2017, 1, 1), dt(2018, 9, 1)]],
                [
                    TeamSpecificFilters(
                        1,
                        ["src-d/go-git"],
                        {PRParticipationKind.AUTHOR: {"mcuadros"}},
                        JIRAFilter.empty(),
                    ),
                ],
            ),
        ]
        calculator = MetricEntriesCalculator(
            1, (DEFAULT_MD_ACCOUNT_ID,), DEFAULT_QUANTILE_STRIDE, mdb, pdb, rdb, cache,
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
        ratio = calc_res[0][0][0][0][0].value
        await wait_deferred()

        # the second time the cache is used and the underlying calc function must not be called
        with mock.patch.object(
            PRFactsCalculator, "__call__", side_effect=PRFactsCalculator.__call__, autospec=True,
        ) as calc_mock:
            second_calc_res = await calc()

        calc_mock.assert_not_called()
        assert second_calc_res[0][0][0][0][0].value == ratio


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
                        1,
                        ["src-d/go-git"],
                        {ReleaseParticipationKind.COMMIT_AUTHOR: [39789]},
                        JIRAFilter.empty(),
                    ),
                ],
            ),
            MetricsLineRequest(
                metrics=[ReleaseMetricID.RELEASE_AGE],
                time_intervals=[[dt(2018, 1, 1), dt(2018, 6, 1)]],
                teams=[
                    TeamSpecificFilters(
                        1,
                        ["src-d/go-git"],
                        {ReleaseParticipationKind.COMMIT_AUTHOR: [39789]},
                        JIRAFilter.empty(),
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
                        1,
                        ["src-d/go-git"],
                        {ReleaseParticipationKind.COMMIT_AUTHOR: [39789]},
                        JIRAFilter.empty(),
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
                        JIRAFilter.empty(),
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
                        JIRAFilter.empty(),
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
                        JIRAFilter.empty(),
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
    settings = Settings.from_account(1, prefixer, sdb, mdb, None, None)
    release_settings = await settings.list_release_matches()
    repos = release_settings.native.keys()
    branches, default_branches = await BranchMiner.load_branches(
        repos, prefixer, 1, meta_ids, mdb, None, None,
    )
    return {
        "release_settings": release_settings,
        "logical_settings": LogicalRepositorySettings.empty(),
        "prefixer": prefixer,
        "branches": branches,
        "default_branches": default_branches,
    }
