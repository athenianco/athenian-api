from __future__ import annotations

import abc
import asyncio
from collections.abc import Collection
import dataclasses
from datetime import datetime
import logging
from typing import Optional, Sequence

from aiohttp import web
import aiomcache
import medvedi as md
import numpy as np
from numpy import typing as npt
import sentry_sdk

from athenian.api.async_utils import gather
from athenian.api.controllers.search_controller.common import (
    OrderBy,
    OrderByMetrics,
    OrderByValues,
    build_metrics_calculator_ensemble,
)
from athenian.api.db import Database
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.datetime_utils import closed_dates_interval_to_datetimes
from athenian.api.internal.features.entries import PRFactsCalculator
from athenian.api.internal.features.github.pull_request_filter import (
    PullRequestListMiner,
    pr_facts_stages_masks,
    pr_stages_mask,
)
from athenian.api.internal.features.github.pull_request_metrics import (
    PullRequestMetricCalculatorEnsemble,
    metric_calculators as pr_metric_calculators,
)
from athenian.api.internal.jira import get_jira_installation
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.bots import bots
from athenian.api.internal.miners.github.pull_request import fetch_prs_numbers
from athenian.api.internal.miners.participation import PRParticipants, PRParticipationKind
from athenian.api.internal.miners.types import JIRAEntityToFetch, PullRequestFacts
from athenian.api.internal.prefixer import Prefixer, RepositoryName
from athenian.api.internal.reposet import resolve_repos_with_request
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
from athenian.api.internal.with_ import resolve_withgroups
from athenian.api.models.metadata.github import NodePullRequest
from athenian.api.models.web import (
    FilterOperator,
    OrderByExpression,
    PullRequestDigest,
    SearchPullRequestsFilter,
    SearchPullRequestsOrderByExpression,
    SearchPullRequestsOrderByPRTrait,
    SearchPullRequestsOrderByStageTiming,
    SearchPullRequestsRequest,
    SearchPullRequestsResponse,
)
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import model_response
from athenian.api.tracing import sentry_span
from athenian.api.unordered_unique import map_array_values

log = logging.getLogger(__name__)


async def search_prs(request: AthenianWebRequest, body: dict) -> web.Response:
    """Search pull requests that satisfy the query."""
    search_request = model_from_body(SearchPullRequestsRequest, body)
    connectors = _SearchPRsConnectors(request.mdb, request.pdb, request.rdb, request.cache)
    account_info = await _build_account_info(search_request.account, request)
    search_filter = await _build_filter(search_request, account_info, request)
    repos_settings = await _build_repos_settings(account_info, search_filter.repositories)
    pr_digests = await _search_pr_digests(
        search_filter, search_request.order_by or (), account_info, repos_settings, connectors,
    )
    return model_response(SearchPullRequestsResponse(pull_requests=pr_digests))


@sentry_span
async def _build_account_info(account: int, request: AthenianWebRequest) -> _SearchPRsAccountInfo:
    meta_ids = await get_metadata_account_ids(account, request.sdb, request.cache)
    prefixer = await Prefixer.load(meta_ids, request.mdb, request.cache)
    settings = Settings.from_account(
        account, prefixer, request.sdb, request.mdb, request.cache, request.app["slack"],
    )
    account_bots = await bots(account, meta_ids, request.mdb, request.sdb, request.cache)
    return _SearchPRsAccountInfo(account, meta_ids, prefixer, account_bots, settings)


@sentry_span
async def _build_repos_settings(
    account_info: _SearchPRsAccountInfo,
    repos: Collection[RepositoryName] | None,
) -> _SearchPRsReposSettings:
    repos_prefixed = None if repos is None else [str(r) for r in repos]
    release_settings, logical_settings = await gather(
        account_info.settings.list_release_matches(repos_prefixed),
        account_info.settings.list_logical_repositories(repos_prefixed),
    )
    return _SearchPRsReposSettings(release_settings, logical_settings)


@sentry_span
async def _build_filter(
    search_req: SearchPullRequestsRequest,
    account_info: _SearchPRsAccountInfo,
    request: AthenianWebRequest,
) -> _SearchPRsFilter:
    dt_from, dt_to = closed_dates_interval_to_datetimes(search_req.date_from, search_req.date_to)

    async def _resolve_repos() -> Collection[RepositoryName] | None:
        if search_req.repositories is None:
            return None
        repos, _ = await resolve_repos_with_request(
            search_req.repositories,
            account_info.account,
            request,
            meta_ids=account_info.meta_ids,
            prefixer=account_info.prefixer,
            pointer=".repositories",
        )
        return repos

    async def _resolve_participants() -> PRParticipants | None:
        if search_req.participants is None:
            return None
        groups = await resolve_withgroups(
            [search_req.participants],
            PRParticipationKind,
            False,
            account_info.account,
            None,
            ".with",
            account_info.prefixer,
            request.sdb,
            group_type=set,
        )
        return groups[0] if groups else {}

    async def _resolve_jira() -> JIRAFilter | None:
        if not search_req.jira:
            return None
        jira_conf = await get_jira_installation(
            account_info.account, request.sdb, request.mdb, request.cache,
        )
        return JIRAFilter.from_web(search_req.jira, jira_conf)

    repositories, participants, jira = await gather(
        _resolve_repos(), _resolve_participants(), _resolve_jira(),
    )

    if filters := search_req.filters:
        filters = [filt.with_converted_value() for filt in filters]

    return _SearchPRsFilter(
        dt_from, dt_to, repositories, participants, jira, search_req.stages, filters,
    )


@dataclasses.dataclass
class _SearchPRsFilter:
    time_from: datetime
    time_to: datetime
    repositories: Collection[RepositoryName] | None = None
    participants: PRParticipants | None = None
    jira: Optional[JIRAFilter] = None
    stages: Optional[list[str]] = None
    extra_filters: Optional[list[SearchPullRequestsFilter]] = None


@dataclasses.dataclass
class _SearchPRsAccountInfo:
    account: int
    meta_ids: tuple[int, ...]
    prefixer: Prefixer
    bots: set[str]
    settings: Settings


@dataclasses.dataclass
class _SearchPRsReposSettings:
    release_settings: ReleaseSettings
    logical_settings: LogicalRepositorySettings


@dataclasses.dataclass
class _SearchPRsConnectors:
    mdb: Database
    pdb: Database
    rdb: Database
    cache: Optional[aiomcache.Client]


@sentry_span
async def _search_pr_digests(
    search_filter: _SearchPRsFilter,
    order_by: Sequence[SearchPullRequestsOrderByExpression],
    account_info: _SearchPRsAccountInfo,
    repos_settings: _SearchPRsReposSettings,
    connectors: _SearchPRsConnectors,
) -> list[PullRequestDigest]:
    mdb, pdb, rdb, cache = connectors.mdb, connectors.pdb, connectors.rdb, connectors.cache
    if search_filter.repositories is None:
        repos = set(repos_settings.release_settings.native.keys())
    else:
        repos = {rname.unprefixed for rname in search_filter.repositories}

    pr_numbers_task = None

    def schedule_fetch_pr_numbers(node_ids: Collection[int]) -> None:
        nonlocal pr_numbers_task
        pr_numbers_task = asyncio.create_task(
            fetch_prs_numbers(node_ids, account_info.meta_ids, mdb),
            name=f"_search_pr_digests/fetch_prs_numbers({len(node_ids)})",
        )

    calc = PRFactsCalculator(
        account_info.account, account_info.meta_ids, mdb, pdb, rdb, cache=cache,
    )
    pr_facts = await calc(
        search_filter.time_from,
        search_filter.time_to,
        repos,
        search_filter.participants or {},
        LabelFilter.empty(),
        search_filter.jira or JIRAFilter.empty(),
        exclude_inactive=True,
        bots=account_info.bots,
        release_settings=repos_settings.release_settings,
        logical_settings=repos_settings.logical_settings,
        prefixer=account_info.prefixer,
        fresh=False,
        with_jira=JIRAEntityToFetch.NOTHING,
        on_prs_known=schedule_fetch_pr_numbers,
    )

    if search_filter.stages is not None:
        pr_facts = _apply_stages_filter(pr_facts, search_filter.stages)

    unchecked_calc_ens = _build_pr_metrics_calculator(pr_facts, search_filter, order_by)

    keep_mask = _apply_extra_filters(
        pr_facts, search_filter.extra_filters or (), unchecked_calc_ens,
    )

    if order_by:
        pr_facts = _apply_order_by(pr_facts, keep_mask, order_by, unchecked_calc_ens)
    else:
        pr_facts.take(keep_mask, inplace=True)

    assert isinstance(pr_numbers_task, asyncio.Task)
    prs_numbers = _align_pr_numbers_to_ids(
        await pr_numbers_task, pr_facts[PullRequestFacts.f.node_id],
    )
    known_mask = prs_numbers != 0

    prefix_logical_repo = account_info.prefixer.prefix_logical_repo
    repo_mapping = {r: prefix_logical_repo(r) for r in repos}
    with sentry_sdk.start_span(op="materialize models", description=str(len(pr_facts))):
        pr_digests = [
            PullRequestDigest(number=number, repository=repo_mapping[repository_full_name])
            for node_id, repository_full_name, number in zip(
                pr_facts[PullRequestFacts.f.node_id][known_mask],
                pr_facts[PullRequestFacts.f.repository_full_name][known_mask],
                prs_numbers[known_mask],
            )
        ]

    unknown_prs = pr_facts[PullRequestFacts.f.node_id][~known_mask]
    if len(unknown_prs):
        log.error(
            "Cannot fetch PR numbers, probably missing entries in node_pullrequest table; PR node"
            " IDs: %s",
            ",".join(map(str, unknown_prs)),
        )
    return pr_digests


def _align_pr_numbers_to_ids(df: md.DataFrame, node_ids: npt.NDArray[int]) -> npt.NDArray[int]:
    """Project each PR node ID to PR number mapped in `df`.

    The result array will have the same length of input array: unmatched PR will
    have `0` as number in the result.
    """
    if not len(df):
        return np.zeros(len(node_ids), dtype=int)

    unsorted_db_node_ids = df[NodePullRequest.node_id.name]
    db_node_ids_sorted_indexes = np.argsort(unsorted_db_node_ids)

    db_node_ids = unsorted_db_node_ids[db_node_ids_sorted_indexes]
    db_numbers = df[NodePullRequest.number.name][db_node_ids_sorted_indexes]

    return map_array_values(node_ids, db_node_ids, db_numbers, 0)


def _build_pr_metrics_calculator(
    pr_facts: md.DataFrame,
    search_filter: _SearchPRsFilter,
    order_by: Sequence[SearchPullRequestsOrderByExpression],
) -> PullRequestMetricCalculatorEnsemble | None:
    metrics = {expr.field for expr in order_by if expr.field in pr_metric_calculators}
    if filters := search_filter.extra_filters:
        metrics |= {f.field for f in filters if f.field in pr_metric_calculators}
    time_from, time_to = search_filter.time_from, search_filter.time_to
    CalculatorCls = PullRequestMetricCalculatorEnsemble
    return build_metrics_calculator_ensemble(pr_facts, metrics, time_from, time_to, CalculatorCls)


def _apply_stages_filter(pr_facts: md.DataFrame, stages: Collection[str]) -> md.DataFrame:
    masks = pr_facts_stages_masks(pr_facts)
    filter_mask = pr_stages_mask(stages)
    return pr_facts.take((masks & filter_mask).view(bool))


def _apply_extra_filters(
    pr_facts: md.DataFrame,
    filters: Sequence[SearchPullRequestsFilter],
    unchecked_metric_calc: Optional[PullRequestMetricCalculatorEnsemble],
) -> npt.NDArray[bool]:
    filter_by_metrics: _FilterByMetrics | None = None

    keep_mask = np.full((len(pr_facts),), True, bool)
    for filt in filters:
        if filt.field in pr_metric_calculators:
            if filter_by_metrics is None:
                assert unchecked_metric_calc is not None
                filter_by_metrics = _FilterByMetrics(unchecked_metric_calc)
            filter_mask = filter_by_metrics.apply(filt, pr_facts, keep_mask)
            keep_mask &= filter_mask
        else:
            raise ValueError(f"Invalid filter field {filt.field}")

    return keep_mask


class _Filter(metaclass=abc.ABCMeta):
    """Handles the extra filters of the pull requests search."""

    @abc.abstractmethod
    def apply(
        self,
        filt: SearchPullRequestsFilter,
        pr_facts: md.DataFrame,
        current_mask: npt.NDArray,
    ) -> npt.NDArray[bool]:
        """Return the mask to select pull requests satisfying the filter."""


class _FilterByMetrics(_Filter):
    """Handles the filters by metric values."""

    _FUNCS = {
        FilterOperator.LT.value: np.less,
        FilterOperator.LE.value: np.less_equal,
        FilterOperator.GT.value: np.greater,
        FilterOperator.GE.value: np.greater_equal,
        FilterOperator.EQ.value: np.equal,
    }

    def __init__(self, calc_ensemble: PullRequestMetricCalculatorEnsemble):
        self._calc_ensemble = calc_ensemble

    def apply(
        self,
        filt: SearchPullRequestsFilter,
        pr_facts: md.DataFrame,
        current_mask: npt.NDArray[bool],
    ) -> npt.NDArray[bool]:
        calc = self._calc_ensemble[filt.field][0]
        values = calc.peek[0]
        func = self._FUNCS[filt.operator]

        # avoid nat - timedelta comparison
        if values.dtype.type == np.timedelta64:
            current_mask &= ~np.isnat(values)

        res_mask = current_mask.copy()

        func(values, filt.value, where=current_mask, out=res_mask)
        return res_mask


@sentry_span
def _apply_order_by(
    pr_facts: md.DataFrame,
    keep_mask: npt.NDArray[bool],
    order_by: Sequence[SearchPullRequestsOrderByExpression],
    unchecked_metric_calc: Optional[PullRequestMetricCalculatorEnsemble],
) -> md.DataFrame:
    assert order_by
    if not len(pr_facts):
        return pr_facts

    ordered_indexes = np.arange(len(pr_facts))
    order_by_metrics: _OrderByPRMetrics | None = None
    order_by_stage_timings: _OrderByStageTimings | None = None
    order_by_traits: _OrderByTraits | None = None

    for expr in reversed(order_by):
        orderer: OrderBy
        if expr.field in _OrderByPRMetrics.FIELDS:
            if order_by_metrics is None:
                assert unchecked_metric_calc is not None
                order_by_metrics = _OrderByPRMetrics(unchecked_metric_calc)
            orderer = order_by_metrics
        elif expr.field in _OrderByStageTimings.FIELDS:
            if order_by_stage_timings is None:
                order_by_stage_timings = _OrderByStageTimings.build(pr_facts, order_by)
            orderer = order_by_stage_timings
        elif expr.field in _OrderByTraits.FIELDS:
            if order_by_traits is None:
                order_by_traits = _OrderByTraits(pr_facts)
            orderer = order_by_traits
        else:
            raise ValueError(f"Invalid order by field {expr.field}")

        ordered_indexes, discard = orderer.apply_expression(expr, ordered_indexes)
        keep_mask[discard] = False

    kept_positions = ordered_indexes[np.flatnonzero(keep_mask[ordered_indexes])]
    return pr_facts.take(kept_positions)


class _OrderByPRMetrics(OrderByMetrics):
    """Handles order by pull request metric values."""

    FIELDS = pr_metric_calculators


class _OrderByStageTimings(OrderBy):
    """Handles order by pull request stage timing."""

    FIELDS = [f.value for f in SearchPullRequestsOrderByStageTiming]

    # values here match what returned by PullRequestListMiner.calc_stage_timings
    _FIELD_TO_STAGE = {
        SearchPullRequestsOrderByStageTiming.PR_WIP_STAGE_TIMING.value: "wip",
        SearchPullRequestsOrderByStageTiming.PR_REVIEW_STAGE_TIMING.value: "review",
        SearchPullRequestsOrderByStageTiming.PR_MERGE_STAGE_TIMING.value: "merge",
        SearchPullRequestsOrderByStageTiming.PR_RELEASE_STAGE_TIMING.value: "release",
    }

    def __init__(self, stage_timings: dict[str, list[npt.NDArray]]):
        self._stage_timings = stage_timings

    @classmethod
    def build(
        cls,
        pr_facts: md.DataFrame,
        order_by: Sequence[OrderByExpression],
    ) -> _OrderByStageTimings:
        fields = [expr.field for expr in order_by if expr.field in cls.FIELDS]
        assert fields

        # PR_TOTAL_STAGE_TIMING pseudo stage needs the computation of all the other stages
        if SearchPullRequestsOrderByStageTiming.PR_TOTAL_STAGE_TIMING.value in fields:
            stages: Collection[str] = cls._FIELD_TO_STAGE.values()
        else:
            stages = [cls._FIELD_TO_STAGE[f] for f in fields]

        stage_calcs, counter_deps = PullRequestListMiner.create_stage_calcs({}, stages=stages)
        stage_timings = PullRequestListMiner.calc_stage_timings(
            pr_facts, stage_calcs, counter_deps,
        )
        return cls(stage_timings)

    def apply_expression(
        self,
        expr: OrderByExpression,
        current_indexes: npt.NDArray[int],
    ) -> tuple[npt.NDArray, npt.NDArray[int]]:
        if expr.field == SearchPullRequestsOrderByStageTiming.PR_TOTAL_STAGE_TIMING.value:
            # sum all other stages for PR_TOTAL_STAGE_TIMING
            all_stages_values = np.array(
                [t[0][current_indexes] for t in self._stage_timings.values()],
            )
            all_stages_values[np.isnat(all_stages_values)] = 0
            values = np.sum(all_stages_values, axis=0)
        else:
            stage = self._FIELD_TO_STAGE[expr.field]
            values = self._stage_timings[stage][0][current_indexes]

        nulls = values != values
        ordered_indexes = self._ordered_indexes(expr, current_indexes, values, nulls)
        discard = self._discard_mask(expr, nulls)
        return ordered_indexes, discard


class _OrderByTraits(OrderByValues):
    FIELDS = [f.value for f in SearchPullRequestsOrderByPRTrait]

    def __init__(self, pr_facts: md.DataFrame):
        self._pr_facts = pr_facts

    def _get_values(self, expr: OrderByExpression) -> npt.NDArray[np.datetime64]:
        if expr.field == SearchPullRequestsOrderByPRTrait.WORK_BEGAN.value:
            return self._pr_facts[PullRequestFacts.f.work_began]
        if expr.field == SearchPullRequestsOrderByPRTrait.FIRST_REVIEW_REQUEST.value:
            return self._pr_facts[PullRequestFacts.f.first_review_request]
        raise RuntimeError(f"Cannot order by pr trait {expr.field}")

    @classmethod
    def _negate_values(cls, values: npt.NDArray) -> npt.NDArray:
        # datetime64 array cannot be negated
        return -(values.astype(np.int64))
