from __future__ import annotations

import abc
from collections.abc import Collection
import dataclasses
from datetime import datetime, timedelta, timezone
import logging
from typing import Optional, Sequence

from aiohttp import web
import aiomcache
import numpy as np
from numpy import typing as npt
import pandas as pd
import sentry_sdk

from athenian.api.async_utils import gather
from athenian.api.db import DatabaseLike
from athenian.api.internal.account import get_metadata_account_ids
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
from athenian.api.internal.features.metric_calculator import DEFAULT_QUANTILE_STRIDE
from athenian.api.internal.jira import get_jira_installation
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.bots import bots
from athenian.api.internal.miners.github.pull_request import fetch_prs_numbers
from athenian.api.internal.miners.types import (
    JIRAEntityToFetch,
    PRParticipants,
    PRParticipationKind,
    PullRequestFacts,
)
from athenian.api.internal.prefixer import Prefixer, RepositoryName
from athenian.api.internal.reposet import resolve_repos_with_request
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
from athenian.api.internal.with_ import resolve_withgroups
from athenian.api.models.web import (
    OrderByDirection,
    PullRequestDigest,
    SearchPullRequestsOrderByExpression,
    SearchPullRequestsOrderByPRTrait,
    SearchPullRequestsOrderByStageTiming,
    SearchPullRequestsRequest,
    SearchPullRequestsResponse,
)
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import model_response
from athenian.api.tracing import sentry_span

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
    min_time = datetime.min.time()
    time_from = datetime.combine(search_req.date_from, min_time, tzinfo=timezone.utc)
    time_to = datetime.combine(
        search_req.date_to + timedelta(days=1), min_time, tzinfo=timezone.utc,
    )

    async def _resolve_repos():
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

    async def _resolve_participants():
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

    async def _resolve_jira():
        if search_req.jira is None:
            return None
        jira_conf = await get_jira_installation(
            account_info.account, request.sdb, request.mdb, request.cache,
        )
        return JIRAFilter.from_web(search_req.jira, jira_conf)

    repositories, participants, jira = await gather(
        _resolve_repos(), _resolve_participants(), _resolve_jira(),
    )

    return _SearchPRsFilter(
        time_from, time_to, repositories, participants, jira, search_req.stages,
    )


@dataclasses.dataclass
class _SearchPRsFilter:
    time_from: datetime
    time_to: datetime
    repositories: Optional[Collection[RepositoryName]] = None
    participants: Optional[PRParticipants] = None
    jira: Optional[JIRAFilter] = None
    stages: Optional[list[str]] = None


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
    mdb: DatabaseLike
    pdb: DatabaseLike
    rdb: DatabaseLike
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
    )

    if search_filter.stages is not None:
        pr_facts = _apply_stages_filter(pr_facts, search_filter.stages)

    if order_by:
        pr_facts = _apply_order_by(pr_facts, search_filter, order_by)

    prs_numbers = await fetch_prs_numbers(
        pr_facts[PullRequestFacts.f.node_id].values, account_info.meta_ids, mdb,
    )
    known_mask = prs_numbers != 0

    prefix_logical_repo = account_info.prefixer.prefix_logical_repo
    repo_mapping = {r: prefix_logical_repo(r) for r in repos}
    with sentry_sdk.start_span(op="materialize models", description=str(len(pr_facts))):
        pr_digests = [
            PullRequestDigest(number=number, repository=repo_mapping[repository_full_name])
            for node_id, repository_full_name, number in zip(
                pr_facts[PullRequestFacts.f.node_id].values[known_mask],
                pr_facts[PullRequestFacts.f.repository_full_name].values[known_mask],
                prs_numbers[known_mask],
            )
        ]

    unknown_prs = pr_facts[PullRequestFacts.f.node_id].values[~known_mask]
    if len(unknown_prs):
        log.error(
            "Cannot fetch PR numbers, probably missing entries in node_pullrequest table; PR node"
            " IDs: %s",
            ",".join(map(str, unknown_prs)),
        )
    return pr_digests


def _apply_stages_filter(pr_facts: pd.DataFrame, stages: Collection[str]) -> pd.DataFrame:
    masks = pr_facts_stages_masks(pr_facts)
    filter_mask = pr_stages_mask(stages)
    return pr_facts.take(np.flatnonzero(masks & filter_mask))


@sentry_span
def _apply_order_by(
    pr_facts: pd.DataFrame,
    search_filter: _SearchPRsFilter,
    order_by: Sequence[SearchPullRequestsOrderByExpression],
) -> pd.DataFrame:
    assert order_by
    if not len(pr_facts):
        return pr_facts

    keep_mask = np.full((len(pr_facts),), True, bool)
    ordered_indexes = np.arange(len(pr_facts))
    order_by_metrics: Optional[_OrderByMetrics] = None
    order_by_stage_timings: Optional[_OrderByStageTimings] = None
    order_by_traits: Optional[_OrderByTraits] = None

    for expr in reversed(order_by):
        orderer: _OrderBy
        if expr.field in _OrderByMetrics.FIELDS:
            if order_by_metrics is None:
                order_by_metrics = _OrderByMetrics.build(pr_facts, search_filter, order_by)
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


class _OrderBy(metaclass=abc.ABCMeta):
    """Handles the order by expressions."""

    @abc.abstractmethod
    def apply_expression(
        self,
        expr: SearchPullRequestsOrderByExpression,
        current_indexes: npt.NDarray[int],
    ) -> tuple[npt.NDArray, npt.NDArray[int]]:
        """Parse an expression and return a tuple with ordered indexes and discard indexes.

        `current_indexes` is the current order of the pull request facts.
        returned `discard` is an array with the indexes of element in `pr_facts` to
        drop from end result due to expression.

        """

    @classmethod
    def _ordered_indexes(
        cls,
        expr: SearchPullRequestsOrderByExpression,
        ordered_indexes: npt.NDarray[int],
        values: npt.NDArray,
        nulls: npt.NDArray[int],
    ) -> npt.NDArray[int]:
        notnulls_values = values[~nulls]
        if expr.direction == OrderByDirection.DESCENDING.value:
            notnulls_values = cls._negate_values(notnulls_values)

        indexes_notnull = ordered_indexes[~nulls][np.argsort(notnulls_values, kind="stable")]

        res_parts = [indexes_notnull, ordered_indexes[nulls]]
        if expr.nulls_first:
            res_parts = res_parts[::-1]
        result = np.concatenate(res_parts)
        return result

    @classmethod
    def _discard_mask(
        cls,
        expr: SearchPullRequestsOrderByExpression,
        nulls: npt.NDArray[int],
    ) -> npt.NDArray[int]:
        if len(nulls) and expr.exclude_nulls:
            return np.flatnonzero(nulls)
        else:
            return np.array([], dtype=int)

    @classmethod
    def _negate_values(cls, values: npt.NDArray) -> npt.NDArray:
        return -values


class _OrderByMetrics(_OrderBy):
    """Handles order by pull request metric values.

    Expressions about a metric field can be fed into this object with `apply_expression`.
    """

    FIELDS = pr_metric_calculators

    def __init__(self, calc_ensemble: PullRequestMetricCalculatorEnsemble):
        self._calc_ensemble = calc_ensemble

    @classmethod
    def build(
        cls,
        pr_facts: pd.DataFrame,
        search_filter: _SearchPRsFilter,
        order_by: Sequence[SearchPullRequestsOrderByExpression],
    ) -> _OrderByMetrics:
        metrics = [expr.field for expr in order_by if expr.field in cls.FIELDS]
        assert metrics
        min_times, max_times = (
            np.array([t.replace(tzinfo=None)], dtype="datetime64[ns]")
            for t in (search_filter.time_from, search_filter.time_to)
        )
        calc_ensemble = PullRequestMetricCalculatorEnsemble(
            *metrics, quantiles=(0, 1), quantile_stride=DEFAULT_QUANTILE_STRIDE,
        )
        groups = np.full((1, len(pr_facts)), True, bool)
        calc_ensemble(pr_facts, min_times, max_times, groups)
        return cls(calc_ensemble)

    def apply_expression(
        self,
        expr: SearchPullRequestsOrderByExpression,
        current_indexes: npt.NDarray[int],
    ) -> tuple[npt.NDArray, npt.NDArray[int]]:
        calc = self._calc_ensemble[expr.field][0]
        values = calc.peek[0][current_indexes]

        if calc.has_nan:
            nulls = values != values
        else:
            nulls = values == calc.nan
        ordered_indexes = self._ordered_indexes(expr, current_indexes, values, nulls)
        discard = self._discard_mask(expr, nulls)
        return ordered_indexes, discard


class _OrderByStageTimings(_OrderBy):
    """Handles order by pull request stage timing."""

    FIELDS = [f.value for f in SearchPullRequestsOrderByStageTiming]

    # values here match what returned by PullRequestListMiner.calc_stage_timings
    _FIELD_TO_STAGE = {
        SearchPullRequestsOrderByStageTiming.PR_WIP_STAGE_TIMING.value: "wip",
        SearchPullRequestsOrderByStageTiming.PR_REVIEW_STAGE_TIMING.value: "review",
        SearchPullRequestsOrderByStageTiming.PR_MERGE_STAGE_TIMING.value: "merge",
        SearchPullRequestsOrderByStageTiming.PR_RELEASE_STAGE_TIMING.value: "release",
    }

    def __init__(self, stage_timings: dict[str, npt.NDArray]):
        self._stage_timings = stage_timings

    @classmethod
    def build(
        cls,
        pr_facts: pd.DataFrame,
        order_by: Sequence[SearchPullRequestsOrderByExpression],
    ) -> _OrderByStageTimings:
        fields = [expr.field for expr in order_by if expr.field in cls.FIELDS]
        assert fields
        stages = [cls._FIELD_TO_STAGE[f] for f in fields]
        stage_calcs, counter_deps = PullRequestListMiner.create_stage_calcs({}, stages=stages)
        stage_timings = PullRequestListMiner.calc_stage_timings(
            pr_facts, stage_calcs, counter_deps,
        )
        return cls(stage_timings)

    def apply_expression(
        self,
        expr: SearchPullRequestsOrderByExpression,
        current_indexes: npt.NDarray[int],
    ) -> tuple[npt.NDArray, npt.NDArray[int]]:
        stage = self._FIELD_TO_STAGE[expr.field]
        values = self._stage_timings[stage][0][current_indexes]
        nulls = values != values
        ordered_indexes = self._ordered_indexes(expr, current_indexes, values, nulls)
        discard = self._discard_mask(expr, nulls)
        return ordered_indexes, discard


class _OrderByTraits(_OrderBy):
    FIELDS = [f.value for f in SearchPullRequestsOrderByPRTrait]

    def __init__(self, pr_facts: pd.DataFrame):
        self._pr_facts = pr_facts

    def apply_expression(
        self,
        expr: SearchPullRequestsOrderByExpression,
        current_indexes: npt.NDarray[int],
    ) -> tuple[npt.NDArray, npt.NDArray[int]]:
        values = self._get_values(expr).copy()[current_indexes]
        nulls = values != values
        discard = self._discard_mask(expr, nulls)
        ordered_indexes = self._ordered_indexes(expr, current_indexes, values, nulls)
        return ordered_indexes, discard

    def _get_values(self, expr: SearchPullRequestsOrderByExpression) -> npt.NDArray[np.datetime64]:
        if expr.field == SearchPullRequestsOrderByPRTrait.WORK_BEGAN.value:
            return self._pr_facts[PullRequestFacts.f.work_began].values
        if expr.field == SearchPullRequestsOrderByPRTrait.FIRST_REVIEW_REQUEST.value:
            return self._pr_facts[PullRequestFacts.f.first_review_request].values
        raise RuntimeError(f"Cannot order by pr trait {expr.field}")

    @classmethod
    def _negate_values(cls, values: npt.NDArray) -> npt.NDArray:
        # datetime64 array cannot be negated
        return -(values.astype(np.int64))
