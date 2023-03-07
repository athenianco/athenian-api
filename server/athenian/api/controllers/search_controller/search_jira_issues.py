from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from typing import Sequence

from aiohttp import web
import aiomcache
import numpy as np
import numpy.typing as npt
import pandas as pd

from athenian.api.async_utils import gather
from athenian.api.controllers.search_controller.common import (
    OrderBy,
    OrderByMetrics,
    OrderByValues,
    build_metrics_calculator_ensemble,
)
from athenian.api.db import Database
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.features.entries import ParticipantsMerge
from athenian.api.internal.features.jira.issue_metrics import (
    JIRAMetricCalculatorEnsemble,
    metric_calculators as jira_metric_calculators,
)
from athenian.api.internal.jira import JIRAConfig, get_jira_installation
from athenian.api.internal.miners.filters import JIRAFilter
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.jira.issue import fetch_jira_issues, resolve_work_began
from athenian.api.internal.miners.participation import JIRAParticipants
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import LogicalRepositorySettings, ReleaseSettings, Settings
from athenian.api.internal.with_ import resolve_jira_with
from athenian.api.models.metadata.jira import AthenianIssue, Issue
from athenian.api.models.web import (
    JIRAIssueDigest,
    OrderByExpression,
    SearchJIRAIssuesOrderByExpression,
    SearchJIRAIssuesOrderByIssueTrait,
    SearchJIRAIssuesRequest,
    SearchJIRAIssuesResponse,
)
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import model_response
from athenian.api.tracing import sentry_span


async def search_jira_issues(request: AthenianWebRequest, body: dict) -> web.Response:
    """Search Jira issues that satisfy the filters."""
    search_request = model_from_body(SearchJIRAIssuesRequest, body)
    account_info = await _AccountInfo.from_request(search_request.account, request)
    connectors = _Connectors(request.sdb, request.mdb, request.pdb, request.cache)
    search_filter = await _SearchFilter.build(search_request, account_info, connectors)
    digests = await _search_jira_issue_digests(
        search_filter, search_request.order_by or (), account_info, connectors,
    )
    return model_response(SearchJIRAIssuesResponse(jira_issues=digests))


@dataclass
class _Connectors:
    sdb: Database
    mdb: Database
    pdb: Database
    cache: aiomcache.Client | None


@dataclass
class _AccountInfo:
    account: int
    meta_ids: tuple[int, ...]
    jira_conf: JIRAConfig
    release_settings: ReleaseSettings
    logical_settings: LogicalRepositorySettings
    default_branches: dict[str, str]

    @classmethod
    async def from_request(cls, account: int, request: AthenianWebRequest) -> "_AccountInfo":
        sdb, pdb, mdb, cache = request.sdb, request.pdb, request.mdb, request.cache
        meta_ids = await get_metadata_account_ids(account, sdb, cache)
        prefixer = await Prefixer.load(meta_ids, mdb, cache)
        settings = Settings.from_account(account, prefixer, sdb, mdb, cache, request.app["slack"])
        release_settings, logical_settings, (_, default_branches), jira_conf = await gather(
            settings.list_release_matches(None),
            settings.list_logical_repositories(None),
            BranchMiner.load_branches(
                None, prefixer, account, meta_ids, mdb, pdb, cache, strip=True,
            ),
            get_jira_installation(account, request.sdb, request.mdb, request.cache),
        )

        return cls(
            account, meta_ids, jira_conf, release_settings, logical_settings, default_branches,
        )


@dataclass
class _SearchFilter:
    time_from: datetime | None
    time_to: datetime | None
    jira: JIRAFilter
    participants: JIRAParticipants

    @classmethod
    async def build(
        cls,
        search_req: SearchJIRAIssuesRequest,
        acc_info: _AccountInfo,
        conns: _Connectors,
    ) -> "_SearchFilter":
        time_from = time_to = None
        if from_ := search_req.date_from:
            time_from = datetime.combine(from_, time.min, tzinfo=timezone.utc)
        if to_ := search_req.date_to:
            time_to = datetime.combine(to_ + timedelta(days=1), time.min, tzinfo=timezone.utc)

        if search_req.filter:
            jira = JIRAFilter.from_web(search_req.filter, acc_info.jira_conf)
        else:
            jira = JIRAFilter.from_jira_config(acc_info.jira_conf).replace(custom_projects=False)

        if search_req.with_:
            all_participants = await resolve_jira_with(
                [search_req.with_], acc_info.account, conns.sdb, conns.mdb, conns.cache,
            )
            participants = all_participants[0]
        else:
            participants = {}

        return cls(time_from, time_to, jira, participants)


async def _search_jira_issue_digests(
    search_filter: _SearchFilter,
    order_by: Sequence[SearchJIRAIssuesOrderByExpression],
    acc_info: _AccountInfo,
    conns: _Connectors,
) -> list[JIRAIssueDigest]:
    reporters, assignees, commenters = ParticipantsMerge.jira([search_filter.participants])
    issues = await fetch_jira_issues(
        search_filter.time_from,
        search_filter.time_to,
        search_filter.jira,
        True,
        reporters,
        assignees,
        commenters,
        False,
        acc_info.default_branches,
        acc_info.release_settings,
        acc_info.logical_settings,
        acc_info.account,
        acc_info.meta_ids,
        conns.mdb,
        conns.pdb,
        conns.cache,
        extra_columns=[Issue.key],
    )
    if order_by:
        unchecked_calc_ens = _build_pr_metrics_calculator(issues, search_filter, order_by)
        issues = _apply_order_by(issues, order_by, unchecked_calc_ens)

    return [JIRAIssueDigest(id=issue_key) for issue_key in issues[Issue.key.name].values]


def _build_pr_metrics_calculator(
    issues: pd.DataFrame,
    search_filter: _SearchFilter,
    order_by: Sequence[SearchJIRAIssuesOrderByExpression],
) -> JIRAMetricCalculatorEnsemble | None:
    metrics = {expr.field for expr in order_by if expr.field in jira_metric_calculators}
    time_from, time_to = search_filter.time_from, search_filter.time_to
    CalculatorCls = JIRAMetricCalculatorEnsemble
    return build_metrics_calculator_ensemble(issues, metrics, time_from, time_to, CalculatorCls)


@sentry_span
def _apply_order_by(
    issues: pd.DataFrame,
    order_by: Sequence[SearchJIRAIssuesOrderByExpression],
    unchecked_metric_calc: JIRAMetricCalculatorEnsemble | None,
) -> pd.DataFrame:
    assert order_by
    if not len(issues):
        return issues

    keep_mask = np.full((len(issues),), True, bool)
    ordered_indexes = np.arange(len(issues))
    order_by_metrics: _OrderByJIRAMetrics | None = None
    order_by_trait: _OrderByIssueTrait | None = None

    for expr in reversed(order_by):
        orderer: OrderBy
        if expr.field in _OrderByJIRAMetrics.FIELDS:
            if order_by_metrics is None:
                assert unchecked_metric_calc is not None
                order_by_metrics = _OrderByJIRAMetrics(unchecked_metric_calc)
            orderer = order_by_metrics
        elif expr.field in _OrderByIssueTrait.FIELDS:
            if order_by_trait is None:
                order_by_trait = _OrderByIssueTrait(issues)
            orderer = order_by_trait
        else:
            raise ValueError(f"Invalid order by field {expr.field}")

        ordered_indexes, discard = orderer.apply_expression(expr, ordered_indexes)
        keep_mask[discard] = False

    kept_positions = ordered_indexes[np.flatnonzero(keep_mask[ordered_indexes])]
    return issues.take(kept_positions)


class _OrderByJIRAMetrics(OrderByMetrics):
    """Handles order by jira metric values."""

    FIELDS = jira_metric_calculators


class _OrderByIssueTrait(OrderByValues):
    """Handles order by extra jira issues traits."""

    FIELDS = [f.value for f in SearchJIRAIssuesOrderByIssueTrait]

    def __init__(self, issues: pd.DataFrame):
        self._issues = issues

    def _get_values(self, expr: OrderByExpression) -> npt.NDArray[np.datetime64]:
        if expr.field == SearchJIRAIssuesOrderByIssueTrait.CREATED.value:
            return self._issues[Issue.created.name].values
        if expr.field == SearchJIRAIssuesOrderByIssueTrait.WORK_BEGAN.value:
            return resolve_work_began(
                self._issues[AthenianIssue.work_began.name].values,
                self._issues["prs_began"].values,
            )
        raise RuntimeError(f"Cannot order by jira issue trait {expr.field}")

    @classmethod
    def _negate_values(cls, values: npt.NDArray) -> npt.NDArray:
        # datetime64 array cannot be negated
        return -(values.astype(np.int64))
