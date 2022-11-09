from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from typing import Any, Collection, Hashable, Iterable, Mapping, Optional, Sequence

import aiomcache
from ariadne import ObjectType
from graphql import GraphQLResolveInfo
from morcilla import Database
import numpy as np
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient

from athenian.api.align.goals.dates import Intervals, goal_dates_to_datetimes
from athenian.api.align.goals.dbaccess import convert_metric_params_datatypes
from athenian.api.align.models import (
    GraphQLMetricValue,
    GraphQLMetricValues,
    GraphQLTeamMetricValue,
    GraphQLTeamTree,
    MetricParamsFields,
    MetricWithParamsInputFields,
    TeamMetricParamsFields,
)
from athenian.api.align.queries.teams import build_team_tree_from_rows
from athenian.api.align.serialization import parse_metric_params
from athenian.api.async_utils import gather
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.features.entries import (
    MetricsLineRequest,
    TeamSpecificFilters,
    make_calculator,
)
from athenian.api.internal.features.github.pull_request_metrics import (
    metric_calculators as pr_metric_calculators,
)
from athenian.api.internal.features.github.release_metrics import (
    metric_calculators as release_metric_calculators,
)
from athenian.api.internal.features.jira.issue_metrics import (
    metric_calculators as jira_metric_calculators,
)
from athenian.api.internal.jira import (
    JIRAConfig,
    check_jira_installation,
    get_jira_installation_or_none,
    load_mapped_jira_users,
    normalize_issue_type,
    normalize_priority,
)
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.bots import bots
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.types import (
    JIRAParticipants,
    JIRAParticipationKind,
    PRParticipationKind,
    ReleaseParticipationKind,
)
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.reposet import resolve_repos_with_request
from athenian.api.internal.settings import Settings
from athenian.api.internal.team import fetch_teams_recursively
from athenian.api.internal.with_ import flatten_teams
from athenian.api.models.state.models import Team
from athenian.api.models.web import InvalidRequestError
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span

query = ObjectType("Query")


@query.field("metricsCurrentValues")
@sentry_span
async def resolve_metrics_current_values(
    obj: Any,
    info: GraphQLResolveInfo,
    accountId: int,
    params: Mapping[str, Any],
) -> Any:
    """Serve metricsCurrentValues()."""
    team_id = params[MetricParamsFields.teamId]
    team_rows, meta_ids, jira_config = await gather(
        fetch_teams_recursively(
            accountId,
            info.context.sdb,
            select_entities=(Team.id, Team.name, Team.members, Team.parent_id),
            root_team_ids=[team_id],
        ),
        get_metadata_account_ids(accountId, info.context.sdb, info.context.cache),
        get_jira_installation_or_none(
            accountId, info.context.sdb, info.context.mdb, info.context.cache,
        ),
    )
    time_interval = _parse_time_interval(params)
    teams_flat = flatten_teams(team_rows)

    repos = await _parse_repositories(params, accountId, meta_ids, info.context)
    jira_filter = _parse_jira_filter(params, jira_config)
    requests = []

    requested_metrics = _parse_requested_metrics(params)
    for requested_metric in requested_metrics:
        for row in team_rows:
            team_detail = RequestedTeamDetails(
                team_id=(row_team_id := row[Team.id.name]),
                goal_id=0,  # change to unique number if the filters are no longer shared
                members=teams_flat[row_team_id],
                repositories=repos,
                jira_filter=jira_filter,
            )
            requests.append(
                TeamMetricsRequest(
                    [requested_metric.resolve_for_team(row_team_id)],
                    [time_interval],
                    [team_detail],
                ),
            )

    team_metrics_all_intervals = await calculate_team_metrics(
        requests,
        accountId,
        meta_ids,
        info.context.sdb,
        info.context.mdb,
        info.context.pdb,
        info.context.rdb,
        info.context.cache,
        info.context.app["slack"],
        jira_config,
    )
    team_metrics = team_metrics_all_intervals[time_interval]

    team_tree = GraphQLTeamTree.from_team_tree(build_team_tree_from_rows(team_rows, team_id))

    models = _build_metrics_response(team_tree, requested_metrics, team_metrics)
    return [m.to_dict() for m in models]


async def _parse_repositories(
    params: Mapping[str, Any],
    account_id: int,
    meta_ids: tuple[int, ...],
    request: AthenianWebRequest,
) -> Optional[tuple[str, ...]]:
    if (repos := params.get(MetricParamsFields.repositories)) is not None:
        repos = tuple(
            r.unprefixed
            for r in (
                await resolve_repos_with_request(repos, account_id, request, meta_ids=meta_ids)
            )[0]
        )
    return repos


def _parse_jira_filter(
    params: Mapping[str, Any],
    unchecked_jira_config: Optional[JIRAConfig],
) -> JIRAFilter:
    jira_projects = frozenset(params.get(MetricParamsFields.jiraProjects) or ())
    jira_priorities = frozenset(
        normalize_priority(p) for p in params.get(MetricParamsFields.jiraPriorities) or ()
    )
    jira_issue_types = frozenset(
        normalize_issue_type(t) for t in params.get(MetricParamsFields.jiraIssueTypes) or ()
    )
    if jira_projects or jira_priorities or jira_issue_types:
        jira_config = check_jira_installation(unchecked_jira_config)
        if jira_projects:
            jira_projects = frozenset(jira_config.translate_project_keys(jira_projects))
            custom_projects = True
        else:
            jira_projects = frozenset(jira_config.projects)
            custom_projects = False
        return JIRAFilter(
            jira_config.acc_id,
            jira_projects,
            LabelFilter.empty(),
            frozenset(),
            frozenset(jira_issue_types or ()),
            frozenset(jira_priorities or ()),
            custom_projects,
            False,
        )
    else:
        return JIRAFilter.empty()


def _parse_time_interval(params: Mapping[str, Any]) -> Intervals:
    date_from, date_to = params[MetricParamsFields.validFrom], params[MetricParamsFields.expiresAt]
    if date_from > date_to:
        raise ResponseError(
            InvalidRequestError(
                pointer=f".{MetricParamsFields.validFrom} or .{MetricParamsFields.expiresAt}",
                detail=(
                    f"{MetricParamsFields.validFrom} must be less than or equal to "
                    f"{MetricParamsFields.expiresAt}"
                ),
            ),
        )
    if date_from > datetime.now(timezone.utc).date():
        raise ResponseError(
            InvalidRequestError(
                pointer=f".{MetricParamsFields.validFrom}",
                detail=f"{MetricParamsFields.validFrom} cannot be in the future",
            ),
        )
    return goal_dates_to_datetimes(date_from, date_to)


def _parse_requested_metrics(params: Mapping[str, Any]) -> list[TeamRequestedMetric]:
    metrics = params.get(MetricParamsFields.metrics)
    metrics_w_params = params.get(MetricParamsFields.metricsWithParams)

    if bool(metrics) == bool(metrics_w_params):
        fields = [MetricParamsFields.metrics, MetricParamsFields.metricsWithParams]
        raise ResponseError(
            InvalidRequestError(
                pointer=" or ".join(f".{f}" for f in fields), detail=f"Use {' or '.join(fields)}",
            ),
        )

    if metrics:
        return [TeamRequestedMetric(m, None, None) for m in metrics]

    assert metrics_w_params  # mypy
    req_metrics = []
    for m_w_params in metrics_w_params:
        name = m_w_params[MetricWithParamsInputFields.name]
        metric_params = convert_metric_params_datatypes(
            parse_metric_params(m_w_params.get(MetricWithParamsInputFields.metricParams)),
        )
        raw_teams_params = m_w_params.get(MetricWithParamsInputFields.teamsMetricParams) or ()
        team_params = {
            p[TeamMetricParamsFields.teamId]: convert_metric_params_datatypes(
                parse_metric_params(p[TeamMetricParamsFields.metricParams]),
            )
            for p in raw_teams_params
        }
        req_metrics.append(TeamRequestedMetric(name, metric_params, team_params))
    return req_metrics


class TeamRequestedMetric:
    """A metric requested, with optional params and params override per team."""

    def __init__(self, name: str, params: Optional[dict], teams_params: Optional[dict[int, dict]]):
        """Init the TeamRequestedMetric."""
        self._name = name
        self._params = params or {}
        self._teams_params = teams_params or {}

    @property
    def name(self) -> str:
        """Get the name of the metric."""
        return self._name

    def resolve_for_team(self, team_id: int) -> MetricWithParams:
        """Resolve the requested metric for a specific team."""
        params = self._teams_params.get(team_id, self._params)
        return MetricWithParams(self._name, params)


@dataclass(frozen=True, slots=True)
class RequestedTeamDetails:
    """Team/goal IDs with resolved team members and all TeamGoal-specific filters."""

    team_id: int
    goal_id: int
    members: Sequence[int]
    # filters start here
    repositories: Optional[tuple[str, ...]] = None
    jira_filter: JIRAFilter = JIRAFilter.empty()
    # add more filters here

    def __hash__(self) -> int:
        """Implement hash() on team ID + goal ID."""
        return hash((self.team_id, self.goal_id))

    def __eq__(self, other: object) -> bool:
        """Implement ==. The filters are defined by the goal ID."""
        if not isinstance(other, RequestedTeamDetails):
            return False
        return self.team_id == other.team_id and self.goal_id == other.goal_id


class MetricWithParams:
    """A metric with the optional associated parameters."""

    def __init__(self, name: str, params: Mapping[str, Hashable]):
        """Init the MetricWithParams."""
        self._name = name
        self._params = params
        params_tuple = tuple(sorted(self.params.items()))
        self._hash = hash((name, params_tuple))

    @property
    def name(self) -> str:
        """Return the name of the metric."""
        return self._name

    @property
    def params(self) -> Mapping[str, Hashable]:
        """Return the parameters for the metric."""
        return self._params

    @property
    def params_key(self) -> Hashable:
        """Return an opaque object that can be used to refer to metric params identity."""
        return MetricWithParams("", self._params)

    def __hash__(self) -> int:
        """Return the MetricWithParams hash."""
        return self._hash

    def __eq__(self, other: object) -> bool:
        """Implement == operator."""
        if not isinstance(other, MetricWithParams):
            return False
        return (self.name, self._params) == (other._name, other._params)


TeamMetricsResult = dict[Intervals, dict[MetricWithParams, dict[tuple[int, int], list[Any]]]]


@dataclass(frozen=True, slots=True)
class TeamMetricsRequest:
    """Request for multiple metrics/intervals/teams usable with calculate_team_metrics."""

    metrics: Sequence[MetricWithParams]
    time_intervals: Sequence[Intervals]
    teams: Collection[RequestedTeamDetails]


@sentry_span
async def calculate_team_metrics(
    requests: Sequence[TeamMetricsRequest],
    account: int,
    meta_ids: tuple[int, ...],
    sdb: Database,
    mdb: Database,
    pdb: Database,
    rdb: Database,
    cache: Optional[aiomcache.Client],
    slack: Optional[SlackWebClient],
    unchecked_jira_config: Optional[JIRAConfig],
) -> TeamMetricsResult:
    """Calculate a set of metrics for each team and time interval.

    The result will be a nested dict structure indexed first by intervals, then by metric
    (with parameters) and finally by (team_id, goal_id).
    Terminal values in the dict structure will be lists of metric values: a metric value will be
    found for each granularity requested in the given interval - for each list of metric
    values its length will equal len(intervals) - 1.

    For example, given the single request:
    TeamMetricsRequest(
       metrics=[MetricWithParams("pr-all-count", {})],
       time_intervals=[dt(2010, 1, 1), dt(2010, 2, 1), dt(2010, 3, 1)],
       teams=[RequestedTeamDetails(team_id=10, goal_id=20, ...)]
    )

    The response will have this shape:

    {
        (dt(2010, 1, 1), dt(2010, 2, 1), dt(2010, 3, 1)): {
          MetricWithParams("pr-all-count", {}): {
            (10, 20): [20, 30],
           }
        }
    }
    Where 20 is the metric value in the first granularity of the interval
    (dt(2010, 1, 1) - dt(2010, 2, 1)) and 30 in the second granularity
    (dt(2010, 2, 1), dt(2010, 3, 1)).

    """
    requests = _simplify_requests(requests)
    QUANTILES = (0, 0.95)
    prefixer = await Prefixer.load(meta_ids, mdb, cache)
    settings = Settings.from_account(account, prefixer, sdb, mdb, cache, slack)
    jira_map_task = asyncio.create_task(
        load_mapped_jira_users(
            account,
            set(chain.from_iterable(td.members for request in requests for td in request.teams)),
            sdb,
            mdb,
            cache,
        ),
    )
    account_bots, release_settings = await gather(
        bots(account, meta_ids, mdb, sdb, cache), settings.list_release_matches(),
    )
    all_repos = tuple(release_settings.native.keys())
    (branches, default_branches), logical_settings, _ = await gather(
        BranchMiner.extract_branches(all_repos, prefixer, meta_ids, mdb, cache),
        settings.list_logical_repositories(),
        jira_map_task,
    )
    jira_map = jira_map_task.result()

    pr_collector = BatchCalcResultCollector()
    release_collector = BatchCalcResultCollector()
    jira_collector = BatchCalcResultCollector()

    for request in requests:
        team_members = _loginify_teams([td.members for td in request.teams], prefixer)
        pr_metric_groups, release_metric_groups, jira_metric_groups = _triage_metrics(
            request.metrics,
        )
        goal_ids = [td.goal_id for td in request.teams]

        for pr_metrics in pr_metric_groups:
            pr_collector.append(
                MetricsLineRequest(
                    metrics=[m.name for m in pr_metrics],
                    # metric_params are the same for every MetricWithParams in the group,
                    # see _triage_metrics
                    metric_params=pr_metrics[0].params,
                    time_intervals=request.time_intervals,
                    teams=[
                        TeamSpecificFilters(
                            team_id=td.team_id,
                            participants={PRParticipationKind.AUTHOR: members},
                            repositories=td.repositories
                            if td.repositories is not None
                            else all_repos,
                            jira_filter=td.jira_filter,
                        )
                        for td, members in zip(request.teams, team_members)
                    ],
                ),
                goal_ids,
            )

        for release_metrics in release_metric_groups:
            release_collector.append(
                MetricsLineRequest(
                    metrics=[m.name for m in release_metrics],
                    metric_params=release_metrics[0].params,
                    time_intervals=request.time_intervals,
                    teams=[
                        TeamSpecificFilters(
                            team_id=td.team_id,
                            participants={
                                ReleaseParticipationKind.PR_AUTHOR: td.members,
                                ReleaseParticipationKind.COMMIT_AUTHOR: td.members,
                                ReleaseParticipationKind.RELEASER: td.members,
                            },
                            repositories=td.repositories
                            if td.repositories is not None
                            else all_repos,
                            jira_filter=td.jira_filter,
                        )
                        for td in request.teams
                    ],
                ),
                goal_ids,
            )

        for jira_metrics in jira_metric_groups:
            jira_collector.append(
                MetricsLineRequest(
                    metrics=[m.name for m in jira_metrics],
                    metric_params=jira_metrics[0].params,
                    time_intervals=request.time_intervals,
                    teams=[
                        TeamSpecificFilters(
                            team_id=td.team_id,
                            participants=_jirafy_team(td.members, jira_map),
                            repositories=td.repositories
                            if td.repositories is not None
                            else all_repos,
                            jira_filter=td.jira_filter,
                        )
                        for td in request.teams
                    ],
                ),
                goal_ids,
            )

    calculator = make_calculator(account, meta_ids, mdb, pdb, rdb, cache)
    tasks = []
    jira_acc_id = unchecked_jira_config.acc_id if unchecked_jira_config else None
    if pr_requests := pr_collector.requests:
        pr_task = calculator.batch_calc_pull_request_metrics_line_github(
            requests=pr_requests,
            quantiles=QUANTILES,
            exclude_inactive=True,
            bots=account_bots,
            release_settings=release_settings,
            logical_settings=logical_settings,
            prefixer=prefixer,
            branches=branches,
            default_branches=default_branches,
            fresh=False,
            jira_acc_id=jira_acc_id,
        )
        tasks.append(pr_task)

    if release_requests := release_collector.requests:
        release_task = calculator.batch_calc_release_metrics_line_github(
            requests=release_requests,
            quantiles=QUANTILES,
            release_settings=release_settings,
            logical_settings=logical_settings,
            prefixer=prefixer,
            branches=branches,
            default_branches=default_branches,
            jira_acc_id=jira_acc_id,
        )
        tasks.append(release_task)

    if jira_requests := jira_collector.requests:
        jira_conf = check_jira_installation(unchecked_jira_config)
        jira_task = calculator.batch_calc_jira_metrics_line_github(
            requests=jira_requests,
            quantiles=QUANTILES,
            exclude_inactive=True,
            release_settings=release_settings,
            logical_settings=logical_settings,
            default_branches=default_branches,
            jira_ids=jira_conf,
        )
        tasks.append(jira_task)

    calc_results = list(await gather(*tasks, op="batch_calculators"))

    team_metrics_res: TeamMetricsResult = {}
    if pr_requests:
        pr_collector.collect(calc_results.pop(0), team_metrics_res)
    if release_requests:
        release_collector.collect(calc_results.pop(0), team_metrics_res)
    if jira_requests:
        jira_collector.collect(calc_results[0], team_metrics_res)

    return team_metrics_res


def _triage_metrics(
    metrics: Sequence[MetricWithParams],
) -> tuple[
    tuple[tuple[MetricWithParams, ...], ...],
    tuple[tuple[MetricWithParams, ...], ...],
    tuple[tuple[MetricWithParams, ...], ...],
]:
    """Partition the metrics in pr, release and jira metrics.

    For each category return a list of groups.
    Metrics in each group will have the same `metric_params`, so they can be
    sent together in a `MetricsLineRequest`.
    """
    pr_metrics: dict[Hashable, set[MetricWithParams]] = {}
    release_metrics: dict[Hashable, set[MetricWithParams]] = {}
    jira_metrics: dict[Hashable, set[MetricWithParams]] = {}
    unidentified = []
    for m in metrics:
        if m.name in pr_metric_calculators:
            pr_metrics.setdefault(m.params_key, set()).add(m)
        elif m.name in release_metric_calculators:
            release_metrics.setdefault(m.params_key, set()).add(m)
        elif m.name in jira_metric_calculators:
            jira_metrics.setdefault(m.params_key, set()).add(m)
        else:
            unidentified.append(m.name)
    if unidentified:
        raise ResponseError(
            InvalidRequestError(
                pointer=f".{MetricParamsFields.metrics}",
                detail=f"The following metrics are not supported: {', '.join(unidentified)}",
            ),
        )
    return (
        tuple(tuple(m) for m in pr_metrics.values()),
        tuple(tuple(m) for m in release_metrics.values()),
        tuple(tuple(m) for m in jira_metrics.values()),
    )


def _loginify_teams(teams: Iterable[Collection[int]], prefixer: Prefixer) -> list[set[str]]:
    result = []
    user_node_to_login = prefixer.user_node_to_login.__getitem__
    for team in teams:
        logins = set()
        for node in team:
            try:
                logins.add(user_node_to_login(node))
            except KeyError:
                continue
        result.append(logins)
    return result


def _jirafy_team(team: Collection[int], jira_map: Mapping[int, str]) -> JIRAParticipants:
    result: JIRAParticipants = {JIRAParticipationKind.ASSIGNEE: (assignees := [])}
    for dev in team:
        try:
            assignees.append(jira_map[dev])
        except KeyError:
            continue
    return result


@sentry_span
def _simplify_requests(requests: Sequence[TeamMetricsRequest]) -> list[TeamMetricsRequest]:
    """Simplify the list of requests and try to group them in less requests."""
    # intervals => team and other filters => metrics with params
    requests_tree: dict[
        tuple[Intervals, ...],
        dict[RequestedTeamDetails, set[MetricWithParams]],
    ] = {}
    for req in requests:
        req_time_intervals = tuple(req.time_intervals)
        requests_tree.setdefault(req_time_intervals, {})
        for td in req.teams:
            for m in req.metrics:
                requests_tree[req_time_intervals].setdefault(td, set()).add(m)

    # intervals => metrics with params => team-specific
    simplified_tree: dict[
        Sequence[Intervals],
        dict[frozenset[MetricWithParams], set[RequestedTeamDetails]],
    ] = {}
    for intervals, intervals_tree in requests_tree.items():
        simplified_tree[intervals] = {}
        for filters, metrics in intervals_tree.items():
            simplified_tree[intervals].setdefault(frozenset(metrics), set()).add(filters)

    # assemble the final groups
    requests = [
        TeamMetricsRequest(tuple(metrics_), intervals_, grouped_filters)
        for intervals_, intervals_tree_ in simplified_tree.items()
        for metrics_, grouped_filters in intervals_tree_.items()
    ]

    return requests


class BatchCalcResultCollector:
    """Convert the results coming from a MetricEntriesCalculator batch method."""

    def __init__(self) -> None:
        """Initialize a new instance of BatchCalcResultCollector."""
        self._requests: list[MetricsLineRequest] = []
        self._goal_ids: list[Sequence[int]] = []

    def append(self, request: MetricsLineRequest, goal_ids: Sequence[int]) -> None:
        """Register another metrics calculation request."""
        self._requests.append(request)
        self._goal_ids.append(goal_ids)

    @sentry_span
    def collect(
        self,
        batch_calc_results: Sequence[np.ndarray],
        team_metrics_res: TeamMetricsResult,
    ) -> None:
        """Collect the results and merge it into TeamMetricsResult `team_metrics_res`.

        `batch_calc_results` must be obtained by calling the batch calc method with
        the requests in `get_requests()`.
        """
        for request, goal_ids, calc_result in zip(
            self._requests, self._goal_ids, batch_calc_results,
        ):
            for intervals_i, intervals in enumerate(request.time_intervals):
                team_metrics_res.setdefault(intervals, {})
                for metric_i, metric in enumerate(request.metrics):
                    metric_w_param = MetricWithParams(metric, request.metric_params)
                    team_metrics_res[intervals].setdefault(metric_w_param, {})
                    for team_i, (team_filters, goal_id) in enumerate(zip(request.teams, goal_ids)):
                        # collect a value for each granularity
                        values = [ar[metric_i].value for ar in calc_result[team_i][intervals_i]]
                        team_metrics_res[intervals][metric_w_param][
                            (team_filters.team_id, goal_id)
                        ] = values

    @property
    def requests(self) -> list[MetricsLineRequest]:
        """Return the registered metric calculation requests."""
        return self._requests


@sentry_span
def _build_metrics_response(
    team_tree: GraphQLTeamTree,
    req_metrics: Sequence[TeamRequestedMetric],
    triaged: dict[MetricWithParams, dict[tuple[int, int], list[Any]]],
) -> list[GraphQLMetricValues]:
    return [
        GraphQLMetricValues(
            metric=req_m.name, value=_build_team_metric_value(team_tree, req_m, triaged),
        )
        for req_m in req_metrics
    ]


def _build_team_metric_value(
    team_tree: GraphQLTeamTree,
    requested_metric: TeamRequestedMetric,
    metric_values: dict[MetricWithParams, dict[tuple[int, int], list[Any]]],
) -> GraphQLTeamMetricValue:
    metric_w_params = requested_metric.resolve_for_team(team_tree.id)
    # intervals used in the metric computation are of length 2, so metric_values dict
    # will include just one value
    value = metric_values[metric_w_params][(team_tree.id, 0)][0]

    return GraphQLTeamMetricValue(
        team=team_tree,
        value=GraphQLMetricValue(value),
        children=[
            _build_team_metric_value(child, requested_metric, metric_values)
            for child in team_tree.children
        ],
    )
