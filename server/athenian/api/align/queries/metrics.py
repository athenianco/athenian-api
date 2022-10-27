from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from typing import Any, Collection, Iterable, Mapping, Optional, Sequence

import aiomcache
from ariadne import ObjectType
from graphql import GraphQLResolveInfo
from morcilla import Database
import numpy as np
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient

from athenian.api.align.goals.dates import Intervals, goal_dates_to_datetimes
from athenian.api.align.models import (
    GraphQLMetricValue,
    GraphQLMetricValues,
    GraphQLTeamMetricValue,
    GraphQLTeamTree,
    MetricParamsFields,
)
from athenian.api.align.queries.teams import build_team_tree_from_rows
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
    teams = [
        RequestedTeamDetails(
            team_id=(row_team_id := row[Team.id.name]),
            goal_id=0,  # change to unique number if the filters are no longer shared
            members=teams_flat[row_team_id],
            repositories=repos,
            jira_filter=jira_filter,
        )
        for row in team_rows
    ]
    team_metrics_all_intervals = await calculate_team_metrics(
        [TeamMetricsRequest(params[MetricParamsFields.metrics], [time_interval], teams)],
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

    models = _build_metrics_response(team_tree, params[MetricParamsFields.metrics], team_metrics)
    return [m.to_dict() for m in models]


async def _parse_repositories(
    params: Mapping[str, Any],
    account_id: int,
    meta_ids: tuple[int, ...],
    request: AthenianWebRequest,
) -> Optional[tuple[str, ...]]:
    if (repos := params.get(MetricParamsFields.repositories)) is not None:
        repos = tuple((await resolve_repos_with_request(repos, account_id, request, meta_ids))[0])
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


TeamMetricsResult = dict[Intervals, dict[str, dict[tuple[int, int], list[Any]]]]


@dataclass(frozen=True, slots=True)
class TeamMetricsRequest:
    """Request for multiple metrics/intervals/teams usable with calculate_team_metrics."""

    metrics: Sequence[str]
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

    The result will be a nested dict structure indexed first by intervals, then by metric and
    finally by (team_id, goal_id).
    Terminal values in the dict structure will be lists of metric values: a metric value will be
    found for each granularity requested in the given interval - for each list of metric
    values its length will equal len(intervals) - 1.

    For example, given the single request:
    TeamMetricsRequest(
       metrics=["pr-all-count"],
       time_intervals=[dt(2010, 1, 1), dt(2010, 2, 1), dt(2010, 3, 1)],
       teams=[RequestedTeamDetails(team_id=10, goal_id=20, ...)]
    )

    The response will have this shape:

    {
        (dt(2010, 1, 1), dt(2010, 2, 1), dt(2010, 3, 1)): {
          "pr-all-count": {
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
        pr_metrics, release_metrics, jira_metrics = _triage_metrics(request.metrics)
        goal_ids = [td.goal_id for td in request.teams]

        if pr_metrics:
            pr_collector.append(
                MetricsLineRequest(
                    pr_metrics,
                    request.time_intervals,
                    [
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

        if release_metrics:
            release_collector.append(
                MetricsLineRequest(
                    release_metrics,
                    request.time_intervals,
                    [
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

        if jira_metrics:
            jira_collector.append(
                MetricsLineRequest(
                    jira_metrics,
                    request.time_intervals,
                    [
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


def _triage_metrics(metrics: Sequence[str]) -> tuple[Sequence[str], Sequence[str], Sequence[str]]:
    pr_metrics = []
    release_metrics = []
    jira_metrics = []
    unidentified = []
    for metric in metrics:
        if metric in pr_metric_calculators:
            pr_metrics.append(metric)
        elif metric in release_metric_calculators:
            release_metrics.append(metric)
        elif metric in jira_metric_calculators:
            jira_metrics.append(metric)
        else:
            unidentified.append(metric)
    if unidentified:
        raise ResponseError(
            InvalidRequestError(
                pointer=f".{MetricParamsFields.metrics}",
                detail=f"The following metrics are not supported: {', '.join(unidentified)}",
            ),
        )
    return pr_metrics, release_metrics, jira_metrics


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
    # intervals => team and other filters => metrics
    requests_tree: dict[tuple[Intervals, ...], dict[RequestedTeamDetails, set[str]]] = {}
    for req in requests:
        req_time_intervals = tuple(req.time_intervals)
        requests_tree.setdefault(req_time_intervals, {})
        for td in req.teams:
            for m in req.metrics:
                requests_tree[req_time_intervals].setdefault(td, set()).add(m)

    # intervals => metrics => team-specific
    simplified_tree: dict[
        Sequence[Intervals],
        dict[tuple[str, ...], set[RequestedTeamDetails]],
    ] = {}
    for intervals, intervals_tree in requests_tree.items():
        simplified_tree[intervals] = {}
        for filters, metrics in intervals_tree.items():
            simplified_tree[intervals].setdefault(tuple(sorted(metrics)), set()).add(filters)

    # assemble the final groups
    requests = [
        TeamMetricsRequest(metrics_, intervals_, grouped_filters)
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
                    team_metrics_res[intervals].setdefault(metric, {})
                    for team_i, (team_filters, goal_id) in enumerate(zip(request.teams, goal_ids)):
                        # collect a value for each granularity
                        values = [ar[metric_i].value for ar in calc_result[team_i][intervals_i]]
                        team_metrics_res[intervals][metric][
                            (team_filters.team_id, goal_id)
                        ] = values

    @property
    def requests(self) -> list[MetricsLineRequest]:
        """Return the registered metric calculation requests."""
        return self._requests


@sentry_span
def _build_metrics_response(
    team_tree: GraphQLTeamTree,
    metrics: Sequence[str],
    triaged: dict[str, dict[tuple[int, int], list[Any]]],
) -> list[GraphQLMetricValues]:
    return [
        GraphQLMetricValues(
            metric=metric, value=_build_team_metric_value(team_tree, triaged[metric]),
        )
        for metric in metrics
    ]


def _build_team_metric_value(
    team_tree: GraphQLTeamTree,
    metric_values: dict[tuple[int, int], list[Any]],
) -> GraphQLTeamMetricValue:
    return GraphQLTeamMetricValue(
        team=team_tree,
        # intervals used in the metric computation are of length 2, so metric_values dict
        # will include just one value
        value=GraphQLMetricValue(metric_values[(team_tree.id, 0)][0]),
        children=[_build_team_metric_value(child, metric_values) for child in team_tree.children],
    )
