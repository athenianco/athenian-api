from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from typing import Any, Collection, Iterable, Mapping, Optional, Sequence, cast

import aiomcache
from ariadne import ObjectType
from graphql import GraphQLResolveInfo
from morcilla import Database
import numpy as np
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient

from athenian.api.align.goals.dates import goal_dates_to_datetimes
from athenian.api.align.models import (
    MetricParamsFields,
    MetricValue,
    MetricValues,
    TeamMetricValue,
    TeamTree,
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
    get_jira_installation,
    get_jira_installation_or_none,
    load_mapped_jira_users,
    normalize_issue_type,
    normalize_priority,
)
from athenian.api.internal.logical_repos import coerce_logical_repos
from athenian.api.internal.miners.access_classes import access_classes
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.bots import bots
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.types import (
    JIRAParticipants,
    JIRAParticipationKind,
    PRParticipationKind,
    ReleaseParticipationKind,
)
from athenian.api.internal.prefixer import Prefixer, RepositoryName
from athenian.api.internal.settings import Settings
from athenian.api.internal.team import fetch_teams_recursively
from athenian.api.internal.with_ import flatten_teams
from athenian.api.models.state.models import Team
from athenian.api.models.web import ForbiddenError, InvalidRequestError
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
    team_rows, meta_ids = await gather(
        fetch_teams_recursively(
            accountId,
            info.context.sdb,
            select_entities=(Team.id, Team.name, Team.members, Team.parent_id),
            root_team_ids=[team_id],
        ),
        get_metadata_account_ids(accountId, info.context.sdb, info.context.cache),
    )
    time_interval = _parse_time_interval(params)
    teams_flat = flatten_teams(team_rows)

    repos = await _parse_repositories(params, accountId, meta_ids, info.context)
    jira_filter = await _parse_jira_filter(params, accountId, info.context)
    teams = {
        row[Team.id.name]: RequestedTeamDetails(teams_flat[row[Team.id.name]], repos, jira_filter)
        for row in team_rows
    }
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
    )
    team_metrics = team_metrics_all_intervals[time_interval]

    team_tree = build_team_tree_from_rows(team_rows, team_id)

    models = _build_metrics_response(team_tree, params[MetricParamsFields.metrics], team_metrics)
    return [m.to_dict() for m in models]


async def _parse_repositories(
    params: Mapping[str, Any],
    account_id: int,
    meta_ids: tuple[int, ...],
    request: AthenianWebRequest,
) -> Optional[tuple[str, ...]]:
    if (repos := params.get(MetricParamsFields.repositories)) is not None:
        repos = tuple(RepositoryName.from_prefixed(r).unprefixed for r in repos)
        checker = access_classes["github"](
            account_id, meta_ids, request.sdb, request.mdb, request.cache,
        )
        await checker.load()
        if denied := await checker.check(coerce_logical_repos(repos).keys()):
            raise ResponseError(
                ForbiddenError(
                    detail=f"Account {account_id} is access denied to repos {'.'.join(denied)}",
                ),
            )
    return repos


async def _parse_jira_filter(
    params: Mapping[str, Any],
    account_id: int,
    request: AthenianWebRequest,
) -> JIRAFilter:
    jira_projects = frozenset(params.get(MetricParamsFields.jiraProjects) or ())
    jira_priorities = frozenset(
        normalize_priority(p) for p in params.get(MetricParamsFields.jiraPriorities) or ()
    )
    jira_issue_types = frozenset(
        normalize_issue_type(t) for t in params.get(MetricParamsFields.jiraIssueTypes) or ()
    )
    if jira_projects or jira_priorities or jira_issue_types:
        jira_config = await get_jira_installation(
            account_id, request.sdb, request.mdb, request.cache,
        )
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


def _parse_time_interval(params: Mapping[str, Any]) -> Interval:
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


Interval = tuple[datetime, datetime]


TeamMetricsResult = dict[Interval, dict[str, dict[int, object]]]


@dataclass(frozen=True, slots=True)
class RequestedTeamDetails:
    """Team members + all team-specific filters."""

    members: Sequence[int]
    repositories: Optional[tuple[str, ...]]
    jira_filter: JIRAFilter = JIRAFilter.empty()
    # add more filters here

    def filters(self) -> tuple[Any, ...]:
        """Return the tuple with contained team-specific filters."""
        return (self.repositories, self.jira_filter)  # add more filters here


@dataclass(frozen=True, slots=True)
class TeamMetricsRequest:
    """Request for multiple metrics/intervals/teams usable with calculate_team_metrics."""

    metrics: Sequence[str]
    time_intervals: Sequence[Interval]
    teams: Mapping[int, RequestedTeamDetails]
    """A mapping of team id to team members and team-specific filters."""


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
) -> TeamMetricsResult:
    """Calculate a set of metrics for each team and time interval.

    The result will be a nested dict structure indexed first by interval, then by metric and
    finally by team id.
    """
    requests = _simplify_requests(requests)
    QUANTILES = (0, 0.95)
    settings = Settings.from_account(account, sdb, mdb, cache, slack)
    jira_map_task = asyncio.create_task(
        load_mapped_jira_users(
            account,
            set(
                chain.from_iterable(
                    td.members for request in requests for td in request.teams.values()
                ),
            ),
            sdb,
            mdb,
            cache,
        ),
    )
    prefixer, account_bots, release_settings = await gather(
        Prefixer.load(meta_ids, mdb, cache),
        bots(account, meta_ids, mdb, sdb, cache),
        settings.list_release_matches(),
    )
    all_repos = release_settings.native.keys()
    (branches, default_branches), logical_settings, _ = await gather(
        BranchMiner.extract_branches(all_repos, prefixer, meta_ids, mdb, cache),
        settings.list_logical_repositories(prefixer),
        jira_map_task,
    )
    jira_map = jira_map_task.result()

    pr_collector = BatchCalcResultCollector()
    release_collector = BatchCalcResultCollector()
    jira_collector = BatchCalcResultCollector()

    for request in requests:
        team_members = _loginify_teams([td.members for td in request.teams.values()], prefixer)
        pr_metrics, release_metrics, jira_metrics = _triage_metrics(request.metrics)

        if pr_metrics:
            pr_collector.requests.append(
                MetricsLineRequest(
                    pr_metrics,
                    request.time_intervals,
                    [
                        TeamSpecificFilters(
                            team_id=team_id,
                            participants={PRParticipationKind.AUTHOR: members},
                            repositories=td.repositories
                            if td.repositories is not None
                            else all_repos,
                            jira_filter=td.jira_filter,
                        )
                        for (team_id, td), members in zip(request.teams.items(), team_members)
                    ],
                ),
            )

        if release_metrics:
            release_collector.requests.append(
                MetricsLineRequest(
                    release_metrics,
                    request.time_intervals,
                    [
                        TeamSpecificFilters(
                            team_id=team_id,
                            repositories=td.repositories
                            if td.repositories is not None
                            else all_repos,
                            participants={
                                ReleaseParticipationKind.PR_AUTHOR: td.members,
                                ReleaseParticipationKind.COMMIT_AUTHOR: td.members,
                                ReleaseParticipationKind.RELEASER: td.members,
                            },
                        )
                        for team_id, td in request.teams.items()
                    ],
                ),
            )

        if jira_metrics:
            jira_collector.requests.append(
                MetricsLineRequest(
                    jira_metrics,
                    request.time_intervals,
                    [
                        TeamSpecificFilters(
                            team_id=team_id,
                            repositories=td.repositories
                            if td.repositories is not None
                            else all_repos,
                            participants=_jirafy_team(td.members, jira_map),
                        )
                        for team_id, td in request.teams.items()
                    ],
                ),
            )

    calculator = make_calculator(account, meta_ids, mdb, pdb, rdb, cache)
    tasks = []
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
        )
        tasks.append(release_task)

    if jira_requests := jira_collector.requests:
        jira_conf = await get_jira_installation_or_none(account, sdb, mdb, cache)
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
def _simplify_requests(requests: Sequence[TeamMetricsRequest]) -> Sequence[TeamMetricsRequest]:
    """Simplify the list of requests and try to group them in less requests."""
    # intervals => team and other filters => metrics
    requests_tree: dict[tuple[Interval, ...], dict[tuple, set[str]]] = {}

    # this can be global across requests, team ids are always mapped to the same team members
    teams_members: dict[int, Sequence[int]] = {}

    for req in requests:
        req_time_intervals = tuple(req.time_intervals)
        requests_tree.setdefault(req_time_intervals, {})

        for team_id, team_details in req.teams.items():
            teams_members[team_id] = team_details.members
            for m in req.metrics:
                requests_tree[req_time_intervals].setdefault(
                    (team_id, team_details.filters()), set(),
                ).add(m)

    # intervals => metrics => team ids and team-specific filters
    simplified_tree: dict[Sequence[Interval], dict[tuple[str, ...], set[tuple]]] = {}
    for intervals, intervals_tree in requests_tree.items():
        simplified_tree[intervals] = {}

        for filters, metrics in intervals_tree.items():
            sorted_metrics = tuple(sorted(metrics))
            simplified_tree[intervals].setdefault(sorted_metrics, set()).add(filters)

    # assemble the final groups
    requests = []
    for intervals_, intervals_tree_ in simplified_tree.items():
        for metrics_, grouped_filters in intervals_tree_.items():
            teams = {
                team_id: RequestedTeamDetails(teams_members[team_id], *filters)
                for team_id, filters in grouped_filters
            }
            requests.append(TeamMetricsRequest(metrics_, intervals_, teams))

    return requests


class BatchCalcResultCollector:
    """Convert the results coming from a MetricEntriesCalculator batch method."""

    def __init__(self) -> None:
        """Initialize a new instance of BatchCalcResultCollector."""
        self._requests: list[MetricsLineRequest] = []

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
        for request, calc_result in zip(self._requests, batch_calc_results):
            for interval_i, interval in enumerate(request.time_intervals):
                int_ = cast(Interval, interval)
                team_metrics_res.setdefault(int_, {})
                for metric_i, metric in enumerate(request.metrics):
                    team_metrics_res[int_].setdefault(metric, {})
                    for team_i, team_filters in enumerate(request.teams):
                        val = calc_result[team_i][interval_i][0][metric_i].value
                        team_metrics_res[int_][metric][team_filters.team_id] = val

    @property
    def requests(self) -> list[MetricsLineRequest]:
        """Return the registered metric calculation requests."""
        return self._requests


@sentry_span
def _build_metrics_response(
    team_tree: TeamTree,
    metrics: Sequence[str],
    triaged: dict[str, dict[int, object]],
) -> list[MetricValues]:
    return [
        MetricValues(metric, _build_team_metric_value(team_tree, triaged[metric]))
        for metric in metrics
    ]


def _build_team_metric_value(
    team_tree: TeamTree,
    metric_values: dict[int, object],
) -> TeamMetricValue:
    return TeamMetricValue(
        team=team_tree,
        value=MetricValue(metric_values[team_tree.id]),
        children=[_build_team_metric_value(child, metric_values) for child in team_tree.children],
    )
