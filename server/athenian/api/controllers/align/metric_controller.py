from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from aiohttp import web

from athenian.api.align.goals.dates import Intervals
from athenian.api.align.goals.dbaccess import convert_metric_params_datatypes
from athenian.api.async_utils import gather
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.datetime_utils import closed_dates_interval_to_datetimes
from athenian.api.internal.jira import (
    JIRAConfig,
    check_jira_installation,
    get_jira_installation_or_none,
    normalize_issue_type,
    normalize_priority,
)
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.reposet import resolve_repos_with_request
from athenian.api.internal.team import fetch_teams_recursively, get_team_from_db
from athenian.api.internal.team_metrics import (
    CalcTeamMetricsRequest,
    MetricWithParams,
    RequestedTeamDetails,
    calculate_team_metrics,
)
from athenian.api.internal.team_tree import build_team_tree_from_rows
from athenian.api.internal.with_ import flatten_teams
from athenian.api.models.state.models import Team
from athenian.api.models.web import (
    InvalidRequestError,
    TeamDigest,
    TeamMetricResponseElement,
    TeamMetricsRequest,
    TeamMetricValueNode,
    TeamTree,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response


async def team_metrics(request: AthenianWebRequest, body: dict) -> web.Response:
    """Compute metric values for the team and the child teams."""
    try:
        metrics_request = TeamMetricsRequest.from_dict(body)
    except ValueError as e:
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e

    team = await get_team_from_db(metrics_request.team, None, request.uid, request.sdb)
    account = team[Team.owner_id.name]

    team_rows, meta_ids, jira_config = await gather(
        fetch_teams_recursively(
            account,
            request.sdb,
            select_entities=(Team.id, Team.name, Team.members, Team.parent_id),
            root_team_ids=[metrics_request.team],
        ),
        get_metadata_account_ids(account, request.sdb, request.cache),
        get_jira_installation_or_none(account, request.sdb, request.mdb, request.cache),
    )
    time_interval = _parse_time_interval(metrics_request)
    repos = await _parse_repositories(metrics_request.repositories, account, meta_ids, request)
    jira_filter = _parse_jira_filter(metrics_request, jira_config)
    teams_flat = flatten_teams(team_rows)

    tm_requests = []
    requested_metrics = _parse_requested_metrics(metrics_request)
    for req_metric in requested_metrics:
        for row in team_rows:
            team_detail = RequestedTeamDetails(
                team_id=(row_team_id := row[Team.id.name]),
                goal_id=0,  # change to unique number if the filters are no longer shared
                members=teams_flat[row_team_id],
                repositories=repos,
                jira_filter=jira_filter,
            )
            tm_requests.append(
                CalcTeamMetricsRequest(
                    [req_metric.resolve_for_team(row_team_id)], [time_interval], [team_detail],
                ),
            )

    team_metrics_all_intervals = await calculate_team_metrics(
        tm_requests,
        account,
        meta_ids,
        request.sdb,
        request.mdb,
        request.pdb,
        request.rdb,
        request.cache,
        request.app["slack"],
        jira_config,
    )
    team_metrics = team_metrics_all_intervals[time_interval]
    team_tree = build_team_tree_from_rows(team_rows, metrics_request.team)

    models = _build_response(requested_metrics, team_tree, team_metrics)
    return model_response(models)


def _parse_time_interval(metrics_request: TeamMetricsRequest) -> Intervals:
    if metrics_request.valid_from > metrics_request.expires_at:
        raise ResponseError(
            InvalidRequestError(
                pointer=".valid_from or .expires_at",
                detail="valid_from must be less than or equal to expires_at",
            ),
        )
    if metrics_request.valid_from > datetime.now(timezone.utc).date():
        raise ResponseError(
            InvalidRequestError(pointer="valid_from", detail="valid_from cannot be in the future"),
        )
    return closed_dates_interval_to_datetimes(
        metrics_request.valid_from, metrics_request.expires_at,
    )


async def _parse_repositories(
    request_repos: Sequence[str],
    account_id: int,
    meta_ids: tuple[int, ...],
    request: AthenianWebRequest,
) -> Optional[tuple[str, ...]]:
    if request_repos is None:
        return None
    return tuple(
        r.unprefixed
        for r in (
            await resolve_repos_with_request(request_repos, account_id, request, meta_ids=meta_ids)
        )[0]
    )


def _parse_jira_filter(
    metrics_request: TeamMetricsRequest,
    unchecked_jira_config: Optional[JIRAConfig],
) -> JIRAFilter:
    jira_projects = frozenset(metrics_request.jira_projects or ())
    jira_priorities = frozenset(
        normalize_priority(p) for p in metrics_request.jira_priorities or ()
    )
    jira_issue_types = frozenset(
        normalize_issue_type(t) for t in metrics_request.jira_issue_types or ()
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


def _parse_requested_metrics(metrics_request: TeamMetricsRequest) -> list[TeamRequestedMetric]:
    req_metrics = []
    for m_w_params in metrics_request.metrics_with_params:
        metric_params = convert_metric_params_datatypes(m_w_params.metric_params)
        teams_params = {
            p.team: convert_metric_params_datatypes(p.metric_params)
            for p in m_w_params.teams_metric_params or ()
        }
        req_metrics.append(TeamRequestedMetric(m_w_params.name, metric_params, teams_params))
    return req_metrics


def _build_response(
    requested_metrics: Sequence[TeamRequestedMetric],
    team_tree: TeamTree,
    metric_values: dict[MetricWithParams, dict[tuple[int, int], list[Any]]],
) -> list[TeamMetricResponseElement]:
    return [
        TeamMetricResponseElement(
            metric=req_metric.name,
            value=_build_team_metric_value_node(req_metric, team_tree, metric_values),
        )
        for req_metric in requested_metrics
    ]


def _build_team_metric_value_node(
    requested_metric: TeamRequestedMetric,
    team_tree: TeamTree,
    metric_values: dict[MetricWithParams, dict[tuple[int, int], list[Any]]],
) -> TeamMetricValueNode:
    metric_w_params = requested_metric.resolve_for_team(team_tree.id)
    # intervals used in the metric computation are of length 2, so metric_values dict
    # will include just one value
    value = metric_values[metric_w_params][(team_tree.id, 0)][0]
    return TeamMetricValueNode(
        team=TeamDigest(id=team_tree.id, name=team_tree.name),
        value=value,
        children=[
            _build_team_metric_value_node(requested_metric, child, metric_values)
            for child in team_tree.children
        ],
    )
