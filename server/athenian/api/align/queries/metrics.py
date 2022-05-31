import asyncio
from datetime import datetime
from itertools import chain
from typing import Any, Collection, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import aiomcache
from ariadne import ObjectType
from graphql import GraphQLResolveInfo
from morcilla import Database
import numpy as np
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient

from athenian.api.align.goals.dates import goal_dates_to_datetimes
from athenian.api.align.models import MetricParamsFields, MetricValue, MetricValues, \
    TeamMetricValue
from athenian.api.align.queries.teams import build_team_tree_nodes_from_rows
from athenian.api.async_utils import gather
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.features.entries import make_calculator
from athenian.api.internal.features.github.pull_request_metrics import \
    metric_calculators as pr_metric_calculators
from athenian.api.internal.features.github.release_metrics import \
    metric_calculators as release_metric_calculators
from athenian.api.internal.features.jira.issue_metrics import \
    metric_calculators as jira_metric_calculators
from athenian.api.internal.jira import get_jira_installation_or_none, load_mapped_jira_users
from athenian.api.internal.miners.filters import JIRAFilter, LabelFilter
from athenian.api.internal.miners.github.bots import bots
from athenian.api.internal.miners.github.branches import BranchMiner
from athenian.api.internal.miners.types import JIRAParticipants, JIRAParticipationKind, \
    PRParticipationKind, ReleaseParticipationKind
from athenian.api.internal.prefixer import Prefixer
from athenian.api.internal.settings import Settings
from athenian.api.internal.team import fetch_teams_recursively
from athenian.api.internal.with_ import flatten_teams
from athenian.api.models.state.models import Team
from athenian.api.models.web import InvalidRequestError
from athenian.api.response import ResponseError
from athenian.api.tracing import sentry_span

query = ObjectType("Query")


@query.field("metricsCurrentValues")
@sentry_span
async def resolve_metrics_current_values(obj: Any,
                                         info: GraphQLResolveInfo,
                                         accountId: int,
                                         params: Mapping[str, Any]) -> Any:
    """Serve metricsCurrentValues()."""
    sdb, mdb, pdb, rdb, cache = \
        info.context.sdb, info.context.mdb, info.context.pdb, info.context.rdb, info.context.cache
    team_rows, meta_ids = await gather(
        fetch_teams_recursively(
            accountId, sdb, select_entities=(Team.id, Team.members, Team.parent_id),
            root_team_ids=[params[MetricParamsFields.teamId]],
        ),
        get_metadata_account_ids(accountId, sdb, cache),
    )
    date_from, date_to = params[MetricParamsFields.validFrom], params[MetricParamsFields.expiresAt]
    if date_from > date_to:
        raise ResponseError(InvalidRequestError(
            pointer=f".{MetricParamsFields.validFrom} or .{MetricParamsFields.expiresAt}",
            detail=f"{MetricParamsFields.validFrom} must be less than or equal to "
                   f"{MetricParamsFields.expiresAt}",
        ))
    time_from, time_to = goal_dates_to_datetimes(date_from, date_to)
    pr_metrics, release_metrics, jira_metrics = _triage_metrics(params[MetricParamsFields.metrics])
    teams_flat = flatten_teams(team_rows)
    teams = [teams_flat[row[Team.id.name]] for row in team_rows]
    metric_values = await _calculate_team_metrics(
        pr_metrics, release_metrics, jira_metrics, time_from, time_to, teams, accountId,
        meta_ids, sdb, mdb, pdb, rdb, cache, info.context.app["slack"])
    team_ids = [row[Team.id.name] for row in team_rows]
    triaged = _triage_metric_values(
        pr_metrics, release_metrics, jira_metrics, team_ids, metric_values)
    nodes, root_team_id = build_team_tree_nodes_from_rows(
        team_rows, params[MetricParamsFields.teamId],
    )
    models = _build_metrics_response(nodes, triaged, root_team_id)
    return [m.to_dict() for m in models]


def _triage_metrics(metrics: List[str]) -> Tuple[List[str], List[str], List[str]]:
    pr_metrics = []
    release_metrics = []
    jira_metrics = []
    unidentified = []
    for metric in metrics:
        if metric in pr_metric_calculators:
            pr_metrics.append(metric)
            continue
        if metric in release_metric_calculators:
            release_metrics.append(metric)
            continue
        if metric in jira_metric_calculators:
            jira_metrics.append(metric)
            continue
        unidentified.append(metric)
    if unidentified:
        raise ResponseError(InvalidRequestError(
            pointer=f".{MetricParamsFields.metrics}",
            detail=f"The following metrics are not supported: {', '.join(unidentified)}",
        ))
    return pr_metrics, release_metrics, jira_metrics


def _loginify_teams(teams: Sequence[Collection[int]],
                    prefixer: Prefixer,
                    ) -> List[Set[str]]:
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


def _jirafy_teams(teams: Sequence[Collection[int]],
                  jira_map: Mapping[int, str],
                  ) -> List[JIRAParticipants]:
    result = []
    for team in teams:
        result.append({JIRAParticipationKind.ASSIGNEE: (assignees := [])})
        for dev in team:
            try:
                assignees.append(jira_map[dev])
            except KeyError:
                continue
    return result


@sentry_span
async def _calculate_team_metrics(
        pr_metrics: Sequence[str],
        release_metrics: Sequence[str],
        jira_metrics: Sequence[str],
        time_from: datetime,
        time_to: datetime,
        teams: Sequence[List[int]],
        account: int,
        meta_ids: Tuple[int, ...],
        sdb: Database,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        cache: Optional[aiomcache.Client],
        slack: Optional[SlackWebClient],
) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, ...]], ...]:
    time_intervals = [[time_from, time_to]]
    quantiles = (0, 0.95)
    settings = Settings.from_account(account, sdb, mdb, cache, slack)
    if jira_metrics:
        jira_map_task = asyncio.create_task(load_mapped_jira_users(
            account, set(chain.from_iterable(teams)), sdb, mdb, cache))
    prefixer, account_bots, jira_ids, release_settings = await gather(
        Prefixer.load(meta_ids, mdb, cache),
        bots(account, meta_ids, mdb, sdb, cache),
        get_jira_installation_or_none(account, sdb, mdb, cache),
        settings.list_release_matches(),
    )
    repos = release_settings.native.keys()
    (branches, default_branches), logical_settings = await gather(
        BranchMiner.extract_branches(repos, prefixer, meta_ids, mdb, cache),
        settings.list_logical_repositories(prefixer),
    )
    calculator = make_calculator(account, meta_ids, mdb, pdb, rdb, cache)
    tasks = []
    if pr_metrics:
        pr_participants = [{
            PRParticipationKind.AUTHOR: team,
        } for team in _loginify_teams(teams, prefixer)]
        tasks.append(calculator.calc_pull_request_metrics_line_github(
            pr_metrics, time_intervals, quantiles, [], [], [repos], pr_participants,
            LabelFilter.empty(), JIRAFilter.empty(), True, account_bots,
            release_settings, logical_settings, prefixer, branches, default_branches, False,
        ))
    if release_metrics:
        release_participants = [{
            ReleaseParticipationKind.PR_AUTHOR: team,
            ReleaseParticipationKind.COMMIT_AUTHOR: team,
            ReleaseParticipationKind.RELEASER: team,
        } for team in teams]
        tasks.append(calculator.calc_release_metrics_line_github(
            release_metrics, time_intervals, quantiles, [repos], release_participants,
            LabelFilter.empty(), JIRAFilter.empty(), release_settings, logical_settings,
            prefixer, branches, default_branches))
    if jira_metrics:
        await jira_map_task
        jira_map = jira_map_task.result()
        tasks.append(calculator.calc_jira_metrics_line_github(
            jira_metrics,
            time_intervals,
            quantiles,
            _jirafy_teams(teams, jira_map),
            LabelFilter.empty(),
            False,
            [], [], [],
            True,
            release_settings, logical_settings,
            default_branches,
            jira_ids,
        ))
    return await gather(*tasks, op="calculators")


def _triage_metric_values(pr_metrics: Sequence[str],
                          release_metrics: Sequence[str],
                          jira_metrics: Sequence[str],
                          teams: Sequence[int],
                          metric_values: Tuple[Union[np.ndarray, Tuple[np.ndarray, ...]], ...],
                          ) -> Dict[str, Dict[int, object]]:
    result = {}
    if pr_metrics:
        pr_metric_values, *metric_values = metric_values
        for i, metric in enumerate(pr_metrics):
            metric_teams = result[metric] = {}
            for team, team_metric_values in zip(teams, pr_metric_values[0][0]):
                metric_teams[team] = team_metric_values[0][0][i].value
    if release_metrics:
        (release_metric_values, *_), *metric_values = metric_values
        for i, metric in enumerate(release_metrics):
            metric_teams = result[metric] = {}
            for team, team_metric_values in zip(teams, release_metric_values):
                metric_teams[team] = team_metric_values[0][0][0][i].value
    if jira_metrics:
        (jira_metric_values, *_), *metric_values = metric_values
        for i, metric in enumerate(jira_metrics):
            metric_teams = result[metric] = {}
            for team, team_metric_values in zip(teams, jira_metric_values):
                metric_teams[team] = team_metric_values[0][0][0][i].value
    return result


def _build_metrics_response(nodes: Dict[int, Dict[str, Any]],
                            triaged: Dict[str, Dict[int, object]],
                            root_team_id: int,
                            ) -> List[MetricValues]:
    return [MetricValues(metric, _build_metric_response(nodes, team_metric_values, root_team_id))
            for metric, team_metric_values in triaged.items()]


def _build_metric_response(nodes: Dict[int, Dict[str, Any]],
                           metric_values: Dict[int, object],
                           team_id: int,
                           ) -> TeamMetricValue:
    return TeamMetricValue(
        team_id=team_id,
        value=MetricValue(metric_values[team_id]),
        children=[_build_metric_response(nodes, metric_values, child)
                  for child in nodes[team_id]["children"]],
    )
