import asyncio
from datetime import date, datetime, time, timedelta, timezone
from itertools import chain
from typing import Any, Collection, List, Mapping, Optional, Sequence, Tuple

import aiomcache
from ariadne import ObjectType
from graphql import GraphQLResolveInfo
from morcilla import Database
from slack_sdk.web.async_client import AsyncWebClient as SlackWebClient

from athenian.api.align.models import MetricParamsFields
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
            accountId, sdb, select_entities=(Team.id, Team.members),
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
    date_to += timedelta(days=1)
    pr_metrics, release_metrics, jira_metrics = _triage_metrics(params[MetricParamsFields.metrics])
    teams = [row[Team.members.name] for row in team_rows]
    await _calculate_team_metrics(
        pr_metrics, release_metrics, jira_metrics, date_from, date_to, teams, accountId,
        meta_ids, sdb, mdb, pdb, rdb, cache, info.context.app["slack"])


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
                    ) -> List[List[str]]:
    result = []
    user_node_to_login = prefixer.user_node_to_login.__getitem__
    for team in teams:
        logins = []
        for node in team:
            try:
                logins.append(user_node_to_login(node))
            except KeyError:
                continue
        result.append(logins)
    return result


def _jirafy_teams(teams: Sequence[Collection[int]],
                  jira_map: Mapping[int, str],
                  ) -> List[JIRAParticipants]:
    result = []
    for team in teams:
        result.append({JIRAParticipationKind.REPORTER: (reporters := [])})
        for dev in team:
            try:
                reporters.append(jira_map[dev])
            except KeyError:
                continue
    return result


@sentry_span
async def _calculate_team_metrics(
        pr_metrics: Sequence[str],
        release_metrics: Sequence[str],
        jira_metrics: Sequence[str],
        date_from: date,
        date_to: date,
        teams: Sequence[Collection[int]],
        account: int,
        meta_ids: Tuple[int, ...],
        sdb: Database,
        mdb: Database,
        pdb: Database,
        rdb: Database,
        cache: Optional[aiomcache.Client],
        slack: Optional[SlackWebClient],
):
    time_from = datetime.combine(date_from, time(), tzinfo=timezone.utc)
    time_to = datetime.combine(date_to, time(), tzinfo=timezone.utc)
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
    repos = list(prefixer.repo_name_to_node.keys())
    logical_settings, (branches, default_branches) = await gather(
        BranchMiner.extract_branches(None, prefixer, meta_ids, mdb, cache),
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
            ReleaseParticipationKind.COMMIT_AUTHOR: team,
        } for team in teams]
        tasks.append(calculator.calc_release_metrics_line_github(
            release_metrics, time_intervals, quantiles, repos, release_participants,
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
    await gather(*tasks, op="calculators")
