import dataclasses
from typing import Optional, Sequence

from aiohttp import web

from athenian.api.align.exceptions import GoalNotFoundError
from athenian.api.align.goals.dbaccess import (
    GoalCreationInfo,
    TeamGoalTargetAssignment,
    delete_goal as db_delete_goal,
    fetch_goal,
    fetch_goal_account,
    insert_goal,
    replace_team_goals,
    update_goal as update_goal_in_db,
)
from athenian.api.auth import disable_default_user
from athenian.api.internal.account import request_user_belongs_to_account
from athenian.api.internal.datetime_utils import closed_dates_interval_to_datetimes
from athenian.api.internal.jira import parse_request_issue_types, parse_request_priorities
from athenian.api.internal.repos import parse_request_repositories
from athenian.api.models.state.models import Goal, TeamGoal
from athenian.api.models.web import (
    BadRequestError,
    CreatedIdentifier,
    GoalCreateRequest,
    GoalUpdateRequest,
    InvalidRequestError,
    PullRequestMetricID,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response
from athenian.api.tracing import sentry_span


@disable_default_user
async def delete_goal(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete an existing goal."""
    async with request.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            account = await fetch_goal_account(id, sdb_conn)
            if not await request_user_belongs_to_account(request, account):
                raise GoalNotFoundError(id)
            await db_delete_goal(account, id, sdb_conn)

    return web.Response(status=204)


@disable_default_user
async def create_goal(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a new goal."""
    try:
        create_request = GoalCreateRequest.from_dict(body)
    except ValueError as e:
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e

    creation_info = await _parse_create_request(request, create_request)
    async with request.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            new_goal_id = await insert_goal(creation_info, sdb_conn)

    return model_response(CreatedIdentifier(id=new_goal_id))


@disable_default_user
async def update_goal(request: AthenianWebRequest, id: int, body: dict) -> web.Response:
    """Update an existing goal."""
    try:
        update_req = GoalUpdateRequest.from_dict(body)
    except ValueError as e:
        raise ResponseError(InvalidRequestError.from_validation_error(e)) from e

    async with request.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            account = await fetch_goal_account(id, sdb_conn)
            if not await request_user_belongs_to_account(request, account):
                raise GoalNotFoundError(id)
            goal = await fetch_goal(account, id, sdb_conn)
            update_info = await _parse_update_request(goal, request, update_req)
            await replace_team_goals(account, id, update_info.team_goals, sdb_conn)
            kw = {
                Goal.repositories.name: update_info.repositories,
                Goal.jira_projects.name: update_info.jira_projects,
                Goal.jira_priorities.name: update_info.jira_priorities,
                Goal.jira_issue_types.name: update_info.jira_issue_types,
                Goal.archived.name: update_info.archived,
                Goal.name.name: update_info.name,
            }
            await update_goal_in_db(account, id, sdb_conn, **kw)

    return model_response(CreatedIdentifier(id=id))


@sentry_span
async def _parse_create_request(
    request: AthenianWebRequest,
    creat_req: GoalCreateRequest,
) -> GoalCreationInfo:
    valid_from, expires_at = closed_dates_interval_to_datetimes(
        creat_req.valid_from, creat_req.expires_at,
    )
    if expires_at < valid_from:
        raise ResponseError(BadRequestError(detail="Goal expires_at cannot precede valid_from"))

    repositories = await parse_request_repositories(
        creat_req.repositories, request, creat_req.account,
    )
    jira_projects = creat_req.jira_projects
    jira_priorities = parse_request_priorities(creat_req.jira_priorities)
    jira_issue_types = parse_request_issue_types(creat_req.jira_issue_types)

    # user cannot directly set TeamGoal filter fields, received goal values are applied
    extra_team_goal_info = {
        TeamGoal.repositories.name: repositories,
        TeamGoal.jira_projects.name: jira_projects,
        TeamGoal.jira_priorities.name: jira_priorities,
        TeamGoal.jira_issue_types.name: jira_issue_types,
    }
    team_goals = [
        TeamGoal(
            team_id=tg.team_id,
            target=tg.target,
            metric_params=tg.metric_params,
            **extra_team_goal_info,
        )
        for tg in creat_req.team_goals
    ]

    if len({team_goal.team_id for team_goal in team_goals}) < len(team_goals):
        raise ResponseError(BadRequestError("More than one team goal with the same teamId"))
    goal = Goal(
        account_id=creat_req.account,
        name=creat_req.name,
        metric=creat_req.metric,
        repositories=repositories,
        jira_projects=jira_projects,
        jira_priorities=jira_priorities,
        jira_issue_types=jira_issue_types,
        metric_params=creat_req.metric_params,
        valid_from=valid_from,
        expires_at=expires_at,
    )
    return GoalCreationInfo(goal, team_goals)


@dataclasses.dataclass(frozen=True, slots=True)
class _GoalUpdateInfo:
    archived: bool
    name: str
    team_goals: Sequence[TeamGoalTargetAssignment]
    repositories: Optional[list[tuple[int, str]]]
    jira_projects: Optional[list[str]]
    jira_priorities: Optional[list[str]]
    jira_issue_types: Optional[list[str]]


@sentry_span
async def _parse_update_request(
    goal: Goal,
    request: AthenianWebRequest,
    update_req: GoalUpdateRequest,
) -> _GoalUpdateInfo:
    duplicates = []
    seen = set()
    team_goals = []
    for tg in update_req.team_goals:
        if tg.team_id in seen:
            duplicates.append(tg.team_id)
        else:
            team_goals.append(TeamGoalTargetAssignment(tg.team_id, tg.target, tg.metric_params))

        seen.add(tg.team_id)

    if duplicates:
        duplicated_repr = ",".join(map(str, duplicates))
        raise ResponseError(
            InvalidRequestError(".team_goals", f"Duplicated teams: {duplicated_repr}"),
        )

    repositories = await parse_request_repositories(
        update_req.repositories, request, goal[Goal.account_id.name],
    )
    jira_priorities = parse_request_priorities(update_req.jira_priorities)
    jira_issue_types = parse_request_issue_types(update_req.jira_issue_types)

    _validate_metric_params(goal[Goal.metric.name], update_req)

    return _GoalUpdateInfo(
        archived=update_req.archived,
        name=update_req.name,
        team_goals=team_goals,
        repositories=repositories,
        jira_projects=update_req.jira_projects,
        jira_priorities=jira_priorities,
        jira_issue_types=jira_issue_types,
    )


_NUMERIC_THRESHOLD_METRICS = [
    PullRequestMetricID.PR_REVIEW_COMMENTS_PER_ABOVE_THRESHOLD_RATIO,
    PullRequestMetricID.PR_SIZE_BELOW_THRESHOLD_RATIO,
]
_TIME_DURATION_THRESHOLD_METRICS = [
    PullRequestMetricID.PR_CYCLE_DEPLOYMENT_TIME_BELOW_THRESHOLD_RATIO,
    PullRequestMetricID.PR_LEAD_TIME_BELOW_THRESHOLD_RATIO,
    PullRequestMetricID.PR_OPEN_TIME_BELOW_THRESHOLD_RATIO,
    PullRequestMetricID.PR_REVIEW_TIME_BELOW_THRESHOLD_RATIO,
    PullRequestMetricID.PR_WAIT_FIRST_REVIEW_TIME_BELOW_THRESHOLD_RATIO,
]


def _validate_metric_params(goal_metric: str, update_req: GoalUpdateRequest) -> None:
    if goal_metric in _NUMERIC_THRESHOLD_METRICS:
        _check_team_goals_threshold_type(goal_metric, update_req, (int, float))
    elif goal_metric in _TIME_DURATION_THRESHOLD_METRICS:
        _check_team_goals_threshold_type(goal_metric, update_req, str)
    else:
        for tg in update_req.team_goals:
            if tg.metric_params:
                raise ResponseError(
                    InvalidRequestError(".team_goals", f'Invalid team goals: "{tg}"'),
                )


def _check_team_goals_threshold_type(
    metric: str,
    update_req: GoalUpdateRequest,
    types: type | tuple[type, ...],
) -> None:
    for tg in update_req.team_goals:
        if tg.metric_params and not isinstance(tg.metric_params.get("threshold"), types):
            msg = "Invalid metric_params {} for metric {metric}"
            raise ResponseError(InvalidRequestError(".team_goals", msg))
