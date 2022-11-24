from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Callable, Optional, Sequence

from ariadne import MutationType
from graphql import GraphQLResolveInfo

from athenian.api.align.exceptions import GoalMutationError
from athenian.api.align.goals.dates import goal_dates_to_datetimes
from athenian.api.align.goals.dbaccess import (
    GoalCreationInfo,
    TeamGoalTargetAssignment,
    assign_team_goals,
    delete_goal,
    delete_team_goals,
    fetch_goal,
    insert_goal,
    update_goal,
)
from athenian.api.align.models import (
    CreateGoalInputFields,
    GoalRemoveStatus,
    MutateGoalResult,
    MutateGoalResultGoal,
    TeamGoalChangeFields,
    TeamGoalInputFields,
    UpdateGoalInputFields,
    UpdateRepositoriesInputFields,
)
from athenian.api.align.serialization import parse_metric_params, parse_union_value
from athenian.api.ariadne import ariadne_disable_default_user
from athenian.api.async_utils import gather
from athenian.api.controllers.goal_controller import parse_request_repositories
from athenian.api.internal.jira import normalize_issue_type, normalize_priority
from athenian.api.models.state.models import Goal, TeamGoal
from athenian.api.models.web import JIRAMetricID, PullRequestMetricID, ReleaseMetricID
from athenian.api.request import AthenianWebRequest
from athenian.api.tracing import sentry_span

mutation = MutationType()


@mutation.field("createGoal")
@sentry_span
@ariadne_disable_default_user
async def resolve_create_goal(
    _: Any,
    info: GraphQLResolveInfo,
    accountId: int,
    input: dict[str, Any],
) -> dict[str, Any]:
    """Create a Goal."""
    creation_info = await _parse_create_goal_input(input, info.context, accountId)

    async with info.context.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            new_goal_id = await insert_goal(creation_info, sdb_conn)

    # TODO: return the complete response
    return MutateGoalResult(MutateGoalResultGoal(new_goal_id)).to_dict()


@mutation.field("removeGoal")
@sentry_span
@ariadne_disable_default_user
async def resolve_remove_goal(
    _: Any,
    info: GraphQLResolveInfo,
    accountId: int,
    goalId: int,
) -> dict[str, Any]:
    """Remove a Goal and referring TeamGoal-s."""
    async with info.context.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            await delete_goal(accountId, goalId, sdb_conn)
    remove_status = GoalRemoveStatus(success=True)
    return remove_status.to_dict()


@mutation.field("updateGoal")
@sentry_span
@ariadne_disable_default_user
async def resolve_update_goal(
    _: Any,
    info: GraphQLResolveInfo,
    accountId: int,
    input: dict,
) -> dict:
    """Update an existing Goal."""
    goal_id = input[UpdateGoalInputFields.goalId]
    update, goal = await gather(
        _parse_update_goal_input(input, info.context, accountId),
        fetch_goal(accountId, goal_id, info.context.sdb),
    )
    if goal is None:
        raise GoalMutationError(f"Goal {goal_id} not found or access denied", HTTPStatus.NOT_FOUND)
    result = MutateGoalResult(MutateGoalResultGoal(goal_id)).to_dict()

    async with info.context.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            if assignments := update.team_goal_assignments:
                await assign_team_goals(accountId, goal_id, assignments, sdb_conn)
            if deletions := update.team_goal_deletions:
                await delete_team_goals(accountId, goal_id, deletions, sdb_conn)

            kw = {}
            goal = await fetch_goal(accountId, goal_id, sdb_conn)
            for update_info, col in (
                (update.repositories, Goal.repositories),
                (update.jira_projects, Goal.jira_projects),
                (update.jira_priorities, Goal.jira_priorities),
                (update.jira_issue_types, Goal.jira_issue_types),
            ):
                if update_info is not None:
                    kw[col.name] = None if update_info.remove else update_info.value
                else:
                    # team goal dependant columns are always sent to update_goal to trigger
                    # the updates in TeamGoal
                    kw[col.name] = goal[col.name]

            if update.archived is not None:
                kw[Goal.archived.name] = update.archived
            if update.name is not None:
                kw[Goal.name.name] = update.name

            await update_goal(accountId, goal_id, sdb_conn, **kw)

    return result


def validate_goal_metric(value: str) -> None:
    """Raise a validation error if the metric is outside of the allowed enum values."""
    if value not in PullRequestMetricID | ReleaseMetricID | JIRAMetricID:
        raise GoalMutationError(f'Unsupported metric "{value}"')


@sentry_span
async def _parse_create_goal_input(
    input: dict[str, Any],
    request: AthenianWebRequest,
    account_id: int,
) -> GoalCreationInfo:
    """Parse CreateGoalInput into GoalCreationInfo."""
    validate_goal_metric(input[CreateGoalInputFields.metric])

    valid_from, expires_at = goal_dates_to_datetimes(
        input[CreateGoalInputFields.validFrom], input[CreateGoalInputFields.expiresAt],
    )
    if expires_at < valid_from:
        raise GoalMutationError("Goal expiresAt cannot precede validFrom")

    repositories = await parse_request_repositories(
        input.get(CreateGoalInputFields.repositories), request, account_id,
    )
    jira_projects = _parse_request_jira_projects(input)
    jira_priorities = _parse_request_jira_priorities(input)
    jira_issue_types = _parse_request_jira_issue_types(input)

    # user cannot directly set TeamGoal filter fields, received goal values are applied
    extra_team_goal_info = {
        TeamGoal.repositories.name: repositories,
        TeamGoal.jira_projects.name: jira_projects,
        TeamGoal.jira_priorities.name: jira_priorities,
        TeamGoal.jira_issue_types.name: jira_issue_types,
    }
    team_goals = [
        _parse_team_goal_input(tg_input, **extra_team_goal_info)
        for tg_input in input[CreateGoalInputFields.teamGoals]
    ]

    if not team_goals:
        raise GoalMutationError("At least one teamGoals is required")

    if len({team_goal.team_id for team_goal in team_goals}) < len(team_goals):
        raise GoalMutationError("More than one team goal with the same teamId")

    metric_params = parse_metric_params(input.get(CreateGoalInputFields.metricParams))

    goal = Goal(
        account_id=account_id,
        name=input[CreateGoalInputFields.name],
        metric=input[CreateGoalInputFields.metric],
        repositories=repositories,
        jira_projects=jira_projects,
        jira_priorities=jira_priorities,
        jira_issue_types=jira_issue_types,
        metric_params=metric_params,
        valid_from=valid_from,
        expires_at=expires_at,
    )
    return GoalCreationInfo(goal, team_goals)


@sentry_span
def _parse_team_goal_input(team_goal_input: dict, **extra: Any) -> TeamGoal:
    """Parse TeamGoalInput into a TeamGoal model."""
    team_id = team_goal_input[TeamGoalInputFields.teamId]
    try:
        target = parse_union_value(team_goal_input[TeamGoalInputFields.target])
    except StopIteration:
        raise GoalMutationError(f"Invalid target for teamId {team_id}")

    metric_params = parse_metric_params(team_goal_input.get(TeamGoalInputFields.metricParams))

    return TeamGoal(team_id=team_id, target=target, metric_params=metric_params, **extra)


@dataclass(frozen=True, slots=True)
class RepositoriesUpdateInfo:
    """The information to update the repositories filter in a goal."""

    value: list[tuple[int, str]]
    remove: bool


@dataclass(frozen=True, slots=True)
class StringsListUpdateInfo:
    """The information to update a list of strings field in a goal."""

    value: list[str]
    remove: bool


@dataclass(frozen=True, slots=True)
class GoalUpdateInfo:
    """The information to update an existing Goal."""

    team_goal_deletions: Sequence[int]
    team_goal_assignments: Sequence[TeamGoalTargetAssignment]
    archived: Optional[bool]
    name: Optional[str]
    repositories: Optional[RepositoriesUpdateInfo]
    jira_projects: Optional[StringsListUpdateInfo]
    jira_priorities: Optional[StringsListUpdateInfo]
    jira_issue_types: Optional[StringsListUpdateInfo]


@sentry_span
async def _parse_update_goal_input(
    input: dict[str, Any],
    request: AthenianWebRequest,
    account_id: int,
) -> GoalUpdateInfo:
    deletions = []
    assignments = []

    duplicates = []
    both = []
    invalid_targets = []
    seen = set()
    for change in input.get(UpdateGoalInputFields.teamGoalChanges, ()):
        team_id = change[TeamGoalChangeFields.teamId]
        if team_id in seen:
            duplicates.append(team_id)
        if change.get(TeamGoalChangeFields.remove) and change.get(TeamGoalChangeFields.target):
            both.append(team_id)

        seen.add(team_id)

        if change.get(TeamGoalChangeFields.remove):
            deletions.append(team_id)
        elif change.get(TeamGoalChangeFields.target) is not None:
            try:
                target = parse_union_value(change[TeamGoalChangeFields.target])
            except StopIteration:
                invalid_targets.append(team_id)
                continue

            metric_params = parse_metric_params(change.get(TeamGoalChangeFields.metricParams))
            assignments.append(TeamGoalTargetAssignment(team_id, target, metric_params))
        else:
            invalid_targets.append(team_id)

    def _ids_repr(ids: Sequence[int]) -> str:
        return ",".join(map(str, ids))

    errors = []
    if duplicates:
        errors.append(f"Multiple changes for teamId-s: {_ids_repr(duplicates)}")
    if both:
        errors.append(f"Both remove and new target present for teamId-s: {_ids_repr(both)}")
    if invalid_targets:
        errors.append(f"Invalid target for teamId-s: {_ids_repr(invalid_targets)}")

    if errors:
        raise GoalMutationError("; ".join(errors))

    if (repositories := input.get(UpdateGoalInputFields.repositories)) is not None:
        if (value := repositories[UpdateRepositoriesInputFields.value]) is None:
            repositories = RepositoriesUpdateInfo([], True)
        else:
            repositories = RepositoriesUpdateInfo(
                await parse_request_repositories(value, request, account_id), False,
            )

    strings_field_parser = _StringsListUpdateInfoParser(input)
    jira_projects = strings_field_parser(UpdateGoalInputFields.jiraProjects, lambda v: v)
    jira_priorities = strings_field_parser(
        UpdateGoalInputFields.jiraPriorities, _parse_request_jira_priorities_value,
    )
    jira_issue_types = strings_field_parser(
        UpdateGoalInputFields.jiraIssueTypes, _parse_request_jira_issue_types_value,
    )

    return GoalUpdateInfo(
        team_goal_deletions=deletions,
        team_goal_assignments=assignments,
        archived=input.get(UpdateGoalInputFields.archived),
        name=input.get(UpdateGoalInputFields.name),
        repositories=repositories,
        jira_projects=jira_projects,
        jira_priorities=jira_priorities,
        jira_issue_types=jira_issue_types,
    )


class _StringsListUpdateInfoParser:
    def __init__(self, input: dict[str, Any]):
        self._input = input

    def __call__(
        self,
        field_name: str,
        _parse_value: Callable[[list[str]], list[str]],
    ) -> Optional[StringsListUpdateInfo]:
        if (field := self._input.get(field_name)) is None:
            return None
        if (value := field.get(UpdateRepositoriesInputFields.value)) is None:
            return StringsListUpdateInfo([], True)
        return StringsListUpdateInfo(_parse_value(value), False)


def _parse_request_jira_projects(input: dict[str, Any]) -> Optional[list[str]]:
    return input.get(CreateGoalInputFields.jiraProjects)


def _parse_request_jira_priorities(input: dict[str, Any]) -> Optional[list[str]]:
    if (priorities := input.get(CreateGoalInputFields.jiraPriorities)) is None:
        return priorities
    else:
        return _parse_request_jira_priorities_value(priorities)


def _parse_request_jira_priorities_value(value: list[str]) -> list[str]:
    return sorted({normalize_priority(p) for p in value})


def _parse_request_jira_issue_types(input: dict[str, Any]) -> Optional[list[str]]:
    if (issue_types := input.get(CreateGoalInputFields.jiraIssueTypes)) is None:
        return None
    else:
        return _parse_request_jira_issue_types_value(issue_types)


def _parse_request_jira_issue_types_value(value: list[str]) -> list[str]:
    return sorted({normalize_issue_type(t) for t in value})
