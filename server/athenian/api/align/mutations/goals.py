from __future__ import annotations

from typing import Any, Dict, Sequence, Union

from ariadne import MutationType
from graphql import GraphQLResolveInfo

from athenian.api.align.exceptions import GoalMutationError
from athenian.api.align.goals.dates import goal_dates_to_datetimes
from athenian.api.align.goals.dbaccess import assign_team_goals, delete_goal, delete_team_goals, \
    GoalCreationInfo, insert_goal, TeamGoalTargetAssignment
from athenian.api.align.goals.templates import TEMPLATES_COLLECTION
from athenian.api.align.models import CreateGoalInputFields, GoalRemoveStatus, MutateGoalResult, \
    MutateGoalResultGoal, TeamGoalChangeFields, TeamGoalInputFields, UpdateGoalInputFields
from athenian.api.models.state.models import Goal, TeamGoal
from athenian.api.tracing import sentry_span
from athenian.api.typing_utils import dataclass

mutation = MutationType()


@mutation.field("createGoal")
@sentry_span
async def resolve_create_goal(
    _: Any,
    info: GraphQLResolveInfo,
    accountId: int,
    input: Dict[str, Any],
) -> Dict[str, Any]:
    """Create a Goal."""
    creation_info = _parse_create_goal_input(input, accountId)

    async with info.context.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            new_goal_id = await insert_goal(creation_info, sdb_conn)

    # TODO: return the complete response
    return MutateGoalResult(MutateGoalResultGoal(new_goal_id)).to_dict()


@mutation.field("removeGoal")
@sentry_span
async def resolve_remove_goal(
    _: Any,
    info: GraphQLResolveInfo,
    accountId: int,
    goalId: int,
) -> Dict[str, Any]:
    """Remove a Goal and referring TeamGoal-s."""
    async with info.context.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            await delete_goal(accountId, goalId, sdb_conn)
    remove_status = GoalRemoveStatus(success=True)
    return remove_status.to_dict()


@mutation.field("updateGoal")
@sentry_span
async def resolve_update_goal(
    _: Any,
    info: GraphQLResolveInfo,
    accountId: int,
    input: dict,
) -> dict:
    """Update an existing Goal."""
    update = _parse_update_goal_input(input, accountId)
    goal_id = input[UpdateGoalInputFields.goalId]
    result = MutateGoalResult(MutateGoalResultGoal(goal_id)).to_dict()

    deletions = update.team_goal_deletions
    assignments = update.team_goal_assignments
    if not deletions and not assignments:
        return result

    async with info.context.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            if deletions:
                await delete_team_goals(accountId, goal_id, deletions, sdb_conn)
            if assignments:
                await assign_team_goals(accountId, goal_id, assignments, sdb_conn)

    return result


def _parse_create_goal_input(input: Dict[str, Any], account_id: int) -> GoalCreationInfo:
    """Parse CreateGoalInput into GoalCreationInfo."""
    template_id = input[CreateGoalInputFields.templateId]
    if template_id not in TEMPLATES_COLLECTION:
        raise GoalMutationError(f"Invalid templateId {template_id}")

    team_goals = [
        _parse_team_goal_input(tg_input)
        for tg_input in input[CreateGoalInputFields.teamGoals]
    ]
    if not team_goals:
        raise GoalMutationError("At least one teamGoals is required")

    if len({team_goal.team_id for team_goal in team_goals}) < len(team_goals):
        raise GoalMutationError("More than one team goal with the same teamId")

    valid_from, expires_at = goal_dates_to_datetimes(
        input[CreateGoalInputFields.validFrom], input[CreateGoalInputFields.expiresAt],
    )
    if expires_at < valid_from:
        raise GoalMutationError("Goal expiresAt cannot precede validFrom")

    goal = Goal(
        account_id=account_id,
        template_id=template_id,
        valid_from=valid_from,
        expires_at=expires_at,
    )
    return GoalCreationInfo(goal, team_goals)


def _parse_team_goal_input(team_goal_input: dict) -> TeamGoal:
    """Parse TeamGoalInput into a Team model."""
    team_id = team_goal_input[TeamGoalInputFields.teamId]
    try:
        target = _parse_team_goal_target(team_goal_input[TeamGoalInputFields.target])
    except StopIteration:
        raise GoalMutationError(f"Invalid target for teamId {team_id}")

    return TeamGoal(team_id=team_id, target=target)


def _parse_team_goal_target(team_goal_target: dict) -> Union[int, float, str]:
    """Get the first non null value in GoalTargetInput."""
    return next(tgt for tgt in team_goal_target.values() if tgt is not None)


@dataclass(frozen=True)
class GoalUpdateInfo:
    """The information to update an existing Goal."""

    team_goal_deletions: Sequence[int]
    team_goal_assignments: Sequence[TeamGoalTargetAssignment]


def _parse_update_goal_input(input: Dict[str, Any], account_id: int) -> GoalUpdateInfo:
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
                target = _parse_team_goal_target(change[TeamGoalChangeFields.target])
            except StopIteration:
                invalid_targets.append(team_id)
            else:
                assignments.append(TeamGoalTargetAssignment(team_id, target))
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

    return GoalUpdateInfo(deletions, assignments)
