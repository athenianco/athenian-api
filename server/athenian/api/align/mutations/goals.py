from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ariadne import MutationType
from graphql import GraphQLResolveInfo

from athenian.api.align.exceptions import GoalMutationError, GoalTemplateNotFoundError
from athenian.api.align.goals.dates import goal_dates_to_datetimes
from athenian.api.align.goals.dbaccess import (
    GoalCreationInfo,
    TeamGoalTargetAssignment,
    assign_team_goals,
    delete_goal,
    delete_team_goals,
    dump_goal_repositories,
    get_goal_template_from_db,
    insert_goal,
    update_goal,
)
from athenian.api.align.models import (
    CreateGoalInputFields,
    GoalRemoveStatus,
    GoalRepositoriesFields,
    MutateGoalResult,
    MutateGoalResultGoal,
    TeamGoalChangeFields,
    TeamGoalInputFields,
    UpdateGoalInputFields,
)
from athenian.api.ariadne import ariadne_disable_default_user
from athenian.api.db import Database
from athenian.api.internal.reposet import RepoIdentitiesMapperFactory
from athenian.api.models.state.models import Goal, GoalTemplate, TeamGoal
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
    creation_info = await _parse_create_goal_input(
        input, accountId, info.context.sdb, RepoIdentitiesMapperFactory(accountId, info.context),
    )

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
    update = _parse_update_goal_input(input)
    goal_id = input[UpdateGoalInputFields.goalId]
    result = MutateGoalResult(MutateGoalResultGoal(goal_id)).to_dict()

    deletions = update.team_goal_deletions
    assignments = update.team_goal_assignments
    if not deletions and not assignments and update.archived is None:
        return result

    async with info.context.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            if deletions:
                await delete_team_goals(accountId, goal_id, deletions, sdb_conn)
            if assignments:
                await assign_team_goals(accountId, goal_id, assignments, sdb_conn)
            if update.archived is not None:
                await update_goal(accountId, goal_id, sdb_conn, archived=update.archived)

    return result


@sentry_span
async def _parse_create_goal_input(
    input: dict[str, Any],
    account: int,
    sdb: Database,
    repo_identities_mapper_factory: RepoIdentitiesMapperFactory,
) -> GoalCreationInfo:
    """Parse CreateGoalInput into GoalCreationInfo."""
    template_id = input[CreateGoalInputFields.templateId]
    template = await get_goal_template_from_db(template_id, sdb)
    if template[GoalTemplate.account_id.name] != account:
        raise GoalTemplateNotFoundError(template_id)

    team_goals = [
        _parse_team_goal_input(tg_input) for tg_input in input[CreateGoalInputFields.teamGoals]
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

    repositories = await _parse_input_repositories(input, repo_identities_mapper_factory)
    goal = Goal(
        account_id=account,
        name=template[GoalTemplate.name.name],
        metric=template[GoalTemplate.metric.name],
        valid_from=valid_from,
        expires_at=expires_at,
        repositories=repositories,
    )
    return GoalCreationInfo(goal, team_goals)


@sentry_span
def _parse_team_goal_input(team_goal_input: dict) -> TeamGoal:
    """Parse TeamGoalInput into a Team model."""
    team_id = team_goal_input[TeamGoalInputFields.teamId]
    try:
        target = _parse_team_goal_target(team_goal_input[TeamGoalInputFields.target])
    except StopIteration:
        raise GoalMutationError(f"Invalid target for teamId {team_id}")

    return TeamGoal(team_id=team_id, target=target)


def _parse_team_goal_target(team_goal_target: dict) -> int | float | str:
    """Get the first non null value in GoalTargetInput."""
    return next(tgt for tgt in team_goal_target.values() if tgt is not None)


@dataclass(frozen=True)
class GoalUpdateInfo:
    """The information to update an existing Goal."""

    team_goal_deletions: Sequence[int]
    team_goal_assignments: Sequence[TeamGoalTargetAssignment]
    archived: Optional[bool]


@sentry_span
def _parse_update_goal_input(input: dict[str, Any]) -> GoalUpdateInfo:
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

    return GoalUpdateInfo(deletions, assignments, input.get("archived"))


async def _parse_input_repositories(
    input: dict[str, Any],
    repo_identities_mapper_factory: RepoIdentitiesMapperFactory,
) -> Optional[list[list]]:
    try:
        req_repos = input[CreateGoalInputFields.repositories][GoalRepositoriesFields.value]
    except KeyError:
        req_repos = None

    if req_repos is None:
        repositories = None
    else:
        mapper = await repo_identities_mapper_factory()
        repositories = dump_goal_repositories(mapper.prefixed_names_to_identities(req_repos))
    return repositories
