from datetime import date, datetime, time, timedelta, timezone
from typing import Any, Dict, Tuple

from ariadne import MutationType
from graphql import GraphQLResolveInfo

from athenian.api.align.goals.dbaccess import delete_goal, GoalCreationInfo, insert_goal
from athenian.api.align.goals.exceptions import GoalMutationError
from athenian.api.align.goals.templates import TEMPLATES_COLLECTION
from athenian.api.align.models import GoalRemoveStatus
from athenian.api.models.state.models import Goal, TeamGoal

mutation = MutationType()


@mutation.field("createGoal")
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
    return {"goal": {"id": new_goal_id}}


@mutation.field("removeGoal")
async def resolve_remove_goal(
    _: Any,
    info: GraphQLResolveInfo,
    accountId: int,
    id: int,
) -> Dict[str, Any]:
    """Remove a Goal and referring TeamGoal-s."""
    async with info.context.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            await delete_goal(accountId, id, sdb_conn)
    remove_status = GoalRemoveStatus(success=True)
    return remove_status.to_dict()


def _parse_create_goal_input(input: Dict[str, Any], account_id: int) -> GoalCreationInfo:
    """Parse CreateGoalInput into GoalCreationInfo."""
    template_id = input["templateId"]
    if template_id not in TEMPLATES_COLLECTION:
        raise GoalMutationError(f"Invalid templateId {template_id}")

    team_goals = [_parse_team_goal_input(tg_input) for tg_input in input["teamGoals"]]
    if not team_goals:
        raise GoalMutationError("At least one teamGoals is required")

    if len({team_goal.team_id for team_goal in team_goals}) < len(team_goals):
        raise GoalMutationError("More than one team goal with the same teamId")

    valid_from, expires_at = _convert_goal_dates(input["validFrom"], input["expiresAt"])

    goal = Goal(
        account_id=account_id,
        template_id=template_id,
        valid_from=valid_from,
        expires_at=expires_at,
    )
    return GoalCreationInfo(goal, team_goals)


def _parse_team_goal_input(team_goal_input: dict) -> TeamGoal:
    """Parse TeamGoalInput into a Team model."""
    try:
        # get the first non null value in GoalTargetInput
        target = next(tgt for tgt in team_goal_input["target"].values() if tgt is not None)
    except StopIteration:
        raise GoalMutationError(f'Invalid target for teamId {team_goal_input["teamId"]}')

    return TeamGoal(team_id=team_goal_input["teamId"], target=target)


def _convert_goal_dates(valid_from: date, expires_at: date) -> Tuple[datetime, datetime]:
    """Convert date objects from API into datetimes."""
    valid_from = datetime.combine(valid_from, time.min, tzinfo=timezone.utc)
    # expiresAt semantic is to include the given day, so datetime is set to the start of the
    # following day
    expires_at = datetime.combine(expires_at + timedelta(days=1), time.min, tzinfo=timezone.utc)
    return (valid_from, expires_at)
