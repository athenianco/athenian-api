from datetime import date, datetime, time, timedelta, timezone
from http import HTTPStatus
from typing import Any, Dict, Sequence, Tuple

from ariadne import MutationType
from graphql import GraphQLResolveInfo
from morcilla import Connection
import sqlalchemy as sa

from athenian.api.align.goals.templates import TEMPLATES_COLLECTION
from athenian.api.models.state.models import Goal, Team, TeamGoal
from athenian.api.models.web import GenericError
from athenian.api.response import ResponseError
from athenian.api.typing_utils import dataclass

mutation = MutationType()


@mutation.field("createGoal")
async def resolve_create_goal(
    _: Any,
    info: GraphQLResolveInfo,
    accountId: int,
    input: Dict[str, Any],
):
    """Create a Goal."""
    parse_result = _parse_create_goal_input(input, accountId)

    async with info.context.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            await _parse_result_db_validation(parse_result, sdb_conn)
            new_goal_id = await _insert_db_goal(parse_result, sdb_conn)

    # TODO: return the complete response
    return {"goal": {"id": new_goal_id}}


class GoalMutationError(ResponseError):
    """An error during a goal mutation handling."""

    def __init__(self, text):
        """Init the GoalMutationError."""
        wrapped_error = GenericError(
            type="/errors/align/GoalMutationError",
            status=HTTPStatus.BAD_REQUEST,
            detail=text,
            title="Goal mutation error",
        )
        super().__init__(wrapped_error)


@dataclass(frozen=True)
class _CreateGoalInputParseResult:
    goal: Goal
    team_goals: Sequence[TeamGoal]


def _parse_create_goal_input(
    input: Dict[str, Any],
    account_id: int,
) -> _CreateGoalInputParseResult:
    """Parse CreateGoalInput into Goal and TeamGoal application models."""
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
    return _CreateGoalInputParseResult(goal, team_goals)


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


async def _parse_result_db_validation(
    parse_result: _CreateGoalInputParseResult, sdb_conn: Connection,
) -> None:
    """Execute validation on parse result using the DB."""
    # check that all team exist and belong to the right account
    team_ids = {team_goal.team_id for team_goal in parse_result.team_goals}
    teams_stmt = sa.select([Team.id]).where(
        sa.and_(Team.id.in_(team_ids), Team.owner_id == parse_result.goal.account_id),
    )
    existing_team_ids_rows = await sdb_conn.fetch_all(teams_stmt)
    existing_team_ids = {r[0] for r in existing_team_ids_rows}

    if team_ids != existing_team_ids:
        missing = [team_id for team_id in team_ids if team_id not in existing_team_ids]
        missing_repr = ",".join(str(team_id) for team_id in missing)
        raise GoalMutationError(f"Some teamId-s don't exist or access denied: {missing_repr}")


async def _insert_db_goal(parse_result: _CreateGoalInputParseResult, sdb_conn: Connection) -> int:
    """Insert the goal and related objects into DB."""
    goal_value = parse_result.goal.create_defaults().explode()
    new_goal_id = await sdb_conn.execute(sa.insert(Goal).values(goal_value))
    team_goals_values = [
        {
            **team_goal.create_defaults().explode(with_primary_keys=True),
            # goal_id can only be set now that Goal has been inserted
            "goal_id": new_goal_id,
        }
        for team_goal in parse_result.team_goals
    ]
    await sdb_conn.execute_many(sa.insert(TeamGoal), team_goals_values)
    return new_goal_id
