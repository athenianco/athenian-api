"""DB access layer utilities for Align Goals objects."""
from typing import Sequence

from morcilla import Connection
import sqlalchemy as sa

from athenian.api.models.state.models import Goal, Team, TeamGoal
from athenian.api.typing_utils import dataclass

from .exceptions import GoalMutationError


@dataclass(frozen=True)
class GoalCreationInfo:
    """The information required to create a Goal.

    - the goal itself
    - the list of TeamGoal-s

    """

    goal: Goal
    team_goals: Sequence[TeamGoal]


async def insert_goal(creation_info: GoalCreationInfo, sdb_conn: Connection) -> int:
    """Insert the goal and related objects into DB."""
    await _validate_goal_creation_info(creation_info, sdb_conn)
    goal_value = creation_info.goal.create_defaults().explode()
    new_goal_id = await sdb_conn.execute(sa.insert(Goal).values(goal_value))
    team_goals_values = [
        {
            **team_goal.create_defaults().explode(with_primary_keys=True),
            # goal_id can only be set now that Goal has been inserted
            "goal_id": new_goal_id,
        }
        for team_goal in creation_info.team_goals
    ]
    await sdb_conn.execute_many(sa.insert(TeamGoal), team_goals_values)
    return new_goal_id


async def _validate_goal_creation_info(
    creation_info: GoalCreationInfo, sdb_conn: Connection,
) -> None:
    """Execute validation on GoalCreationInfo using the DB."""
    # check that all team exist and belong to the right account
    team_ids = {team_goal.team_id for team_goal in creation_info.team_goals}
    teams_stmt = sa.select([Team.id]).where(
        sa.and_(Team.id.in_(team_ids), Team.owner_id == creation_info.goal.account_id),
    )
    existing_team_ids_rows = await sdb_conn.fetch_all(teams_stmt)
    existing_team_ids = {r[0] for r in existing_team_ids_rows}

    if team_ids != existing_team_ids:
        missing = [team_id for team_id in team_ids if team_id not in existing_team_ids]
        missing_repr = ",".join(str(team_id) for team_id in missing)
        raise GoalMutationError(f"Some teamId-s don't exist or access denied: {missing_repr}")
