"""DB access layer utilities for Align Goals objects."""
from http import HTTPStatus
from typing import Sequence

import sqlalchemy as sa

from athenian.api.db import conn_in_transaction, Connection, DatabaseLike
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


async def insert_goal(creation_info: GoalCreationInfo, sdb_conn: DatabaseLike) -> int:
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


async def delete_goal(account_id: int, goal_id: int, sdb_conn: Connection) -> None:
    """Delete a goal from DB with related team goals."""
    assert conn_in_transaction(sdb_conn)
    where_clause = sa.and_(Goal.account_id == account_id, Goal.id == goal_id)
    # no rowcount support in morcilla, no delete ... returning support in sqlalchemy / sqlite,
    # so two queries are needed
    select_stmt = sa.select(sa.func.count(Goal.id)).where(where_clause)
    if await sdb_conn.fetch_val(select_stmt) == 0:
        raise GoalMutationError(f"Goal {goal_id} not found", HTTPStatus.NOT_FOUND)

    await sdb_conn.execute(sa.delete(Goal).where(where_clause))


async def _validate_goal_creation_info(
    creation_info: GoalCreationInfo, sdb_conn: DatabaseLike,
) -> None:
    """Execute validation on GoalCreationInfo using the DB."""
    # check that all team exist and belong to the right account
    team_ids = {team_goal.team_id for team_goal in creation_info.team_goals}
    teams_stmt = sa.select([Team.id]).where(
        sa.and_(Team.id.in_(team_ids), Team.owner_id == creation_info.goal.account_id),
    )
    existing_team_ids_rows = await sdb_conn.fetch_all(teams_stmt)
    existing_team_ids = {r[0] for r in existing_team_ids_rows}

    if missing_team_ids := team_ids - existing_team_ids:
        missing_repr = ",".join(str(team_id) for team_id in missing_team_ids)
        raise GoalMutationError(
            f"Some teamId-s don't exist or access denied: {missing_repr}", HTTPStatus.FORBIDDEN,
        )
