"""DB access layer utilities for Align Goals objects."""

from datetime import datetime, timezone
from http import HTTPStatus
from typing import Sequence, Union

import sqlalchemy as sa

from athenian.api.db import conn_in_transaction, Connection, DatabaseLike, dialect_specific_insert
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


@dataclass(frozen=True)
class TeamGoalTargetAssignment:
    """The assignment of a new goal target for a team."""

    team_id: int
    target: Union[int, float, str]


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


async def delete_team_goals(
    account_id: int, goal_id: int, team_ids: Sequence[int], sdb_conn: Connection,
) -> None:
    """Delete a set of TeamGoal-s from DB."""
    assert team_ids
    assert conn_in_transaction(sdb_conn)
    await _validate_team_goal_deletions(account_id, goal_id, team_ids, sdb_conn)

    delete_stmt = sa.delete(TeamGoal).where(
        sa.and_(TeamGoal.goal_id == goal_id, TeamGoal.team_id.in_(team_ids)),
    )
    await sdb_conn.execute(delete_stmt)


async def assign_team_goals(
    account_id: int,
    goal_id: int,
    assignments: Sequence[TeamGoalTargetAssignment],
    sdb_conn: Connection,
) -> None:
    """Assign new TeamGoal-s targets for an existing Goal."""
    await _validate_team_goal_assignments(account_id, goal_id, assignments, sdb_conn)

    now = datetime.now(timezone.utc)
    values = [
        {
            TeamGoal.goal_id.name: goal_id,
            TeamGoal.team_id.name: assign.team_id,
            TeamGoal.target.name: assign.target,
            TeamGoal.created_at.name: now,
            TeamGoal.updated_at.name: now,
        }
        for assign in assignments
    ]

    insert = dialect_specific_insert(sdb_conn)
    stmt = insert(TeamGoal)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=[TeamGoal.goal_id, TeamGoal.team_id],
        set_={
            TeamGoal.target.name: stmt.excluded.target,
            TeamGoal.updated_at.name: stmt.excluded.updated_at,
        },
    )
    await sdb_conn.execute_many(upsert_stmt, values)


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


async def _validate_team_goal_deletions(
    account_id: int, goal_id: int, team_ids: Sequence[int], sdb_conn: DatabaseLike,
) -> None:
    where_clause = sa.and_(
        Goal.account_id == account_id,
        Team.owner_id == account_id,
        TeamGoal.goal_id == goal_id,
        TeamGoal.team_id.in_(team_ids),
    )
    select_stmt = sa.select(
        [TeamGoal.team_id],
    ).join_from(TeamGoal, Goal).join(Team).where(where_clause)
    found = {row[0] for row in await sdb_conn.fetch_all(select_stmt)}
    if missing := [team_id for team_id in team_ids if team_id not in found]:
        missing_repr = ",".join(map(str, missing))
        raise GoalMutationError(
            f"TeamGoal-s to remove not found for teams: {missing_repr}", HTTPStatus.NOT_FOUND,
        )


async def _validate_team_goal_assignments(
    account_id: int,
    goal_id: int,
    assignments: Sequence[TeamGoalTargetAssignment],
    sdb_conn: Connection,
) -> None:
    goal_exists = await sdb_conn.fetch_val(sa.select([1]).where(sa.and_(
        Goal.account_id == account_id, Goal.id == goal_id,
    )))
    if not goal_exists:
        raise GoalMutationError(
            f"Goal {goal_id} doesn't exist or access denied", HTTPStatus.NOT_FOUND,
        )

    teams_stmt = sa.select([Team.id]).where(
        sa.and_(
            Team.id.in_(a.team_id for a in assignments), Team.owner_id == account_id,
        ),
    )
    found_teams = set(r[0] for r in await sdb_conn.fetch_all(teams_stmt))
    if missing_teams := [a.team_id for a in assignments if a.team_id not in found_teams]:
        missing_teams_repr = ",".join(map(str, missing_teams))
        raise GoalMutationError(
            f"Team-s don't exist or access denied: {missing_teams_repr}", HTTPStatus.NOT_FOUND,
        )
