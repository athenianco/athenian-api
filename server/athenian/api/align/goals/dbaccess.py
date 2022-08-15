"""DB access layer utilities for Align Goals objects."""

from dataclasses import dataclass
from datetime import datetime, timezone
from http import HTTPStatus
import logging
from typing import Iterable, Sequence

import sqlalchemy as sa

from athenian.api.align.exceptions import GoalMutationError, GoalTemplateNotFoundError
from athenian.api.align.goals.templates import TEMPLATES_COLLECTION
from athenian.api.db import (
    Connection,
    DatabaseLike,
    Row,
    conn_in_transaction,
    dialect_specific_insert,
    integrity_errors,
)
from athenian.api.models.state.models import Goal, GoalTemplate, Team, TeamGoal
from athenian.api.tracing import sentry_span


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
    target: int | float | str


@sentry_span
async def insert_goal(creation_info: GoalCreationInfo, sdb_conn: DatabaseLike) -> int:
    """Insert the goal and related objects into DB."""
    await _validate_goal_creation_info(creation_info, sdb_conn)

    goal_value = creation_info.goal.create_defaults().explode()
    try:
        new_goal_id = await sdb_conn.execute(sa.insert(Goal).values(goal_value))
    except integrity_errors:  # uc_goal_name constraint can fail here
        raise GoalMutationError(
            "There is an existing goal with the same name for the same time interval",
        )

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


@sentry_span
async def delete_goal(account_id: int, goal_id: int, sdb_conn: Connection) -> None:
    """Delete a goal from DB with related team goals."""
    assert await conn_in_transaction(sdb_conn)
    where_clause = sa.and_(Goal.account_id == account_id, Goal.id == goal_id)
    # no rowcount support in morcilla, no delete ... returning support in sqlalchemy / sqlite,
    # so two queries are needed
    select_stmt = sa.select(sa.func.count(Goal.id)).where(where_clause)
    if await sdb_conn.fetch_val(select_stmt) == 0:
        raise GoalMutationError(f"Goal {goal_id} not found", HTTPStatus.NOT_FOUND)

    await sdb_conn.execute(sa.delete(Goal).where(where_clause))


@sentry_span
async def delete_team_goals(
    account_id: int,
    goal_id: int,
    team_ids: Sequence[int],
    sdb_conn: Connection,
) -> None:
    """Delete a set of TeamGoal-s from DB."""
    assert team_ids
    assert await conn_in_transaction(sdb_conn)
    await _validate_team_goal_deletions(account_id, goal_id, team_ids, sdb_conn)

    delete_stmt = sa.delete(TeamGoal).where(
        sa.and_(TeamGoal.goal_id == goal_id, TeamGoal.team_id.in_(team_ids)),
    )
    await sdb_conn.execute(delete_stmt)


@sentry_span
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

    insert = await dialect_specific_insert(sdb_conn)
    stmt = insert(TeamGoal)
    upsert_stmt = stmt.on_conflict_do_update(
        index_elements=TeamGoal.__table__.primary_key.columns,
        set_={
            TeamGoal.target.name: stmt.excluded.target,
            TeamGoal.updated_at.name: stmt.excluded.updated_at,
        },
    )
    await sdb_conn.execute_many(upsert_stmt, values)


@sentry_span
async def update_goal(
    account_id: int,
    goal_id: int,
    sdb_conn: Connection,
    *,
    archived: bool,
) -> None:
    """Update the properties of an existing Goal."""
    assert await conn_in_transaction(sdb_conn)

    # morcilla/asyncpg don't return update rowcount, two queries are needed
    where = sa.and_(Goal.account_id == account_id, Goal.id == goal_id)
    select_stmt = sa.select(Goal).where(where).with_for_update()
    if await sdb_conn.fetch_one(select_stmt) is None:
        raise GoalMutationError(f"Goal {goal_id} not found", HTTPStatus.NOT_FOUND)

    values = {Goal.archived.name: archived, Goal.updated_at: datetime.now(timezone.utc)}
    update_stmt = sa.update(Goal).where(where).values(values)
    await sdb_conn.execute(update_stmt)


@sentry_span
async def fetch_team_goals(
    account: int,
    team_ids: Iterable[int],
    sdb: DatabaseLike,
) -> Sequence[Row]:
    """Fetch the TeamGoals from DB related to a set of teams.

    TeamGoal linked to archived Goal are not included.
    Result is ordered by Goal id.
    """
    stmt = (
        sa.select(TeamGoal.team_id, TeamGoal.target, Goal)
        .join_from(TeamGoal, Goal, TeamGoal.goal_id == Goal.id)
        .where(
            sa.and_(TeamGoal.team_id.in_(team_ids), Goal.account_id == account, ~Goal.archived),
        )
        .order_by(Goal.id, TeamGoal.team_id)
    )
    return await sdb.fetch_all(stmt)


@sentry_span
async def delete_empty_goals(account: int, sdb_conn: DatabaseLike) -> None:
    """Delete all account Goal-s having no more TeamGoal-s assigned."""
    delete_stmt = sa.delete(Goal).where(
        sa.and_(
            Goal.account_id == account,
            sa.not_(sa.exists().where(TeamGoal.goal_id == Goal.id)),
            # inefficient, generates a subquery:
            # Goal.id.not_in(sa.select(TeamGoal.goal_id).distinct()),
        ),
    )
    await sdb_conn.execute(delete_stmt)


async def get_goal_template_from_db(template_id: int, sdb: DatabaseLike) -> Row:
    """Return a GoalTemplate."""
    stmt = sa.select(GoalTemplate).where(GoalTemplate.id == template_id)
    template = await sdb.fetch_one(stmt)
    if template is None:
        raise GoalTemplateNotFoundError(template_id)
    return template


async def get_goal_templates_from_db(account: int, sdb: DatabaseLike) -> Row:
    """Return all account GoalTemplate-s, ordered by id."""
    stmt = (
        sa.select(GoalTemplate).where(GoalTemplate.account_id == account).order_by(GoalTemplate.id)
    )
    return await sdb.fetch_all(stmt)


async def insert_goal_template(account_id: int, name: str, metric: str, sdb: DatabaseLike) -> int:
    """Insert a new Goal Template."""
    model = GoalTemplate(account_id=account_id, name=name, metric=metric)
    values = model.create_defaults().explode()
    return await sdb.execute(sa.insert(GoalTemplate).values(values))


async def delete_goal_template_from_db(template_id: int, sdb: DatabaseLike) -> None:
    """Delete a Goal Template."""
    await sdb.execute(sa.delete(GoalTemplate).where(GoalTemplate.id == template_id))


async def update_goal_template_in_db(template_id, name: str, sdb: DatabaseLike) -> None:
    """Update a Goal Template."""
    values = {GoalTemplate.name: name, GoalTemplate.updated_at: datetime.now(timezone.utc)}
    stmt = sa.update(GoalTemplate).where(GoalTemplate.id == template_id).values(values)
    await sdb.execute(stmt)


@sentry_span
async def create_default_goal_templates(account: int, sdb_conn: DatabaseLike) -> None:
    """Create the set of default goal templates for the account."""
    log = logging.getLogger(f"{__name__}.create_default_goal_templates")
    log.info("creating for account %d", account)
    models = [
        GoalTemplate(metric=template_def["metric"], name=template_def["name"], account_id=account)
        for template_def in TEMPLATES_COLLECTION.values()
    ]
    values = [model.create_defaults().explode(with_primary_keys=False) for model in models]
    # skip existing templates with the same name
    insert = await dialect_specific_insert(sdb_conn)
    stmt = insert(GoalTemplate).on_conflict_do_nothing()
    await sdb_conn.execute_many(stmt, values)


@sentry_span
async def _validate_goal_creation_info(
    creation_info: GoalCreationInfo,
    sdb_conn: DatabaseLike,
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


@sentry_span
async def _validate_team_goal_deletions(
    account_id: int,
    goal_id: int,
    team_ids: Sequence[int],
    sdb_conn: DatabaseLike,
) -> None:
    where_clause = sa.and_(
        Goal.account_id == account_id, Team.owner_id == account_id, TeamGoal.goal_id == goal_id,
    )
    select_stmt = (
        sa.select(TeamGoal.team_id).join_from(TeamGoal, Goal).join(Team).where(where_clause)
    )
    found = {row[0] for row in await sdb_conn.fetch_all(select_stmt)}

    if missing := [team_id for team_id in team_ids if team_id not in found]:
        missing_repr = ",".join(map(str, missing))
        raise GoalMutationError(
            f"TeamGoal-s to remove not found for teams: {missing_repr}", HTTPStatus.NOT_FOUND,
        )

    if len(set(team_ids)) == len(found):
        raise GoalMutationError("Impossible to remove all TeamGoal-s from the Goal")


async def _validate_team_goal_assignments(
    account_id: int,
    goal_id: int,
    assignments: Sequence[TeamGoalTargetAssignment],
    sdb_conn: Connection,
) -> None:
    goal_exists = await sdb_conn.fetch_val(
        sa.select([1]).where(sa.and_(Goal.account_id == account_id, Goal.id == goal_id)),
    )
    if not goal_exists:
        raise GoalMutationError(
            f"Goal {goal_id} doesn't exist or access denied", HTTPStatus.NOT_FOUND,
        )

    teams_stmt = sa.select([Team.id]).where(
        sa.and_(Team.id.in_(a.team_id for a in assignments), Team.owner_id == account_id),
    )
    found_teams = set(r[0] for r in await sdb_conn.fetch_all(teams_stmt))
    if missing_teams := [a.team_id for a in assignments if a.team_id not in found_teams]:
        missing_teams_repr = ",".join(map(str, missing_teams))
        raise GoalMutationError(
            f"Team-s don't exist or access denied: {missing_teams_repr}", HTTPStatus.NOT_FOUND,
        )
