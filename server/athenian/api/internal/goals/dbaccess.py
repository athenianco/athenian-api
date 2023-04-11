"""DB access layer utilities for Align Goals objects."""

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from http import HTTPStatus
import logging
from typing import Any, Iterable, Mapping, Optional, Sequence

import sqlalchemy as sa

from athenian.api.db import (
    Connection,
    DatabaseLike,
    Row,
    conn_in_transaction,
    dialect_specific_insert,
    is_postgresql,
)
from athenian.api.internal.goals.exceptions import (
    GoalMutationError,
    GoalNotFoundError,
    GoalTemplateNotFoundError,
    TeamGoalNotFoundError,
)
from athenian.api.internal.goals.templates import TEMPLATES_COLLECTION
from athenian.api.internal.team import fetch_teams_recursively
from athenian.api.models.state.models import Goal, GoalTemplate, Team, TeamGoal
from athenian.api.serialization import deserialize_timedelta
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
    metric_params: Optional[dict]

    def equals_db_row(self, row: Mapping[str, Any]) -> bool:
        """Return True if this assignment is already equal to the DB row."""
        return (
            row[TeamGoal.target.name] == self.target
            and row[TeamGoal.metric_params.name] == self.metric_params
        )


@sentry_span
async def fetch_goal(account: int, id: int, sdb: DatabaseLike) -> Row:
    """Load the Goal by ID and account ID."""
    return await sdb.fetch_one(
        sa.select(Goal).where(
            Goal.account_id == account,
            Goal.id == id,
        ),
    )


async def fetch_goal_account(goal_id: int, sdb: DatabaseLike) -> int:
    """Fetch the account owner of the goal, or fail if the Goal is not found."""
    res = await sdb.fetch_val(sa.select(Goal.account_id).where(Goal.id == goal_id))
    if res is None:
        raise GoalNotFoundError(goal_id)
    return res


@sentry_span
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


@sentry_span
async def delete_goal(account_id: int, goal_id: int, sdb_conn: Connection) -> None:
    """Delete a goal from DB with related team goals."""
    assert await conn_in_transaction(sdb_conn)
    where_clause = sa.and_(Goal.account_id == account_id, Goal.id == goal_id)
    # no rowcount support in morcilla, no delete ... returning support in sqlalchemy / sqlite,
    # so two queries are needed
    select_stmt = sa.select(Goal.id).where(where_clause)
    if await sdb_conn.fetch_val(select_stmt) is None:
        raise GoalMutationError(f"Goal {goal_id} not found or access denied", HTTPStatus.NOT_FOUND)

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
            TeamGoal.metric_params.name: assign.metric_params,
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
            TeamGoal.metric_params.name: stmt.excluded.metric_params,
            TeamGoal.updated_at.name: stmt.excluded.updated_at,
        },
    )
    await sdb_conn.execute_many(upsert_stmt, values)


@sentry_span
async def replace_team_goals(
    account_id: int,
    goal_id: int,
    assignments: Sequence[TeamGoalTargetAssignment],
    sdb_conn: Connection,
) -> None:
    """Replace the TeamGoal-s assigned to a Goal."""
    current_stmt = (
        sa.select(TeamGoal.team_id, TeamGoal.target, TeamGoal.metric_params)
        .join_from(TeamGoal, Goal, TeamGoal.goal_id == Goal.id)
        .where(Goal.account_id == account_id, TeamGoal.goal_id == goal_id)
    )
    current = {r[TeamGoal.team_id.name]: r for r in await sdb_conn.fetch_all(current_stmt)}
    new_teams = {assign.team_id for assign in assignments}
    to_delete = [t for t in current if t not in new_teams]

    changed = []
    for assign in assignments:
        try:
            existing_row = current[assign.team_id]
        except KeyError:
            changed.append(assign)  # newly assigned team
        else:
            if not assign.equals_db_row(existing_row):
                changed.append(assign)  # team goal must be updated

    await assign_team_goals(account_id, goal_id, changed, sdb_conn)
    if to_delete:
        await delete_team_goals(account_id, goal_id, to_delete, sdb_conn)


@sentry_span
async def update_goal(
    account_id: int,
    goal_id: int,
    sdb_conn: Connection,
    **kwargs,
) -> None:
    """Update the properties of an existing Goal.

    Must check access in advance!
    """
    assert await conn_in_transaction(sdb_conn)
    values = {**kwargs, Goal.updated_at: datetime.now(timezone.utc)}
    await sdb_conn.execute(
        sa.update(Goal).where(Goal.account_id == account_id, Goal.id == goal_id).values(values),
    )
    # for now: update all the connected team goals
    # we will remove this once we switch to independent team goals
    # fmt: off
    dependent_values = {
        k: v for k, v in values.items() if k in (
            Goal.repositories.name,
            Goal.jira_projects.name,
            Goal.jira_priorities.name,
            Goal.jira_issue_types.name,
        )
    }
    # fmt: on
    if dependent_values:
        dependent_values[TeamGoal.updated_at] = datetime.now(timezone.utc)
        await sdb_conn.execute(
            sa.update(TeamGoal).where(TeamGoal.goal_id == goal_id).values(dependent_values),
        )


class GoalColumnAlias(Enum):
    """Aliases for Goal columns returned by fetch_team_goals.

    Aliases are needed since column names are shared between TeamGoal and the joined Goal table.

    """

    REPOSITORIES = f"goal_{Goal.repositories.name}"
    JIRA_PROJECTS = f"goal_{Goal.jira_projects.name}"
    JIRA_PRIORITIES = f"goal_{Goal.jira_priorities.name}"
    JIRA_ISSUE_TYPES = f"goal_{Goal.jira_issue_types.name}"
    METRIC_PARAMS = f"goal_{Goal.metric_params.name}"


AliasedGoalColumns = {getattr(Goal, f.value[5:].lower()).name: f.value for f in GoalColumnAlias}
TeamGoalColumns = {f: f for f in AliasedGoalColumns}


@sentry_span
async def fetch_team_goals(
    account: int,
    team_ids: Iterable[int],
    sdb: DatabaseLike,
) -> Sequence[Row]:
    """Fetch the TeamGoals from DB related to a set of teams.

    TeamGoal linked to archived Goal are not included.
    Result is ordered by Goal id and team id.
    Columns from joined Goal are included, with `goal_` prefix in case of conflict.
    """
    goal_rows = (
        Goal.id,
        Goal.valid_from,
        Goal.expires_at,
        Goal.name,
        Goal.metric,
        Goal.repositories.label(GoalColumnAlias.REPOSITORIES.value),
        Goal.jira_projects.label(GoalColumnAlias.JIRA_PROJECTS.value),
        Goal.jira_priorities.label(GoalColumnAlias.JIRA_PRIORITIES.value),
        Goal.jira_issue_types.label(GoalColumnAlias.JIRA_ISSUE_TYPES.value),
        Goal.metric_params.label(GoalColumnAlias.METRIC_PARAMS.value),
    )
    stmt = (
        sa.select(TeamGoal, *goal_rows)
        .join_from(TeamGoal, Goal, TeamGoal.goal_id == Goal.id)
        .where(TeamGoal.team_id.in_(team_ids), Goal.account_id == account, ~Goal.archived)
        .order_by(Goal.id, TeamGoal.team_id)
    )
    return await sdb.fetch_all(stmt)


async def get_goal_template_from_db(template_id: int, sdb: DatabaseLike) -> Row:
    """Return a GoalTemplate."""
    stmt = sa.select(GoalTemplate).where(GoalTemplate.id == template_id)
    template = await sdb.fetch_one(stmt)
    if template is None:
        raise GoalTemplateNotFoundError(template_id)
    return template


async def get_goal_templates_from_db(
    account: int,
    exclude_with_metric_params: bool,
    sdb: DatabaseLike,
) -> list[Row]:
    """Return all account GoalTemplate-s, ordered by id."""
    where = GoalTemplate.account_id == account
    if exclude_with_metric_params:
        exclude_expr = GoalTemplate.metric_params.is_(None)
        # with sqlite both SQL NULL and JSON null can be found on db
        if not await is_postgresql(sdb):
            exclude_expr = sa.or_(exclude_expr, GoalTemplate.metric_params == sa.JSON.NULL)
        where = sa.and_(where, exclude_expr)
    stmt = sa.select(GoalTemplate).where(where).order_by(GoalTemplate.id)
    return await sdb.fetch_all(stmt)


async def insert_goal_template(sdb: DatabaseLike, **kwargs: Any) -> int:
    """Insert a new Goal Template."""
    model = GoalTemplate(**kwargs)
    values = model.create_defaults().explode()
    return await sdb.execute(sa.insert(GoalTemplate).values(values))


async def delete_goal_template_from_db(template_id: int, sdb: DatabaseLike) -> None:
    """Delete a Goal Template."""
    await sdb.execute(sa.delete(GoalTemplate).where(GoalTemplate.id == template_id))


async def update_goal_template_in_db(template_id: int, sdb: DatabaseLike, **values: Any) -> None:
    """Update a Goal Template."""
    values[GoalTemplate.updated_at.name] = datetime.now(timezone.utc)
    stmt = sa.update(GoalTemplate).where(GoalTemplate.id == template_id).values(values)
    await sdb.execute(stmt)


@sentry_span
async def create_default_goal_templates(account: int, sdb_conn: DatabaseLike) -> None:
    """Create the set of default goal templates for the account."""
    log = logging.getLogger(f"{__name__}.create_default_goal_templates")
    log.info("creating for account %d", account)
    models = [
        GoalTemplate(
            metric=template_def["metric"],
            name=template_def["name"],
            account_id=account,
            metric_params=template_def.get("metric_params"),
        )
        for template_def in TEMPLATES_COLLECTION
    ]
    values = [model.create_defaults().explode(with_primary_keys=False) for model in models]
    # skip existing templates with the same name
    insert = await dialect_specific_insert(sdb_conn)
    stmt = insert(GoalTemplate).on_conflict_do_nothing()
    await sdb_conn.execute_many(stmt, values)


class MetricParamNames(Enum):
    """Names of the possible parameters for a metric."""

    threshold = "threshold"


def convert_metric_params_datatypes(metric_params: Optional[dict]) -> dict:
    """Convert the metric_params column to the format used in MetricWithParams and calculation."""
    if not metric_params:
        return {}
    parsed = metric_params.copy()
    if isinstance(threshold := metric_params.get(MetricParamNames.threshold.name), str):
        parsed[MetricParamNames.threshold.name] = deserialize_timedelta(threshold)
    return parsed


async def unassign_team_from_goal(
    account: int,
    goal_id: int,
    team: int,
    sdb_conn: Connection,
) -> bool:
    """Unassign a team from a goal.

    Delete the goal if that was the last assigned team.
    Raise TeamGoalNotFoundError if the team is not assigned to the goal.
    Return True if the goal still exists after the operation.
    """
    where = [Goal.account_id == account, Goal.id == goal_id]
    join_on = sa.and_(Goal.id == TeamGoal.goal_id, TeamGoal.team_id == team)
    select_stmt = (
        sa.select(TeamGoal.team_id)
        .select_from(sa.outerjoin(Goal, TeamGoal, join_on))
        .where(*where)
    )
    assigned_team = await sdb_conn.fetch_val(select_stmt)
    if assigned_team is None:
        raise TeamGoalNotFoundError(goal_id, team)

    delete_stmt = sa.delete(TeamGoal).where(TeamGoal.goal_id == goal_id, TeamGoal.team_id == team)
    await sdb_conn.execute(delete_stmt)

    return not await _delete_goal_if_empty(account, goal_id, sdb_conn)


async def unassign_team_from_goal_recursive(
    account: int,
    goal_id: int,
    team: int,
    sdb_conn: Connection,
) -> bool:
    """Unassign a team and all its descendants from a goal.

    Delete the goal if the last assigned team was removed this way.
    Return True if the goal still exists after the operation.

    """
    teams = await fetch_teams_recursively(account, sdb_conn, root_team_ids=[team])
    team_ids = [t[Team.id.name] for t in teams]
    delete_stmt = sa.delete(TeamGoal).where(
        TeamGoal.goal_id == goal_id, TeamGoal.team_id.in_(team_ids),
    )
    await sdb_conn.execute(delete_stmt)
    return not await _delete_goal_if_empty(account, goal_id, sdb_conn)


async def _delete_goal_if_empty(account: int, goal_id: int, sdb_conn: Connection) -> bool:
    """Delete the goal when it has no assigned teams."""
    if not await sdb_conn.fetch_val(sa.select(1).where(TeamGoal.goal_id == goal_id)):
        delete_goal_stmt = sa.delete(Goal).where(Goal.id == goal_id, Goal.account_id == account)
        await sdb_conn.execute(delete_goal_stmt)
        return True
    return False


@sentry_span
async def _validate_goal_creation_info(
    creation_info: GoalCreationInfo,
    sdb_conn: DatabaseLike,
) -> None:
    """Execute validation on GoalCreationInfo using the DB."""
    # check that all team exist and belong to the right account
    team_ids = {team_goal.team_id for team_goal in creation_info.team_goals}
    teams_stmt = sa.select(Team.id).where(
        sa.and_(Team.id.in_(team_ids), Team.owner_id == creation_info.goal.account_id),
    )
    existing_team_ids_rows = await sdb_conn.fetch_all(teams_stmt)
    existing_team_ids = {r[0] for r in existing_team_ids_rows}

    if missing_team_ids := team_ids - existing_team_ids:
        missing_repr = ",".join(str(team_id) for team_id in missing_team_ids)
        # TODO(gaetano-guerriero): change to a more generic exception when graphql is dropped
        raise GoalMutationError(
            f"Some teams don't exist or access denied: {missing_repr}", HTTPStatus.NOT_FOUND,
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
        sa.select(1).where(sa.and_(Goal.account_id == account_id, Goal.id == goal_id)),
    )
    if not goal_exists:
        raise GoalMutationError(
            f"Goal {goal_id} doesn't exist or access denied", HTTPStatus.NOT_FOUND,
        )

    teams_stmt = sa.select(Team.id).where(
        sa.and_(Team.id.in_(a.team_id for a in assignments), Team.owner_id == account_id),
    )
    found_teams = {r[0] for r in await sdb_conn.fetch_all(teams_stmt)}
    if missing_teams := [a.team_id for a in assignments if a.team_id not in found_teams]:
        missing_teams_repr = ",".join(map(str, missing_teams))
        raise GoalMutationError(
            f"Team-s don't exist or access denied: {missing_teams_repr}", HTTPStatus.NOT_FOUND,
        )
