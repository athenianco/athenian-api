from http import HTTPStatus
from typing import Any, Collection, Mapping, Optional, Sequence, Union

import sqlalchemy as sa
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.db import DatabaseLike
from athenian.api.models.state.models import Team
from athenian.api.models.web import GenericError
from athenian.api.response import ResponseError


async def get_root_team(account_id: int, sdb_conn: DatabaseLike) -> Mapping[Union[int, str], Any]:
    """Return the root team for the account."""
    stmt = sa.select(Team).where(sa.and_(Team.owner_id == account_id, Team.parent_id == None))  # noqa F821
    root_teams = await sdb_conn.fetch_all(stmt)
    if not root_teams:
        raise TeamNotFoundError(0)
    if len(root_teams) > 1:
        raise MultipleRootTeamsError(account_id)
    return root_teams[0]


async def get_team_from_db(
    account_id: int, team_id: int, sdb_conn: DatabaseLike,
) -> Mapping[Union[int, str], Any]:
    """Return a team owned by an account."""
    stmt = sa.select(Team).where(sa.and_(Team.owner_id == account_id, Team.id == team_id))  # noqa F821
    team = await sdb_conn.fetch_one(stmt)
    if team is None:
        raise TeamNotFoundError(team_id)
    return team


async def fetch_teams_recursively(
    account: int,
    sdb: DatabaseLike,
    select_entities: Sequence[InstrumentedAttribute] = (Team.id,),
    root_team_ids: Optional[Collection] = None,
    max_depth: int = None,
) -> Sequence[Mapping[Union[int, str], Any]]:
    """Return the recursively collected list of teams for the account.

    If `root_team_ids` is passed those will be taken as base for the recursion.
    If `root_team_ids` is None all root teams will be taken as base for the recursion.

    The returned list of teams will include duplicates when
    on of the `root_team_ids` is ancestor of another.

    Returned columns can be selected with `select_entities`. The ID of the root team
    used to fetch the row will always be included as last column.
    """
    # a recursive CTE is used to link children teams and track depth

    # Team.id is required inside cte for join even if not requested by caller
    if Team.id not in select_entities:
        cte_select_entities: Sequence[InstrumentedAttribute] = (*select_entities, Team.id)
    else:
        cte_select_entities = select_entities

    # base team is the specified root_team_id if present, else all account's root teams
    recursive_base_where = Team.owner_id == account
    if root_team_ids is None:
        recursive_base_where = sa.and_(recursive_base_where, Team.parent_id.is_(None))
    else:
        recursive_base_where = sa.and_(recursive_base_where, Team.id.in_(root_team_ids))

    recursive_base = sa.select(
        *cte_select_entities,
        sa.cast(1, sa.Integer).label("depth"),
        Team.id.label("root_id"),
    ).where(
        recursive_base_where,
    )
    cte = recursive_base.cte("teams_cte", recursive=True)
    recursive_step_where = sa.and_(Team.parent_id == cte.c.id, Team.owner_id == account)
    # stop recursion on depth if requested by caller
    if max_depth is not None:
        recursive_step_where = sa.and_(recursive_step_where, cte.c.depth < max_depth)
    recursive_step = sa.select(
        *cte_select_entities, cte.c.depth + 1, cte.c.root_id,
    ).where(recursive_step_where)
    cte = cte.union_all(recursive_step)

    # selected entities have the same names but must be taken from cte selectable
    result_select_entities = (getattr(cte.c, entity.name) for entity in select_entities)
    stmt = sa.select(*result_select_entities, cte.c.root_id)
    return await sdb.fetch_all(stmt)


class TeamNotFoundError(ResponseError):
    """A team was not found."""

    def __init__(self, team_id: int):
        """Init the TeamNotFoundError."""
        wrapped_error = GenericError(
            type="/errors/teams/TeamNotFound",
            status=HTTPStatus.NOT_FOUND,
            detail=f"Team {team_id} not found or access denied",
            title="Team not found",
        )
        super().__init__(wrapped_error)


class MultipleRootTeamsError(ResponseError):
    """An account has multiple root teams."""

    def __init__(self, account_id: int):
        """Init the MultipleRootTeamsError."""
        wrapped_error = GenericError(
            type="/errors/teams/MultipleRootTeamsError",
            status=HTTPStatus.INTERNAL_SERVER_ERROR,
            detail=f"Account {account_id} has multiple root teams",
            title="Multiple root teams",
        )
        super().__init__(wrapped_error)
