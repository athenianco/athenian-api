from http import HTTPStatus
import logging
from typing import Any, Collection, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

import aiomcache
import morcilla
import sqlalchemy as sa
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api.async_utils import gather
from athenian.api.db import DatabaseLike
from athenian.api.internal.jira import load_mapped_jira_users
from athenian.api.models.metadata.github import User
from athenian.api.models.state.models import Team
from athenian.api.models.web import Contributor, GenericError
from athenian.api.response import ResponseError


class TeamNotFoundError(ResponseError):
    """A team was not found."""

    def __init__(self, *team_id: int):
        """Init the TeamNotFoundError."""
        wrapped_error = GenericError(
            type="/errors/teams/TeamNotFound",
            status=HTTPStatus.NOT_FOUND,
            detail=f"Team{'s' if len(team_id) > 1 else ''} {', '.join(map(str, team_id))} "
                   f"not found or access denied",
            title="Team not found",
        )
        super().__init__(wrapped_error)


class RootTeamNotFoundError(ResponseError):
    """A team was not found."""

    def __init__(self):
        """Init the RootTeamNotFoundError."""
        wrapped_error = GenericError(
            type="/errors/teams/TeamNotFound",
            status=HTTPStatus.NOT_FOUND,
            detail="Root team not found or access denied",
            title="Root team not found",
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


async def get_root_team(account_id: int, sdb_conn: DatabaseLike) -> Mapping[Union[int, str], Any]:
    """Return the root team for the account."""
    stmt = sa.select(Team).where(sa.and_(Team.owner_id == account_id, Team.parent_id.is_(None)))
    root_teams = await sdb_conn.fetch_all(stmt)
    if not root_teams:
        raise RootTeamNotFoundError()
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


async def get_all_team_members(gh_user_ids: Iterable[int],
                               account: int,
                               meta_ids: Tuple[int, ...],
                               mdb: morcilla.Database,
                               sdb: morcilla.Database,
                               cache: Optional[aiomcache.Client],
                               ) -> Dict[int, Contributor]:
    """Return contributor objects for given github user identifiers."""
    user_rows, mapped_jira = await gather(
        mdb.fetch_all(sa.select([User]).where(sa.and_(
            User.acc_id.in_(meta_ids),
            User.node_id.in_(gh_user_ids),
        ))),
        load_mapped_jira_users(account, gh_user_ids, sdb, mdb, cache),
    )
    user_by_node = {u[User.node_id.name]: u for u in user_rows}
    all_contributors = {}
    missing = []
    for m in gh_user_ids:
        try:
            ud = user_by_node[m]
        except KeyError:
            missing.append(m)
            c = Contributor(login=str(m))
        else:
            login = ud[User.html_url.name].split("://", 1)[1]
            c = Contributor(login=login,
                            name=ud[User.name.name],
                            email=ud[User.email.name],
                            picture=ud[User.avatar_url.name],
                            jira_user=mapped_jira.get(m))
        all_contributors[m] = c

    if missing:
        logging.getLogger("team.get_all_team_members").error(
            "Some users are missing in %s: %s", meta_ids, missing)
    return all_contributors


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
    one of the `root_team_ids` is ancestor of another.

    Returned columns can be selected with `select_entities`. The ID of the root team
    used to fetch the row is always included as the last column.
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
        Team.id.label(Team.root_id),
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

    rows = await sdb.fetch_all(stmt)
    if missing := set(root_team_ids or set()) - {r[Team.root_id] for r in rows}:
        raise TeamNotFoundError(*missing)
    return rows
