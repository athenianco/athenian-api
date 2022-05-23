from datetime import datetime, timezone
from http import HTTPStatus
from itertools import chain
import logging
from sqlite3 import IntegrityError, OperationalError
from typing import Any, Collection, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from aiohttp import web
import aiomcache
from asyncpg import UniqueViolationError
import morcilla
import sqlalchemy as sa
from sqlalchemy import and_, delete, insert, select, update
from sqlalchemy.orm.attributes import InstrumentedAttribute

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.auth import disable_default_user
from athenian.api.balancing import weight
from athenian.api.db import DatabaseLike
from athenian.api.internal.account import copy_teams_as_needed, get_metadata_account_ids, \
    get_user_account_status_from_request
from athenian.api.internal.jira import load_mapped_jira_users
from athenian.api.models.metadata.github import User
from athenian.api.models.state.models import Team
from athenian.api.models.web import BadRequestError, Contributor, CreatedIdentifier, \
    DatabaseConflict, ForbiddenError, GenericError, NotFoundError, Team as TeamListItem, \
    TeamCreateRequest, TeamUpdateRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


@disable_default_user
async def create_team(request: AthenianWebRequest, body: dict) -> web.Response:
    """Create a team.

    :param body: Team creation request body.
    """
    body = TeamCreateRequest.from_dict(body)
    account = body.account
    parent = body.parent
    name = _check_name(body.name)
    async with request.sdb.connection() as sdb_conn:
        meta_ids = await get_metadata_account_ids(body.account, sdb_conn, request.cache)
        members = await _resolve_members(body.members, meta_ids, request.mdb)
        await _check_parent(account, parent, sdb_conn)
        t = Team(owner_id=account, name=name, members=members, parent_id=parent).create_defaults()
        try:
            tid = await sdb_conn.execute(insert(Team).values(t.explode()))
        except (UniqueViolationError, IntegrityError, OperationalError) as err:
            raise ResponseError(DatabaseConflict(
                detail="Team '%s' already exists: %s: %s" % (name, type(err).__name__, err)),
            ) from None
        return model_response(CreatedIdentifier(tid))


@disable_default_user
async def delete_team(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete a team.

    :param id: Numeric identifier of the team to delete.
    """
    async with request.sdb.connection() as sdb_conn:
        account = await sdb_conn.fetch_val(select([Team.owner_id]).where(Team.id == id))
        if account is None:
            return ResponseError(NotFoundError("Team %d was not found." % id)).response
        await get_user_account_status_from_request(request, account)
        await sdb_conn.execute(update(Team)
                               .where(Team.parent_id == id)
                               .values({Team.parent_id: None,
                                        Team.updated_at: datetime.now(timezone.utc)}))
        await sdb_conn.execute(delete(Team).where(Team.id == id))
    return web.Response()


async def get_team(request: AthenianWebRequest, id: int) -> web.Response:
    """List the team's members. The user must belong to the account that owns the team.

    :param id: Numeric identifier of the team to list.
    """
    async with request.sdb.connection() as sdb_conn:
        team = await sdb_conn.fetch_one(select([Team]).where(Team.id == id))
        if team is None:
            return ResponseError(NotFoundError("Team %d was not found." % id)).response
        account = team[Team.owner_id.name]
        await get_user_account_status_from_request(request, account)
        meta_ids = await get_metadata_account_ids(account, sdb_conn, request.cache)
    members = await get_all_team_members(
        team[Team.members.name], account, meta_ids, request.mdb, request.sdb, request.cache)
    model = TeamListItem(id=team[Team.id.name],
                         name=team[Team.name.name],
                         parent=team[Team.parent_id.name],
                         members=sorted((members[m] for m in team[Team.members.name]
                                         if m in members),
                                        key=lambda u: u.login))
    return model_response(model)


async def list_teams(request: AthenianWebRequest, id: int) -> web.Response:
    """List the teams belonging to the current user.

    :param id: Numeric identifier of the account.
    """
    account = id
    async with request.sdb.connection() as sdb_conn:
        await get_user_account_status_from_request(request, account)
        teams, meta_ids = await gather(
            sdb_conn.fetch_all(
                select([Team]).where(Team.owner_id == account).order_by(Team.name)),
            get_metadata_account_ids(account, sdb_conn, request.cache),
        )
    return await _list_loaded_teams(teams, account, meta_ids, request)


async def _list_loaded_teams(teams: List[Mapping[str, Any]],
                             account: int,
                             meta_ids: Tuple[int, ...],
                             request: AthenianWebRequest,
                             ) -> web.Response:
    gh_user_ids = set(chain.from_iterable([t[Team.members.name] for t in teams]))
    all_members = await get_all_team_members(
        gh_user_ids, account, meta_ids, request.mdb, request.sdb, request.cache)
    items = [TeamListItem(id=t[Team.id.name],
                          name=t[Team.name.name],
                          parent=t[Team.parent_id.name],
                          members=[all_members[m] for m in t[Team.members.name]
                                   if m in all_members])
             for t in teams]
    return model_response(items)


@disable_default_user
async def update_team(request: AthenianWebRequest, id: int,
                      body: Union[dict, bytes] = None) -> web.Response:
    """Update a team.

    :param id: Numeric identifier of the team to update.
    :param body: Team update request body.
    """
    body = TeamUpdateRequest.from_dict(body)
    name = _check_name(body.name)
    async with request.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            team = await sdb_conn.fetch_one(select(Team).where(Team.id == id).with_for_update())
            if team is None:
                return ResponseError(NotFoundError(f"Team {id} was not found.")).response
            account = team[Team.owner_id.name]
            await get_user_account_status_from_request(request, account)
            if body.parent is None and team[Team.parent_id.name] is not None:
                raise ResponseError(BadRequestError(detail="Team parent cannot be unset."))
            if id == body.parent:
                raise ResponseError(BadRequestError(
                    detail="Team cannot be a the parent of itself.",
                ))
            meta_ids = await get_metadata_account_ids(account, sdb_conn, request.cache)
            members = await _resolve_members(body.members, meta_ids, request.mdb)
            await _check_parent(account, body.parent, sdb_conn)
            await _check_parent_cycle(id, body.parent, sdb_conn)
            values = {
                Team.updated_at.name: datetime.now(timezone.utc),
                Team.name.name: name,
                Team.members.name: members,
                Team.parent_id.name: body.parent,
            }
            try:
                await sdb_conn.execute(update(Team).where(Team.id == id).values(values))
            except (UniqueViolationError, IntegrityError, OperationalError) as err:
                return ResponseError(DatabaseConflict(
                    detail="Team '%s' already exists: %s: %s" % (name, type(err).__name__, err)),
                ).response
    return web.json_response({})


def _check_name(name: str) -> str:
    if not name:
        raise ResponseError(BadRequestError("Name of the team cannot be empty."))
    if len(name) > 255:
        raise ResponseError(BadRequestError(
            "Name of the team cannot be longer than 255 Python chars."))
    return name


async def _resolve_members(members: List[str],
                           meta_ids: Tuple[int, ...],
                           mdb: DatabaseLike,
                           ) -> List[int]:
    to_fetch = set()
    members = set(members)
    for m in members:
        if len(splitted := m.rsplit("/", 1)) == 2:
            to_fetch.add(splitted[1])

    rows = await mdb.fetch_all(select([User.html_url, User.node_id])
                               .where(and_(User.acc_id.in_(meta_ids),
                                           User.login.in_(to_fetch)))
                               .order_by(User.node_id))
    exist = {r[0].split("://", 1)[1] for r in rows}
    invalid_members = sorted(members - exist)

    if invalid_members or len(members) == 0:
        raise ResponseError(BadRequestError(
            detail="Invalid members of the team: %s" % ", ".join(invalid_members)))

    return [r[1] for r in rows]


async def _check_parent(account: int, parent_id: Optional[int], sdb: DatabaseLike) -> None:
    if parent_id is None:
        return
    parent_owner = await sdb.fetch_val(select([Team.owner_id]).where(Team.id == parent_id))
    if parent_owner != account:  # including None
        raise ResponseError(BadRequestError(detail="Team's parent does not exist."))


async def _check_parent_cycle(team_id: int, parent_id: Optional[int], sdb: DatabaseLike) -> None:
    while parent_id not in (visited := {None, team_id}):
        visited.add(parent_id := await sdb.fetch_val(
            select([Team.parent_id]).where(Team.id == parent_id)))
    if parent_id is not None:
        visited.remove(None)
        raise ResponseError(BadRequestError(detail="Detected a team parent cycle: %s." % visited))


async def get_all_team_members(gh_user_ids: Iterable[int],
                               account: int,
                               meta_ids: Tuple[int, ...],
                               mdb: morcilla.Database,
                               sdb: morcilla.Database,
                               cache: Optional[aiomcache.Client],
                               ) -> Dict[int, Contributor]:
    """Return contributor objects for given github user identifiers."""
    user_rows, mapped_jira = await gather(
        mdb.fetch_all(select([User]).where(and_(
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
        logging.getLogger("%s.get_all_team_members" % metadata.__package__).error(
            "Some users are missing in %s: %s", meta_ids, missing)
    return all_contributors


@disable_default_user
@weight(0.5)
async def resync_teams(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete all the teams belonging to the account and then clone from GitHub.

    The "Bots" team will remain intact. The rest of the teams will be identical to what's
    on GitHub.

    :param id: Numeric identifier of the account.
    """
    account = id
    if not await get_user_account_status_from_request(request, account):
        raise ResponseError(ForbiddenError(
            detail="User %s may not resynchronize teams %d" % (request.uid, account)))
    async with request.sdb.connection() as sdb_conn:
        meta_ids = await get_metadata_account_ids(account, sdb_conn, request.cache)
        async with sdb_conn.transaction():
            await sdb_conn.execute(delete(Team).where(and_(Team.owner_id == account,
                                                           Team.name != Team.BOTS)))
            teams, _ = await copy_teams_as_needed(
                account, meta_ids, sdb_conn, request.mdb, request.cache)
    return await _list_loaded_teams(teams, account, meta_ids, request)


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
