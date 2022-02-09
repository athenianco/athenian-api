from datetime import datetime, timezone
from itertools import chain
import logging
from sqlite3 import IntegrityError, OperationalError
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from aiohttp import web
import aiomcache
from asyncpg import UniqueViolationError
import morcilla
from sqlalchemy import and_, delete, insert, select, update

from athenian.api import metadata
from athenian.api.async_utils import gather
from athenian.api.auth import disable_default_user
from athenian.api.balancing import weight
from athenian.api.controllers.account import copy_teams_as_needed, get_metadata_account_ids, \
    get_user_account_status
from athenian.api.controllers.jira import load_mapped_jira_users
from athenian.api.controllers.miners.github.user import mine_users
from athenian.api.db import DatabaseLike
from athenian.api.models.metadata.github import User
from athenian.api.models.state.models import Team
from athenian.api.models.web import BadRequestError, Contributor, CreatedIdentifier, \
    DatabaseConflict, NotFoundError, Team as TeamListItem, TeamCreateRequest, TeamUpdateRequest
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
        members = await _check_members(body.members, meta_ids, request.mdb)
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
    user = request.uid
    async with request.sdb.connection() as sdb_conn:
        account = await sdb_conn.fetch_val(select([Team.owner_id]).where(Team.id == id))
        if account is None:
            return ResponseError(NotFoundError("Team %d was not found." % id)).response
        await get_user_account_status(user, account, request.sdb, request.mdb, request.user,
                                      request.app["slack"], request.cache)
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
    user = request.uid
    async with request.sdb.connection() as sdb_conn:
        team = await sdb_conn.fetch_one(select([Team]).where(Team.id == id))
        if team is None:
            return ResponseError(NotFoundError("Team %d was not found." % id)).response
        account = team[Team.owner_id.name]
        await get_user_account_status(user, account, request.sdb, request.mdb, request.user,
                                      request.app["slack"], request.cache)
        meta_ids = await get_metadata_account_ids(account, sdb_conn, request.cache)
    members = await _get_all_team_members(
        [team], account, meta_ids, request.mdb, request.sdb, request.cache)
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
        await get_user_account_status(request.uid, account, request.sdb, request.mdb, request.user,
                                      request.app["slack"], request.cache)
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
    all_members = await _get_all_team_members(
        teams, account, meta_ids, request.mdb, request.sdb, request.cache)
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
    user = request.uid
    name = _check_name(body.name)
    async with request.sdb.connection() as sdb_conn:
        account = await sdb_conn.fetch_val(select([Team.owner_id]).where(Team.id == id))
        if account is None:
            return ResponseError(NotFoundError("Team %d was not found." % id)).response
        await get_user_account_status(user, account, request.sdb, request.mdb, request.user,
                                      request.app["slack"], request.cache)
        if id == body.parent:
            raise ResponseError(BadRequestError(detail="Team cannot be a the parent of itself."))
        meta_ids = await get_metadata_account_ids(account, sdb_conn, request.cache)
        members = await _check_members(body.members, meta_ids, request.mdb)
        await _check_parent(account, body.parent, sdb_conn)
        await _check_parent_cycle(id, body.parent, sdb_conn)
        t = Team(
            owner_id=account,
            name=name,
            members=members,
            parent_id=body.parent,
        ).create_defaults()
        try:
            await sdb_conn.execute(update(Team).where(Team.id == id).values(t.explode()))
        except (UniqueViolationError, IntegrityError, OperationalError) as err:
            return ResponseError(DatabaseConflict(
                detail="Team '%s' already exists: %s: %s" % (name, type(err).__name__, err)),
            ).response
    return web.Response()


def _check_name(name: str) -> str:
    if not name:
        raise ResponseError(BadRequestError("Name of the team cannot be empty."))
    if len(name) > 255:
        raise ResponseError(BadRequestError(
            "Name of the team cannot be longer than 255 Python chars."))
    return name


async def _check_members(members: List[str],
                         meta_ids: Tuple[int, ...],
                         mdb: DatabaseLike,
                         ) -> List[str]:
    to_fetch = set()
    members = set(members)
    for m in members:
        if len(splitted := m.rsplit("/", 1)) == 2:
            to_fetch.add(splitted[1])

    rows = await mdb.fetch_all(select([User.html_url])
                               .where(and_(User.acc_id.in_(meta_ids),
                                           User.login.in_(to_fetch))))
    exist = {r[0].split("://", 1)[1] for r in rows}
    invalid_members = sorted(members - exist)

    if invalid_members or len(members) == 0:
        raise ResponseError(BadRequestError(
            detail="Invalid members of the team: %s" % ", ".join(invalid_members)))

    return sorted(members)


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


async def _get_all_team_members(teams: Iterable[Mapping],
                                account: int,
                                meta_ids: Tuple[int, ...],
                                mdb: morcilla.Database,
                                sdb: morcilla.Database,
                                cache: Optional[aiomcache.Client]) -> Dict[str, Contributor]:
    all_members_prefixed = set(chain.from_iterable([t[Team.members.name] for t in teams]))
    all_members = {m.rsplit("/", 1)[1]: m for m in all_members_prefixed}
    user_by_login = {
        u[User.login.name]: u for u in await mine_users(all_members, meta_ids, mdb, cache)
    }
    mapped_jira = await load_mapped_jira_users(
        account, [u[User.node_id.name] for u in user_by_login.values()], sdb, mdb, cache)
    all_contributors = {}
    for m in all_members:
        try:
            ud = user_by_login[m]
        except KeyError:
            login = all_members[m]
            c = Contributor(login=login)
        else:
            login = ud[User.html_url.name].split("://", 1)[1]
            c = Contributor(login=login,
                            name=ud[User.name.name],
                            email=ud[User.email.name],
                            picture=ud[User.avatar_url.name],
                            jira_user=mapped_jira.get(ud[User.node_id.name]))
        all_contributors[login] = c

    if missing := all_members_prefixed - all_contributors.keys():
        logging.getLogger("%s._get_all_team_members" % metadata.__package__).error(
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
    await get_user_account_status(request.uid, account, request.sdb, request.mdb, request.user,
                                  request.app["slack"], request.cache)
    async with request.sdb.connection() as sdb_conn:
        meta_ids = await get_metadata_account_ids(account, sdb_conn, request.cache)
        async with sdb_conn.transaction():
            await sdb_conn.execute(delete(Team).where(and_(Team.owner_id == account,
                                                           Team.name != Team.BOTS)))
            teams, _ = await copy_teams_as_needed(
                account, meta_ids, sdb_conn, request.mdb, request.cache)
    return await _list_loaded_teams(teams, account, meta_ids, request)
