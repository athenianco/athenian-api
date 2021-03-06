from datetime import datetime, timezone
from itertools import chain
from sqlite3 import IntegrityError, OperationalError
from typing import Dict, Iterable, List, Mapping, Optional, Tuple, Union

from aiohttp import web
import aiomcache
from asyncpg import UniqueViolationError
import databases
from sqlalchemy import delete, insert, select, update

from athenian.api.auth import disable_default_user
from athenian.api.controllers.account import get_metadata_account_ids, get_user_account_status
from athenian.api.controllers.miners.github.users import mine_users
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.metadata.github import User
from athenian.api.models.state.models import Team
from athenian.api.models.web import BadRequestError, CreatedIdentifier, DatabaseConflict, \
    NotFoundError
from athenian.api.models.web.contributor import Contributor
from athenian.api.models.web.team import Team as TeamListItem
from athenian.api.models.web.team_create_request import TeamCreateRequest
from athenian.api.models.web.team_update_request import TeamUpdateRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError
from athenian.api.typing_utils import DatabaseLike


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
        members = _check_members(body.members)
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
        await get_user_account_status(user, account, sdb_conn, request.cache)
        await sdb_conn.execute(update(Team)
                               .where(Team.parent_id == id)
                               .values({Team.parent_id: None,
                                        Team.updated_at: datetime.now(timezone.utc),
                                        Team.members_count: Team.members_count,
                                        Team.members_checksum: Team.members_checksum}))
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
        account = team[Team.owner_id.key]
        await get_user_account_status(user, account, sdb_conn, request.cache)
        meta_ids = await get_metadata_account_ids(account, sdb_conn, request.cache)
    members = await _get_all_members([team], meta_ids, request.mdb, request.cache)
    model = TeamListItem(id=team[Team.id.key],
                         name=team[Team.name.key],
                         parent=team[Team.parent_id.key],
                         members=sorted((members[m] for m in team[Team.members.key]),
                                        key=lambda u: u.login))
    return model_response(model)


async def list_teams(request: AthenianWebRequest, id: int) -> web.Response:
    """List the teams belonging to the current user.

    :param id: Numeric identifier of the account.
    """
    user = request.uid
    account = id
    async with request.sdb.connection() as sdb_conn:
        await get_user_account_status(user, account, sdb_conn, request.cache)
        teams = await sdb_conn.fetch_all(
            select([Team]).where(Team.owner_id == account).order_by(Team.name))
        meta_ids = await get_metadata_account_ids(account, sdb_conn, request.cache)
    all_members = await _get_all_members(teams, meta_ids, request.mdb, request.cache)
    items = [TeamListItem(id=t[Team.id.key],
                          name=t[Team.name.key],
                          parent=t[Team.parent_id.key],
                          members=[all_members[m] for m in t[Team.members.key]])
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
        await get_user_account_status(user, account, sdb_conn, request.cache)
        if id == body.parent:
            raise ResponseError(BadRequestError(detail="Team cannot be a the parent of itself."))
        members = _check_members(body.members)
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


def _check_members(members: List[str]) -> List[str]:
    invalid_members = []
    prefix = PREFIXES["github"]
    for m in members:
        # Very basic check
        splitted = m.split("/")
        if not m.startswith(prefix) or len(splitted) > 2 or not splitted[1]:
            invalid_members.append(m)

    if invalid_members or len(members) == 0:
        raise ResponseError(BadRequestError(
            detail="Invalid members of the team: %s" % ", ".join(invalid_members)))

    return sorted(set(members))


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


async def _get_all_members(teams: Iterable[Mapping],
                           meta_ids: Tuple[int, ...],
                           mdb: databases.Database,
                           cache: Optional[aiomcache.Client]) -> Dict[str, Contributor]:
    prefix = PREFIXES["github"]
    all_members = set(chain.from_iterable([t[Team.members.key] for t in teams]))
    all_members = {m.split("/", 1)[1] for m in all_members if m.startswith(prefix)}
    user_by_login = {
        u[User.login.key]: u for u in await mine_users(all_members, meta_ids, mdb, cache)
    }
    all_contributors = {}
    for m in all_members:
        ud = user_by_login.get(m)
        login = prefix + m
        if ud is not None:
            c = Contributor(login=login,
                            name=ud[User.name.key],
                            email=ud[User.email.key],
                            picture=ud[User.avatar_url.key])
        else:
            c = Contributor(login=login)
        all_contributors[login] = c

    return all_contributors
