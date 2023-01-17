from datetime import datetime, timezone
from itertools import chain
from sqlite3 import IntegrityError, OperationalError
from typing import Any, Mapping, Optional, Union

from aiohttp import web
from asyncpg import UniqueViolationError
from sqlalchemy import insert, select, update

from athenian.api.async_utils import gather
from athenian.api.auth import disable_default_user
from athenian.api.balancing import weight
from athenian.api.db import DatabaseLike
from athenian.api.internal.account import (
    get_metadata_account_ids,
    get_user_account_status_from_request,
)
from athenian.api.internal.team import (
    delete_team as db_delete_team,
    get_all_team_members,
    get_root_team,
)
from athenian.api.internal.team_sync import sync_teams
from athenian.api.models.metadata.github import User
from athenian.api.models.state.models import Team
from athenian.api.models.web import (
    BadRequestError,
    CreatedIdentifier,
    DatabaseConflict,
    ForbiddenError,
    NotFoundError,
    Team as TeamListItem,
    TeamCreateRequest,
    TeamUpdateRequest,
)
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response


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
        if not members:
            raise ResponseError(BadRequestError(detail="Empty member list is not allowed."))

        await _check_parent(account, parent, sdb_conn)
        # parent defaults to root team, for retro-compatibility
        if parent is None:
            parent_team_row = await get_root_team(account, sdb_conn)
            parent = parent_team_row[Team.id.name]
        t = Team(owner_id=account, name=name, members=members, parent_id=parent).create_defaults()
        try:
            tid = await sdb_conn.execute(insert(Team).values(t.explode()))
        except (UniqueViolationError, IntegrityError, OperationalError) as err:
            raise ResponseError(
                DatabaseConflict(
                    detail="Team '%s' already exists: %s: %s" % (name, type(err).__name__, err),
                ),
            ) from None
        return model_response(CreatedIdentifier(id=tid))


@disable_default_user
async def delete_team(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete a team.

    :param id: Numeric identifier of the team to delete.
    """
    async with request.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            team = await sdb_conn.fetch_one(select(Team).where(Team.id == id))
            if team is None:
                return ResponseError(NotFoundError(f"Team {id} was not found.")).response
            await get_user_account_status_from_request(request, team[Team.owner_id.name])
            await db_delete_team(team, sdb_conn)

    return web.json_response({})


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
        team[Team.members.name], account, meta_ids, request.mdb, request.sdb, request.cache,
    )
    model = TeamListItem(
        id=team[Team.id.name],
        name=team[Team.name.name],
        parent=team[Team.parent_id.name],
        members=sorted(
            (members[m] for m in team[Team.members.name] if m in members), key=lambda u: u.login,
        ),
    )
    return model_response(model)


async def list_teams(request: AthenianWebRequest, id: int) -> web.Response:
    """List the teams belonging to the current user.

    :param id: Numeric identifier of the account.
    """
    account = id
    async with request.sdb.connection() as sdb_conn:
        await get_user_account_status_from_request(request, account)
        teams, meta_ids = await gather(
            sdb_conn.fetch_all(select([Team]).where(Team.owner_id == account).order_by(Team.name)),
            get_metadata_account_ids(account, sdb_conn, request.cache),
        )
    return await _list_loaded_teams(teams, account, meta_ids, request)


async def _list_loaded_teams(
    teams: list[Mapping[str, Any]],
    account: int,
    meta_ids: tuple[int, ...],
    request: AthenianWebRequest,
) -> web.Response:
    gh_user_ids = set(chain.from_iterable([t[Team.members.name] for t in teams]))
    all_members = await get_all_team_members(
        gh_user_ids, account, meta_ids, request.mdb, request.sdb, request.cache,
    )
    items = [
        TeamListItem(
            id=t[Team.id.name],
            name=t[Team.name.name],
            parent=t[Team.parent_id.name],
            members=[all_members[m] for m in t[Team.members.name] if m in all_members],
        )
        for t in teams
    ]
    return model_response(items)


@disable_default_user
async def update_team(
    request: AthenianWebRequest,
    id: int,
    body: Union[dict, bytes] = None,
) -> web.Response:
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

            new_parent_id = body.parent
            if id == new_parent_id:
                raise ResponseError(
                    BadRequestError(detail="Team cannot be a the parent of itself."),
                )

            if team[Team.parent_id.name] is None:
                if new_parent_id is not None:
                    raise ResponseError(BadRequestError(detail="Cannot set parent for root team."))
            else:
                # null parent means root team as parent, for retro-compatibility
                if new_parent_id is None:
                    root_team = await get_root_team(account, sdb_conn)
                    new_parent_id = root_team[Team.id.name]

            meta_ids = await get_metadata_account_ids(account, sdb_conn, request.cache)
            members = await _resolve_members(body.members, meta_ids, request.mdb)

            if team[Team.parent_id.name] is not None and not members:
                raise ResponseError(BadRequestError(detail="Empty member list is not allowed."))

            await _check_parent(account, new_parent_id, sdb_conn)
            await _check_parent_cycle(id, new_parent_id, sdb_conn)

            values = {
                Team.updated_at.name: datetime.now(timezone.utc),
                Team.name.name: name,
                Team.members.name: members,
                Team.parent_id.name: new_parent_id,
            }
            try:
                await sdb_conn.execute(update(Team).where(Team.id == id).values(values))
            except (UniqueViolationError, IntegrityError, OperationalError) as err:
                return ResponseError(
                    DatabaseConflict(
                        detail="Team '%s' already exists: %s: %s"
                        % (name, type(err).__name__, err),
                    ),
                ).response
    return web.json_response({})


def _check_name(name: str) -> str:
    if not name:
        raise ResponseError(BadRequestError("Name of the team cannot be empty."))
    if len(name) > 255:
        raise ResponseError(
            BadRequestError("Name of the team cannot be longer than 255 Python chars."),
        )
    return name


async def _resolve_members(
    members: list[str],
    meta_ids: tuple[int, ...],
    mdb: DatabaseLike,
) -> list[int]:
    to_fetch = set()
    members = set(members)
    for m in members:
        if len(splitted := m.rsplit("/", 1)) == 2:
            to_fetch.add(splitted[1])

    rows = await mdb.fetch_all(
        select(User.html_url, User.node_id)
        .where(User.acc_id.in_(meta_ids), User.login.in_(to_fetch))
        .order_by(User.node_id),
    )
    exist = {r[0].split("://", 1)[1] for r in rows}
    invalid_members = sorted(members - exist)

    if invalid_members:
        raise ResponseError(
            BadRequestError(detail="Invalid members of the team: %s" % ", ".join(invalid_members)),
        )

    return [r[1] for r in rows]


async def _check_parent(account: int, parent_id: Optional[int], sdb: DatabaseLike) -> None:
    if parent_id is None:
        return
    parent_owner = await sdb.fetch_val(select([Team.owner_id]).where(Team.id == parent_id))
    if parent_owner != account:  # including None
        raise ResponseError(BadRequestError(detail="Team's parent does not exist."))


async def _check_parent_cycle(team_id: int, parent_id: Optional[int], sdb: DatabaseLike) -> None:
    while parent_id not in (visited := {None, team_id}):
        visited.add(
            parent_id := await sdb.fetch_val(select([Team.parent_id]).where(Team.id == parent_id)),
        )
    if parent_id is not None:
        visited.remove(None)
        raise ResponseError(BadRequestError(detail="Detected a team parent cycle: %s." % visited))


@disable_default_user
@weight(0.5)
async def resync_teams(
    request: AthenianWebRequest,
    id: int,
    unmapped: bool = False,
) -> web.Response:
    """Delete all the teams belonging to the account and then clone from GitHub.

    The "Bots" team and the "Root" artificial team will remain intact.
    The rest of the teams will be identical to what's on GitHub.
    Goals assignments will be removed except for "Bots" and "Root" teams, and empty goals
    will be removed too.
    "Root" team will need to be already present for this operation to succeed.

    :param id: Numeric identifier of the account.
    :param unmapped: Value indicating whether we should remove teams that are not backed by GitHub.
    """
    account = id
    if not await get_user_account_status_from_request(request, account):
        raise ResponseError(
            ForbiddenError(
                detail="User %s may not resynchronize teams %d" % (request.uid, account),
            ),
        )
    meta_ids = await get_metadata_account_ids(account, request.sdb, request.cache)
    await sync_teams(account, meta_ids, request.sdb, request.mdb, force=True, unmapped=unmapped)
    teams = await request.sdb.fetch_all(
        select([Team]).where(Team.owner_id == account).order_by(Team.name),
    )
    return await _list_loaded_teams(teams, account, meta_ids, request)
