from http import HTTPStatus
from sqlite3 import IntegrityError, OperationalError
from typing import List, Union

from aiohttp import web
from asyncpg import UniqueViolationError
from sqlalchemy import insert

from athenian.api.controllers.account import get_user_account_status
from athenian.api.models.metadata import PREFIXES
from athenian.api.models.state.models import Team
from athenian.api.models.web import BadRequestError, CreatedIdentifier, DatabaseConflict
from athenian.api.models.web.team_create_request import TeamCreateRequest
from athenian.api.models.web.team_update_request import TeamUpdateRequest
from athenian.api.request import AthenianWebRequest
from athenian.api.response import model_response, ResponseError


async def create_team(request: AthenianWebRequest,
                      body: Union[dict, bytes] = None) -> web.Response:
    """Create a team.

    :param body: Team creation request body.

    """
    body = TeamCreateRequest.from_dict(body)
    user = request.uid
    account = body.account
    name = body.name
    async with request.sdb.connection() as sdb_conn:
        try:
            await get_user_account_status(user, account, sdb_conn, request.cache)
            members = _check_members(body.members)
        except ResponseError as e:
            return e.response
        t = Team(owner=account, name=name, members=members).create_defaults()
        try:
            tid = await sdb_conn.execute(insert(Team).values(t.explode()))
        except (UniqueViolationError, IntegrityError, OperationalError) as err:
            return ResponseError(DatabaseConflict(
                detail="Team '%s' already exists: %s: %s" % (name, type(err).__name__, err)),
            ).response
        return model_response(CreatedIdentifier(tid))


async def delete_team(request: AthenianWebRequest, id: int) -> web.Response:
    """Delete a team.

    :param id: Numeric identifier of the team to delete.
    """
    return web.Response(status=HTTPStatus.NOT_IMPLEMENTED)


async def get_team(request: AthenianWebRequest, id: int) -> web.Response:
    """List the team's members. The user must belong to the account that owns the team.

    :param id: Numeric identifier of the team to list.
    """
    return web.Response(status=HTTPStatus.NOT_IMPLEMENTED)


async def list_teams(request: AthenianWebRequest, id: int) -> web.Response:
    """List the teams belonging to the current user.

    :param id: Numeric identifier of the account.
    """
    return web.Response(status=200)


async def update_team(request: AthenianWebRequest, id: int,
                      body: Union[dict, bytes] = None) -> web.Response:
    """Update a team.

    :param id: Numeric identifier of the team to update.
    :param body: Team update request body.
    """
    body = TeamUpdateRequest.from_dict(body)
    return web.Response(status=HTTPStatus.NOT_IMPLEMENTED)


def _check_members(members: List[str]) -> List[str]:
    invalid_members = []
    prefix = PREFIXES["github"]
    for m in members:
        # Very basic check
        splitted = m.split("/")
        if not m.startswith(prefix) or len(splitted) > 2 or not splitted[1]:
            invalid_members.append(m)

    if invalid_members:
        raise ResponseError(BadRequestError(
            detail="Invalid members of the team: %s" % ", ".join(invalid_members)))

    return sorted(set(members))
