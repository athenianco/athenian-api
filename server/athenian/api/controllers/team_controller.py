from typing import Union

from aiohttp import web

from athenian.api.models.web.team_create_request import TeamCreateRequest
from athenian.api.models.web.team_update_request import TeamUpdateRequest


async def create_team(request: web.Request,
                      body: Union[dict, bytes] = None) -> web.Response:
    """Create a team.

    :param body: Team creation request body.

    """
    body = TeamCreateRequest.from_dict(body)
    return web.Response(status=200)


async def delete_team(request: web.Request, id: int) -> web.Response:
    """Delete a team.

    :param id: Numeric identifier of the team to delete.

    """
    return web.Response(status=200)


async def get_team(request: web.Request, id: int) -> web.Response:
    """List the team's members. The user must belong to the account that owns the team.

    :param id: Numeric identifier of the team to list.

    """
    return web.Response(status=200)


async def list_teams(request: web.Request, id: int) -> web.Response:
    """List the teams belonging to the current user.

    :param id: Numeric identifier of the account.

    """
    return web.Response(status=200)


async def update_team(request: web.Request, id: int,
                      body: Union[dict, bytes] = None) -> web.Response:
    """Update a team.

    :param id: Numeric identifier of the team to update.
    :param body: Team update request body.

    """
    body = TeamUpdateRequest.from_dict(body)
    return web.Response(status=200)
