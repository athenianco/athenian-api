from aiohttp import web
from sqlalchemy import select

from athenian.api import FriendlyJson
from athenian.api.controllers.response import ResponseError
from athenian.api.models.state.models import UserTeam
from athenian.api.models.web import ForbiddenError


async def get_user(request: web.Request) -> web.Response:
    """Return details about the current user."""
    user = vars(request.user)
    teams = await request.sdb.fetch_all(select([UserTeam]).where(UserTeam.user_id == user["id"]))
    user["teams"] = {x[UserTeam.team_id.key]: x[UserTeam.is_admin.key] for x in teams}
    return web.json_response(user, status=200, dumps=FriendlyJson.dumps)


async def get_team(request: web.Request, id: int) -> web.Response:
    """Return details about the current user."""
    user_id = request.user.id
    users = await request.sdb.fetch_all(select([UserTeam]).where(UserTeam.team_id == id))
    for user in users:
        if user[UserTeam.user_id.key] == user_id:
            break
    else:
        return ResponseError(ForbiddenError(
            detail="User %s is not allowed to access team %d" % (user_id, id))).response
    return web.json_response({}, status=200, dumps=FriendlyJson.dumps)
