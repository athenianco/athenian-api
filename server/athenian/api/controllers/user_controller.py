from aiohttp import web
from sqlalchemy import select

from athenian.api.controllers.response import ResponseError, response
from athenian.api.models.state.models import UserTeam
from athenian.api.models.web import ForbiddenError
from athenian.api.models.web.team import Team
from athenian.api.request import AthenianWebRequest


async def get_user(request: AthenianWebRequest) -> web.Response:
    """Return details about the current user."""
    await request.user.load_teams(request.sdb)
    return response(request.user)


async def get_team(request: AthenianWebRequest, id: int) -> web.Response:
    """Return details about the current user."""
    user_id = request.user.id
    users = await request.sdb.fetch_all(select([UserTeam]).where(UserTeam.team_id == id))
    for user in users:
        if user[UserTeam.user_id.key] == user_id:
            break
    else:
        return ResponseError(ForbiddenError(
            detail="User %s is not allowed to access team %d" % (user_id, id))).response
    admins = []
    regulars = []
    for user in users:
        l = admins if user[UserTeam.is_admin.key] else regulars
        l.append(user[UserTeam.user_id.key])
    users = await request.auth.get_users(regulars + admins)
    team = Team(regulars=users[:len(regulars)], admins=users[len(regulars):])
    return response(team)
