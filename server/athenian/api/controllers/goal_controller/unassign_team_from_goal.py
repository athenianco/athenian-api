from http import HTTPStatus

from aiohttp import web

from athenian.api.auth import disable_default_user
from athenian.api.internal.account import request_user_belongs_to_account
from athenian.api.internal.goals.dbaccess import (
    fetch_goal_account,
    unassign_team_from_goal as unassign_team_from_goal_in_db,
    unassign_team_from_goal_recursive,
)
from athenian.api.internal.goals.exceptions import GoalNotFoundError
from athenian.api.models.web import CreatedIdentifier, GoalUnassignTeamRequest
from athenian.api.request import AthenianWebRequest, model_from_body
from athenian.api.response import model_response


@disable_default_user
async def unassign_team_from_goal(
    request: AthenianWebRequest,
    id: int,
    body: dict,
) -> web.Response:
    """Unassign a team from a goal. The Goal is deleted if the goal has no more teams assigned."""
    unassign_req = model_from_body(GoalUnassignTeamRequest, body)
    async with request.sdb.connection() as sdb_conn:
        async with sdb_conn.transaction():
            account = await fetch_goal_account(id, sdb_conn)
            if not await request_user_belongs_to_account(request, account):
                raise GoalNotFoundError(id)
            if unassign_req.recursive:
                still_existing = await unassign_team_from_goal_recursive(
                    account, id, unassign_req.team, sdb_conn,
                )
            else:
                still_existing = await unassign_team_from_goal_in_db(
                    account, id, unassign_req.team, sdb_conn,
                )

    if still_existing:
        return model_response(CreatedIdentifier(id=id))
    return web.Response(status=HTTPStatus.NO_CONTENT)
