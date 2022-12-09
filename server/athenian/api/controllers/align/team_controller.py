from typing import Optional

from aiohttp import web

from athenian.api.db import DatabaseLike
from athenian.api.internal.team import fetch_teams_recursively, get_root_team, get_team_from_db
from athenian.api.internal.team_tree import build_team_tree_from_rows
from athenian.api.models.state.models import Team
from athenian.api.models.web import BadRequestError, TeamTree
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response
from athenian.api.tracing import sentry_span


async def get_team_tree(
    request: AthenianWebRequest,
    team_id: int,
    account: Optional[int] = None,
) -> web.Response:
    """Retrieve the tree describing requested team."""
    # team_id 0 means root team
    if team_id == 0:
        if account is None:
            msg = "Parameter account is required with team_id 0"
            raise ResponseError(BadRequestError(detail=msg))
        team = await get_root_team(account, request.sdb)
    else:
        team = await get_team_from_db(team_id, account, request.uid, request.sdb)
    account = team[Team.owner_id.name]

    actual_team_id = None if team_id == 0 else team_id
    team_tree = await _fetch_team_tree(account, actual_team_id, request.sdb)
    return model_response(team_tree)


@sentry_span
async def _fetch_team_tree(
    account: int,
    root_team_id: Optional[int],
    sdb: DatabaseLike,
) -> TeamTree:
    """Build the TeamTree for the Team root_team_id."""
    team_select = [Team.id, Team.parent_id, Team.name, Team.members]
    team_rows = await fetch_teams_recursively(
        account, sdb, team_select, [root_team_id] if root_team_id else None,
    )
    return build_team_tree_from_rows(team_rows, root_team_id)
