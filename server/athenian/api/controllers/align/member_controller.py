from typing import Optional

from aiohttp import web

from athenian.api.internal.account import (
    get_metadata_account_ids,
    get_user_account_status_from_request,
)
from athenian.api.internal.team import (
    fetch_team_members_recursively,
    get_all_team_members,
    get_root_team,
    get_team_from_db,
)
from athenian.api.models.state.models import Team
from athenian.api.models.web import BadRequestError, Contributor
from athenian.api.request import AthenianWebRequest
from athenian.api.response import ResponseError, model_response


async def list_team_members(
    request: AthenianWebRequest,
    team_id: int,
    recursive: bool,
    account: Optional[int] = None,
) -> web.Response:
    """List the members of a team."""
    if account is not None:
        await get_user_account_status_from_request(request, account)

    sdb, mdb, cache = request.sdb, request.mdb, request.cache

    if team_id == 0:
        if account is None:
            msg = "Parameter account is required with team_id 0"
            raise ResponseError(BadRequestError(detail=msg))
        team = await get_root_team(account, sdb)
    else:
        team = await get_team_from_db(team_id, account, request.uid, sdb)
        account = team[Team.owner_id.name]

    meta_ids = await get_metadata_account_ids(account, sdb, cache)

    if recursive:
        member_ids = await fetch_team_members_recursively(account, sdb, team[Team.id.name])
    else:
        member_ids = team[Team.members.name]
    members = await get_all_team_members(member_ids, account, meta_ids, mdb, sdb, cache)

    def sort_key(member: Contributor) -> tuple[bool, str, str]:
        # first users with a name
        return (member.name is None, (member.name or "").lower(), member.login.lower())

    return model_response(sorted(members.values(), key=sort_key))
