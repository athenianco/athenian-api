from typing import Any

from ariadne import QueryType
from graphql import GraphQLResolveInfo

from athenian.api.align.models import Member
from athenian.api.async_utils import gather
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.team import (
    fetch_team_members_recursively,
    get_all_team_members,
    get_root_team,
    get_team_from_db,
)
from athenian.api.models.state.models import Team
from athenian.api.tracing import sentry_span

query = QueryType()


@query.field("members")
@sentry_span
async def resolve_members(
    obj: Any,
    info: GraphQLResolveInfo,
    accountId: int,
    teamId: int,
    recursive: bool,
) -> Any:
    """Serve members()."""
    sdb, mdb, cache = info.context.sdb, info.context.mdb, info.context.cache

    team, meta_ids = await gather(
        # teamId 0 means root team
        get_root_team(accountId, sdb)
        if teamId == 0
        else get_team_from_db(teamId, accountId, None, sdb),
        get_metadata_account_ids(accountId, sdb, cache),
    )

    if recursive:
        member_ids = await fetch_team_members_recursively(accountId, sdb, team[Team.id.name])
    else:
        member_ids = team[Team.members.name]
    members = await get_all_team_members(member_ids, accountId, meta_ids, mdb, sdb, cache)

    def sort_key(member: Member) -> tuple[bool, str, str]:
        # first users with a name
        return (member.name is None, (member.name or "").lower(), member.login.lower())

    # Contributor-s returned by get_all_team_members need to be wrapped in Member
    # to include the `jiraUser` field
    return sorted((Member(**c.to_dict()) for c in members.values()), key=sort_key)
