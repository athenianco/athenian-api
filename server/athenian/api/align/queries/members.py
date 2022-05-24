from operator import attrgetter
from typing import Any

from ariadne import QueryType
from graphql import GraphQLResolveInfo

from athenian.api.controllers.invitation_controller import get_organizations_members
from athenian.api.controllers.team_controller import get_all_team_members
from athenian.api.internal.account import get_metadata_account_ids
from athenian.api.internal.team import get_root_team, get_team_from_db
from athenian.api.models.state.models import Team
from athenian.api.tracing import sentry_span

query = QueryType()


@query.field("members")
@sentry_span
async def resolve_members(obj: Any, info: GraphQLResolveInfo, accountId: int, teamId: int) -> Any:
    """Serve members()."""
    cache = info.context.cache
    sdb = info.context.sdb
    mdb = info.context.mdb

    # teamId 0 means root team
    if teamId == 0:
        team = await get_root_team(accountId, sdb)
    else:
        team = await get_team_from_db(accountId, teamId, sdb)

    meta_ids = await get_metadata_account_ids(accountId, sdb, cache)

    if team[Team.parent_id.name] is None:
        # root team is ad hoc handled, members column is empty and all members from
        # all organizations are retrieved from mdb
        member_ids = await get_organizations_members(meta_ids, mdb)
    else:
        member_ids = team[Team.members.name]

    members = await get_all_team_members(member_ids, accountId, meta_ids, mdb, sdb, cache)

    # Contributor web model is exactly the same as GraphQL Member
    return sorted(members.values(), key=attrgetter("login"))
