from typing import Any, Optional

from ariadne import QueryType
from graphql import GraphQLResolveInfo

from athenian.api.align.models import GraphQLTeamTree
from athenian.api.db import DatabaseLike
from athenian.api.internal.team import fetch_teams_recursively
from athenian.api.internal.team_tree import build_team_tree_from_rows
from athenian.api.models.state.models import Team
from athenian.api.tracing import sentry_span

query = QueryType()


@query.field("teams")
@sentry_span
async def resolve_teams(obj: Any, info: GraphQLResolveInfo, accountId: int, teamId: int) -> Any:
    """Serve teams."""
    sdb = info.context.sdb

    # teamId 0 means all teams
    actual_team_id = None if teamId == 0 else teamId

    team_tree = await _fetch_team_tree(accountId, actual_team_id, sdb)
    return team_tree.to_dict()


@sentry_span
async def _fetch_team_tree(
    account: int,
    root_team_id: Optional[int],
    sdb: DatabaseLike,
) -> GraphQLTeamTree:
    """Build the GraphQLTeamTree for the Team root_team_id."""
    team_select = [Team.id, Team.parent_id, Team.name, Team.members]
    team_rows = await fetch_teams_recursively(
        account, sdb, team_select, [root_team_id] if root_team_id else None,
    )
    tree = build_team_tree_from_rows(team_rows, root_team_id)
    return GraphQLTeamTree.from_team_tree(tree)
