"""Interaction with teams on metadata DB."""
from collections import defaultdict
import graphlib
from typing import Any, Iterable, Sequence

import sqlalchemy as sa

from athenian.api.db import DatabaseLike, Row
from athenian.api.models.metadata.github import Team, TeamMember


async def get_meta_teams_members(
    meta_team_ids: Sequence[int],
    meta_ids: Sequence[int],
    mdb: DatabaseLike,
) -> dict[int, list[int]]:
    """Return the members for the given metadata teams.

    A dict of metadata team id => list of members is returned.
    """
    member_rows = await mdb.fetch_all(
        sa.select(TeamMember.parent_id, TeamMember.child_id).where(
            TeamMember.acc_id.in_(meta_ids),
            TeamMember.parent_id.in_(meta_team_ids),
        ),
    )
    members = defaultdict(list)
    for row in member_rows:
        members[row[TeamMember.parent_id.name]].append(row[TeamMember.child_id.name])
    return members


def get_meta_teams_topological_order(meta_team_rows: Iterable[Row]) -> Iterable[Any]:
    """Return the team IDs in topological order according to parentship relation.

    Invalid parents are ignored: a parent not having a row will not be included in the order.
    Raise an error if the graph includes cycles.
    """
    all_team_ids = {r[Team.id.name] for r in meta_team_rows}
    graph: graphlib.TopologicalSorter = graphlib.TopologicalSorter()
    for row in meta_team_rows:
        if (parent_id := row[Team.parent_team_id.name]) is not None and parent_id in all_team_ids:
            graph.add(row[Team.id.name], parent_id)
        else:
            graph.add(row[Team.id.name])
    return graph.static_order()
