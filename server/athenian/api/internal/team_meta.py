"""Interaction with teams on metadata DB."""
from collections import defaultdict
from typing import Sequence

import sqlalchemy as sa

from athenian.api.db import DatabaseLike
from athenian.api.models.metadata.github import TeamMember


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
            TeamMember.parent_id.in_(meta_team_ids),
            TeamMember.acc_id.in_(meta_ids),
        ),
    )
    members = defaultdict(list)
    for row in member_rows:
        members[row[TeamMember.parent_id.name]].append(row[TeamMember.child_id.name])
    return members
