from operator import attrgetter
from typing import Any, Iterable, Optional, Sequence

from athenian.api.db import Row
from athenian.api.internal.team import (
    MultipleRootTeamsError,
    RootTeamNotFoundError,
    TeamNotFoundError,
)
from athenian.api.models.state.models import Team
from athenian.api.models.web import TeamTree
from athenian.api.tracing import sentry_span


@sentry_span
def build_team_tree_from_rows(rows: Sequence[Row], root_team_id: Optional[int]) -> TeamTree:
    """Build the TeamTree for the Team root_team_id starting from the retrieved team rows.

    Team rows can be fetched with `fetch_teams_recursively`, and should at least include
    id, parent_id, name and members as columns.
    """
    nodes, root_team_id = _build_team_tree_nodes_from_rows(rows, root_team_id)
    return _build_team_tree_from_node(nodes[root_team_id], nodes)


@sentry_span
def _build_team_tree_nodes_from_rows(
    team_rows: Iterable[Row],
    root_team_id: Optional[int],
) -> tuple[dict[int, dict[str, Any]], int]:
    """
    Convert the flat Team rows to a tree with nested teams.

    Each node contains the same elements as the original rows + "children" that point at
    the nested teams.
    """
    no_parent_teams = []
    nodes: dict[int, dict[str, Any]] = {}

    # iter the rows building children relations from the parent_id relation, in order
    # to then process teams top-down
    for team_row in team_rows:
        id_ = team_row[Team.id.name]

        parent_id = team_row[Team.parent_id.name]
        if parent_id:
            # setdefault is needed if child is seen before parent, parent dict will be filled later
            nodes.setdefault(parent_id, {})
            nodes[parent_id].setdefault("children", []).append(id_)
        else:
            no_parent_teams.append(id_)

        nodes.setdefault(id_, {})
        nodes[id_].update(team_row)
        nodes[id_].setdefault("children", [])

    # if root_team_id is None all teams have been retrieved, but there should be a single root team
    if root_team_id is None:
        if len(no_parent_teams) > 1:
            raise MultipleRootTeamsError(1)
        if not no_parent_teams:
            raise RootTeamNotFoundError()
        root_team_id = no_parent_teams[0]
    elif root_team_id not in nodes:
        raise TeamNotFoundError(root_team_id)
    return nodes, root_team_id


def _build_team_tree_from_node(team_info: dict[str, Any], all_teams: dict) -> TeamTree:
    children = sorted(
        (
            _build_team_tree_from_node(all_teams[child_id], all_teams)
            for child_id in team_info["children"]
        ),
        key=attrgetter("name"),
    )
    return TeamTree(
        id=team_info["id"],
        name=team_info["name"],
        children=children,
        members=team_info["members"],
    )
