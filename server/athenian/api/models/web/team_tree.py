from itertools import chain
from typing import Optional

from athenian.api.models.web.base_model_ import Model


class TeamTree(Model):
    """A team with the tree of child teams."""

    id: int
    name: str
    members_count: int
    total_teams_count: int
    total_members_count: int
    children: list["TeamTree"]
    members: list[int]
    total_members: list[int]

    def __init__(
        self,
        id: int,
        name: str,
        children: list["TeamTree"],
        members: list[int],
        members_count: Optional[int] = None,
        total_members: Optional[list[int]] = None,
        total_teams_count: Optional[int] = None,
        total_members_count: Optional[int] = None,
    ):
        """Init the TeamTree."""
        self._id = id
        self._name = name
        self._children = children
        self._members = members

        if members_count is None:
            members_count = len(members)
        self._members_count = members_count

        if total_members is None:
            total_members = sorted(
                set(chain(members, *(child.total_members for child in children))),
            )
        self._total_members = total_members

        if total_teams_count is None:
            total_teams_count = sum(child.total_teams_count for child in children) + len(children)
        self._total_teams_count = total_teams_count

        if total_members_count is None:
            total_members_count = len(total_members)
        self._total_members_count = total_members_count

    def with_children(self, children: list["TeamTree"]) -> "TeamTree":
        """Return a copy of the object with children property replaced.

        Properties depending from `children` retain the original value of the object.
        """
        copy = self.copy()
        copy._children = children
        return copy

    def flatten_team_ids(self) -> list[int]:
        """Return the flatten team id list of this team and all descendants."""
        return [
            self.id,
            *chain.from_iterable(child.flatten_team_ids() for child in self.children),
        ]
