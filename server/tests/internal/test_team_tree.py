from typing import Optional

import pytest

from athenian.api.db import Database
from athenian.api.internal.team import (
    MultipleRootTeamsError,
    RootTeamNotFoundError,
    TeamNotFoundError,
    fetch_teams_recursively,
)
from athenian.api.internal.team_tree import build_team_tree_from_rows
from athenian.api.models.state.models import Team
from athenian.api.models.web import TeamTree
from tests.testutils.db import models_insert
from tests.testutils.factory.state import TeamFactory


class TestBuildTeamTreeFromRows:
    async def test_lonely_team(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, name="TEAM A", members=[1, 2]))

        team_tree = await self._fetch_and_build(sdb, 1, 1)
        assert team_tree.id == 1
        assert team_tree.name == "TEAM A"
        assert team_tree.children == []
        assert team_tree.total_teams_count == 0
        assert team_tree.total_members_count == 2

    async def test_multiple_teams(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, owner_id=2, members=[100]),
            TeamFactory(id=2, members=[1]),
            TeamFactory(id=3, members=[2]),
            TeamFactory(id=4, parent_id=3, members=[3, 4], name="TEAM 4"),
            TeamFactory(id=5, parent_id=3, members=[4, 5]),
            TeamFactory(id=6, parent_id=4, members=[6]),
            TeamFactory(id=7, parent_id=6, members=[7, 8, 9]),
            TeamFactory(id=8, parent_id=4, members=[6, 7]),
        )

        team_tree = await self._fetch_and_build(sdb, 1, 3)

        assert team_tree.id == 3
        assert team_tree.total_teams_count == 5
        assert team_tree.total_members_count == 8
        assert team_tree.members == [2]
        assert team_tree.total_members == [2, 3, 4, 5, 6, 7, 8, 9]
        assert len(team_tree.children) == 2

        assert team_tree.children[0].id == 4
        assert team_tree.children[0].name == "TEAM 4"
        assert team_tree.children[0].total_teams_count == 3
        assert team_tree.children[0].total_members_count == 6
        assert team_tree.children[0].members == [3, 4]
        assert team_tree.children[0].total_members == [3, 4, 6, 7, 8, 9]
        assert len(team_tree.children[0].children) == 2

        assert team_tree.children[1].id == 5
        assert team_tree.children[1].total_teams_count == 0
        assert team_tree.children[1].total_members_count == 2
        assert team_tree.children[1].members == [4, 5]
        assert team_tree.children[1].total_members == [4, 5]
        assert len(team_tree.children[1].children) == 0

        assert (team_6 := team_tree.children[0].children[0]).id == 6
        assert team_6.total_teams_count == 1
        assert team_6.total_members_count == 4
        assert team_6.members == [6]
        assert team_6.total_members == [6, 7, 8, 9]
        assert len(team_6.children) == 1

        assert team_6.children[0].id == 7
        assert team_6.children[0].total_teams_count == 0
        assert team_6.children[0].total_members_count == 3
        assert team_6.children[0].members == [7, 8, 9]
        assert team_6.children[0].total_members == [7, 8, 9]
        assert len(team_6.children[0].children) == 0

        assert (team_8 := team_tree.children[0].children[1]).id == 8
        assert team_8.total_teams_count == 0
        assert team_8.total_members_count == 2
        assert team_8.members == [6, 7]
        assert team_8.total_members == [6, 7]
        assert len(team_8.children) == 0

    async def test_none_root_team(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1), TeamFactory(id=2, parent_id=1))
        team_tree = await self._fetch_and_build(sdb, 1, None)
        assert team_tree.id == 1
        assert len(team_tree.children) == 1
        assert team_tree.children[0].id == 2

    async def test_none_root_team_no_team_existing(self, sdb: Database) -> None:
        with pytest.raises(RootTeamNotFoundError):
            await self._fetch_and_build(sdb, 1, None)

    async def test_none_root_team_multiple_roots(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1), TeamFactory(id=2))
        with pytest.raises(MultipleRootTeamsError):
            await self._fetch_and_build(sdb, 1, None)

    async def test_explicit_team_not_found(self, sdb: Database) -> None:
        with pytest.raises(TeamNotFoundError):
            await self._fetch_and_build(sdb, 1, 1)

    async def _fetch_and_build(
        self,
        sdb: Database,
        account: int,
        team_id: Optional[int],
    ) -> TeamTree:
        team_select = [Team.id, Team.parent_id, Team.name, Team.members]
        team_rows = await fetch_teams_recursively(
            account, sdb, team_select, [team_id] if team_id else None,
        )
        return build_team_tree_from_rows(team_rows, team_id)
