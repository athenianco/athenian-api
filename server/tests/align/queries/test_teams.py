from operator import itemgetter
from typing import Sequence

from aiohttp.test_utils import TestClient
import pytest

from athenian.api.align.queries.teams import _fetch_team_tree
from athenian.api.db import Database
from athenian.api.internal.team import (
    MultipleRootTeamsError,
    RootTeamNotFoundError,
    TeamNotFoundError,
)
from tests.align.utils import (
    align_graphql_request,
    assert_extension_error,
    build_fragment,
    build_recursive_fields_structure,
)
from tests.testutils.db import model_insert_stmt, models_insert
from tests.testutils.factory.state import TeamFactory


class TestFetchTeamTree:
    async def test_lonely_team(self, sdb: Database) -> None:
        team_model = TeamFactory(id=1, name="TEAM A", members=[1, 2])
        await sdb.execute(model_insert_stmt(team_model))

        team_tree = await _fetch_team_tree(1, 1, sdb)
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

        team_tree = await _fetch_team_tree(1, 3, sdb)

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
        await models_insert(
            sdb,
            TeamFactory(id=1),
            TeamFactory(id=2, parent_id=1),
        )
        team_tree = await _fetch_team_tree(1, None, sdb)
        assert team_tree.id == 1
        assert len(team_tree.children) == 1
        assert team_tree.children[0].id == 2

    async def test_none_root_team_no_team_existing(self, sdb: Database) -> None:
        with pytest.raises(RootTeamNotFoundError):
            await _fetch_team_tree(1, None, sdb)

    async def test_none_root_team_multiple_roots(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1), TeamFactory(id=2))
        with pytest.raises(MultipleRootTeamsError):
            await _fetch_team_tree(1, None, sdb)

    async def test_explicit_team_not_found(self, sdb: Database) -> None:
        with pytest.raises(TeamNotFoundError):
            await _fetch_team_tree(1, 1, sdb)


class BaseTeamsTest:
    _ALL_FIELDS = ("id", "name", "membersCount", "totalTeamsCount", "totalMembersCount")

    def _query(self, fields, depth):
        fragment = build_fragment("teamFields", "Team", fields)

        recursive_part = build_recursive_fields_structure(["...teamFields"], depth)

        return f"""
        {fragment}
        query ($accountId: Int!, $teamId: Int!) {{
            teams(accountId: $accountId, teamId: $teamId) {{
                {recursive_part}
            }}
        }}
        """

    async def _request(
        self,
        account_id: int,
        team_id: int,
        client: TestClient,
        headers: dict,
        fields: Sequence[str] = _ALL_FIELDS,
        depth: int = 9,
    ) -> dict:
        body = {
            "query": self._query(fields, depth),
            "variables": {"accountId": account_id, "teamId": team_id},
        }
        return await align_graphql_request(client, headers=headers, json=body)


class TestTeamsErrors(BaseTeamsTest):
    async def test_team_not_found(self, client: TestClient, headers: dict) -> None:
        res = await self._request(1, 999, client, headers)
        assert_extension_error(res, "Team 999 not found or access denied")

    async def test_team_account_mismatch(
        self,
        client: TestClient,
        headers: dict,
        sdb: Database,
    ) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(id=1, owner_id=2, members=[1])))
        res = await self._request(1, 1, client, headers)
        assert_extension_error(res, "Team 1 not found or access denied")

    async def test_zero_team_id_no_root_teams(
        self,
        client: TestClient,
        headers: dict,
        sdb: Database,
    ):
        res = await self._request(1, 0, client, headers)
        assert_extension_error(res, "Root team not found or access denied")

    async def test_zero_team_id_multiple_root_teams(
        self,
        client: TestClient,
        headers: dict,
        sdb: Database,
    ):
        await models_insert(sdb, TeamFactory(id=1), TeamFactory(id=2))
        res = await self._request(1, 0, client, headers)
        assert_extension_error(res, "Account 1 has multiple root teams")


class TestTeams(BaseTeamsTest):
    async def test_single_team(self, client: TestClient, headers: dict, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[1]),
            TeamFactory(id=2, parent_id=1, members=[]),
        )
        res = await self._request(1, 2, client, headers)
        teams = res["data"]["teams"]
        assert teams["id"] == 2
        assert teams["membersCount"] == 0
        assert teams["totalTeamsCount"] == 0
        assert teams["totalMembersCount"] == 0
        assert teams["children"] == []

    async def test_limited_depth(self, client: TestClient, headers: dict, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[1]),
            TeamFactory(id=2, parent_id=1, members=[2]),
            TeamFactory(id=3, parent_id=1, members=[3]),
            TeamFactory(id=4, parent_id=3, members=[4]),
        )

        res = await self._request(1, 1, client, headers, depth=2)
        teams = res["data"]["teams"]
        # team 4 is at 3rd level and is not returned, but counts are correct
        assert teams["id"] == 1
        assert teams["membersCount"] == 1
        assert teams["totalTeamsCount"] == 3
        assert teams["totalMembersCount"] == 4

        assert teams["children"][0]["id"] == 2
        assert teams["children"][0]["membersCount"] == 1
        assert teams["children"][0]["totalTeamsCount"] == 0
        assert teams["children"][0]["totalMembersCount"] == 1
        assert "children" not in teams["children"][0]

        assert teams["children"][1]["id"] == 3
        assert teams["children"][1]["membersCount"] == 1
        assert teams["children"][1]["totalTeamsCount"] == 1
        assert teams["children"][1]["totalMembersCount"] == 2
        assert "children" not in teams["children"][1]

    async def test_hierarchy(self, client: TestClient, headers: dict, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[1]),
            TeamFactory(id=2, parent_id=1, members=[2, 3]),
            TeamFactory(id=3, parent_id=1, members=[3, 4]),
            TeamFactory(id=4, parent_id=3, members=[4]),
            TeamFactory(id=5, parent_id=3, members=[4, 5, 6]),
            TeamFactory(id=6, members=[7]),
        )
        res = await self._request(1, 1, client, headers)
        teams = res["data"]["teams"]

        assert teams["id"] == 1
        assert teams["membersCount"] == 1
        assert teams["totalTeamsCount"] == 4
        assert teams["totalMembersCount"] == 6

        team_1_children = sorted(teams["children"], key=itemgetter("id"))

        assert team_1_children[0]["id"] == 2
        assert team_1_children[0]["membersCount"] == 2
        assert team_1_children[0]["totalTeamsCount"] == 0
        assert team_1_children[0]["totalMembersCount"] == 2
        assert team_1_children[0]["children"] == []

        assert team_1_children[1]["id"] == 3
        assert team_1_children[1]["membersCount"] == 2
        assert team_1_children[1]["totalTeamsCount"] == 2
        assert team_1_children[1]["totalMembersCount"] == 4

        team_3_children = sorted(team_1_children[1]["children"], key=itemgetter("id"))

        assert team_3_children[0]["id"] == 4
        assert team_3_children[0]["membersCount"] == 1
        assert team_3_children[0]["totalTeamsCount"] == 0
        assert team_3_children[0]["totalMembersCount"] == 1
        assert team_3_children[0]["children"] == []

        assert team_3_children[1]["id"] == 5
        assert team_3_children[1]["membersCount"] == 3
        assert team_3_children[1]["totalTeamsCount"] == 0
        assert team_3_children[1]["totalMembersCount"] == 3
        assert team_3_children[1]["children"] == []

    async def test_no_team_id(self, client: TestClient, headers: dict, sdb: Database):
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[1]),
            TeamFactory(id=2, parent_id=1, members=[2]),
            TeamFactory(id=3, parent_id=2, members=[2]),
        )
        res = await self._request(1, 0, client, headers)
        teams = res["data"]["teams"]

        assert teams["id"] == 1
        assert len(teams["children"]) == 1
        assert teams["children"][0]["id"] == 2
        assert len(teams["children"][0]["children"]) == 1
        assert teams["children"][0]["children"][0]["id"] == 3
