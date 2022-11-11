from operator import itemgetter
from typing import Optional

from athenian.api.db import Database
from tests.testutils.auth import force_request_auth
from tests.testutils.db import models_insert
from tests.testutils.factory.state import TeamFactory, UserAccountFactory
from tests.testutils.requester import Requester


class BaseGetTeamTreeTest(Requester):
    async def _request(
        self,
        team_id: int,
        assert_status: int = 200,
        account: Optional[int] = None,
        user_id: Optional[str] = None,
    ) -> dict:
        path = f"/private/align/team_tree/{team_id}"
        params = {}
        if account is not None:
            params["account"] = str(account)
        with force_request_auth(user_id, self.headers) as headers:
            response = await self.client.request(
                method="GET", path=path, headers=headers, params=params,
            )

        assert response.status == assert_status
        return await response.json()


class TestGetTeamTreeErrors(BaseGetTeamTreeTest):
    async def test_team_not_found(self) -> None:
        res = await self._request(999, 404)
        assert res["type"] == "/errors/teams/TeamNotFound"
        assert res["title"] == "Team not found"
        assert res["detail"] == "Team 999 not found or access denied"

    async def test_team_account_mismatch(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1, owner_id=3, members=[1]))
        res = await self._request(1, 404)
        assert res["type"] == "/errors/teams/TeamNotFound"
        assert res["title"] == "Team not found"
        assert res["detail"] == "Team 1 not found or access denied"

    async def test_team_correct_account_mismatch(self, sdb: Database) -> None:
        await models_insert(sdb, TeamFactory(id=1))
        res = await self._request(1, 404, account=3)
        assert res["type"] == "/errors/AccountNotFound"

    async def test_zero_team_id_account_mismatch(self, sdb: Database) -> None:
        await models_insert(
            sdb, UserAccountFactory(user_id="u0", account_id=2), TeamFactory(id=10),
        )
        res = await self._request(0, 404, account=1, user_id="u0")
        assert res["type"] == "/errors/AccountNotFound"

    async def test_zero_team_id_no_account_param(self, sdb: Database):
        res = await self._request(0, 400)
        assert res["type"] == "/errors/BadRequest"
        assert res["detail"] == "Parameter account is required with team_id 0"

    async def test_zero_team_id_no_root_teams(self, sdb: Database):
        res = await self._request(0, 404, account=1)
        assert res["type"] == "/errors/teams/TeamNotFound"
        assert res["title"] == "Root team not found"
        assert res["detail"] == "Root team not found or access denied"

    async def test_zero_team_id_multiple_root_teams(self, sdb: Database):
        await models_insert(sdb, TeamFactory(id=1), TeamFactory(id=2))
        res = await self._request(0, 500, account=1)
        assert res["type"] == "/errors/teams/MultipleRootTeamsError"
        assert res["detail"] == "Account 1 has multiple root teams"


class TestGetTeamTree(BaseGetTeamTreeTest):
    async def test_single_team(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[1]),
            TeamFactory(id=2, parent_id=1, members=[]),
        )
        res = await self._request(2)
        assert res["id"] == 2
        assert res["members_count"] == 0
        assert res["total_teams_count"] == 0
        assert res["total_members_count"] == 0
        assert res["children"] == []

    async def test_hierarchy(self, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[1]),
            TeamFactory(id=2, parent_id=1, members=[2, 3]),
            TeamFactory(id=3, parent_id=1, members=[3, 4]),
            TeamFactory(id=4, parent_id=3, members=[4]),
            TeamFactory(id=5, parent_id=3, members=[4, 5, 6]),
            TeamFactory(id=6, members=[7]),
        )
        res = await self._request(1)

        assert res["id"] == 1
        assert res["members_count"] == 1
        assert res["total_teams_count"] == 4
        assert res["total_members_count"] == 6

        team_1_children = sorted(res["children"], key=itemgetter("id"))

        assert team_1_children[0]["id"] == 2
        assert team_1_children[0]["members_count"] == 2
        assert team_1_children[0]["total_teams_count"] == 0
        assert team_1_children[0]["total_members_count"] == 2
        assert team_1_children[0]["children"] == []

        assert team_1_children[1]["id"] == 3
        assert team_1_children[1]["members_count"] == 2
        assert team_1_children[1]["total_teams_count"] == 2
        assert team_1_children[1]["total_members_count"] == 4

        team_3_children = sorted(team_1_children[1]["children"], key=itemgetter("id"))

        assert team_3_children[0]["id"] == 4
        assert team_3_children[0]["members_count"] == 1
        assert team_3_children[0]["total_teams_count"] == 0
        assert team_3_children[0]["total_members_count"] == 1
        assert team_3_children[0]["children"] == []

        assert team_3_children[1]["id"] == 5
        assert team_3_children[1]["members_count"] == 3
        assert team_3_children[1]["total_teams_count"] == 0
        assert team_3_children[1]["total_members_count"] == 3
        assert team_3_children[1]["children"] == []

    async def test_no_team_id(self, sdb: Database):
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[1]),
            TeamFactory(id=2, parent_id=1, members=[2]),
            TeamFactory(id=3, parent_id=2, members=[2]),
        )
        res = await self._request(0, account=1)
        assert res["id"] == 1
        assert len(res["children"]) == 1
        assert res["children"][0]["id"] == 2
        assert len(res["children"][0]["children"]) == 1
        assert res["children"][0]["children"][0]["id"] == 3
