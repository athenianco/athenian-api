from typing import Sequence

from aiohttp.test_utils import TestClient

from athenian.api.db import Database
from tests.align.utils import align_graphql_request, assert_extension_error, get_extension_error
from tests.testutils.db import model_insert_stmt
from tests.testutils.factory.state import TeamFactory


class BaseMembersTest:
    def _query(self, fields=None):
        if fields is None:
            fields = ("login", "name", "email", "picture", "jiraUser")
        fields_section = "\n".join(fields)
        fragment = f"""
            fragment memberFields on Member {{
               {fields_section}
            }}
        """
        return (
            fragment
            + """
            query ($accountId: Int!, $teamId: Int!) {
              members(accountId: $accountId, teamId: $teamId) {
                ...memberFields
              }
            }
        """
        )

    async def _request(
        self,
        account_id: int,
        team_id: int,
        client: TestClient,
        headers: dict,
        fields: Sequence[str] = None,
    ) -> dict:
        body = {
            "query": self._query(fields),
            "variables": {"accountId": account_id, "teamId": team_id},
        }
        return await align_graphql_request(client, headers=headers, json=body)


class TestMembersErrors(BaseMembersTest):
    async def test_auth_failure(
        self, client: TestClient, headers: dict, sdb: Database,
    ) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(id=1, members=[1, 2])))
        headers["Authorization"] = "Bearer invalid"
        res = await self._request(1, 1, client, headers)
        error = get_extension_error(res)
        assert "Error decoding token headers" in error

    async def test_team_not_found(self, client: TestClient, headers: dict) -> None:
        res = await self._request(1, 999, client, headers)
        assert_extension_error(res, "Team 999 not found or access denied")

    async def test_root_team_multiple_roots(
        self, client: TestClient, headers: dict, sdb: Database,
    ):
        for model in (TeamFactory(), TeamFactory()):
            await sdb.execute(model_insert_stmt(model))

        res = await self._request(1, 0, client, headers)
        assert_extension_error(res, "Account 1 has multiple root teams")

    async def test_root_team_no_team_exists(
        self, client: TestClient, headers: dict, sdb: Database,
    ):
        res = await self._request(1, 0, client, headers)
        assert_extension_error(res, "Root team not found or access denied")


class TestMembers(BaseMembersTest):
    async def test_specific_non_root_team_few_fields(
        self, client: TestClient, headers: dict, sdb: Database,
    ) -> None:
        # members id are from the 6 MB metadata db fixture
        for model in (
            TeamFactory(id=1, members=[]),
            TeamFactory(id=2, members=[40078, 39652900], parent_id=1),
        ):
            await sdb.execute(model_insert_stmt(model))

        res = await self._request(1, 2, client, headers, ("login", "name"))
        assert res["data"]["members"] == [
            {"login": "github.com/leantrace", "name": "Alexander Schamne"},
            {"login": "github.com/reujab", "name": "Christopher Knight"},
        ]

    async def test_specific_non_root_team(
        self, client: TestClient, headers: dict, sdb: Database,
    ) -> None:
        for model in (
            TeamFactory(id=1, members=[]),
            TeamFactory(id=2, members=[40078, 39652900], parent_id=1),
        ):
            await sdb.execute(model_insert_stmt(model))

        res = await self._request(1, 2, client, headers)
        res_members = res["data"]["members"]
        assert res_members[0]["email"] == "alexander.schamne@gmail.com"
        assert res_members[1]["email"] == "reujab@gmail.com"

    async def test_explicit_root_team(
        self, client: TestClient, headers: dict, sdb: Database,
    ) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(id=1, members=[], parent_id=None)))
        res = await self._request(1, 1, client, headers, ("login",))
        # members are retrieved from mdb, result will depened on the 6 MB fixture
        logins = [m["login"] for m in res["data"]["members"]]
        assert "github.com/vmarkovtsev" in logins
        assert "github.com/gkwillie" in logins

    async def test_implicit_root_team(
        self, client: TestClient, headers: dict, sdb: Database,
    ) -> None:
        for model in (
            TeamFactory(id=1, members=[]),
            TeamFactory(id=2, parent_id=1, members=[3]),
            TeamFactory(id=3, parent_id=1, members=[3, 4]),
        ):
            await sdb.execute(model_insert_stmt(model))
        res = await self._request(1, 0, client, headers, ("login",))
        logins = [m["login"] for m in res["data"]["members"]]
        assert "github.com/vmarkovtsev" in logins
        assert "github.com/gkwillie" in logins
