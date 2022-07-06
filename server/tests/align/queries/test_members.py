from typing import Optional, Sequence

from aiohttp.test_utils import TestClient

from athenian.api.db import Database
from tests.align.utils import (
    align_graphql_request,
    assert_extension_error,
    get_extension_error_obj,
)
from tests.conftest import DEFAULT_HEADERS
from tests.testutils.db import model_insert_stmt, models_insert
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
        extra_headers: Optional[dict] = None,
        fields: Sequence[str] = None,
    ) -> dict:
        body = {
            "query": self._query(fields),
            "variables": {"accountId": account_id, "teamId": team_id},
        }
        headers = {**DEFAULT_HEADERS, **extra_headers} if extra_headers else DEFAULT_HEADERS
        return await align_graphql_request(client, headers=headers, json=body)


class TestMembersErrors(BaseMembersTest):
    async def test_auth_failure(self, client: TestClient, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(id=1, members=[1, 2])))
        headers = {"Authorization": "Bearer invalid"}
        res = await self._request(1, 1, client, headers)
        error = get_extension_error_obj(res)["detail"]
        assert "Error decoding token headers" in error

    async def test_team_not_found(self, client: TestClient) -> None:
        res = await self._request(1, 999, client)
        assert_extension_error(res, "Team 999 not found or access denied")

    async def test_root_team_multiple_roots(self, client: TestClient, sdb: Database):
        for model in (TeamFactory(), TeamFactory()):
            await sdb.execute(model_insert_stmt(model))

        res = await self._request(1, 0, client)
        assert_extension_error(res, "Account 1 has multiple root teams")

    async def test_root_team_no_team_exists(self, client: TestClient, sdb: Database):
        res = await self._request(1, 0, client)
        assert_extension_error(res, "Root team not found or access denied")


class TestMembers(BaseMembersTest):
    async def test_specific_non_root_team_few_fields(
        self,
        client: TestClient,
        sdb: Database,
    ) -> None:
        # members id are from the 6 MB metadata db fixture
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[]),
            TeamFactory(id=2, members=[40078, 39652900], parent_id=1),
        )

        res = await self._request(1, 2, client, fields=("login", "name"))
        assert res["data"]["members"] == [
            {"login": "github.com/leantrace", "name": "Alexander Schamne"},
            {"login": "github.com/reujab", "name": "Christopher Knight"},
        ]

    async def test_specific_non_root_team(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[]),
            TeamFactory(id=2, members=[40078, 39652900], parent_id=1),
        )

        res = await self._request(1, 2, client)
        res_members = res["data"]["members"]
        assert res_members[0]["email"] == "alexander.schamne@gmail.com"
        assert res_members[1]["email"] == "reujab@gmail.com"

    async def test_explicit_root_team(self, client: TestClient, sdb: Database) -> None:
        await sdb.execute(model_insert_stmt(TeamFactory(id=1, members=[40078], parent_id=None)))
        res = await self._request(1, 1, client, fields=("login",))
        logins = [m["login"] for m in res["data"]["members"]]
        assert logins == ["github.com/reujab"]

    async def test_implicit_root_team(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[]),
            TeamFactory(id=2, parent_id=1, members=[3]),
            TeamFactory(id=3, parent_id=1, members=[3, 4]),
        )
        res = await self._request(1, 0, client, fields=("login",))
        assert res["data"]["members"] == []
