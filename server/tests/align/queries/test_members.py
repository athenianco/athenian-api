from typing import Optional, Sequence

from aiohttp.test_utils import TestClient

from athenian.api.db import Database
from tests.align.utils import (
    align_graphql_request,
    assert_extension_error,
    get_extension_error_obj,
)
from tests.conftest import DEFAULT_HEADERS
from tests.testutils.db import DBCleaner, model_insert_stmt, models_insert
from tests.testutils.factory import metadata as md_factory
from tests.testutils.factory.state import MappedJIRAIdentityFactory, TeamFactory


class BaseMembersTest:
    _DEFAULT_FIELDS = ("login", "name", "email", "picture", "jiraUser")

    def _query(self, fields: Sequence[str]):
        fields_section = "\n".join(fields)
        fragment = f"""
            fragment memberFields on Member {{
               {fields_section}
            }}
        """
        return (
            fragment
            + """
            query ($accountId: Int!, $teamId: Int!, $recursive: Boolean) {
              members(accountId: $accountId, teamId: $teamId, recursive: $recursive) {
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
        fields: Sequence[str] = _DEFAULT_FIELDS,
        recursive: Optional[bool] = None,
    ) -> dict:
        variables = {"accountId": account_id, "teamId": team_id}
        if recursive is not None:
            variables["recursive"] = recursive
        body = {"query": self._query(fields), "variables": variables}
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

    async def test_team_not_found_recursive(self, client: TestClient) -> None:
        res = await self._request(1, 999, client, recursive=True)
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

    async def test_recursive_param(self, client: TestClient, sdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[39652900]),
            TeamFactory(id=2, parent_id=1, members=[40078]),
        )

        res = await self._request(1, 1, client, recursive=True)
        expected = ["github.com/leantrace", "github.com/reujab"]
        assert [m["login"] for m in res["data"]["members"]] == expected

        res = await self._request(1, 1, client, recursive=False)
        expected = ["github.com/leantrace"]
        assert [m["login"] for m in res["data"]["members"]] == expected

        res = await self._request(1, 1, client)
        expected = ["github.com/leantrace"]
        assert [m["login"] for m in res["data"]["members"]] == expected

    async def test_result_ordering(self, client: TestClient, sdb: Database, mdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[100, 101, 102, 103]),
        )

        async with DBCleaner(mdb) as mdb_cleaner:
            models = [
                md_factory.UserFactory(node_id=100, html_url="https://c", name="Foo Bar"),
                md_factory.UserFactory(node_id=101, html_url="https://a", name=None),
                md_factory.UserFactory(node_id=102, html_url="https://b", name="zz-Top"),
                md_factory.UserFactory(node_id=103, html_url="https://d", name="aa"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb, *models)

            res = await self._request(1, 1, client, recursive=True)

        members = res["data"]["members"]
        assert [m["login"] for m in members] == ["d", "c", "b", "a"]
        assert [m["name"] for m in members] == ["aa", "Foo Bar", "zz-Top", None]

    async def test_jira_user_field(self, client: TestClient, sdb: Database, mdb: Database) -> None:
        await models_insert(
            sdb,
            TeamFactory(id=1, members=[100, 101]),
            MappedJIRAIdentityFactory(github_user_id=100, jira_user_id="200"),
        )

        async with DBCleaner(mdb) as mdb_cleaner:
            models = [
                md_factory.UserFactory(node_id=100),
                md_factory.UserFactory(node_id=101),
                md_factory.JIRAUserFactory(id="200", display_name="My JIRA name"),
            ]
            mdb_cleaner.add_models(*models)
            await models_insert(mdb, *models)
            res = await self._request(1, 1, client)

        members = res["data"]["members"]
        assert [m["jiraUser"] for m in members] == ["My JIRA name", None]
